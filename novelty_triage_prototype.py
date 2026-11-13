
"""
Novelty Triage Prototype
-------------------------
Prototype for the research workflow:

1. OOD / novelty detection using Energy-Based OOD
2. Characterization using Expected-Gradient-style feature attributions
   inspired by:
   https://arxiv.org/abs/1906.10670
3. Uncertainty score using normalized predictive entropy
4. Priority ranking for human attention
5. Traceable pruning history
6. Human review + revisit of earlier stages

IMPORTANT:
- The incoming CSV's `animal` column is treated as hidden ground truth.
  It is NOT used by the model.
- This is a proof-of-concept, not a claim that the chosen weights/thresholds
  are optimal for real deployments.
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, precision_score

# -----------------------------
# Configuration
# -----------------------------
SEED = 42
TRAIN_PATH = Path("animal_training_dataset(2).csv")
INCOMING_PATH = Path("animal_incoming_unknown_dataset(1).csv")

TOP_K_HUMAN = 20
OOD_QUANTILE = 0.95
HUMAN_QUANTILE = 0.80

EPOCHS = 400
LR = 0.01
TEMPERATURE = 1.0
EG_BASELINES = 8
EG_STEPS = 16

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cpu"


# -----------------------------
# Utility
# -----------------------------
def load_data(train_path=TRAIN_PATH, incoming_path=INCOMING_PATH):
    train = pd.read_csv(train_path)
    incoming = pd.read_csv(incoming_path)

    required = {"observation_id", "animal"}
    if not required.issubset(train.columns) or not required.issubset(incoming.columns):
        raise ValueError("Both CSVs must contain observation_id and animal columns.")

    if set(train.columns) != set(incoming.columns):
        raise ValueError(
            "Training and incoming CSVs must use the same feature columns."
        )

    feature_cols = [c for c in train.columns if c not in ["observation_id", "animal"]]

    # Model learns only known classes from training.
    known_classes = sorted(train["animal"].unique().tolist())

    # Incoming labels are retained only for evaluation/demo ground truth.
    incoming["true_unknown"] = ~incoming["animal"].isin(known_classes)

    return train, incoming, feature_cols, known_classes


def make_one_hot(train, incoming, feature_cols):
    combined = pd.concat(
        [train[feature_cols], incoming[feature_cols]],
        axis=0,
        ignore_index=True
    )

    # Feature names are deterministic and easy to map back to original attributes.
    encoded = pd.get_dummies(
        combined,
        columns=feature_cols,
        prefix_sep="==",
        dtype=float
    )

    x_train = encoded.iloc[:len(train)].to_numpy(dtype=np.float32)
    x_incoming = encoded.iloc[len(train):].to_numpy(dtype=np.float32)

    return x_train, x_incoming, list(encoded.columns)


class TinyMLP(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_classifier(x_train, y_train, n_classes):
    model = TinyMLP(x_train.shape[1], n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    X = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y_train, dtype=torch.long, device=DEVICE)

    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    return model


def energy_score(logits, temperature=TEMPERATURE):
    # Standard energy definition: E(x) = -T log sum exp(logit / T)
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def normalized_entropy(probs):
    # H(p) / log(C), producing [0,1] for C classes.
    eps = 1e-12
    h = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1)
    return h / np.log(probs.shape[1])


# -----------------------------
# Expected-Gradient-style characterization
# -----------------------------
def expected_gradient_attributions(
    model,
    x,
    background,
    target_class,
    n_baselines=EG_BASELINES,
    n_steps=EG_STEPS
):
    """
    Expected-Gradient-style attribution:
    sample reference points from a background dataset and sample points
    along paths from each reference to x, averaging gradient * path difference.

    This follows the core Expected Gradients idea used in:
    https://arxiv.org/abs/1906.10670
    """
    model.eval()

    x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE)
    bg = np.asarray(background, dtype=np.float32)

    rng = np.random.default_rng(SEED + int(target_class) + int(x.sum() * 1000) % 10000)
    chosen = rng.choice(len(bg), size=min(n_baselines, len(bg)), replace=False)

    total_attr = torch.zeros_like(x_t)

    for idx in chosen:
        b_t = torch.tensor(bg[idx], dtype=torch.float32, device=DEVICE)
        delta = x_t - b_t

        for _ in range(n_steps):
            alpha = float(rng.uniform(0.0, 1.0))
            z = (b_t + alpha * delta).detach().clone().requires_grad_(True)

            model.zero_grad(set_to_none=True)
            out = model(z.unsqueeze(0))[0, target_class]
            grad = torch.autograd.grad(out, z, retain_graph=False)[0]

            total_attr += grad * delta

    total_attr /= (len(chosen) * n_steps)
    return total_attr.detach().cpu().numpy()


def aggregate_attributions(attributions, encoded_columns, original_feature_cols):
    grouped = {c: 0.0 for c in original_feature_cols}

    for value, encoded_name in zip(attributions, encoded_columns):
        original = encoded_name.split("==", 1)[0]
        if original in grouped:
            grouped[original] += abs(float(value))

    total = sum(grouped.values()) or 1.0
    grouped = {k: v / total for k, v in grouped.items()}
    return grouped


def make_characterization(row, group_importance, feature_cols):
    ordered = sorted(group_importance.items(), key=lambda kv: kv[1], reverse=True)

    parts = []
    for feature, importance in ordered[:3]:
        value = row[feature]
        parts.append(f"{feature}={value}")
    return " + ".join(parts)


# -----------------------------
# Main pipeline
# -----------------------------
def run_pipeline(
    train_path=TRAIN_PATH,
    incoming_path=INCOMING_PATH,
    output_dir=Path(".")
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train, incoming, feature_cols, known_classes = load_data(
        train_path, incoming_path
    )

    # Encode features across both datasets so the feature space matches.
    x_train, x_incoming, encoded_columns = make_one_hot(
        train, incoming, feature_cols
    )

    class_to_idx = {c: i for i, c in enumerate(known_classes)}
    y_train = np.array([class_to_idx[a] for a in train["animal"]], dtype=np.int64)

    model = train_classifier(
        x_train=x_train,
        y_train=y_train,
        n_classes=len(known_classes)
    )

    model.eval()
    with torch.no_grad():
        train_logits = model(torch.tensor(x_train, dtype=torch.float32))
        incoming_logits = model(torch.tensor(x_incoming, dtype=torch.float32))

        train_energy = energy_score(train_logits).numpy()
        incoming_energy = energy_score(incoming_logits).numpy()

        incoming_probs = torch.softmax(incoming_logits, dim=1)
        incoming_entropy = normalized_entropy(incoming_probs).numpy()

        predicted_idx = incoming_probs.argmax(dim=1).numpy()
        predicted_conf = incoming_probs.max(dim=1).values.numpy()

    # Energy OOD: higher energy = more OOD.
    ood_threshold = float(np.quantile(train_energy, OOD_QUANTILE))

    energy_scale = float(np.std(train_energy))
    if energy_scale < 1e-8:
        energy_scale = 1.0

    # Smooth 0..1 visualization score.
    novelty_score = 1.0 / (
        1.0 + np.exp(-(incoming_energy - np.mean(train_energy)) / energy_scale)
    )

    ood_flag = incoming_energy > ood_threshold

    # Characterization using Expected-Gradient-style feature attribution.
    background = x_train[np.random.default_rng(SEED).choice(
        len(x_train),
        size=min(EG_BASELINES, len(x_train)),
        replace=False
    )]

    char_records = []
    for i, row in incoming.iterrows():
        target = int(predicted_idx[i])
        attrs = expected_gradient_attributions(
            model=model,
            x=x_incoming[i],
            background=background,
            target_class=target
        )
        grouped = aggregate_attributions(
            attrs,
            encoded_columns,
            feature_cols
        )

        top_features = sorted(
            grouped.items(),
            key=lambda kv: kv[1],
            reverse=True
        )

        char_records.append({
            "observation_id": row["observation_id"],
            "predicted_class": known_classes[target],
            "characterization": make_characterization(row, grouped, feature_cols),
            "top_feature_1": top_features[0][0],
            "top_feature_1_importance": top_features[0][1],
            "top_feature_2": top_features[1][0],
            "top_feature_2_importance": top_features[1][1],
            "top_feature_3": top_features[2][0],
            "top_feature_3_importance": top_features[2][1],
        })

    characterization = pd.DataFrame(char_records)

    # Priority: novelty + uncertainty.
    # Equal weights keep the proof-of-concept interpretable.
    priority_score = 0.60 * novelty_score + 0.40 * incoming_entropy

    results = incoming.copy()
    results["predicted_class"] = [known_classes[i] for i in predicted_idx]
    results["prediction_confidence"] = predicted_conf
    results["energy"] = incoming_energy
    results["novelty_score"] = novelty_score
    results["ood_candidate"] = ood_flag
    results["uncertainty_score"] = incoming_entropy
    results["priority_score"] = priority_score

    results = results.merge(characterization, on=["observation_id", "predicted_class"], how="left")
    results["priority_rank"] = results["priority_score"].rank(
        method="first", ascending=False
    ).astype(int)

    # Human review queue: among OOD candidates, highest priority first.
    ood_results = results[results["ood_candidate"]].copy()

    if len(ood_results) > 0:
        # Use either top-k or the high-priority quantile, whichever is smaller.
        quantile_cut = float(
            ood_results["priority_score"].quantile(HUMAN_QUANTILE)
        )
        queue = ood_results[
            ood_results["priority_score"] >= quantile_cut
        ].sort_values("priority_score", ascending=False).head(TOP_K_HUMAN)
    else:
        queue = results.nlargest(TOP_K_HUMAN, "priority_score")

    human_ids = set(queue["observation_id"])

    results["final_status"] = np.where(
        results["observation_id"].isin(human_ids),
        "HUMAN_REVIEW",
        np.where(results["ood_candidate"], "DEFERRED", "LOW_PRIORITY")
    )

    results["pruning_reason"] = np.where(
        results["final_status"] == "HUMAN_REVIEW",
        "Selected by OOD + uncertainty priority",
        np.where(
            results["ood_candidate"],
            "OOD candidate retained for later review",
            "Not sufficiently novel for current attention budget"
        )
    )

    queue = results[
        results["final_status"] == "HUMAN_REVIEW"
    ].sort_values("priority_score", ascending=False).copy()

    # Traceable pruning history.
    history_rows = []
    for _, r in results.iterrows():
        oid = r["observation_id"]

        history_rows.append({
            "observation_id": oid,
            "stage": "0_INPUT",
            "status": "RECEIVED",
            "score": np.nan,
            "reason": "Incoming observation"
        })

        history_rows.append({
            "observation_id": oid,
            "stage": "1_OOD",
            "status": "OOD_CANDIDATE" if r["ood_candidate"] else "NOT_OOD",
            "score": r["novelty_score"],
            "reason": f"Energy={r['energy']:.4f}; threshold={ood_threshold:.4f}"
        })

        history_rows.append({
            "observation_id": oid,
            "stage": "2_CHARACTERIZATION",
            "status": "CHARACTERIZED",
            "score": np.nan,
            "reason": r["characterization"]
        })

        history_rows.append({
            "observation_id": oid,
            "stage": "3_UNCERTAINTY",
            "status": "HIGHER_UNCERTAINTY" if r["uncertainty_score"] >= 0.5 else "LOWER_UNCERTAINTY",
            "score": r["uncertainty_score"],
            "reason": f"Predictive entropy={r['uncertainty_score']:.4f}"
        })

        history_rows.append({
            "observation_id": oid,
            "stage": "4_PRIORITY",
            "status": "HUMAN_REVIEW" if oid in human_ids else "DEFERRED",
            "score": r["priority_score"],
            "reason": f"Rank={int(r['priority_rank'])}"
        })

    history = pd.DataFrame(history_rows)

    # Evaluation using hidden ground truth in incoming CSV.
    y_true = results["true_unknown"].astype(int).to_numpy()
    metrics = {}
    if len(np.unique(y_true)) == 2:
        metrics["ood_roc_auc"] = float(roc_auc_score(y_true, results["novelty_score"]))

    top_k = min(TOP_K_HUMAN, len(results))
    topk = results.nlargest(top_k, "priority_score")
    metrics["top_k"] = int(top_k)
    metrics["top_k_unknown_count"] = int(topk["true_unknown"].sum())
    metrics["top_k_unknown_precision"] = float(topk["true_unknown"].mean())

    # Export.
    results.to_csv(output_dir / "triage_results.csv", index=False)
    queue.to_csv(output_dir / "human_review_queue.csv", index=False)
    history.to_csv(output_dir / "pruning_history.csv", index=False)
    characterization.to_csv(output_dir / "characterization.csv", index=False)

    with open(output_dir / "prototype_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Model weights for reproducibility.
    torch.save(model.state_dict(), output_dir / "animal_mlp_energy_model.pt")

    print("\n=== NOVELTY TRIAGE PROTOTYPE ===")
    print(f"Training rows: {len(train)}")
    print(f"Incoming rows: {len(incoming)}")
    print(f"Known classes: {known_classes}")
    print(f"OOD threshold (95th percentile of training energy): {ood_threshold:.4f}")
    print(f"OOD candidates: {int(results['ood_candidate'].sum())}")
    print(f"Human review queue: {len(queue)}")
    print("\nTop human-review observations:")
    cols = [
        "observation_id", "predicted_class", "novelty_score",
        "uncertainty_score", "priority_score", "characterization"
    ]
    print(queue[cols].head(TOP_K_HUMAN).to_string(index=False))

    print("\nEvaluation using hidden incoming labels:")
    print(json.dumps(metrics, indent=2))

    print("\nSaved:")
    for p in [
        "triage_results.csv",
        "human_review_queue.csv",
        "pruning_history.csv",
        "characterization.csv",
        "prototype_metrics.json",
        "animal_mlp_energy_model.pt",
    ]:
        print(" -", output_dir / p)

    return {
        "model": model,
        "train": train,
        "incoming": incoming,
        "results": results,
        "queue": queue,
        "history": history,
        "metrics": metrics,
        "feature_cols": feature_cols,
        "known_classes": known_classes,
        "encoded_columns": encoded_columns,
    }


# -----------------------------
# Human review / revisit tools
# -----------------------------
def review_observation(observation_id, results, history):
    r = results[results["observation_id"] == observation_id]
    if r.empty:
        print(f"Observation {observation_id} not found.")
        return

    r = r.iloc[0]

    print("\n" + "=" * 70)
    print(f"HUMAN REVIEW: {observation_id}")
    print("=" * 70)

    print("\nRaw observation:")
    raw_cols = [
        c for c in results.columns
        if c in ["animal", "no_of_legs", "nose_type", "color", "habitat", "fins", "wings"]
    ]
    print(r[raw_cols].to_string())

    print("\nModel assessment:")
    print(f"Predicted known class : {r['predicted_class']}")
    print(f"Energy                : {r['energy']:.4f}")
    print(f"Novelty score         : {r['novelty_score']:.4f}")
    print(f"Uncertainty score     : {r['uncertainty_score']:.4f}")
    print(f"Priority score        : {r['priority_score']:.4f}")
    print(f"Priority rank         : {int(r['priority_rank'])}")

    print("\nCharacterization:")
    print(r["characterization"])

    print("\nWhy it was pruned / selected:")
    print(r["pruning_reason"])

    print("\nFull pruning history:")
    print(history[history["observation_id"] == observation_id].to_string(index=False))


def revisit_stage(observation_id, history, stage=None):
    h = history[history["observation_id"] == observation_id]
    if h.empty:
        print(f"Observation {observation_id} not found.")
        return

    if stage is not None:
        h = h[h["stage"] == stage]

    print(h.to_string(index=False))
    return h


if __name__ == "__main__":
    # Run this file from the folder containing both CSVs.
    out = run_pipeline(
        train_path=TRAIN_PATH,
        incoming_path=INCOMING_PATH,
        output_dir=Path(".")
    )

    # Example:
    # review_observation("IN_0001", out["results"], out["history"])
    # revisit_stage("IN_0001", out["history"], stage="2_CHARACTERIZATION")
