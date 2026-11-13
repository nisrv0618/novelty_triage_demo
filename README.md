# Novelty Triage Prototype

A proof-of-concept workflow for identifying potentially novel (out-of-distribution, or OOD) observations and prioritizing them for human attention. The prototype combines energy-based OOD scoring, feature-attribution-based characterization, predictive uncertainty, priority ranking, and a traceable history of triage decisions.

> **Scope:** This is a controlled demonstration using small animal-category CSV datasets. It is not a validated production system, and its saved metrics should not be interpreted as evidence of performance on arbitrary real-world data.

## What the prototype does

1. **Load known-class training data and incoming observations.** The script checks that both CSV files contain `observation_id` and `animal`, and that their columns match.
2. **Train a small neural classifier.** A PyTorch `TinyMLP` is trained on the known classes in the training CSV.
3. **Score potential OOD observations.** The workflow uses an energy-based score derived from classifier logits.
4. **Estimate predictive uncertainty.** Normalized predictive entropy is calculated from the classifier's probabilities.
5. **Characterize observations.** Expected-Gradient-style feature attributions are calculated and aggregated to help describe influential features.
6. **Prioritize and record decisions.** Novelty and uncertainty contribute to a priority score; the script exports ranked results, a human-review queue, and stage-by-stage pruning/triage history.
7. **Support review and revisit.** Helper functions allow an observation's review information and earlier triage stages to be inspected.

## Repository contents

| File | Purpose |
|---|---|
| `novelty_triage_prototype.py` | Main implementation and pipeline |
| `novelty_triage_prototype.ipynb` | Notebook version / interactive exploration |
| `animal_training_dataset(2).csv` | Known-class training dataset |
| `animal_incoming_unknown_dataset(1).csv` | Incoming dataset; `animal` is retained as evaluation ground truth |
| `animal_mlp_energy_model.pt` | Saved PyTorch model weights from the supplied run |
| `triage_results.csv` | Per-observation triage results |
| `human_review_queue.csv` | Prioritized human-review candidates |
| `pruning_history.csv` | Trace of triage/pruning stages |
| `characterization.csv` | Feature-characterization output |
| `prototype_metrics.json` | Saved evaluation metrics |
| `prototype_dashboard.png` | Combined dashboard |
| `plots/` | Individual visualization images |

## Requirements

- Python 3.10 or newer is recommended.
- Dependencies are listed in `requirements.txt`.

## Setup

Clone or download this repository, then open a terminal in the project directory.

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the pipeline

The script expects the two input CSV files to be in the project directory with their supplied filenames. Run:

```bash
python novelty_triage_prototype.py
```

The pipeline writes its result files and model weights to the output directory configured in the script. The supplied CSV outputs, metrics, model weights, and plots are included as artifacts from the provided prototype run; rerunning the pipeline may regenerate them.

## Evaluation snapshot

The supplied `prototype_metrics.json` reports:

| Metric | Saved value |
|---|---:|
| OOD ROC-AUC | 0.8015 |
| Top-K | 20 |
| Unknown observations among top 20 | 19 |
| Top-20 unknown precision | 0.95 |

These figures describe this particular demonstration dataset and run. They do not establish generalization to other datasets or deployment conditions. The incoming CSV's `animal` column is used as hidden ground truth for evaluation and is not intended as an input feature to the classifier.

## Important methodological notes

- The training data defines the known classes; incoming labels are retained for controlled evaluation.
- The human-review queue and the top-20 evaluation metric are distinct outputs. Do not interpret the top-20 precision as the measured accuracy of every item in the review queue.
- The pruning history records triage stages and statuses. The prototype preserves observation-level results rather than physically deleting observations from the source data.
- Energy-based OOD scores and normalized entropy are scoring signals, not guarantees that an observation is genuinely novel.
- Feature attributions are explanatory aids and should not be treated as causal explanations.
- Thresholds, model architecture, and weighting choices are prototype settings, not validated universal defaults.

## Relationship to the astronomy adaptation

This repository documents the animal-dataset prototype. A separate astronomy project adapts the broader triage workflow to simulated LSST/Rubin-style light curves. The astronomy repository should be treated as a domain-specific adaptation, not as the source of this original animal-data demonstration.

## Acknowledgements

The feature-attribution approach is described as Expected-Gradient-style and is inspired by the method in [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1906.10670). This prototype is an implementation inspired by that work, not an official implementation by the paper's authors.
