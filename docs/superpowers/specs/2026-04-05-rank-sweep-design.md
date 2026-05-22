# Rank Sweep Analysis -- Triplet Accuracy vs Rank

## Goal

Create a stable analysis task that computes triplet prediction accuracy as a function of SRF rank for the THINGS behavioral dataset. This provides data for a figure panel showing how accuracy scales with dimensionality, with SPoSE and VICE as horizontal baselines.

## Motivation

An existing sandbox run (`sandbox/things/rank_sweep_100pct/`) used a non-canonical validation set (`50pct_part0/test_10.txt`) and only alpha=1.0, producing numbers incomparable to the rest of the paper. This analysis replaces that with canonical evaluation.

## Location

`experiments/analyses/things_behavior/rank_sweep/`

## Design

### Data loading

Reuse `resources.py` to load:
- Train triplets (canonical, from `load_triplets()`)
- Validation triplets (canonical)
- SPoSE embedding (66d)
- VICE embedding (66d)

### RSM construction

Build two RSMs from train triplets via `compute_similarity_matrix_from_triplets()`:
- alpha=0 (no smoothing)
- alpha=1 (Laplace smoothing)

### SRF rank sweep

For each (alpha, rank, seed):
- Fit SRF on RSM via `fit_srf_model()`
- Evaluate triplet prediction accuracy on canonical validation triplets via `compute_triplet_prediction_accuracy()`

### Baselines

Evaluate SPoSE and VICE once on the same canonical validation triplets. These are rank-independent reference points.

### Parallelism

All (alpha, rank, seed) combinations run in parallel via `joblib.Parallel(n_jobs=-1)`.

### Parameters

- **Rank grid**: [5, 10, 15, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 66]
- **Seeds**: 5 (0-4)
- **Alphas**: [0, 1]
- Total SRF fits: 2 alphas x 16 ranks x 5 seeds = 160

### Output

`outputs/results.csv` with columns:
- `model`: "SRF", "SPoSE", or "VICE"
- `alpha`: 0 or 1 (NA for baselines)
- `rank`: dimensionality
- `seed`: random seed
- `val_acc`: triplet prediction accuracy on canonical validation set

### Dependencies

- `experiments/analyses/things_behavior/common.py` -- `compute_triplet_prediction_accuracy`, `fit_srf_model`, `compute_similarity_matrix_from_triplets`
- `experiments/analyses/things_behavior/resources.py` -- canonical data loading

### No plot.py

The figure script in `experiments/figures/plot_things/` will consume this CSV directly.
