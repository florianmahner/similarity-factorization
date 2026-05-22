# THINGS-behavior — bias-aware similarity pipeline

End-to-end pipeline that uses the **bias-aware** (Fisher-weighted, item-bias
shrunk) similarity matrix for both rank selection and triplet prediction.
Kept isolated from the normal-RSM pipeline (in
`experiments/analyses/things_behavior/` and
`experiments/datasets/dimensionality/`) so the two stories don't entangle.

## Matrix definition (formal)

Let $c_{ij}$ = number of triplets in which the pair $(i,j)$ was retained as
the most-similar pair, and $s_{ij}$ = number of triplets in which both items
appeared. With Laplace smoothing parameter $\alpha = 1$:

1. **Raw Laplace estimate:** $m_{ij} = (c_{ij} + \alpha) / (s_{ij} + 2\alpha)$.

2. **Fit item baseline** by weighted least squares on logits,
   $\text{logit}(m_{ij}) \approx \beta_0 + b_i + b_j$, with Fisher
   (inverse-variance) weights $w_{ij} = s_{ij}\, m_{ij}(1 - m_{ij})$.
   Baseline: $p_{ij} = \sigma(\beta_0 + b_i + b_j)$.

3. **Shrink** each pair toward the baseline with prior weight $\lambda = 10$:
   $$\tilde m_{ij} = \frac{c_{ij} + \alpha + \lambda\, p_{ij}}{s_{ij} + 2\alpha + \lambda}$$

4. **Unobserved** pairs ($s_{ij} = 0$) fall back to $p_{ij}$. Diagonal = 1.

Implemented in `_matrix.py` -> `build_bias_aware_things_matrix(...)`, which
just wraps the existing `src/similarity/triplet_rsm.py` helpers with the
fixed paper choices ($\alpha=1$, Fisher, $\lambda=10$).

## SRF hyperparameters (fixed throughout this pipeline)

| Stage | rho | max_outer | max_inner | tol |
|---|---|---|---|---|
| Dimensionality CV | 3.0 | 200 | 30 | 0.0 |
| Final triplet prediction | 3.0 | 2000 | 50 | 1e-4 |

`rho = 3.0` matches the normal-RSM CV and the consensus pipeline.

## How to reproduce

```bash
# 1. Rank selection on bias-aware matrix
poetry run python experiments/analyses/things_behavior/bias_aware/dimensionality/run.py

# 2. Triplet prediction at the CV-selected rank
poetry run python experiments/analyses/things_behavior/bias_aware/triplet_prediction/run.py
```

The triplet_prediction script auto-loads the rank written by the CV step.

## Outputs

```
dimensionality/outputs/
├── cross_validation.json   # 5-fold x 10-repeat val_mse curve + argmin_rank
└── log.txt                 # CV progress log
triplet_prediction/outputs/
├── summary.csv             # SRF vs SPoSE vs noise ceiling
├── final_results.csv       # per-seed val_acc
├── selection.json          # rank, srf_kwargs, matrix params
└── triplet_prediction_barplot.{pdf,png}
```
