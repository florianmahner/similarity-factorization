# Joint Bandwidth-Rank Selection via Kappa Sharpness

**Date:** 2026-03-27
**Location:** `experiments/sandbox/things/bandwidth_kappa_stability/run.py`
**Dataset:** DINOv3 features (1854 images)

## Problem

The RBF kernel bandwidth and factorization rank are circularly dependent:
- Bandwidth determines the similarity matrix S and its eigenspectrum
- The eigenspectrum determines the kappa rank estimate k*
- The rank determines factorization quality (stability)
- But the "right" bandwidth is the one that produces a stable factorization

The median heuristic (sigma = median of pairwise distances) is a common default but
empirically gives poor factorization stability (mean reliability ~0.48 at 1.0x median
vs ~0.95 at 0.4x median for CLIP RN50 features).

## Goal

Test whether **kappa changepoint sharpness** -- a quantity computable without any
factorization -- predicts which bandwidth produces the most stable SRF embedding.
If it does, bandwidth selection can be folded into the existing kappa pipeline at
negligible extra cost.

## Mathematical Framework

### Setup

Features X of shape (n, d). For bandwidth multiplier m:

    sigma(m) = m * median(||x_i - x_j||)
    S(m)_ij  = exp(-||x_i - x_j||^2 / (2 * sigma(m)^2))

### Stage 1: Kappa changepoint sharpness

For each m in {0.2, 0.4, 0.6, 0.8, 1.0}:

1. Build S(m) from features X
2. Compute coherence surface I(k, p) via eigenspace coherence with bootstrap
   masking (B=20, B_null=15, n_p=20, p in [0.05, 0.95])
3. Estimate kappa_hat(k) = median_p[ (1 - I_med(k,p)) * p/(1-p) ] over the
   high-p band (p >= quantile(p_list, 0.85))
4. Find k*(m) via largest-jump changepoint on the smoothed kappa curve
5. Compute the changepoint signal-to-noise ratio:

       kappa_smooth = median_filter(kappa_hat, window=5)
       delta[k]     = kappa_smooth[k+1] - kappa_smooth[k]
       SNR(m)       = max_k |delta[k]| / MAD(kappa_hat)

   where MAD(x) = median(|x - median(x)|).

   SNR(m) measures how much the largest jump in the kappa curve stands out
   relative to the overall variability. This is analogous to a z-score: a
   bandwidth that produces a clean signal/noise transition will have high SNR,
   while a bandwidth that blurs the transition or creates spurious structure
   will have low SNR.

   The MAD normalization is robust to outliers and scale-free, making it
   comparable across bandwidths that produce kappa curves at different scales.

Select m* = argmax_m SNR(m) as the kappa-optimal bandwidth.

### Stage 2: Profile stability (validation)

For each m in {0.2, 0.4, 0.6, 0.8, 1.0}:

1. Using S(m) and k*(m) from Stage 1
2. Fit 5 independent SRF models at rank k*(m) with different random seeds
3. For each pair (i, j) of the C(5,2) = 10 run pairs:
   a. Compute absolute correlation matrix: C_ab = |corr(W_i[:, a], W_j[:, b])|
   b. Find optimal column assignment via Hungarian algorithm:
      assignment = linear_sum_assignment(-C)
   c. Extract matched correlations r_d for each dimension d
4. Per-dimension reliability: r_d = mean over all 10 pairs of the matched
   correlation for dimension d (averaged in Fisher-z space for correctness)
5. Stability(m) = mean_d(r_d)

### Stage 3: Validation criterion

Compare rankings of bandwidths by SNR(m) vs Stability(m).
Success criteria:
- argmax SNR(m) == argmax Stability(m), or
- Spearman rank correlation between SNR and Stability > 0.8

## Parameters

| Parameter      | Value                      | Rationale                              |
|----------------|----------------------------|----------------------------------------|
| sigma_mults    | [0.2, 0.4, 0.6, 0.8, 1.0] | Covers sharp to standard median        |
| B              | 20                         | Reduced from 50; sufficient for shape  |
| B_null         | 15                         | Reduced from 30                        |
| n_p            | 20                         | Reduced from 25                        |
| k_max          | 100                        | Standard                               |
| n_srf_runs     | 5                          | User-specified                         |
| smooth_window  | 5                          | Matches kappa_changepoint default      |
| hi_band_quant  | 0.85                       | Matches kappa default                  |

## Outputs

- `results.csv`: one row per bandwidth multiplier
  - Columns: sigma_mult, sigma, k_star, snr, stability_mean, stability_min,
    sim_mean, sim_min
- `kappa_curves.npz`: kappa_hat arrays keyed by sigma_mult (for diagnostic plotting)
- `stability_per_dim.npz`: per-dimension reliability arrays keyed by sigma_mult
- Console summary table

## Data Source

DINOv3 features: `data/features/dinov3/dinov3_features.npy`
