# API reference

Function-by-function reference for the Recipe-K library in
[`RECIPE_K/src/_common.py`](../src/_common.py). All snippets assume
`RECIPE_K/src/` is on `sys.path`.

## Top-level entry points

### `spectral_pass(S, k_list=None, p_grid=None, B=20, smooth_window=10, cliff_log_jump_min=2.0, cliff_ratio_min=30.0, show_progress=False)`

Runs the bootstrap $I^{\rm proj}$ spectral pass on a symmetric matrix $S$.

**Returns a dict with at minimum:**

| Key | Type | Meaning |
|---|---|---|
| `n` | int | side length of $S$ |
| `k_list` | list[int] | candidate ranks scanned |
| `p_grid` | ndarray | sampling probabilities scanned |
| `kappa_hat` | ndarray (K,) | per-rank leakage rate $\hat\kappa_r$ |
| `evals_ref` | ndarray (Kmax,) | top reference eigenvalues |
| `k_cut` | int | F-statistic spectral changepoint |
| `F_max` | float | F-statistic at the changepoint |
| `k_cliff` | int or None | sharp-cliff detector output |
| `k_smooth_legacy` | int | legacy smoothed-window changepoint |
| `cliff_log_jump`, `cliff_salience` | float | cliff diagnostics |
| `rho`, `alpha`, `flag` | float, float, str | detectability statistics; `flag ∈ {sharp, borderline, smooth}` |
| `Iproj_boot` | ndarray (K, P, B) | per-(rank, p, bootstrap) overlaps |
| `I_med` | ndarray (K, P) | bootstrap median of `Iproj_boot` |
| `trPkS_boot`, `trPkS_med` | ndarray | Rayleigh-trace numerator (since 2026-05-11) |
| `tr_S`, `tr_S_truncated` | float | `np.trace(S)` and $\sum_{a\le K_\max}\lambda_a$ |

### `recipe_K(spectral_out, delta=0.1, k_cv=5, p_max=None, p_floor=0.5, p_floor_mode='adaptive', M_min=2000, p_max_floor=0.95, use_rayleigh_trace=True, apply_pav=True, tr_S_mode='full')`

The model-selection-layer entry point. Consumes a `spectral_pass` output and
returns the calibrated operating point.

**Parameter cheat-sheet (defaults in *italics*):**

| Param | Allowed | Notes |
|---|---|---|
| `delta` | float in (0, 1), *0.10* | spectral fidelity tolerance |
| `k_cv` | int $\ge 2$, *5* | inner CV folds (used in inflation only) |
| `p_max` | float or None | upper safety cap on `p_cv`; auto-set from `n` when None |
| `p_floor` | float, *0.5* | only used when `p_floor_mode='constant'` |
| `p_floor_mode` | `'adaptive' | 'adaptive_legacy' | 'constant'`, *'adaptive'* | Wigner-proxy operator-norm safety floor mode — **RECIPE_K flips this default to 'adaptive'** vs canonical 'constant' to match the manuscript §4.4 prescription |
| `M_min` | int, *2000* | minimum held-out count for the M-curve diagnostic |
| `p_max_floor` | float, *0.95* | lower bound on auto-computed `p_max` |
| `use_rayleigh_trace` | bool, *True* | use exact $\mathrm{tr}(P_kS)$ from `trPkS_med`; else fall back to per-rank-projector proxy |
| `apply_pav` | bool, *True* | PAV-monotonize the deficit curve before inversion |
| `tr_S_mode` | `'full' | 'truncated'`, *'full'* | denominator: `np.trace(S)` vs $\sum_{a\le K_\max}\lambda_a$ |

**Returns** a dict with the operating point and all diagnostics — see
[`overview.md`](overview.md#outputs-returned-by-recipek) for the full field
list.

## Spectral helpers

### `estimate_k_cut_fstat(kappa_hat, k_list, min_seg=2)`

Two-segment F-statistic changepoint of the leakage profile. Returns
`(k_cut, F_max, i_cut, F_arr)`.

### `detect_cliff(kappa_hat, k_list, log_jump_min=2.0, ratio_min=30.0)`

Sharp-cliff detector (single-step log-jump). Returns `(k_cliff, log_jump,
salience)` or `(None, 0, 0)` if no cliff exceeds the thresholds.

### `estimate_k_smooth(kappa_hat, k_list, smooth_window=10)`

Legacy smoothed-window changepoint. Kept for regression checks.

### `detectability_rho(evals_ref, k)`, `power_law_alpha(evals_ref, k, kw=30)`, `detectability_flag(rho, alpha)`

Detectability ratio, power-law tail exponent, and the
`{sharp, borderline, smooth}` flag used by `status` in `recipe_K`.

### `variance_weighted_kappa(kappa_hat, evals_ref, k_list, k_cut)`

Variance-weighted aggregate $\sum_{r\le k}\hat\kappa_r\lambda_r^2 / \sum_{r\le k}\lambda_r^2$.

### `estimate_incoherence(S, k_max)`

Vector-form incoherence parameter $\mu = n\max_a\max_i u_{a,i}^2$ used in
§2.3 of the manuscript.

## CV helpers

### `split_omega_into_folds(M_outer, k_inner, rng)`

Symmetric entrywise fold split of the observed pool $\Omega=\sim M_{\rm outer}$
into `k_inner` equal-sized validation masks.

### `split_bcv_symmetric(M_outer, k_inner, rng, max_rescue_iters=10)`

Symmetric block-CV split for principal-submatrix validation.

### `fit_admm_score(S, holdout_mask, V_mask, M_mask, U_mask, rank, bounds, seed)`

Fit ADMM SymmNMF with `NaN` at holdout entries; return MSE on V, M, U.

### `nested_cv(S, p_outer, ranks, k_inner=5, n_reps=1, seed=0, n_jobs=2)`, `nested_cv_bcv(...)`

Convenience wrappers for entrywise / block nested CV with V/M/U scoring.

### `argmin_safe(curve, ranks)`, `argmin_1se_safe(curve, ranks, sem=None)`

Argmin selection with NaN-handling; one-standard-error reporting.

## Other utilities

### `env_info()`

Returns a dict with numpy/scipy/sklearn versions, Python version, and host
info — useful for archiving alongside experiment outputs.

### `load_v3_dataset(label)`

Loads one of the small canonical v3 datasets: `"10-block n=200"`,
`"DCSBM"`, `"Multiscale"`, `"Temporal"`, `"Horn-fail"`, etc.

## Representation fitters (`_representations.py`)

`fit_softimpute`, `fit_ppca_em`, `fit_wnmf_hals`, `fit_robustpca`,
`fit_symmnmf_admm`, `fit_laplacian_eigenmaps_heat`,
`fit_laplacian_eigenmaps_embed`. All have the same signature
`fit_<name>(S, rank, holdout_mask, seed=0, **kwargs) -> S_hat`.

`score_repr_VM(S, S_hat, V_mask, M_mask)` returns `(mse_V, mse_M)` for a
fitted reconstruction.

## Benchmarks (`_benchmarks.py`)

```python
from _benchmarks import BENCHMARKS
for name, fn in BENCHMARKS:
    print(name, fn(S))
```

Includes Kneedle, eigengap, Donoho-Gavish, ScreeNOT, Horn parallel analysis,
EVBMF, TwoNN, MLE, DANCo, Owen-Perry bicross, and `recipe_K_kcut`.
