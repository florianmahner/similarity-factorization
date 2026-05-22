# Extract Coherence Framework into src/coherence/

**Date:** 2026-03-17
**Status:** Design

## Problem

The coherence computation lives in a single 2915-line sandbox file
(`sandbox/coherence/kachun/v2/_compute_coherence.py`) mixing masking helpers,
eigensolvers, bootstrap workers, the main orchestrator, correlation utilities,
cluster consensus analysis, kappa diagnostics, signal-noise liftoff, and plotly
visualization. Multiple sandbox scripts import from it via sys.path hacks.

Additionally, `src/tools/coherence.py` is misnamed (it contains bounds estimation,
not coherence), and `src/coherence.py` is an empty file.

## Goal

Extract the v2 coherence framework into a clean `src/coherence/` package with
one file per logical group. The code must be mathematically equivalent to v2
(same numerical output for same inputs). Follow project coding standards
(snake_case, pathlib, type hints, atomic functions, no complex one-liners).

## File Structure

```
src/coherence/
    __init__.py        # Public API exports
    masking.py         # NaN-aware Bernoulli masking (~110 lines)
    eigensolve.py      # Top-k eigensolvers with warm starts (~200 lines)
    workers.py         # Bootstrap worker functions (~180 lines)
    compute.py         # Main compute_coherence orchestrator (~300 lines)
    analysis.py        # Activation, kappa, baseline correction (~150 lines)
    cluster.py         # Cluster consensus across p (~400 lines)
    _correlation.py    # Pearson, Spearman, MI matrices (~300 lines)
```

Also:
- Rename `src/tools/coherence.py` to `src/tools/bounds.py`
- Delete empty `src/coherence.py`
- Update all imports that referenced the old paths

## Mapping: v2 functions to new locations

### masking.py

| v2 function | New name | Lines |
|-------------|----------|-------|
| `_symmetrize_with_nan` | `symmetrize_with_nan` | 29-60 |
| `_prepare_observation_mask` | `prepare_observation_mask` | 62-92 |
| `_masked_unbiased_spd_missing` | `masked_bootstrap_sample` | 104-139 |
| `_masked_unbiased_spd_missing_from_uniform` | `masked_bootstrap_from_uniform` | 217-251 |

Drop: `_apply_unbiased_missingness_scaling` (line 95, unused)

### eigensolve.py

| v2 function | New name | Lines |
|-------------|----------|-------|
| `_try_import_eigsh` | `_try_import_eigsh` | 20-26 |
| `_try_import_lobpcg` | `_try_import_lobpcg` | 208-214 |
| `_topk_eigenvectors` | `topk_eigenvectors` | 142-156 |
| `_topk_eigenvectors_exact_warm` | `topk_eigenvectors_warm` | 254-336 |
| `_randomized_topk_eigenspace_symmetric` | `randomized_topk_eigenspace` | 159-187 |
| `_orthonormalize_columns` | `orthonormalize_columns` | 190-204 |

### workers.py

| v2 function | New name | Lines |
|-------------|----------|-------|
| `_eig_worker_one_p` | `worker_one_p` | 432-513 |
| `_fast_iproj_worker_one_boot` | `worker_one_boot` | 339-425 |

Workers import from masking and eigensolve within the package.

### compute.py

| v2 function | New name | Lines |
|-------------|----------|-------|
| `compute_incremental_coherence_multi_k_eig_anisotropic` | `compute_coherence` | 520-1007 |

This is the main entry point. It:
1. Symmetrizes S, builds observation mask
2. Computes reference eigenspace
3. Dispatches workers in parallel
4. Aggregates bootstrap results
5. Computes CIs, null thresholds, baseline correction
6. Returns result dict

Drop: `test_fast_coherence` (lines 1010-1318, variant not on main path)

### analysis.py

| v2 function | New name | Lines |
|-------------|----------|-------|
| `_baseline_correct_array` | `baseline_correct` | 1451-1465 |
| `_estimate_kappa_hat` | `estimate_kappa` | 2429-2462 |
| `_smooth_median` | `smooth_median` | 2469-2480 |
| `kappa_changepoint` | `kappa_changepoint` | 2482-2543 |
| `_bh_fdr_reject` | `bh_fdr` | 2368-2395 |
| `_find_first_run_ge_threshold` | `find_first_run_above` | 2581-2595 |
| (new) | `per_component_activation` | (extract from compute.py lines 729-747) |

Drop:
- `eps_from_signal_quantile` (line 2545, tied to v3 liftoff)
- `analyze_iproj_signal_alignment_v3` (line 2598, doesn't work for continuous spectra)
- `IprojAnalysisResult` (line 2576, tied to v3)
- `_try_beta_cdf` (line 2410, unused beta_null mode)
- `_quantile`, `_safe_log`, `_mad` (trivial wrappers)
- `_find_first_p_ge_threshold`, `_default_select_k_for_curves` (plotting helpers)

### cluster.py

| v2 function | New name | Lines |
|-------------|----------|-------|
| `compute_trend_correlation` | `compute_trend_correlation` | 1468-1512 |
| `compute_cumulative_trend_correlation` | `compute_cumulative_trend_correlation` | 1515-1577 |
| `analyze_cluster_consensus_across_p` | `cluster_consensus_across_p` | 1696-2056 |

Drop: `plot_consensus_from_p_slider` (lines 2059-2357, plotly visualization stays in sandbox)

### _correlation.py

| v2 function | New name | Lines |
|-------------|----------|-------|
| `_rankdata_average_1d` | `rankdata_average` | 1324-1346 |
| `_pearson_corr_matrix` | `pearson_corr_matrix` | 1349-1366 |
| `_spearman_corr_matrix` | `spearman_corr_matrix` | 1369-1373 |
| `_mutual_info_matrix` | `mutual_info_matrix` | 1376-1436 |
| `_compute_all_trend_similarity_matrices` | `compute_trend_similarity` | 1439-1448 |
| `_pearson_corr` | `pearson_corr` | 1580-1590 |
| `_spearman_corr` | `spearman_corr` | 1593-1596 |
| `_fd_bins` | `fd_bins` | 1598-1611 |
| `_mutual_info_discrete` | `mutual_info_discrete` | 1614-1662 |
| `_mutual_info_matrix_pairwise` | `mutual_info_matrix_pairwise` | 1665-1675 |
| `_get_pair_metric` | `get_pair_metric` | 1678-1693 |

## What stays in sandbox

- `test_fast_coherence` -- variant, not main path
- `plot_consensus_from_p_slider` -- plotly interactive visualization
- `analyze_iproj_signal_alignment_v3` -- kappa liftoff (doesn't work for continuous spectra)
- `IprojAnalysisResult`, `eps_from_signal_quantile` -- tied to v3 liftoff
- `_try_beta_cdf` -- unused beta_null mode
- All plotly imports and visualization code

## Additional changes

1. **Rename** `src/tools/coherence.py` to `src/tools/bounds.py`
2. **Delete** empty `src/coherence.py`
3. **Update imports** in:
   - `sandbox/coherence/subspace_rank/_subspace_coherence.py` (imports v2 helpers)
   - Any other files that import from `src.tools.coherence`

## Verification strategy

For each extracted function, write a test that:
1. Calls the v2 original and the new version with identical inputs
2. Asserts np.allclose on outputs (atol=1e-12)

Run the full coherence diagnostic on THINGS behavioral using the new package
and verify the activation_p profile matches the v2 output exactly.

## Coding standards to apply during extraction

- Remove leading underscores from public functions
- Add Python 3.12+ type hints to all function signatures
- Use `NDArray[np.floating]` for array types
- Keep docstrings but convert to numpy style if not already
- Replace `os.cpu_count()` patterns with a `_default_n_jobs()` helper
- No plotly imports anywhere in src/
- No `sys.path` manipulation
- Variables always lowercase
