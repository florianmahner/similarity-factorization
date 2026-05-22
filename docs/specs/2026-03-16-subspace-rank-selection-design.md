# Subspace Coherence for Rank Selection

**Date:** 2026-03-16
**Location:** `sandbox/coherence/subspace_rank/`
**Status:** Design

## Problem

Ka Chun's projected coherence framework (v2) measures per-eigenvector recoverability
I^proj_k(p) to estimate the rank of a similarity matrix. On continuous-spectrum data
like THINGS behavioral (1854x1854), it fails in two ways:

1. **Kappa changepoint gives k\*=25** -- finds the steepest spectral gap, not the
   rank boundary. Too conservative.
2. **CV curve and count-based activation give k\*=100+** -- the random-direction null
   is too permissive at high p. Both track k_max. Too liberal.

The expected answer (from SRF cross-validation and domain knowledge) is k\*=40-75.

### Root cause

Per-eigenvector I^proj_k is unstable for continuous spectra. When eigenvalues near
rank k are close together, small masking perturbations swap which eigenvectors land
in the top-k set. This produces:

- Non-monotone coherence curves (anomalous bumps at k=45 in THINGS data)
- High bootstrap variance in the degenerate region (k=30-60)
- Artificially low I^proj for individually "swapped-out" eigenvectors even though
  the subspace they belong to is stable

SRF/NMF uses the subspace, not individual eigenvectors. The coherence test should
measure subspace quality.

## Approach: Subspace Coherence S_k

### Definition

Given the cross-Gram matrix G(p) = U_boot(p)^T @ U_ref of shape (Kmax, Kmax):

```
S_k(p) = (1/k) * ||G[:k, :k]||_F^2
       = (1/k) * sum_{i<k} sum_{j<k} (u_i^boot . u_j^ref)^2
       = (1/k) * trace(P_k^boot P_k^ref)
```

This is the mean squared canonical correlation between the top-k reference
eigenspace and the top-k bootstrap eigenspace.

### Properties

- S_k in [0, 1]. Equals 1 iff the two k-subspaces are identical.
- Invariant to rotations within each subspace (immune to eigenvector swapping).
  No sign alignment needed -- squaring G entries eliminates sign ambiguity.
- Empirically decreasing past the true rank: adding noise dimensions dilutes
  the subspace average. Note: this is NOT a mathematical guarantee of monotonicity.
  S_{k+1} = (k*S_k + new_terms) / (k+1) where new_terms come from the (k+1)-th
  row and column of G. For noise dimensions these terms are ~ k/n, which is less
  than S_k for signal k, causing the decrease. But brief local increases are
  possible if a particular eigenvector pair happens to align well.
- For degenerate eigenvalues in the reference: the subspace is unique even when
  individual eigenvectors are not, so S_k is well-defined regardless.
- Computable from G^2 via 2D cumulative sum:
  `block_sum = cumsum(cumsum(G^2, axis=0), axis=1)`
  `S_k = block_sum[k-1, k-1] / k`

### Baseline correction

Random k-dimensional subspaces have E[S_k] = k/n. Corrected statistic:

```
x_k(p) = (S_k(p) - k/n) / (1 - k/n)
```

Maps [k/n, 1] to [0, 1]. Random subspaces give x ~ 0. Perfect recovery gives x = 1.

### Null calibration

For each bootstrap replicate b at each p:

1. Draw B_null random Gaussian matrices V_m of shape (n, k), normalize columns
2. Compute S_k^null(b,m,p) = (1/k) * ||U_boot[:,:k]^T V_m||_F^2
3. Pool across (b, m) to get the null distribution at each (k, p)
4. Threshold tau_k(p) = quantile(S_k^null, alpha) where alpha = 0.95

**Key difference from v2**: v2 draws 1 random direction per null sample. Here we
draw k random directions to test the k-dimensional subspace null. This gives
a tighter null (averaging over k projections reduces variance).

**Computational cost**: Generating the null requires B * P * B_null matrix
multiplies of shape (k, n) @ (n, k). For n=1854, Kmax=120, B=100, P=25,
B_null=20: 50,000 evaluations of a (120, 1854) @ (1854, 120) product.
Each is ~50M FLOPs, total ~2.5 TFLOP. At 50 GFLOP/s this is ~50 seconds,
comparable to the eigensolve cost. Use normalized Gaussian columns (no QR
orthogonalization) to avoid the O(n*k^2) QR cost per sample -- the Frobenius
norm statistic is asymptotically equivalent for non-orthogonalized random
matrices since E[||U^T V||_F^2 / k] = k/n for both cases when n >> k.
B_null=20 (not 50) is sufficient since we pool across B replicates,
giving B*B_null = 2000 total null samples per (k, p) cell.

### Rank selection (k\*)

For each candidate dimension k:

1. Compute the bootstrap distribution of S_k(p_max) across B replicates
2. Compute the lower CI: S_k_lo = quantile(S_k_boot, 0.025)
3. Compare to null threshold: reject H0 if S_k_lo > tau_k(p_max)
4. Apply BH-FDR at q=0.05 across all tested k values
5. k\* = largest k that rejects H0

If no dimension passes the FDR threshold, return k\*=0 with a warning
(the data may have no recoverable subspace structure, or B/B_null is too low).

Secondary diagnostic: kappa changepoint on S_k (same formula as v2 but with S_k
instead of I^proj). This identifies the steepest spectral feature but is not used
as the primary rank estimate.

### Sampling fraction (p\*)

1. Identify signal set K_sig = {k : k <= k\*}
2. Noise reference curve: at each p, take the 90th quantile of S_k upper CIs
   across noise dimensions (k > k\*). If no noise dimensions exist, use the
   upper half of the k_list as fallback.
3. Signal reference curve: the lower CI of S_{k\*}(p) across p (the boundary
   dimension, the hardest signal to recover).
4. p\* = first p where signal_ref(p) > noise_ref(p) + lift_margin.
   lift_margin defaults to 0.0 (signal just needs to exceed noise).
   This is the minimum sampling fraction at which the boundary signal
   dimension separates from the noise floor.

## Architecture

### File structure

```
sandbox/coherence/subspace_rank/
  _subspace_coherence.py    # Core: bootstrap, S_k, null, analysis
  run_things_behavior.py    # THINGS behavioral experiment
  run_comparison.py         # Side-by-side comparison with v2 I^proj
```

### Module structure (_subspace_coherence.py)

**Layer 1 -- Bootstrap computation (reuse from v2)**

Import and reuse from Ka Chun's v2:
- `_symmetrize_with_nan`
- `_prepare_observation_mask`
- `_masked_unbiased_spd_missing_from_uniform`
- `_topk_eigenvectors_exact_warm`
- `_orthonormalize_columns`
- `_randomized_topk_eigenspace_symmetric`

These are pure helpers with no v2-specific assumptions. Import them directly
rather than copying. Import path:

```python
import sys
from pathlib import Path
_V2_DIR = str(Path(__file__).resolve().parent.parent / "kachun" / "v2")
sys.path.insert(0, _V2_DIR)
from _compute_coherence import (
    _symmetrize_with_nan, _prepare_observation_mask, ...
)
```

**Layer 2 -- Worker function (new)**

`_subspace_worker_one_boot(args) -> (b, S_boot_b, Iproj_boot_b, null_S_boot_b, info)`

For one bootstrap replicate b, sweep p_list in increasing order:
- At each p: build masked matrix, eigensolve with warm start
- Compute G = U_boot^T @ U_ref
- Compute G2 = G * G
- Compute S_k via 2D cumulative sum of G2 (new)
- Compute I^proj_k via 1D cumulative sum of G2 diagonal (existing, for comparison)
- Compute null: for each of B_null random orthonormal bases, compute S_k^null
- Return S_boot (K_count, P), Iproj_boot (K_count, P), null_S_boot (K_count, P, B_null)

**Layer 3 -- Main computation function (new)**

`compute_subspace_coherence(S, k_list, p_list, B, B_null, ...) -> result_dict`

Orchestrates:
1. Symmetrize S, build observation mask
2. Compute reference eigenspace
3. Dispatch workers (parallel via joblib)
4. Aggregate results: S_boot (K, P, B), Iproj_boot (K, P, B), null_S_boot (K, P, B, B_null)
5. Compute CIs, tau thresholds, baseline correction

**Layer 4 -- Rank selection function (new)**

`select_rank(result_dict, fdr_q=0.05, ci_level=0.95, ...) -> rank_result`

1. Per-dimension hypothesis test: S_k_lo vs tau_k at p_max
2. BH-FDR correction
3. k\* = last significant dimension
4. Kappa changepoint as secondary diagnostic
5. p\* via signal-noise liftoff

Returns a dataclass with k\*, p\*, per-dimension p-values, FDR results,
kappa curve, signal/noise classification.

**Layer 5 -- Plotting (new)**

`plot_rank_selection(result_dict, rank_result, output_dir) -> list[Path]`

Generates:
1. Heatmap of S_k(p) across k and p
2. S_k curve at p_max with null threshold and k\* line
3. Kappa curve with changepoint
4. Signal-noise liftoff for p\*
5. Comparison panel: S_k vs I^proj_k at p_max (shows why S_k is better)

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| k_list | range(5, 121, 5) | Candidate dimensions to test |
| p_list | linspace(0.05, 0.95, 25) | Sampling fraction grid |
| B | 100 | Bootstrap replicates |
| B_null | 20 | Null samples per (b, p) for S_k (pooled: B*B_null=2000 total) |
| fdr_q | 0.05 | FDR level for rank selection |
| ci_level | 0.95 | Confidence level for bootstrap CIs |
| alpha_tau | 0.95 | Null quantile for threshold |
| use_baseline_correction | True | Apply (S_k - k/n)/(1 - k/n) |
| keep_diag | True | Preserve diagonal during masking |
| solver | "auto" | Eigensolve strategy (lobpcg/eigh) |
| n_jobs | None | Parallel workers |

### Outputs

The main `result_dict` contains:
- `S_boot`: (K, P, B) -- subspace coherence bootstrap samples
- `Iproj_boot`: (K, P, B) -- per-eigenvector coherence (for comparison)
- `null_S_boot`: (K, P, B, B_null) -- null distribution for S_k
- `S_mean`, `S_ci_lo`, `S_ci_hi`: (K, P) -- summary statistics
- `tau_kp`: (K, P) -- null thresholds
- `evals_ref`, `U_ref_K`: reference eigenvalues/vectors
- `p`, `k_list`: grids

The `rank_result` contains:
- `k_star`: int -- estimated rank
- `p_star`: float -- recommended sampling fraction
- `pvalues`: (K,) -- per-dimension p-values
- `significant`: (K,) bool -- FDR-adjusted significance
- `kappa_hat`: (K,) -- kappa curve (secondary diagnostic)
- `kappa_k_cut`: int -- kappa changepoint (secondary)

## Expected behavior on THINGS behavioral

| Statistic | v2 kappa | v2 CV curve | v3 S_k (expected) | SRF CV |
|-----------|----------|-------------|-------------------|--------|
| k\* | 25 | 105 | 50-70 | 80 |
| Stable across k_max? | Yes | No (tracks k_max) | Yes | N/A |
| Monotone in k? | N/A | No (bumps) | Empirically yes | N/A |

## Validation plan

1. **Synthetic block matrix** (clear spectral gap): should recover the true rank
   exactly, matching v2 kappa. Validates correctness.
2. **Synthetic continuous spectrum** (polynomial eigenvalue decay with noise floor):
   known true rank, continuous decay. Tests whether S_k correctly identifies the
   rank boundary when there is no sharp spectral gap. Use
   lambda_k = max(1/k^alpha, sigma^2) with alpha=1, sigma=0.1, true rank ~ 30.
3. **THINGS behavioral** (real data, continuous spectrum): should give k\* in
   [35, 85] and be stable across k_max choices. Success criterion: k\* does not
   change by more than 10 when k_max varies from 80 to 200.
4. **Stability test**: vary k_max from 80 to 200 in steps of 20, verify k\* is
   invariant (the key test that v2 count-based activation and CV curve fail).
5. **Comparison plots**: S_k vs I^proj_k curves side-by-side on the same data,
   showing the monotonicity improvement and tighter null separation.
6. **Sensitivity**: check k\* with B in {20, 50, 100} and B_null in {10, 20, 50}
   to verify the defaults are adequate.

## What this does NOT change

- The bootstrap masking framework (reused as-is)
- The eigensolve machinery (reused as-is)
- pysrf or any third-party code
- Any existing experiments or stable code
