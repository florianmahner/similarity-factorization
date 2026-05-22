# Subspace Rank Selection Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement subspace coherence S_k for rank selection as an improvement over per-eigenvector I^proj_k, producing k* and p* for similarity matrix factorization.

**Architecture:** A single computation module (`_subspace_coherence.py`) that reuses bootstrap masking and eigensolve helpers from Ka Chun's v2 code, adds the S_k statistic (block Frobenius norm of cross-Gram matrix), null calibration via random Gaussian subspaces, and rank selection via FDR-corrected hypothesis testing. Two run scripts exercise it on THINGS behavioral data and compare against v2.

**Tech Stack:** NumPy, SciPy (eigsh/lobpcg), joblib (parallelism), matplotlib/seaborn (plots via `src/utils/figure_theme`). Reuses v2 helpers from `sandbox/coherence/kachun/v2/_compute_coherence.py`.

**Spec:** `docs/specs/2026-03-16-subspace-rank-selection-design.md`

---

## File Structure

```
sandbox/coherence/subspace_rank/
  _subspace_coherence.py    # Core module: worker, orchestrator, rank selection, plotting
  run_things_behavior.py    # Main experiment: THINGS behavioral (1854x1854)
  run_comparison.py         # Compare S_k vs I^proj on same data, vary k_max
```

### File responsibilities

| File | Responsibility | Lines (est.) |
|------|---------------|-------------|
| `_subspace_coherence.py` | Bootstrap worker (S_k + I^proj + null), main computation function, rank selection with FDR, kappa diagnostic, plotting | ~500-600 |
| `run_things_behavior.py` | Load THINGS data, run subspace coherence, select rank, save plots | ~80 |
| `run_comparison.py` | Run at multiple k_max values, compare S_k vs v2 I^proj, stability table | ~120 |

---

## Chunk 1: Core Module

### Task 1: Create sandbox directory and v2 import bridge

**Files:**
- Create: `sandbox/coherence/subspace_rank/_subspace_coherence.py`

- [ ] **Step 1: Create the directory and module with v2 imports**

```python
"""Subspace coherence for rank selection.

Measures how well the top-k eigensubspace of a similarity matrix survives
random subsampling. Uses the block Frobenius norm of the cross-Gram matrix
(mean squared canonical correlation) instead of per-eigenvector projected
coherence, making it robust to eigenvector swapping in continuous spectra.
"""

import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.linalg as la
from joblib import Parallel, delayed
from tqdm import tqdm

# Import bootstrap helpers from Ka Chun's v2
_V2_DIR = str(Path(__file__).resolve().parent.parent / "kachun" / "v2")
if _V2_DIR not in sys.path:
    sys.path.insert(0, _V2_DIR)

from _compute_coherence import (
    _symmetrize_with_nan,
    _prepare_observation_mask,
    _masked_unbiased_spd_missing_from_uniform,
    _topk_eigenvectors_exact_warm,
    _orthonormalize_columns,
    _randomized_topk_eigenspace_symmetric,
)
```

- [ ] **Step 2: Verify the import works**

Run: `poetry run python -c "import sys; sys.path.insert(0, 'sandbox/coherence/subspace_rank'); import _subspace_coherence; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add sandbox/coherence/subspace_rank/_subspace_coherence.py
git commit -m "scaffold: create subspace_rank module with v2 import bridge"
```

---

### Task 2: Implement the bootstrap worker

This is the core computation. For one bootstrap replicate, sweep p_list and compute S_k, I^proj_k, and null S_k at each p.

**Files:**
- Modify: `sandbox/coherence/subspace_rank/_subspace_coherence.py`

- [ ] **Step 1: Add the S_k computation helper**

This function extracts S_k and I^proj_k from the cross-Gram matrix G. It is the mathematical core.

```python
def _compute_subspace_and_iproj(g2, k_idx):
    """Extract S_k and I^proj_k from squared cross-Gram matrix G^2.

    Parameters
    ----------
    g2 : (Kmax, Kmax) array
        Element-wise squared cross-Gram matrix: G[i,j]^2 = (u_i^boot . u_j^ref)^2
    k_idx : (K_count,) int array
        Zero-based indices into k_list (i.e., k_list - 1).

    Returns
    -------
    s_k : (K_count,) array
        Subspace coherence: S_k = (1/k) * ||G[:k,:k]||_F^2
    iproj_k : (K_count,) array
        Per-eigenvector coherence: I^proj_k = sum_{i<=k} G[i,k]^2
    """
    kmax = g2.shape[0]

    # I^proj: diagonal of row-cumsum (same as v2)
    iproj_all = np.cumsum(g2, axis=0)[np.arange(kmax), np.arange(kmax)]

    # S_k: 2D cumsum, take diagonal, divide by k
    block_sum = np.cumsum(np.cumsum(g2, axis=0), axis=1)
    s_all = block_sum[np.arange(kmax), np.arange(kmax)] / np.arange(1, kmax + 1)

    return (
        np.clip(s_all[k_idx], 0.0, 1.0),
        np.clip(iproj_all[k_idx], 0.0, 1.0),
    )
```

- [ ] **Step 2: Add the null S_k computation helper**

For B_null random Gaussian matrices, compute S_k^null = (1/k) * ||U_boot[:,:k]^T V||_F^2.

```python
def _compute_null_subspace(u_boot, k_idx, b_null, rng):
    """Compute null distribution for S_k using random Gaussian subspaces.

    Parameters
    ----------
    u_boot : (n, Kmax) array
        Top-Kmax bootstrap eigenvectors.
    k_idx : (K_count,) int array
        Zero-based indices into k_list.
    b_null : int
        Number of null samples.
    rng : numpy Generator
        Random state.

    Returns
    -------
    null_s : (K_count, b_null) array
        Null S_k values for each tested k.
    """
    n, kmax = u_boot.shape
    k_count = len(k_idx)
    null_s = np.zeros((k_count, b_null), dtype=float)

    for m in range(b_null):
        # Random Gaussian matrix, normalize columns
        v = rng.standard_normal((n, kmax))
        v /= np.maximum(la.norm(v, axis=0, keepdims=True), 1e-12)

        # Cross-Gram with random subspace
        g_null = u_boot.T @ v  # (Kmax, Kmax)
        g2_null = g_null * g_null

        # Block Frobenius norm via 2D cumsum
        block_sum = np.cumsum(np.cumsum(g2_null, axis=0), axis=1)
        s_null_all = block_sum[np.arange(kmax), np.arange(kmax)] / np.arange(1, kmax + 1)

        null_s[:, m] = np.clip(s_null_all[k_idx], 0.0, 1.0)

    return null_s
```

- [ ] **Step 3: Add the main worker function**

One worker per bootstrap replicate. Sweeps p_list with warm-started eigensolve and nested masks.

```python
def _subspace_worker_one_boot(args):
    """Compute S_k, I^proj_k, and null S_k for one bootstrap replicate."""
    (
        b, p_list, s0, w, u_ref_k, k_idx,
        random_seed_base, keep_diag,
        solver, solver_tol, solver_maxiter, residual_tol,
        use_nested_masks, b_null,
    ) = args

    n = s0.shape[0]
    kmax = u_ref_k.shape[1]
    k_count = len(k_idx)
    p_list = np.asarray(p_list, float)
    p_count = len(p_list)
    iu = np.triu_indices(n, k=1)

    rng = np.random.default_rng((int(random_seed_base) + 9176 * int(b)) & 0xFFFFFFFF)
    edge_u = rng.random(iu[0].size) if use_nested_masks else None

    s_boot_b = np.zeros((k_count, p_count), float)
    iproj_boot_b = np.zeros((k_count, p_count), float)
    null_s_boot_b = np.zeros((k_count, p_count, b_null), float) if b_null > 0 else None

    x_init = np.array(u_ref_k, copy=True)

    for j, p in enumerate(p_list):
        if use_nested_masks:
            eu = edge_u
        else:
            eu = rng.random(iu[0].size)

        a = _masked_unbiased_spd_missing_from_uniform(
            s0, w, float(p), eu, iu, keep_diag=keep_diag,
        )

        _, u_k, info = _topk_eigenvectors_exact_warm(
            a, kmax, X_init=x_init,
            solver=solver, tol=solver_tol,
            maxiter=solver_maxiter, residual_tol=residual_tol,
        )

        x_init = np.array(u_k, copy=True)

        # Cross-Gram matrix
        g = u_k.T @ u_ref_k  # (Kmax, Kmax)
        g2 = g * g

        # Subspace coherence + per-eigenvector coherence
        s_boot_b[:, j], iproj_boot_b[:, j] = _compute_subspace_and_iproj(g2, k_idx)

        # Null calibration
        if null_s_boot_b is not None:
            null_rng = np.random.default_rng(
                (int(random_seed_base) + 7919 * int(b) + 6271 * int(j)) & 0xFFFFFFFF
            )
            null_s_boot_b[:, j, :] = _compute_null_subspace(
                u_k, k_idx, b_null, null_rng,
            )

    return b, s_boot_b, iproj_boot_b, null_s_boot_b
```

- [ ] **Step 4: Quick sanity check on a small random matrix**

Run: `poetry run python -c "
import sys; sys.path.insert(0, 'sandbox/coherence/subspace_rank')
import numpy as np
from _subspace_coherence import _compute_subspace_and_iproj
g = np.eye(5)
g2 = g * g
s, ip = _compute_subspace_and_iproj(g2, np.arange(5))
print('S_k (identity G):', s)
print('I^proj (identity G):', ip)
assert np.allclose(s, 1.0), 'S_k should be 1 for identity G'
assert np.allclose(ip, 1.0), 'I^proj should be 1 for identity G'
print('PASS')
"`
Expected: All S_k and I^proj = 1.0, PASS

- [ ] **Step 5: Commit**

```bash
git add sandbox/coherence/subspace_rank/_subspace_coherence.py
git commit -m "feat: add subspace coherence worker with S_k, I^proj, and null computation"
```

---

### Task 3: Implement the main orchestrator function

Dispatches workers in parallel, aggregates results, computes CIs and null thresholds.

**Files:**
- Modify: `sandbox/coherence/subspace_rank/_subspace_coherence.py`

- [ ] **Step 1: Add the main computation function**

```python
def compute_subspace_coherence(
    s,
    k_list,
    p_list,
    b=100,
    random_state=0,
    keep_diag=True,
    solver="auto",
    solver_tol=1e-10,
    solver_maxiter=200,
    residual_tol=1e-8,
    use_nested_masks=True,
    b_null=20,
    alpha_tau=0.95,
    ci_level=0.95,
    use_baseline_correction=True,
    n_jobs=None,
    show_progress=True,
    ref_oversample=10,
    ref_n_iter=2,
):
    """Compute subspace coherence across multiple k and masking fractions p.

    Parameters
    ----------
    s : (n, n) array
        Symmetric similarity matrix, may contain NaN.
    k_list : list[int]
        Candidate dimensions to test.
    p_list : array-like
        Sampling fraction grid, sorted increasing in (0, 1].
    b : int
        Bootstrap replicates.
    random_state : int
        Seed for reproducibility.
    keep_diag : bool
        Preserve diagonal during masking.
    solver, solver_tol, solver_maxiter, residual_tol
        Eigensolve parameters (passed to _topk_eigenvectors_exact_warm).
    use_nested_masks : bool
        Reuse one uniform draw across p for nested Bernoulli path.
    b_null : int
        Null samples per (bootstrap, p) cell. Pooled total = b * b_null.
    alpha_tau : float
        Quantile for null threshold (0.95 = 95th percentile).
    ci_level : float
        Confidence level for bootstrap CIs.
    use_baseline_correction : bool
        Apply (S_k - k/n) / (1 - k/n) correction.
    n_jobs : int or None
        Parallel workers. None = cpu_count - 1.
    show_progress : bool
        Show tqdm progress bar.
    ref_oversample, ref_n_iter : int
        Parameters for randomized reference eigenspace (used when S has NaN).

    Returns
    -------
    result : dict
        Keys: S_boot, Iproj_boot, null_S_boot, S_mean, S_ci_lo, S_ci_hi,
        tau_kp, evals_ref, U_ref_K, p, k_list, n, diagnostics.
    """
    # Symmetrize and prepare mask
    s_sym = _symmetrize_with_nan(s)
    n = s_sym.shape[0]

    k_list = np.asarray(sorted(set(k_list)), int)
    p_list = np.asarray(p_list, float)
    kmax = int(k_list.max())
    k_count = len(k_list)
    p_count = len(p_list)
    k_idx = k_list - 1

    cpu = os.cpu_count() or 2
    if n_jobs is None:
        n_jobs_eff = max(1, cpu - 1)
    else:
        n_jobs_eff = int(max(1, min(int(n_jobs), max(1, cpu - 1))))

    s0, w, q = _prepare_observation_mask(s_sym, keep_diag=keep_diag)

    # Reference eigenspace
    has_missing = q < (1.0 - 1e-15)
    if has_missing:
        s_ref = w * s0
        if keep_diag:
            np.fill_diagonal(s_ref, np.diag(s0))
        s_ref = 0.5 * (s_ref + s_ref.T)
    else:
        s_ref = 0.5 * (s0 + s0.T)

    evals_all, evecs_all = la.eigh(s_ref)
    idx_ref = np.argsort(evals_all)[::-1][:kmax]
    evals_ref = evals_all[idx_ref].astype(float)
    u_ref_k = _orthonormalize_columns(evecs_all[:, idx_ref].astype(float))

    # Dispatch workers
    tasks = [
        (
            boot, p_list, s0, w, u_ref_k, k_idx,
            int(random_state or 0), bool(keep_diag),
            str(solver), float(solver_tol), int(solver_maxiter), float(residual_tol),
            bool(use_nested_masks), int(b_null),
        )
        for boot in range(b)
    ]

    if n_jobs_eff == 1 or b <= 1:
        iterator = tqdm(tasks, desc="subspace coherence") if show_progress else tasks
        results = [_subspace_worker_one_boot(t) for t in iterator]
    else:
        results = Parallel(n_jobs=n_jobs_eff, backend="loky", prefer="processes")(
            delayed(_subspace_worker_one_boot)(t)
            for t in tqdm(tasks, desc=f"subspace coherence (n_jobs={n_jobs_eff})")
            if show_progress
        ) if show_progress else Parallel(
            n_jobs=n_jobs_eff, backend="loky", prefer="processes",
        )(delayed(_subspace_worker_one_boot)(t) for t in tasks)

    results.sort(key=lambda x: x[0])

    # Aggregate
    s_boot = np.zeros((k_count, p_count, b), float)
    iproj_boot = np.zeros((k_count, p_count, b), float)
    null_s_boot = np.zeros((k_count, p_count, b, b_null), float) if b_null > 0 else None

    for boot_idx, s_b, ip_b, null_b in results:
        s_boot[:, :, boot_idx] = s_b
        iproj_boot[:, :, boot_idx] = ip_b
        if null_s_boot is not None and null_b is not None:
            null_s_boot[:, :, boot_idx, :] = null_b

    # Baseline correction
    if use_baseline_correction:
        k_over_n = (k_list / float(n)).reshape(-1, 1, 1)
        denom = np.maximum(1.0 - k_over_n, 1e-12)
        x_boot = np.clip((s_boot - k_over_n) / denom, 0.0, 1.0)

        if null_s_boot is not None:
            k_over_n_4d = k_over_n[:, :, :, np.newaxis]
            denom_4d = np.maximum(1.0 - k_over_n_4d, 1e-12)
            null_x = np.clip((null_s_boot - k_over_n_4d) / denom_4d, 0.0, 1.0)
        else:
            null_x = None
    else:
        x_boot = np.clip(s_boot, 0.0, 1.0)
        null_x = np.clip(null_s_boot, 0.0, 1.0) if null_s_boot is not None else None

    # Summary statistics
    ci_lo_q = (1.0 - float(ci_level)) / 2.0
    ci_hi_q = 1.0 - ci_lo_q
    s_mean = x_boot.mean(axis=2)
    s_ci_lo = np.quantile(x_boot, ci_lo_q, axis=2)
    s_ci_hi = np.quantile(x_boot, ci_hi_q, axis=2)

    # Null thresholds
    tau_kp = None
    if null_x is not None:
        null_flat = null_x.reshape(k_count, p_count, -1)
        tau_kp = np.quantile(null_flat, float(alpha_tau), axis=2)

    return {
        "S_boot": s_boot,
        "x_boot": x_boot,
        "Iproj_boot": iproj_boot,
        "null_S_boot": null_s_boot,
        "S_mean": s_mean,
        "S_ci_lo": s_ci_lo,
        "S_ci_hi": s_ci_hi,
        "tau_kp": tau_kp,
        "evals_ref": evals_ref,
        "U_ref_K": u_ref_k,
        "p": p_list,
        "k_list": k_list,
        "n": n,
        "q_offdiag_obs_rate": float(q),
        "use_baseline_correction": use_baseline_correction,
        "B": b,
        "B_null": b_null,
        "alpha_tau": alpha_tau,
        "ci_level": ci_level,
    }
```

- [ ] **Step 2: Commit**

```bash
git add sandbox/coherence/subspace_rank/_subspace_coherence.py
git commit -m "feat: add compute_subspace_coherence orchestrator with parallel dispatch"
```

---

### Task 4: Implement rank selection with FDR

**Files:**
- Modify: `sandbox/coherence/subspace_rank/_subspace_coherence.py`

- [ ] **Step 1: Add the RankResult dataclass and BH-FDR helper**

```python
@dataclass
class RankResult:
    k_star: int
    p_star: float | None
    pvalues: np.ndarray
    significant: np.ndarray
    kappa_hat: np.ndarray
    kappa_k_cut: int
    signal_mask: np.ndarray
    noise_ref_curve: np.ndarray | None
    signal_ref_curve: np.ndarray | None


def _bh_fdr(pvalues, q=0.05):
    """Benjamini-Hochberg FDR correction. Returns boolean rejection mask."""
    pv = np.asarray(pvalues, float)
    m = pv.size
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(pv)
    sorted_pv = pv[order]
    thresholds = np.arange(1, m + 1) / m * q
    reject_sorted = np.zeros(m, dtype=bool)
    # Find largest i where p_(i) <= i/m * q
    candidates = np.where(sorted_pv <= thresholds)[0]
    if len(candidates) > 0:
        cutoff = candidates[-1]
        reject_sorted[: cutoff + 1] = True
    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject
```

- [ ] **Step 2: Add kappa computation (reused concept from v2)**

```python
def _estimate_kappa(s_mean, p_list, hi_band_quantile=0.85):
    """Kappa from subspace coherence: ell_k(p) = (1 - S_k(p)) * p/(1-p)."""
    p_list = np.asarray(p_list, float)
    p_hi = float(np.quantile(p_list, hi_band_quantile))
    hi_idx = np.where(p_list >= p_hi)[0]
    if hi_idx.size == 0:
        hi_idx = np.array([len(p_list) - 1], int)
    scale = p_list[hi_idx] / np.maximum(1.0 - p_list[hi_idx], 1e-12)
    ell = (1.0 - s_mean[:, hi_idx]) * scale[None, :]
    kappa_hat = np.median(ell, axis=1)
    return kappa_hat


def _kappa_changepoint(kappa_hat, k_list, smooth_window=3, min_k=2):
    """Find k_cut via largest jump in smoothed kappa."""
    k = kappa_hat.size
    if k < 3:
        return int(k_list[-1])
    # Median smoothing
    sm = np.copy(kappa_hat)
    half = smooth_window // 2
    for i in range(k):
        a, b_end = max(0, i - half), min(k, i + half + 1)
        sm[i] = np.median(kappa_hat[a:b_end])
    d = sm[1:] - sm[:-1]
    valid = np.where(np.asarray(k_list[:-1]) >= min_k)[0]
    if valid.size == 0:
        valid = np.arange(k - 1)
    i_star = int(valid[np.argmax(d[valid])])
    return int(k_list[i_star])
```

- [ ] **Step 3: Add the select_rank function**

```python
def select_rank(
    result,
    fdr_q=0.05,
    noise_quantile=0.90,
    lift_margin=0.0,
    hi_band_quantile=0.85,
    smooth_window=3,
):
    """Select rank k* and sampling fraction p* from subspace coherence results.

    Parameters
    ----------
    result : dict
        Output of compute_subspace_coherence.
    fdr_q : float
        FDR level for BH correction.
    noise_quantile : float
        Quantile for noise reference curve.
    lift_margin : float
        Minimum gap between signal and noise for p*.
    hi_band_quantile : float
        High-p band for kappa estimation.
    smooth_window : int
        Smoothing window for kappa changepoint.

    Returns
    -------
    RankResult
    """
    k_list = result["k_list"]
    p_list = result["p"]
    s_ci_lo = result["S_ci_lo"]
    s_ci_hi = result["S_ci_hi"]
    s_mean = result["S_mean"]
    tau_kp = result["tau_kp"]
    k_count = len(k_list)

    # Per-dimension p-values at p_max
    # p-value = fraction of null samples >= observed lower CI
    if result["null_S_boot"] is not None:
        x_boot = result["x_boot"]
        null_x = result["null_S_boot"]
        if result["use_baseline_correction"]:
            n = result["n"]
            k_over_n = (k_list / float(n)).reshape(-1, 1, 1, 1)
            denom = np.maximum(1.0 - k_over_n, 1e-12)
            null_x = np.clip((null_x - k_over_n) / denom, 0.0, 1.0)

        # At p_max: observed = median of bootstrap S_k
        obs_at_pmax = np.median(x_boot[:, -1, :], axis=1)  # (K,)
        null_at_pmax = null_x[:, -1, :, :].reshape(k_count, -1)  # (K, B*B_null)
        pvalues = np.array([
            float(np.mean(null_at_pmax[kk] >= obs_at_pmax[kk]))
            for kk in range(k_count)
        ])
    else:
        # Fallback: use CI vs tau
        pvalues = np.ones(k_count)
        for kk in range(k_count):
            if tau_kp is not None and s_ci_lo[kk, -1] > tau_kp[kk, -1]:
                pvalues[kk] = 0.01  # placeholder

    significant = _bh_fdr(pvalues, q=fdr_q)

    if np.any(significant):
        k_star = int(k_list[np.where(significant)[0][-1]])
    else:
        k_star = 0

    # Kappa diagnostic
    kappa_hat = _estimate_kappa(s_mean, p_list, hi_band_quantile)
    kappa_k_cut = _kappa_changepoint(kappa_hat, k_list, smooth_window)

    # Signal/noise classification
    signal_mask = k_list <= k_star if k_star > 0 else np.zeros(k_count, dtype=bool)
    noise_mask = ~signal_mask

    # p* via signal-noise liftoff
    p_star = None
    noise_ref_curve = None
    signal_ref_curve = None

    if k_star > 0 and np.any(noise_mask):
        noise_pool = s_ci_hi[noise_mask]
        noise_ref_curve = np.quantile(noise_pool, noise_quantile, axis=0)

        boundary_idx = np.where(signal_mask)[0][-1]
        signal_ref_curve = s_ci_lo[boundary_idx]

        gap = signal_ref_curve - noise_ref_curve
        above = gap >= lift_margin
        idxs = np.where(above)[0]
        if len(idxs) > 0:
            p_star = float(p_list[idxs[0]])

    return RankResult(
        k_star=k_star,
        p_star=p_star,
        pvalues=pvalues,
        significant=significant,
        kappa_hat=kappa_hat,
        kappa_k_cut=kappa_k_cut,
        signal_mask=signal_mask,
        noise_ref_curve=noise_ref_curve,
        signal_ref_curve=signal_ref_curve,
    )
```

- [ ] **Step 4: Commit**

```bash
git add sandbox/coherence/subspace_rank/_subspace_coherence.py
git commit -m "feat: add select_rank with FDR-corrected hypothesis testing and kappa diagnostic"
```

---

### Task 5: Implement plotting functions

**Files:**
- Modify: `sandbox/coherence/subspace_rank/_subspace_coherence.py`

- [ ] **Step 1: Add plotting function**

```python
def plot_rank_selection(result, rank_result, output_dir):
    """Generate all diagnostic plots and save to output_dir.

    Returns list of saved file paths.
    """
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from utils.figure_theme import CMAP, GRAY, create_figure, despine

    output_dir = Path(output_dir)
    k_list = result["k_list"]
    p_list = result["p"]
    s_mean = result["S_mean"]
    s_ci_lo = result["S_ci_lo"]
    s_ci_hi = result["S_ci_hi"]
    tau_kp = result["tau_kp"]
    iproj_mean = np.median(result["Iproj_boot"], axis=2)
    k_star = rank_result.k_star
    saved = []

    # 1. Heatmap of S_k(p)
    fig, ax = create_figure("wide")
    im = ax.imshow(
        s_mean, aspect="auto", origin="lower", interpolation="nearest",
        extent=[p_list[0], p_list[-1], k_list[0] - 2.5, k_list[-1] + 2.5],
        cmap="magma",
    )
    if k_star > 0:
        ax.axhline(k_star + 2.5, color="white", linewidth=1.5, linestyle="--")
        ax.text(p_list[1], k_star + 4, f"k* = {k_star}", color="white", fontsize=8)
    ax.set_xlabel("Sampling fraction $p$")
    ax.set_ylabel("Dimension $k$")
    ax.set_title("Subspace coherence $S_k(p)$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = output_dir / "subspace_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    # 2. S_k at p_max with null threshold
    s_at_pmax = np.median(result["x_boot"][:, -1, :], axis=1)
    s_lo_pmax = np.quantile(result["x_boot"][:, -1, :], 0.05, axis=1)
    s_hi_pmax = np.quantile(result["x_boot"][:, -1, :], 0.95, axis=1)
    tau_at_pmax = tau_kp[:, -1] if tau_kp is not None else None

    fig, ax = create_figure("wide")
    ax.fill_between(k_list, s_lo_pmax, s_hi_pmax, color=CMAP[1], alpha=0.15)
    ax.plot(k_list, s_at_pmax, marker="o", markersize=4, linewidth=1.5,
            color=CMAP[1], label="$S_k(p_{\\max})$")
    if tau_at_pmax is not None:
        ax.plot(k_list, tau_at_pmax, linestyle="--", linewidth=1.5,
                color=CMAP[0], label="Null threshold $\\tau$")
    if k_star > 0:
        ax.axvline(k_star, linestyle=":", linewidth=1, color=GRAY["dark"],
                   label=f"$k^*$ = {k_star}")
    ax.axvline(rank_result.kappa_k_cut, linestyle=":", linewidth=1, color=GRAY["light"],
               label=f"Kappa $k_{{cut}}$ = {rank_result.kappa_k_cut}")
    ax.set_xlabel("Dimension $k$")
    ax.set_ylabel(f"Subspace coherence at $p = {p_list[-1]:.2f}$")
    ax.set_title("Subspace rank selection")
    ax.legend(fontsize=6)
    despine(ax)
    path = output_dir / "subspace_rank_curve.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    # 3. Kappa curve
    fig, ax = create_figure("wide")
    ax.plot(k_list, rank_result.kappa_hat, marker=".", markersize=5,
            linewidth=1, color=GRAY["dark"])
    ax.scatter(k_list[rank_result.signal_mask],
              rank_result.kappa_hat[rank_result.signal_mask],
              s=30, color=CMAP[1], zorder=5, label="Signal")
    ax.scatter(k_list[~rank_result.signal_mask],
              rank_result.kappa_hat[~rank_result.signal_mask],
              s=15, color=GRAY["light"], zorder=4, label="Noise")
    ax.axvline(rank_result.kappa_k_cut + 2.5, linestyle="--", linewidth=1.5,
               color=CMAP[0], label=f"Kappa $k_{{cut}}$ = {rank_result.kappa_k_cut}")
    ax.set_xlabel("Dimension $k$")
    ax.set_ylabel("$\\hat{\\kappa}_k$")
    ax.set_title("Kappa diagnostic (secondary)")
    ax.legend(fontsize=7)
    despine(ax)
    path = output_dir / "kappa_curve.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    # 4. Comparison: S_k vs I^proj at p_max
    iproj_at_pmax = np.median(result["Iproj_boot"][:, -1, :], axis=1)

    fig, axes = create_figure("full_width", ncols=2)
    ax1, ax2 = axes

    ax1.plot(k_list, s_at_pmax, marker="o", markersize=3, linewidth=1.2,
             color=CMAP[1], label="$S_k$ (subspace)")
    if tau_at_pmax is not None:
        ax1.plot(k_list, tau_at_pmax, linestyle="--", linewidth=1, color=CMAP[0],
                 label="Null $\\tau$")
    if k_star > 0:
        ax1.axvline(k_star, linestyle=":", color=GRAY["dark"], linewidth=0.8)
    ax1.set_xlabel("Dimension $k$")
    ax1.set_ylabel("Subspace coherence")
    ax1.set_title("$S_k$ (this work)")
    ax1.legend(fontsize=5)
    despine(ax1)

    ax2.plot(k_list, iproj_at_pmax, marker="o", markersize=3, linewidth=1.2,
             color=CMAP[2], label="$I^{proj}_k$ (per-eigenvector)")
    ax2.set_xlabel("Dimension $k$")
    ax2.set_ylabel("Per-eigenvector coherence")
    ax2.set_title("$I^{proj}_k$ (Ka Chun v2)")
    ax2.legend(fontsize=5)
    despine(ax2)

    path = output_dir / "subspace_vs_iproj.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    # 5. Signal-noise liftoff for p*
    if rank_result.signal_ref_curve is not None and rank_result.noise_ref_curve is not None:
        fig, ax = create_figure("wide")
        ax.plot(p_list, rank_result.signal_ref_curve, linewidth=1.5, color=CMAP[1],
                label=f"Signal ref ($k$={k_star})")
        ax.plot(p_list, rank_result.noise_ref_curve, linewidth=1.5, linestyle="--",
                color=CMAP[0], label="Noise ref")
        gap = rank_result.signal_ref_curve - rank_result.noise_ref_curve
        ax.plot(p_list, gap, linewidth=1, linestyle=":", color=GRAY["dark"], label="Gap")
        ax.axhline(0, color=GRAY["light"], linewidth=0.5)
        if rank_result.p_star is not None:
            ax.axvline(rank_result.p_star, color="k", linewidth=1, linestyle="--",
                       label=f"$p^*$ = {rank_result.p_star:.3f}")
        ax.set_xlabel("Sampling fraction $p$")
        ax.set_ylabel("Coherence")
        ax.set_title(f"Signal-noise liftoff ($k^*$={k_star})")
        ax.legend(fontsize=6)
        despine(ax)
        path = output_dir / "signal_noise_liftoff.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(path)

    return saved
```

- [ ] **Step 2: Commit**

```bash
git add sandbox/coherence/subspace_rank/_subspace_coherence.py
git commit -m "feat: add plot_rank_selection for all diagnostic plots"
```

---

## Chunk 2: Run Scripts and Validation

### Task 6: THINGS behavioral experiment

**Files:**
- Create: `sandbox/coherence/subspace_rank/run_things_behavior.py`

- [ ] **Step 1: Write the run script**

```python
"""Subspace coherence rank selection on THINGS behavioral similarity."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import logging

from src.similarity import build_similarity
from src.utils import get_output_dir
from omegaconf import OmegaConf

from _subspace_coherence import compute_subspace_coherence, select_rank, plot_rank_selection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()


def main():
    cfg = OmegaConf.create({
        "name": "things_behavior",
        "type": "triplet",
        "path": "data/things",
        "triplet_number": "4.7mio",
        "n_objects": 1854,
    })
    s = build_similarity(cfg)
    n = s.shape[0]
    log.info(f"THINGS behavioral: {s.shape}, NaN={np.sum(np.isnan(s))}")

    k_list = list(range(5, 121, 5))
    p_list = np.linspace(0.05, 0.95, 25)

    log.info(f"Running subspace coherence: k={k_list[0]}..{k_list[-1]}, B=100, B_null=20")
    result = compute_subspace_coherence(
        s, k_list, p_list,
        b=100, b_null=20,
        use_baseline_correction=True,
        n_jobs=None,
    )

    rank = select_rank(result, fdr_q=0.05)

    log.info(f"\n=== RANK SELECTION ===")
    log.info(f"  k* (FDR q=0.05): {rank.k_star}")
    log.info(f"  p* (liftoff):    {rank.p_star}")
    log.info(f"  Kappa k_cut:     {rank.kappa_k_cut} (secondary)")

    log.info(f"\nPer-dimension results:")
    for i, k in enumerate(result["k_list"]):
        sig = "***" if rank.significant[i] else ""
        log.info(f"  k={k:3d}: p-value={rank.pvalues[i]:.4f} {sig}")

    saved = plot_rank_selection(result, rank, OUTPUT_DIR)
    log.info(f"\nPlots saved to {OUTPUT_DIR}")

    # Save numerical results
    np.savez(
        OUTPUT_DIR / "subspace_coherence_results.npz",
        S_boot=result["S_boot"],
        Iproj_boot=result["Iproj_boot"],
        S_mean=result["S_mean"],
        k_list=result["k_list"],
        p_list=result["p"],
        k_star=rank.k_star,
        p_star=rank.p_star if rank.p_star is not None else np.nan,
        pvalues=rank.pvalues,
        significant=rank.significant,
        kappa_hat=rank.kappa_hat,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the experiment**

Run: `./scripts/submit sandbox/coherence/subspace_rank/run_things_behavior.py --bg`
Monitor: `dash`
Expected: Completes in ~5-15 minutes. Check log for k* in [35, 85].

- [ ] **Step 3: Inspect results**

Check the output plots, especially:
- `subspace_rank_curve.png`: S_k should decay monotonically, k* line should be in 40-75 range
- `subspace_vs_iproj.png`: S_k should be smoother than I^proj (no bumps at k=45)
- `subspace_heatmap.png`: triangular pattern similar to v2 but smoother

- [ ] **Step 4: Commit**

```bash
git add sandbox/coherence/subspace_rank/run_things_behavior.py
git commit -m "feat: add THINGS behavioral experiment for subspace rank selection"
```

---

### Task 7: Comparison and stability test

**Files:**
- Create: `sandbox/coherence/subspace_rank/run_comparison.py`

- [ ] **Step 1: Write the comparison script**

```python
"""Compare subspace coherence S_k vs per-eigenvector I^proj across k_max values.

Tests stability: k* should not change when k_max varies from 80 to 200.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
import logging

from src.similarity import build_similarity
from src.utils import get_output_dir
from src.utils.figure_theme import CMAP, GRAY, create_figure, despine
from omegaconf import OmegaConf

from _subspace_coherence import compute_subspace_coherence, select_rank

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()


def main():
    cfg = OmegaConf.create({
        "name": "things_behavior",
        "type": "triplet",
        "path": "data/things",
        "triplet_number": "4.7mio",
        "n_objects": 1854,
    })
    s = build_similarity(cfg)
    log.info(f"THINGS behavioral: {s.shape}")

    p_list = np.linspace(0.05, 0.95, 25)

    # Run once at k_max=200 and then subset results
    k_list_full = list(range(5, 201, 5))
    log.info(f"Running subspace coherence: k=5..200, B=50, B_null=20")

    result_full = compute_subspace_coherence(
        s, k_list_full, p_list,
        b=50, b_null=20,
        use_baseline_correction=True,
        n_jobs=None,
    )

    # Test stability across k_max
    k_max_values = [80, 100, 120, 140, 160, 180, 200]
    k_arr_full = result_full["k_list"]
    results_subspace = []

    for km in k_max_values:
        idx_end = int(np.searchsorted(k_arr_full, km, side="right"))
        sub = {
            key: result_full[key]
            for key in result_full
            if key not in ("S_boot", "Iproj_boot", "null_S_boot", "x_boot",
                           "S_mean", "S_ci_lo", "S_ci_hi", "tau_kp", "k_list")
        }
        sub["k_list"] = k_arr_full[:idx_end]
        sub["S_boot"] = result_full["S_boot"][:idx_end]
        sub["Iproj_boot"] = result_full["Iproj_boot"][:idx_end]
        sub["x_boot"] = result_full["x_boot"][:idx_end]
        sub["S_mean"] = result_full["S_mean"][:idx_end]
        sub["S_ci_lo"] = result_full["S_ci_lo"][:idx_end]
        sub["S_ci_hi"] = result_full["S_ci_hi"][:idx_end]
        if result_full["null_S_boot"] is not None:
            sub["null_S_boot"] = result_full["null_S_boot"][:idx_end]
        if result_full["tau_kp"] is not None:
            sub["tau_kp"] = result_full["tau_kp"][:idx_end]

        rank = select_rank(sub, fdr_q=0.05)
        results_subspace.append((km, rank.k_star, rank.p_star, rank.kappa_k_cut))

    # Print comparison table
    log.info("\n=== STABILITY TABLE ===")
    log.info(f"{'k_max':>6} | {'S_k k*':>8} | {'S_k p*':>8} | {'Kappa k_cut':>12}")
    log.info("-" * 45)
    for km, ks, ps, kc in results_subspace:
        ps_str = f"{ps:.3f}" if ps is not None else "None"
        log.info(f"{km:>6} | {ks:>8} | {ps_str:>8} | {kc:>12}")

    # Plot: k* vs k_max
    fig, ax = create_figure("wide")
    k_max_arr = np.array([r[0] for r in results_subspace])
    k_star_arr = np.array([r[1] for r in results_subspace])
    kappa_arr = np.array([r[3] for r in results_subspace])

    ax.plot(k_max_arr, k_star_arr, marker="o", markersize=5, linewidth=1.5,
            color=CMAP[1], label="$S_k$ (subspace, FDR)")
    ax.plot(k_max_arr, kappa_arr, marker="s", markersize=5, linewidth=1.5,
            color=CMAP[0], label="Kappa changepoint")
    ax.plot(k_max_arr, k_max_arr, linestyle=":", linewidth=0.8, color=GRAY["light"],
            label="$k^* = k_{max}$ (ceiling)")
    ax.set_xlabel("Maximum tested rank $k_{\\max}$")
    ax.set_ylabel("Estimated $k^*$")
    ax.set_title("Rank estimate stability across $k_{\\max}$")
    ax.legend(fontsize=6)
    despine(ax)
    fig.savefig(OUTPUT_DIR / "stability_kstar_vs_kmax.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    log.info(f"\nPlots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the comparison**

Run: `./scripts/submit sandbox/coherence/subspace_rank/run_comparison.py --bg`
Monitor: `dash`
Expected: k* should be roughly constant across k_max values (within +/- 10).

- [ ] **Step 3: Evaluate results against success criteria**

Check:
- k* in [35, 85] for THINGS behavioral
- k* varies by less than 10 across k_max = 80..200
- k* does NOT track k_max (unlike v2 count-based activation and CV curve)
- Kappa k_cut is stable at ~25 (consistent with v2, secondary diagnostic)

- [ ] **Step 4: Commit**

```bash
git add sandbox/coherence/subspace_rank/run_comparison.py
git commit -m "feat: add stability comparison across k_max for subspace rank selection"
```

---

## Post-Implementation Notes

**If k\* is outside [35, 85] on THINGS:** The null may still be too permissive or too conservative. Adjustments:
- If k* is too high (>85): increase alpha_tau from 0.95 to 0.99, or require effect size
- If k* is too low (<35): decrease alpha_tau, or check that baseline correction is working
- If k* tracks k_max: the fundamental approach needs rethinking (escalate to Approach C from the design)

**If results look good, next steps:**
- Run validation tests 1-2 from the spec (synthetic block matrix, synthetic continuous spectrum)
- Run sensitivity analysis (B, B_null variations)
- Consider promoting the core S_k computation to `src/tools/` if it proves generally useful
