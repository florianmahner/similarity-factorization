# Coherence Module Rewrite (Kappa-Only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 3100-line `pysrf/coherence.py` with a clean, modular ~400-line module exposing `estimate_rank(s, k_max, ...)` via kappa changepoint, with floating-point identical results to the original.

**Architecture:** Single flat file (`pysrf/coherence.py`) organized in four layers: (1) matrix preparation, (2) reference eigenspace, (3) bootstrap coherence engine, (4) kappa estimation. Each layer is a set of small private helpers. One public function `estimate_rank` ties them together. The old file is preserved as `tests/coherence_full.py` for regression testing. Helpers are named for reuse when activation/cluster methods are added later.

**Tech Stack:** numpy, numpy.linalg, scipy.sparse.linalg (optional eigsh), joblib, tqdm

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `tests/coherence_full.py` | Create (copy of current `pysrf/coherence.py`) | Reference implementation for regression tests |
| `pysrf/coherence.py` | Rewrite | Clean kappa-only coherence module |
| `tests/test_coherence.py` | Create | Regression + unit tests ensuring identical results |
| `pysrf/__init__.py` | Modify | Export `estimate_rank` |

---

### Task 1: Preserve the old implementation as test reference

**Files:**
- Create: `tests/coherence_full.py`

- [ ] **Step 1: Copy current coherence.py to tests/**

```bash
cp pysrf/coherence.py tests/coherence_full.py
```

- [ ] **Step 2: Verify the copy imports correctly**

```bash
poetry run python -c "
import sys, os
sys.path.insert(0, 'tests')
from coherence_full import _estimate_kappa_hat, kappa_changepoint
print('old imports ok')
"
```

Expected: `old imports ok`

- [ ] **Step 3: Commit**

```bash
git add tests/coherence_full.py
git commit -m "test: preserve original coherence.py as regression reference"
```

---

### Task 2: Write regression test scaffold using the old implementation

This test generates a small deterministic matrix, runs the old bootstrap + kappa pipeline, and hard-codes the expected outputs. The new implementation must match these exactly.

**Files:**
- Create: `tests/test_coherence.py`

- [ ] **Step 1: Generate reference values from the old code**

Run this script to capture the exact floating-point outputs:

```bash
poetry run python -c "
import sys
sys.path.insert(0, 'tests')
import numpy as np
from coherence_full import (
    _symmetrize_with_nan,
    _prepare_observation_mask,
    _topk_eigenvectors,
    _masked_unbiased_spd_missing,
    _estimate_kappa_hat,
    kappa_changepoint,
    compute_incremental_coherence_multi_k_eig_anisotropic,
)

# Deterministic low-rank matrix (n=30, true rank=5)
rng = np.random.default_rng(42)
w = np.abs(rng.standard_normal((30, 5)))
s = w @ w.T
s += 0.01 * rng.standard_normal((30, 30))
s = (s + s.T) / 2

k_list = list(range(1, 16))
p_list = np.linspace(0.1, 0.95, 10)

result = compute_incremental_coherence_multi_k_eig_anisotropic(
    s,
    k_list=k_list,
    p_list=p_list,
    B=5,
    random_state=0,
    compute_null=False,
    B_null=0,
    n_jobs=1,
    show_progress=False,
    visualize=False,
)

iproj_median = np.median(result['Iproj_boot'], axis=2)
kappa, _ = _estimate_kappa_hat(iproj_median, p_list, hi_band_quantile=0.85)
k_star, _ = kappa_changepoint(kappa, np.array(k_list))

print(f'k_star = {k_star}')
print(f'kappa[:5] = {kappa[:5].tolist()}')
print(f'iproj_median[0,:3] = {iproj_median[0,:3].tolist()}')
print(f'iproj_median.shape = {iproj_median.shape}')
print(f'evals_ref[:3] = {result[\"evals_ref\"][:3].tolist()}')

# Save arrays for test fixture
np.savez(
    'tests/coherence_reference.npz',
    iproj_boot=result['Iproj_boot'],
    iproj_median=iproj_median,
    kappa=kappa,
    k_star=np.array(k_star),
    evals_ref=result['evals_ref'],
    k_list=np.array(k_list),
    p_list=p_list,
    s=s,
)
print('saved tests/coherence_reference.npz')
"
```

- [ ] **Step 2: Write the regression test file**

```python
# tests/test_coherence.py
"""Regression tests for pysrf.coherence against the original implementation."""

import numpy as np
import pytest
from pathlib import Path

from helpers import make_symmetric_matrix


@pytest.fixture(scope="module")
def reference():
    """Load reference outputs from the original coherence implementation."""
    path = Path(__file__).parent / "coherence_reference.npz"
    return dict(np.load(path, allow_pickle=False))


@pytest.fixture(scope="module")
def test_matrix():
    """Deterministic low-rank matrix matching the reference generation."""
    rng = np.random.default_rng(42)
    w = np.abs(rng.standard_normal((30, 5)))
    s = w @ w.T
    s += 0.01 * rng.standard_normal((30, 30))
    s = (s + s.T) / 2
    return s


# ---- Layer 1: Matrix preparation ----

class TestSymmetrize:
    def test_already_symmetric(self, test_matrix):
        from pysrf.coherence import _symmetrize
        result = _symmetrize(test_matrix)
        assert np.allclose(result, result.T)

    def test_nan_handling(self):
        from pysrf.coherence import _symmetrize
        s = np.array([[1.0, np.nan, 3.0],
                       [2.0, 1.0, np.nan],
                       [np.nan, 4.0, 1.0]])
        result = _symmetrize(s)
        assert np.isfinite(result).all() or np.isnan(result).sum() == 0
        assert np.allclose(result, result.T, equal_nan=True)


class TestObservationMask:
    def test_full_observation(self, test_matrix):
        from pysrf.coherence import _observation_mask
        s_filled, mask, obs_rate = _observation_mask(test_matrix)
        assert np.all(mask == 1.0)
        assert abs(obs_rate - 1.0) < 1e-10

    def test_with_missing(self):
        from pysrf.coherence import _observation_mask
        s = np.array([[1.0, np.nan], [np.nan, 1.0]])
        s_filled, mask, obs_rate = _observation_mask(s)
        assert s_filled[0, 1] == 0.0
        assert mask[0, 1] == 0.0


# ---- Layer 2: Reference eigenspace ----

class TestReferenceEigenpairs:
    def test_shape_and_order(self, test_matrix, reference):
        from pysrf.coherence import _reference_eigenpairs
        k = 15
        evals, evecs = _reference_eigenpairs(test_matrix, k)
        assert evals.shape == (k,)
        assert evecs.shape == (30, k)
        # Eigenvalues descending
        assert np.all(np.diff(evals) <= 1e-10)

    def test_eigenvalues_match_reference(self, test_matrix, reference):
        from pysrf.coherence import _reference_eigenpairs
        k = 15
        evals, _ = _reference_eigenpairs(test_matrix, k)
        np.testing.assert_allclose(evals, reference["evals_ref"][:k], atol=1e-10)


# ---- Layer 3: Bootstrap coherence ----

class TestBootstrapSample:
    def test_symmetry(self, test_matrix):
        from pysrf.coherence import _observation_mask, _bootstrap_sample
        s_filled, mask, _ = _observation_mask(test_matrix)
        rng = np.random.default_rng(0)
        iu = np.triu_indices(30, k=1)
        a = _bootstrap_sample(s_filled, mask, 0.5, rng, iu)
        np.testing.assert_allclose(a, a.T)

    def test_diagonal_preserved(self, test_matrix):
        from pysrf.coherence import _observation_mask, _bootstrap_sample
        s_filled, mask, _ = _observation_mask(test_matrix)
        rng = np.random.default_rng(0)
        iu = np.triu_indices(30, k=1)
        a = _bootstrap_sample(s_filled, mask, 0.5, rng, iu)
        np.testing.assert_allclose(np.diag(a), np.diag(s_filled))


class TestEigenspaceOverlap:
    def test_perfect_overlap(self):
        from pysrf.coherence import _eigenspace_overlap
        rng = np.random.default_rng(0)
        u = np.linalg.qr(rng.standard_normal((10, 3)))[0]
        iproj = _eigenspace_overlap(u, u)
        np.testing.assert_allclose(iproj, np.ones(3), atol=1e-10)

    def test_zero_overlap(self):
        from pysrf.coherence import _eigenspace_overlap
        u_ref = np.eye(6, 3)
        u_boot = np.eye(6, 3, k=3)
        iproj = _eigenspace_overlap(u_boot, u_ref)
        np.testing.assert_allclose(iproj, np.zeros(3), atol=1e-10)


class TestBootstrapCoherence:
    def test_iproj_shape(self, test_matrix):
        from pysrf.coherence import _bootstrap_coherence
        k_max = 5
        p_list = np.linspace(0.1, 0.95, 4)
        iproj_boot, evals_ref = _bootstrap_coherence(
            test_matrix, k_max=k_max, p_list=p_list, n_boot=3,
            random_state=0, n_jobs=1,
        )
        assert iproj_boot.shape == (k_max, 4, 3)
        assert evals_ref.shape == (k_max,)

    def test_iproj_bounded(self, test_matrix):
        from pysrf.coherence import _bootstrap_coherence
        iproj_boot, _ = _bootstrap_coherence(
            test_matrix, k_max=5, p_list=np.linspace(0.1, 0.95, 4),
            n_boot=3, random_state=0, n_jobs=1,
        )
        assert np.all(iproj_boot >= 0.0)
        assert np.all(iproj_boot <= 1.0)

    def test_matches_reference(self, test_matrix, reference):
        from pysrf.coherence import _bootstrap_coherence
        k_list = reference["k_list"].astype(int)
        k_max = int(k_list.max())
        p_list = reference["p_list"]
        iproj_boot, _ = _bootstrap_coherence(
            test_matrix, k_max=k_max, p_list=p_list, n_boot=5,
            random_state=0, n_jobs=1,
        )
        # Select only the k indices that match k_list
        k_idx = k_list - 1
        iproj_selected = iproj_boot[k_idx]
        np.testing.assert_allclose(
            iproj_selected, reference["iproj_boot"], atol=1e-12,
        )


# ---- Layer 4: Kappa estimation ----

class TestScaledLeakage:
    def test_shape(self, reference):
        from pysrf.coherence import _scaled_leakage
        iproj_median = reference["iproj_median"]
        p_list = reference["p_list"]
        kappa = _scaled_leakage(iproj_median, p_list, hi_quantile=0.85)
        assert kappa.shape == (iproj_median.shape[0],)

    def test_matches_reference(self, reference):
        from pysrf.coherence import _scaled_leakage
        kappa = _scaled_leakage(
            reference["iproj_median"], reference["p_list"], hi_quantile=0.85,
        )
        np.testing.assert_allclose(kappa, reference["kappa"], atol=1e-12)


class TestLargestJump:
    def test_matches_reference(self, reference):
        from pysrf.coherence import _largest_jump
        k_list = reference["k_list"].astype(int)
        k_star = _largest_jump(reference["kappa"], k_list)
        assert k_star == int(reference["k_star"])

    def test_monotonic_kappa(self):
        from pysrf.coherence import _largest_jump
        kappa = np.array([0.01, 0.02, 0.03, 0.5, 0.8, 0.9])
        k_list = np.arange(1, 7)
        k_star = _largest_jump(kappa, k_list)
        assert k_star == 3  # jump from 0.03 to 0.5


# ---- Public API ----

class TestEstimateRank:
    def test_returns_k_star(self, test_matrix, reference):
        from pysrf.coherence import estimate_rank
        result = estimate_rank(
            test_matrix,
            k_max=15,
            p_list=reference["p_list"],
            n_boot=5,
            random_state=0,
            n_jobs=1,
        )
        assert result["k_star"] == int(reference["k_star"])

    def test_result_keys(self, test_matrix):
        from pysrf.coherence import estimate_rank
        result = estimate_rank(
            test_matrix, k_max=5,
            p_list=np.linspace(0.1, 0.95, 5),
            n_boot=3, random_state=0, n_jobs=1,
        )
        assert "k_star" in result
        assert "kappa" in result
        assert "iproj_median" in result
        assert "evals_ref" in result
        assert "k_list" in result
        assert "p_list" in result

    def test_kappa_matches_reference(self, test_matrix, reference):
        from pysrf.coherence import estimate_rank
        result = estimate_rank(
            test_matrix,
            k_max=15,
            p_list=reference["p_list"],
            n_boot=5,
            random_state=0,
            n_jobs=1,
        )
        np.testing.assert_allclose(
            result["kappa"], reference["kappa"], atol=1e-12,
        )
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
poetry run pytest tests/test_coherence.py -v --tb=short 2>&1 | head -40
```

Expected: All tests fail with `ImportError` (new module not written yet).

- [ ] **Step 4: Commit**

```bash
git add tests/test_coherence.py tests/coherence_reference.npz
git commit -m "test: add coherence regression tests against original implementation"
```

---

### Task 3: Write the new coherence module — Layer 1 (matrix preparation)

**Files:**
- Create: `pysrf/coherence.py` (overwrite)

- [ ] **Step 1: Write Layer 1 helpers**

```python
"""Eigenspace coherence for dimensionality estimation.

Estimates the number of signal dimensions (k*) in a symmetric
similarity matrix by measuring how stable each eigenspace
dimension is under random entry masking.

The method works by:
1. Computing a reference eigenspace from the full matrix
2. Repeatedly masking entries at rate p and re-computing eigenvectors
3. Measuring overlap (Iproj) between bootstrap and reference eigenvectors
4. Estimating scaled leakage (kappa) per dimension
5. Finding the changepoint where kappa jumps from signal to noise
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.linalg as la

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 1: Matrix preparation
# ---------------------------------------------------------------------------


def _symmetrize(s: np.ndarray) -> np.ndarray:
    """Symmetrize a matrix while preserving NaN semantics.

    For each pair (i, j):
    - average if both finite
    - copy the finite value if only one side is finite
    - keep NaN if both missing

    Diagonal NaNs are replaced with zero.
    """
    s = np.asarray(s, dtype=float)
    st = s.T
    a = np.isfinite(s)
    b = np.isfinite(st)

    out = np.full_like(s, np.nan, dtype=float)

    both = a & b
    out[both] = 0.5 * (s[both] + st[both])

    only_a = a & ~b
    out[only_a] = s[only_a]

    only_b = ~a & b
    out[only_b] = st[only_b]

    d = np.diag(out).copy()
    d[~np.isfinite(d)] = 0.0
    np.fill_diagonal(out, d)
    return out


def _observation_mask(
    s_sym: np.ndarray, eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build observation mask from a symmetrized matrix.

    Parameters
    ----------
    s_sym : (n, n) array
        Symmetrized similarity matrix (may contain NaN).

    Returns
    -------
    s_filled : (n, n) array
        Input with NaN replaced by zero.
    mask : (n, n) array
        Symmetric 0/1 observation mask with diagonal forced to 1.
    obs_rate : float
        Off-diagonal observation rate, clipped to [eps, 1].
    """
    n = s_sym.shape[0]
    mask = np.isfinite(s_sym).astype(float)
    mask = ((mask + mask.T) > 0).astype(float)
    np.fill_diagonal(mask, 1.0)

    s_filled = np.nan_to_num(s_sym, nan=0.0)

    if n <= 1:
        obs_rate = 1.0
    else:
        iu = np.triu_indices(n, k=1)
        obs_rate = float(mask[iu].mean()) if iu[0].size else 1.0
    obs_rate = float(np.clip(obs_rate, eps, 1.0))

    return s_filled, mask, obs_rate
```

- [ ] **Step 2: Run Layer 1 tests**

```bash
poetry run pytest tests/test_coherence.py::TestSymmetrize tests/test_coherence.py::TestObservationMask -v
```

Expected: All Layer 1 tests pass.

- [ ] **Step 3: Commit**

```bash
git add pysrf/coherence.py
git commit -m "feat(coherence): add matrix preparation helpers"
```

---

### Task 4: Write Layer 2 (reference eigenspace)

**Files:**
- Modify: `pysrf/coherence.py`

- [ ] **Step 1: Append Layer 2 to coherence.py**

```python
# ---------------------------------------------------------------------------
# Layer 2: Reference eigenspace
# ---------------------------------------------------------------------------


def _reference_eigenpairs(
    s: np.ndarray, k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k eigenpairs of a symmetric matrix in descending order.

    Parameters
    ----------
    s : (n, n) array
        Symmetric similarity matrix (fully observed, no NaN).
    k : int
        Number of top eigenpairs to return.

    Returns
    -------
    evals : (k,) array
        Top-k eigenvalues, descending.
    evecs : (n, k) array
        Corresponding eigenvectors with deterministic sign convention.
    """
    s = 0.5 * (s + s.T)
    evals_all, evecs_all = la.eigh(s)
    idx = np.argsort(evals_all)[::-1][:k]
    evals = evals_all[idx]
    evecs = evecs_all[:, idx]
    evecs = _sign_normalize(evecs)
    return evals, evecs


def _sign_normalize(q: np.ndarray) -> np.ndarray:
    """Fix sign ambiguity: largest-magnitude entry in each column is positive."""
    for j in range(q.shape[1]):
        i = int(np.argmax(np.abs(q[:, j])))
        if q[i, j] < 0.0:
            q[:, j] *= -1.0
    return q
```

- [ ] **Step 2: Run Layer 2 tests**

```bash
poetry run pytest tests/test_coherence.py::TestReferenceEigenpairs -v
```

Expected: All Layer 2 tests pass.

- [ ] **Step 3: Commit**

```bash
git add pysrf/coherence.py
git commit -m "feat(coherence): add reference eigenspace computation"
```

---

### Task 5: Write Layer 3 (bootstrap coherence engine)

**Files:**
- Modify: `pysrf/coherence.py`

- [ ] **Step 1: Append bootstrap helpers to coherence.py**

```python
# ---------------------------------------------------------------------------
# Layer 3: Bootstrap coherence engine
# ---------------------------------------------------------------------------


def _bootstrap_sample(
    s_filled: np.ndarray,
    mask: np.ndarray,
    p: float,
    rng: np.random.Generator,
    iu: tuple[np.ndarray, np.ndarray],
    eps: float = 1e-12,
) -> np.ndarray:
    """One symmetric Bernoulli-masked bootstrap replicate.

    Off-diagonal entries sampled with probability p, rescaled by 1/p.
    Missing entries suppressed through the observation mask.
    Diagonal is preserved.

    Parameters
    ----------
    s_filled : (n, n) array
        Similarity matrix with NaN replaced by zero.
    mask : (n, n) array
        Observation mask (0/1).
    p : float
        Masking probability in (0, 1].
    rng : Generator
        Random number generator.
    iu : tuple
        Upper triangle indices (precomputed for efficiency).

    Returns
    -------
    a : (n, n) symmetric array
        Bootstrap-masked matrix.
    """
    n = s_filled.shape[0]
    p = float(p)
    scale = 1.0 / max(p, eps)

    bern = (rng.random(iu[0].size) < p).astype(float)
    vals = bern * mask[iu] * s_filled[iu] * scale

    a = np.zeros((n, n), dtype=float)
    a[iu] = vals
    a[(iu[1], iu[0])] = vals
    np.fill_diagonal(a, np.diag(s_filled))
    a = 0.5 * (a + a.T)
    return a


def _eigenspace_overlap(
    u_boot: np.ndarray, u_ref: np.ndarray,
) -> np.ndarray:
    """Incremental projection overlap between bootstrap and reference eigenvectors.

    Computes Iproj[k] = sum_{j<=k} (u_boot_j . u_ref_j)^2 for each k,
    measuring how much of the k-th reference eigendirection is captured
    by the first k bootstrap eigenvectors.

    Parameters
    ----------
    u_boot : (n, k_max) array
        Bootstrap eigenvectors.
    u_ref : (n, k_max) array
        Reference eigenvectors.

    Returns
    -------
    iproj : (k_max,) array
        Overlap values in [0, 1] for each dimension.
    """
    g = u_boot.T @ u_ref
    g2 = g * g
    k_max = g2.shape[0]
    iproj = np.cumsum(g2, axis=0)[np.arange(k_max), np.arange(k_max)]
    return np.clip(iproj, 0.0, 1.0)


def _top_eigenvectors(a: np.ndarray, k: int) -> np.ndarray:
    """Top-k eigenvectors of a symmetric matrix.

    Tries scipy.sparse.linalg.eigsh for efficiency; falls back to dense eigh.

    Returns
    -------
    evecs : (n, k) array
        Top-k eigenvectors in descending eigenvalue order.
    """
    n = a.shape[0]
    eigsh = _try_eigsh()
    if eigsh is not None and k < n:
        try:
            vals, vecs = eigsh(a, k=k, which="LA", tol=1e-6)
            idx = np.argsort(vals)[::-1]
            return vecs[:, idx]
        except Exception:
            pass

    vals, vecs = la.eigh(a)
    idx = np.argsort(vals)[::-1][:k]
    return vecs[:, idx]


def _try_eigsh():
    """Return scipy.sparse.linalg.eigsh if available, else None."""
    try:
        from scipy.sparse.linalg import eigsh
        return eigsh
    except Exception:
        return None
```

- [ ] **Step 2: Append the main bootstrap loop**

```python
def _worker_one_p(args: tuple) -> tuple[int, np.ndarray]:
    """Bootstrap coherence for one masking rate p.

    Computes Iproj for all bootstrap replicates at a single p value.
    Designed for parallel dispatch via joblib.

    Returns
    -------
    (i, iproj_p) where iproj_p has shape (k_max, n_boot).
    """
    (i, p, s_filled, mask, u_ref, k_max, n_boot, seed_base) = args

    n = s_filled.shape[0]
    iu = np.triu_indices(n, k=1)
    iproj_p = np.zeros((k_max, n_boot), dtype=float)

    for b in range(n_boot):
        seed = (seed_base + 1000003 * i + 9176 * b) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)

        a = _bootstrap_sample(s_filled, mask, p, rng, iu)
        u_boot = _top_eigenvectors(a, k_max)
        iproj_p[:, b] = _eigenspace_overlap(u_boot, u_ref)

    return i, iproj_p


def _bootstrap_coherence(
    s: np.ndarray,
    k_max: int,
    p_list: np.ndarray,
    n_boot: int = 20,
    random_state: int = 0,
    n_jobs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute bootstrap eigenspace coherence across masking rates.

    Parameters
    ----------
    s : (n, n) array
        Symmetric similarity matrix (may contain NaN).
    k_max : int
        Maximum number of dimensions to test.
    p_list : (n_p,) array
        Masking probabilities in (0, 1].
    n_boot : int
        Number of bootstrap replicates per masking rate.
    random_state : int
        Seed for reproducibility.
    n_jobs : int or None
        Parallel workers (None = cpu_count - 1).

    Returns
    -------
    iproj_boot : (k_max, n_p, n_boot) array
        Bootstrap overlap values.
    evals_ref : (k_max,) array
        Reference eigenvalues.
    """
    import os
    from joblib import Parallel, delayed

    s_sym = _symmetrize(s)
    s_filled, mask, _ = _observation_mask(s_sym)
    evals_ref, u_ref = _reference_eigenpairs(s_filled, k_max)

    p_list = np.asarray(p_list, dtype=float)
    n_p = len(p_list)

    cpu = os.cpu_count() or 2
    if n_jobs is None:
        n_jobs_eff = max(1, cpu - 1)
    else:
        n_jobs_eff = max(1, min(int(n_jobs), cpu - 1))

    tasks = [
        (i, float(p_list[i]), s_filled, mask, u_ref, k_max, n_boot, random_state)
        for i in range(n_p)
    ]

    if n_jobs_eff == 1:
        results = [_worker_one_p(t) for t in tasks]
    else:
        results = Parallel(n_jobs=n_jobs_eff, backend="loky")(
            delayed(_worker_one_p)(t) for t in tasks
        )

    results.sort(key=lambda x: x[0])
    iproj_boot = np.zeros((k_max, n_p, n_boot), dtype=float)
    for i, iproj_p in results:
        iproj_boot[:, i, :] = iproj_p

    return iproj_boot, evals_ref
```

- [ ] **Step 3: Run Layer 3 tests**

```bash
poetry run pytest tests/test_coherence.py::TestBootstrapSample tests/test_coherence.py::TestEigenspaceOverlap tests/test_coherence.py::TestBootstrapCoherence -v
```

Expected: All Layer 3 tests pass. The `test_matches_reference` test is the critical floating-point regression check.

- [ ] **Step 4: Commit**

```bash
git add pysrf/coherence.py
git commit -m "feat(coherence): add bootstrap coherence engine"
```

---

### Task 6: Write Layer 4 (kappa estimation) and public API

**Files:**
- Modify: `pysrf/coherence.py`

- [ ] **Step 1: Append kappa helpers**

```python
# ---------------------------------------------------------------------------
# Layer 4: Kappa estimation
# ---------------------------------------------------------------------------


def _scaled_leakage(
    iproj_median: np.ndarray,
    p_list: np.ndarray,
    hi_quantile: float = 0.85,
) -> np.ndarray:
    """Estimate kappa per dimension from the high-p band.

    For each dimension k, the scaled leakage is:
        ell_k(p) = (1 - Iproj_median[k, p]) * p / (1 - p)

    Kappa is the median of ell_k over masking rates in the high-p band.
    Signal dimensions have low kappa; noise dimensions leak and have high kappa.

    Parameters
    ----------
    iproj_median : (k_max, n_p) array
        Median Iproj across bootstrap replicates.
    p_list : (n_p,) array
        Masking probabilities.
    hi_quantile : float
        Quantile of p_list defining the high-p band (default 0.85).

    Returns
    -------
    kappa : (k_max,) array
        Scaled leakage per dimension.
    """
    p_list = np.asarray(p_list, dtype=float)
    p_hi = float(np.quantile(p_list, hi_quantile))
    hi_idx = np.where(p_list >= p_hi)[0]
    if hi_idx.size == 0:
        hi_idx = np.array([len(p_list) - 1], dtype=int)

    scale = p_list[hi_idx] / np.maximum(1.0 - p_list[hi_idx], 1e-12)
    ell = (1.0 - iproj_median[:, hi_idx]) * scale[np.newaxis, :]
    return np.median(ell, axis=1).astype(float)


def _smooth_median(x: np.ndarray, w: int) -> np.ndarray:
    """Running median smoother."""
    if w <= 1:
        return x.copy()
    n = x.size
    out = np.empty(n, dtype=float)
    half = w // 2
    for i in range(n):
        out[i] = np.median(x[max(0, i - half):min(n, i + half + 1)])
    return out


def _largest_jump(
    kappa: np.ndarray,
    k_list: np.ndarray,
    smooth_window: int = 5,
    min_k: int = 2,
) -> int:
    """Find the changepoint where kappa jumps from signal to noise.

    Returns k_cut: dimensions {k <= k_cut} are signal.

    Parameters
    ----------
    kappa : (K,) array
        Scaled leakage per dimension.
    k_list : (K,) array
        Dimension indices.
    smooth_window : int
        Window for median smoothing before differentiation.
    min_k : int
        Minimum k to consider for the changepoint.

    Returns
    -------
    k_cut : int
        Last signal dimension.
    """
    kappa = np.asarray(kappa, dtype=float)
    k_list = np.asarray(k_list, dtype=int)

    if kappa.size < 3:
        return int(k_list[-1])

    sm = _smooth_median(kappa, max(1, smooth_window))
    d = sm[1:] - sm[:-1]

    valid = np.where(k_list[:-1] >= min_k)[0]
    if valid.size == 0:
        valid = np.arange(kappa.size - 1)

    i_star = int(valid[np.argmax(d[valid])])
    return int(k_list[i_star])
```

- [ ] **Step 2: Append public API**

```python
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_rank(
    s: np.ndarray,
    k_max: int,
    p_list: np.ndarray | None = None,
    n_boot: int = 20,
    random_state: int = 0,
    n_jobs: int | None = None,
    hi_quantile: float = 0.85,
    smooth_window: int = 5,
) -> dict:
    """Estimate the number of signal dimensions via kappa changepoint.

    Computes bootstrap eigenspace coherence, estimates scaled leakage
    (kappa) per dimension, and finds the changepoint where kappa jumps
    from signal to noise levels.

    Parameters
    ----------
    s : (n, n) array
        Symmetric similarity matrix (may contain NaN for missing entries).
    k_max : int
        Maximum number of dimensions to test.
    p_list : (n_p,) array or None
        Masking probabilities in (0, 1]. If None, uses linspace(0.1, 0.95, 20).
    n_boot : int
        Number of bootstrap replicates per masking rate.
    random_state : int
        Seed for reproducibility.
    n_jobs : int or None
        Parallel workers (None = cpu_count - 1).
    hi_quantile : float
        Quantile of p_list defining the high-p band for kappa estimation.
    smooth_window : int
        Window for median smoothing in changepoint detection.

    Returns
    -------
    result : dict
        k_star : int
            Estimated number of signal dimensions.
        kappa : (k_max,) array
            Scaled leakage per dimension.
        iproj_median : (k_max, n_p) array
            Median bootstrap overlap per dimension and masking rate.
        iproj_boot : (k_max, n_p, n_boot) array
            Raw bootstrap overlap values.
        evals_ref : (k_max,) array
            Reference eigenvalues.
        k_list : (k_max,) array
            Dimension indices 1..k_max.
        p_list : (n_p,) array
            Masking probabilities used.
    """
    if p_list is None:
        p_list = np.linspace(0.1, 0.95, 20)
    p_list = np.asarray(p_list, dtype=float)

    k_list = np.arange(1, k_max + 1)

    iproj_boot, evals_ref = _bootstrap_coherence(
        s, k_max=k_max, p_list=p_list, n_boot=n_boot,
        random_state=random_state, n_jobs=n_jobs,
    )

    iproj_median = np.median(iproj_boot, axis=2)
    kappa = _scaled_leakage(iproj_median, p_list, hi_quantile=hi_quantile)
    k_star = _largest_jump(kappa, k_list, smooth_window=smooth_window)

    return {
        "k_star": k_star,
        "kappa": kappa,
        "iproj_median": iproj_median,
        "iproj_boot": iproj_boot,
        "evals_ref": evals_ref,
        "k_list": k_list,
        "p_list": p_list,
    }
```

- [ ] **Step 3: Run all kappa + public API tests**

```bash
poetry run pytest tests/test_coherence.py::TestScaledLeakage tests/test_coherence.py::TestLargestJump tests/test_coherence.py::TestEstimateRank -v
```

Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add pysrf/coherence.py
git commit -m "feat(coherence): add kappa estimation and estimate_rank API"
```

---

### Task 7: Update __init__.py and run full test suite

**Files:**
- Modify: `pysrf/__init__.py`

- [ ] **Step 1: Add estimate_rank to exports**

Add to the imports in `pysrf/__init__.py`:

```python
from .coherence import estimate_rank
```

Add `"estimate_rank"` to the `__all__` list.

- [ ] **Step 2: Run all coherence tests**

```bash
poetry run pytest tests/test_coherence.py -v
```

Expected: All tests pass.

- [ ] **Step 3: Run full test suite to check nothing is broken**

```bash
poetry run pytest tests/ -v --tb=short
```

Expected: All existing tests still pass (coherence was not previously imported anywhere in the test suite).

- [ ] **Step 4: Commit**

```bash
git add pysrf/__init__.py
git commit -m "feat: export estimate_rank from pysrf"
```

---

### Task 8: Verify the regression — exact floating-point match

This is the critical validation step. Run the old and new implementations side by side and assert identical outputs.

**Files:**
- No new files (verification only)

- [ ] **Step 1: Run end-to-end comparison**

```bash
poetry run python -c "
import sys
sys.path.insert(0, 'tests')
import numpy as np
from pysrf.coherence import estimate_rank

# Same matrix as reference generation
rng = np.random.default_rng(42)
w = np.abs(rng.standard_normal((30, 5)))
s = w @ w.T
s += 0.01 * rng.standard_normal((30, 30))
s = (s + s.T) / 2

ref = dict(np.load('tests/coherence_reference.npz', allow_pickle=False))

result = estimate_rank(
    s, k_max=15,
    p_list=ref['p_list'],
    n_boot=5,
    random_state=0,
    n_jobs=1,
)

# Check k_star
assert result['k_star'] == int(ref['k_star']), f'{result[\"k_star\"]} != {int(ref[\"k_star\"])}'

# Check kappa (exact float match)
assert np.allclose(result['kappa'], ref['kappa'], atol=1e-12), 'kappa mismatch'

# Check iproj_median (exact float match)
assert np.allclose(result['iproj_median'], ref['iproj_median'], atol=1e-12), 'iproj_median mismatch'

# Check eigenvalues
assert np.allclose(result['evals_ref'], ref['evals_ref'][:15], atol=1e-10), 'evals mismatch'

print('ALL REGRESSION CHECKS PASSED')
"
```

Expected: `ALL REGRESSION CHECKS PASSED`

- [ ] **Step 2: Verify line count is reasonable**

```bash
wc -l pysrf/coherence.py
```

Expected: ~350-450 lines (down from ~3100).

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "test: verify coherence rewrite produces identical results"
```
