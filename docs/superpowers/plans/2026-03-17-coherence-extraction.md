# Coherence Extraction Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the 2915-line v2 coherence sandbox file into a clean `src/coherence/` package with one module per logical group, maintaining numerical equivalence.

**Architecture:** Copy functions from `sandbox/coherence/kachun/v2/_compute_coherence.py` into `src/coherence/{masking,eigensolve,workers,compute,analysis,cluster,_correlation}.py`. Remove leading underscores from public APIs, add type hints, strip plotly code. Update all sandbox imports. Validate numerical equivalence with `np.allclose` tests.

**Tech Stack:** NumPy, SciPy (eigsh/lobpcg), joblib, pytest. Source file: `sandbox/coherence/kachun/v2/_compute_coherence.py` (v2). Tests: `tests/test_coherence/`.

**Spec:** `docs/specs/2026-03-17-coherence-extraction-design.md`

---

## File Structure

### New files to create

| File | Responsibility | Approx lines |
|------|---------------|--------------|
| `src/coherence/__init__.py` | Public API exports | ~30 |
| `src/coherence/masking.py` | NaN-aware symmetrization, observation masks, Bernoulli bootstrap sampling | ~110 |
| `src/coherence/eigensolve.py` | Top-k eigensolvers: exact, warm-start, randomized | ~200 |
| `src/coherence/workers.py` | Bootstrap worker functions dispatched by joblib | ~180 |
| `src/coherence/compute.py` | Main `compute_coherence` orchestrator | ~300 |
| `src/coherence/analysis.py` | Baseline correction, kappa estimation, activation detection, FDR | ~150 |
| `src/coherence/cluster.py` | Cluster consensus across p-prefixes | ~400 |
| `src/coherence/_correlation.py` | Pearson, Spearman, MI matrices (pairwise and row-wise) | ~300 |
| `tests/test_coherence/__init__.py` | Test package | 0 |
| `tests/test_coherence/test_masking.py` | Masking numerical equivalence tests | ~80 |
| `tests/test_coherence/test_eigensolve.py` | Eigensolver equivalence tests | ~80 |
| `tests/test_coherence/test_correlation.py` | Correlation equivalence tests | ~80 |
| `tests/test_coherence/test_analysis.py` | Analysis function equivalence tests | ~80 |
| `tests/test_coherence/test_cluster.py` | Cluster/trend correlation equivalence tests | ~80 |
| `tests/test_coherence/test_compute.py` | End-to-end coherence integration test | ~60 |

### Spec deviations

- **`src/tools/coherence.py` rename**: The spec says to rename `src/tools/coherence.py` to `src/tools/bounds.py`. This file does not exist in the current codebase (`src/tools/` contains only `__init__.py`, `metrics.py`, `rsa.py`, `stats.py`). The two sandbox scripts (`sandbox/samuel/{coherence,explore}/run.py`) that import from it are already broken. This rename is **N/A**.
- **`run_things_cv.py`**: Exists in `sandbox/coherence/kachun/v2/` but does not import from `_compute_coherence`. No changes needed.

### Files to modify

| File | Change |
|------|--------|
| `src/coherence.py` | Delete (empty file) |
| `tests/test_structure.py` | Add `test_coherence_imports()` |

### Files with import updates (sandbox -- do NOT modify until Task 10)

| File | Current import | New import |
|------|---------------|------------|
| `sandbox/coherence/subspace_rank/_subspace_coherence.py` | `from _compute_coherence import _symmetrize_with_nan, ...` | `from src.coherence.masking import symmetrize_with_nan, ...` etc. |
| `sandbox/coherence/kachun/v2/run_*.py` (5 files) | `from _compute_coherence import compute_incremental_coherence_multi_k_eig_anisotropic` | `from src.coherence import compute_coherence` |
| `sandbox/coherence/subspace_rank/run_diagnostic*.py` (4 files) | same | same |
| `sandbox/swow/ppmi_coherence/run.py` | same | same |

---

## Task 1: Scaffold package and masking module

**Files:**
- Delete: `src/coherence.py`
- Create: `src/coherence/__init__.py`
- Create: `src/coherence/masking.py`
- Test: `tests/test_coherence/test_masking.py`

**Source lines:** `_compute_coherence.py` lines 29-139, 217-251

- [ ] **Step 1: Delete the empty `src/coherence.py`**

```bash
rm src/coherence.py
```

- [ ] **Step 2: Create `src/coherence/__init__.py` (stub)**

```python
"""Coherence analysis for symmetric matrix factorization under missing data."""
```

We will populate the public API exports in Task 8 after all modules exist.

- [ ] **Step 3: Create `tests/test_coherence/__init__.py`**

Empty file.

- [ ] **Step 4: Write failing tests for masking functions**

Create `tests/test_coherence/test_masking.py`. Each test calls both the v2 original and new version with identical inputs, asserts `np.allclose`.

```python
"""Numerical equivalence tests: src/coherence/masking vs v2 _compute_coherence."""
import sys
from pathlib import Path

import numpy as np
import pytest

# Import v2 originals via sys.path (temporary, for testing only)
_V2 = str(Path(__file__).resolve().parents[2] / "sandbox" / "coherence" / "kachun" / "v2")
if _V2 not in sys.path:
    sys.path.insert(0, _V2)
from _compute_coherence import (
    _symmetrize_with_nan as v2_symmetrize,
    _prepare_observation_mask as v2_prepare_mask,
    _masked_unbiased_spd_missing as v2_masked_bootstrap,
    _masked_unbiased_spd_missing_from_uniform as v2_masked_from_uniform,
)

from src.coherence.masking import (
    symmetrize_with_nan,
    prepare_observation_mask,
    masked_bootstrap_sample,
    masked_bootstrap_from_uniform,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sim_with_nan(rng):
    """30x30 similarity matrix with ~20% NaN off-diagonal."""
    n = 30
    s = rng.standard_normal((n, n))
    s = s @ s.T / n
    mask = rng.random((n, n)) < 0.2
    np.fill_diagonal(mask, False)
    s[mask] = np.nan
    return s


def test_symmetrize_with_nan(sim_with_nan):
    new = symmetrize_with_nan(sim_with_nan)
    old = v2_symmetrize(sim_with_nan)
    assert np.allclose(new, old, atol=1e-12, equal_nan=True)


def test_prepare_observation_mask(sim_with_nan):
    s_sym = symmetrize_with_nan(sim_with_nan)
    s0_new, w_new, q_new = prepare_observation_mask(s_sym)
    s0_old, w_old, q_old = v2_prepare_mask(s_sym)
    assert np.allclose(s0_new, s0_old, atol=1e-12)
    assert np.allclose(w_new, w_old, atol=1e-12)
    assert abs(q_new - q_old) < 1e-12


def test_masked_bootstrap_sample(sim_with_nan):
    s_sym = symmetrize_with_nan(sim_with_nan)
    s0, w, q = prepare_observation_mask(s_sym)
    p = 0.5
    rng_new = np.random.default_rng(99)
    rng_old = np.random.default_rng(99)
    new = masked_bootstrap_sample(s0, w, q, p, rng_new)
    old = v2_masked_bootstrap(s0, w, q, p, rng_old)
    assert np.allclose(new, old, atol=1e-12)


def test_masked_bootstrap_from_uniform(sim_with_nan):
    s_sym = symmetrize_with_nan(sim_with_nan)
    s0, w, q = prepare_observation_mask(s_sym)
    n = s0.shape[0]
    iu = np.triu_indices(n, k=1)
    rng = np.random.default_rng(77)
    edge_u = rng.random(iu[0].size)
    p = 0.4
    new = masked_bootstrap_from_uniform(s0, w, p, edge_u, iu)
    old = v2_masked_from_uniform(s0, w, p, edge_u, iu)
    assert np.allclose(new, old, atol=1e-12)
```

- [ ] **Step 5: Run tests to confirm they fail**

```bash
poetry run pytest tests/test_coherence/test_masking.py -v
```

Expected: FAIL (ImportError -- `src.coherence.masking` does not exist yet).

- [ ] **Step 6: Implement `src/coherence/masking.py`**

Copy functions from v2 lines 29-139, 217-251. Changes:
- Remove leading underscores from public names
- Add type hints (use `NDArray[np.floating]` for arrays)
- Skip `_apply_unbiased_missingness_scaling` (line 95, unused per spec)
- Keep internal logic identical

```python
"""NaN-aware Bernoulli masking for coherence bootstrap."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def symmetrize_with_nan(s: NDArray[np.floating]) -> NDArray[np.floating]:
    """Symmetrize a matrix while preserving missingness semantics.

    For each pair (i, j):
    - average if both entries are finite,
    - copy the finite value if only one side is finite,
    - keep NaN if both are missing.
    The diagonal is forced finite by replacing missing values with zero.
    """
    s = np.asarray(s, float)
    st = s.T
    a = np.isfinite(s)
    b = np.isfinite(st)

    out = np.full_like(s, np.nan, dtype=float)

    both = a & b
    out[both] = 0.5 * (s[both] + st[both])

    only_a = a & (~b)
    out[only_a] = s[only_a]

    only_b = (~a) & b
    out[only_b] = st[only_b]

    d = np.diag(out).copy()
    d[~np.isfinite(d)] = 0.0
    np.fill_diagonal(out, d)
    return out


def prepare_observation_mask(
    s_sym: NDArray[np.floating],
    keep_diag: bool = True,
    eps: float = 1e-12,
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    """Build zero-filled matrix, binary observation mask, and off-diagonal observed rate.

    Parameters
    ----------
    s_sym : (n, n) symmetric array, may contain NaN
    keep_diag : if True, diagonal always counted as observed
    eps : numerical stability floor

    Returns
    -------
    s0 : (n, n) copy of s_sym with NaN replaced by 0
    w : (n, n) binary mask (1 where observed, 0 where NaN)
    q : fraction of off-diagonal entries that are observed
    """
    # Copy from v2 lines 62-92 exactly
    s_sym = np.asarray(s_sym, float)
    n = s_sym.shape[0]

    w = np.isfinite(s_sym).astype(float)
    s0 = np.where(np.isfinite(s_sym), s_sym, 0.0)

    if keep_diag:
        np.fill_diagonal(w, 1.0)
        np.fill_diagonal(s0, np.where(np.isfinite(np.diag(s_sym)), np.diag(s_sym), 0.0))

    n_offdiag = n * (n - 1)
    if n_offdiag == 0:
        q = 1.0
    else:
        diag_mask = np.eye(n, dtype=float)
        q = float(np.sum(w * (1.0 - diag_mask)) / n_offdiag)
    q = max(q, eps)
    return s0, w, q


def masked_bootstrap_sample(
    s0: NDArray[np.floating],
    w: NDArray[np.floating],
    q: float,
    p: float,
    rng: np.random.Generator,
    keep_diag: bool = True,
    eps: float = 1e-12,
    iu: tuple[NDArray[np.intp], NDArray[np.intp]] | None = None,
) -> NDArray[np.floating]:
    """Sample a symmetric Bernoulli-masked matrix for one bootstrap replicate.

    Off-diagonal entries are sampled with probability p and rescaled by 1/p.
    Missing entries are suppressed through w.
    """
    n = s0.shape[0]
    p = float(p)

    m = np.zeros((n, n), dtype=float)
    if iu is None:
        iu = np.triu_indices(n, k=1)

    mask_u = (rng.random(iu[0].size) < p).astype(float)
    m[iu] = mask_u
    m[(iu[1], iu[0])] = mask_u

    mw = m * w

    a = mw * s0
    scale_off = 1.0 / max(p, eps)
    a[iu] *= scale_off
    a[(iu[1], iu[0])] *= scale_off

    if keep_diag:
        np.fill_diagonal(a, np.diag(s0))
    else:
        dmask = (rng.random(n) < p).astype(float)
        np.fill_diagonal(a, dmask * np.diag(s0) / max(p, eps))

    a = 0.5 * (a + a.T)
    return a


def masked_bootstrap_from_uniform(
    s0: NDArray[np.floating],
    w: NDArray[np.floating],
    p: float,
    edge_u: NDArray[np.floating],
    iu: tuple[NDArray[np.intp], NDArray[np.intp]],
    keep_diag: bool = True,
    diag_u: NDArray[np.floating] | None = None,
    eps: float = 1e-12,
) -> NDArray[np.floating]:
    """Build one symmetric bootstrap matrix from pre-sampled uniforms.

    Reusing the same edge_u across increasing p yields a nested mask path.
    """
    n = s0.shape[0]
    p = float(p)
    scale = 1.0 / max(p, eps)

    vals = ((edge_u < p).astype(float) * w[iu] * s0[iu]) * scale
    a = np.zeros((n, n), dtype=float)
    a[iu] = vals
    a[(iu[1], iu[0])] = vals

    if keep_diag:
        np.fill_diagonal(a, np.diag(s0))
    else:
        if diag_u is None:
            raise ValueError("diag_u is required when keep_diag=False.")
        dmask = (diag_u < p).astype(float)
        np.fill_diagonal(a, dmask * np.diag(s0) * scale)

    a = 0.5 * (a + a.T)
    return a
```

- [ ] **Step 7: Run tests to confirm they pass**

```bash
poetry run pytest tests/test_coherence/test_masking.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add src/coherence/__init__.py src/coherence/masking.py tests/test_coherence/__init__.py tests/test_coherence/test_masking.py
git rm src/coherence.py
git commit -m "feat(coherence): extract masking module from v2 sandbox"
```

---

## Task 2: Eigensolver module

**Files:**
- Create: `src/coherence/eigensolve.py`
- Test: `tests/test_coherence/test_eigensolve.py`

**Source lines:** `_compute_coherence.py` lines 20-26, 142-204, 208-336

- [ ] **Step 1: Write failing tests**

Create `tests/test_coherence/test_eigensolve.py`:

```python
"""Numerical equivalence tests: src/coherence/eigensolve vs v2."""
import sys
from pathlib import Path

import numpy as np
import pytest

_V2 = str(Path(__file__).resolve().parents[2] / "sandbox" / "coherence" / "kachun" / "v2")
if _V2 not in sys.path:
    sys.path.insert(0, _V2)
from _compute_coherence import (
    _topk_eigenvectors as v2_topk,
    _randomized_topk_eigenspace_symmetric as v2_randomized,
    _orthonormalize_columns as v2_orthonorm,
    _topk_eigenvectors_exact_warm as v2_warm,
)

from src.coherence.eigensolve import (
    topk_eigenvectors,
    randomized_topk_eigenspace,
    orthonormalize_columns,
    topk_eigenvectors_warm,
)


@pytest.fixture
def spd_matrix():
    rng = np.random.default_rng(123)
    a = rng.standard_normal((50, 50))
    return a @ a.T / 50


def test_topk_eigenvectors(spd_matrix):
    vals_new, vecs_new = topk_eigenvectors(spd_matrix, 5)
    vals_old, vecs_old = v2_topk(spd_matrix, 5)
    assert np.allclose(vals_new, vals_old, atol=1e-10)
    # Eigenvectors may differ by sign; compare absolute dot products
    dots = np.abs(np.sum(vecs_new * vecs_old, axis=0))
    assert np.allclose(dots, 1.0, atol=1e-10)


def test_randomized_topk(spd_matrix):
    vals_new, vecs_new = randomized_topk_eigenspace(spd_matrix, 5)
    vals_old, vecs_old = v2_randomized(spd_matrix, 5)
    assert np.allclose(vals_new, vals_old, atol=1e-10)


def test_orthonormalize(spd_matrix):
    x = spd_matrix[:, :5]
    new = orthonormalize_columns(x)
    old = v2_orthonorm(x)
    assert np.allclose(new, old, atol=1e-12)


def test_warm_start(spd_matrix):
    vals_new, vecs_new, info_new = topk_eigenvectors_warm(spd_matrix, 5)
    vals_old, vecs_old, info_old = v2_warm(spd_matrix, 5)
    assert np.allclose(vals_new, vals_old, atol=1e-10)
    dots = np.abs(np.sum(vecs_new * vecs_old, axis=0))
    assert np.allclose(dots, 1.0, atol=1e-10)
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
poetry run pytest tests/test_coherence/test_eigensolve.py -v
```

Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `src/coherence/eigensolve.py`**

Copy from v2 lines 20-26, 142-204, 208-336. Key changes:
- `_try_import_eigsh` -> `_try_import_eigsh` (keep private, internal helper)
- `_try_import_lobpcg` -> `_try_import_lobpcg` (keep private)
- `_topk_eigenvectors` -> `topk_eigenvectors`
- `_randomized_topk_eigenspace_symmetric` -> `randomized_topk_eigenspace`
- `_orthonormalize_columns` -> `orthonormalize_columns`
- `_topk_eigenvectors_exact_warm` -> `topk_eigenvectors_warm`
- Add type hints to all public signatures
- `orthonormalize_columns` is used internally by `topk_eigenvectors_warm`, keep reference working

```python
"""Top-k eigensolvers with warm starts for coherence bootstrap."""
from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
import numpy.linalg as la
from numpy.typing import NDArray


@lru_cache(maxsize=1)
def _try_import_eigsh():
    """Return scipy.sparse.linalg.eigsh when SciPy is available, else None."""
    try:
        from scipy.sparse.linalg import eigsh
        return eigsh
    except Exception:
        return None


@lru_cache(maxsize=1)
def _try_import_lobpcg():
    """Return scipy.sparse.linalg.lobpcg when SciPy is available, else None."""
    try:
        from scipy.sparse.linalg import lobpcg
        return lobpcg
    except Exception:
        return None


def topk_eigenvectors(
    a: NDArray[np.floating],
    k: int,
    eigsh=None,
    tol: float = 1e-6,
    maxiter: int | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return top-k eigenpairs of a symmetric matrix in descending order."""
    # Copy from v2 lines 142-156 exactly
    n = a.shape[0]
    k = int(k)
    if k <= 0:
        return np.array([], float), np.zeros((n, 0), float)

    if eigsh is not None and k < n:
        vals, vecs = eigsh(a, k=k, which="LA", tol=tol, maxiter=maxiter)
        idx = np.argsort(vals)[::-1]
        return vals[idx], vecs[:, idx]
    else:
        vals, vecs = la.eigh(a)
        idx = np.argsort(vals)[::-1][:k]
        return vals[idx], vecs[:, idx]


def randomized_topk_eigenspace(
    a: NDArray[np.floating],
    k: int,
    oversample: int = 10,
    n_iter: int = 2,
    random_state: int = 0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute approximate top-k symmetric eigenspace via random projection."""
    # Copy from v2 lines 159-187 exactly
    n = a.shape[0]
    k = int(k)
    l = int(min(n, k + int(oversample)))

    rng = np.random.default_rng(int(random_state) & 0xFFFFFFFF)
    omega = rng.standard_normal((n, l))
    y = a @ omega

    for _ in range(int(n_iter)):
        y = a @ (a @ y)

    q, _ = la.qr(y, mode="reduced")
    b = q.T @ a @ q
    b = 0.5 * (b + b.T)

    evals, evecs_small = la.eigh(b)
    idx = np.argsort(evals)[::-1][:k]
    evals_top = evals[idx]
    evecs_top = q @ evecs_small[:, idx]
    return evals_top, evecs_top


def orthonormalize_columns(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return a QR-orthonormal basis with deterministic column signs."""
    x = np.asarray(x, float)
    if x.ndim != 2:
        raise ValueError("X must be a 2D array.")
    n, k = x.shape
    if k == 0:
        return np.zeros((n, 0), float)

    q, _ = la.qr(x, mode="reduced")
    for j in range(q.shape[1]):
        idx = int(np.argmax(np.abs(q[:, j])))
        if q[idx, j] < 0.0:
            q[:, j] *= -1.0
    return q


def topk_eigenvectors_warm(
    a: NDArray[np.floating],
    k: int,
    x_init: NDArray[np.floating] | None = None,
    solver: str = "auto",
    tol: float = 1e-10,
    maxiter: int = 200,
    residual_tol: float = 1e-8,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict]:
    """Exact top-k eigensolve with warm-start fast path and dense fallback.

    The returned eigenpairs are validated by residual norm. If the warm-started
    iterative solve does not meet the residual tolerance, falls back to dense eigh.
    """
    # Copy from v2 lines 254-336 exactly
    a = 0.5 * (np.asarray(a, float) + np.asarray(a, float).T)
    n = a.shape[0]
    k = int(k)
    if k <= 0:
        return np.array([], float), np.zeros((n, 0), float), {
            "solver": "none", "fallback": False, "max_residual": 0.0,
        }

    lobpcg = _try_import_lobpcg()
    use_lobpcg = (
        solver in ("auto", "lobpcg")
        and lobpcg is not None
        and k < n
        and n > (8 * k)
    )

    if use_lobpcg:
        if x_init is None or np.shape(x_init) != (n, k):
            rng = np.random.default_rng(0)
            x = rng.standard_normal((n, k))
        else:
            x = np.array(x_init, dtype=float, copy=True)
        x = orthonormalize_columns(x)

        try:
            with warnings.catch_warnings(record=True) as wlog:
                warnings.simplefilter("always")
                vals, vecs = lobpcg(
                    a, x, largest=True, tol=float(tol),
                    maxiter=int(maxiter), verbosityLevel=0,
                )
            order = np.argsort(vals)[::-1]
            vals = np.asarray(vals, float)[order]
            vecs = np.asarray(vecs, float)[:, order]
            vecs = orthonormalize_columns(vecs)
            resid = la.norm(a @ vecs - vecs * vals[np.newaxis, :], axis=0)
            max_resid = float(np.max(resid)) if resid.size else 0.0
            warning_messages = [str(w.message) for w in wlog]
            warning_flag = any(msg for msg in warning_messages)
            if np.isfinite(vals).all() and (not warning_flag) and max_resid <= float(residual_tol):
                return vals, vecs, {
                    "solver": "lobpcg", "fallback": False,
                    "max_residual": max_resid, "warning_count": 0,
                }
        except Exception:
            pass

    evals_all, evecs_all = la.eigh(a)
    idx = np.argsort(evals_all)[::-1][:k]
    vals = np.asarray(evals_all[idx], float)
    vecs = orthonormalize_columns(np.asarray(evecs_all[:, idx], float))
    resid = la.norm(a @ vecs - vecs * vals[np.newaxis, :], axis=0)
    max_resid = float(np.max(resid)) if resid.size else 0.0
    return vals, vecs, {
        "solver": "eigh", "fallback": bool(use_lobpcg),
        "max_residual": max_resid, "warning_count": 1 if use_lobpcg else 0,
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
poetry run pytest tests/test_coherence/test_eigensolve.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/coherence/eigensolve.py tests/test_coherence/test_eigensolve.py
git commit -m "feat(coherence): extract eigensolve module from v2 sandbox"
```

---

## Task 3: Correlation module

**Files:**
- Create: `src/coherence/_correlation.py`
- Test: `tests/test_coherence/test_correlation.py`

**Source lines:** `_compute_coherence.py` lines 1324-1694

- [ ] **Step 1: Write failing tests**

Create `tests/test_coherence/test_correlation.py`:

```python
"""Numerical equivalence tests: src/coherence/_correlation vs v2."""
import sys
from pathlib import Path

import numpy as np
import pytest

_V2 = str(Path(__file__).resolve().parents[2] / "sandbox" / "coherence" / "kachun" / "v2")
if _V2 not in sys.path:
    sys.path.insert(0, _V2)
from _compute_coherence import (
    _rankdata_average_1d as v2_rankdata,
    _pearson_corr_matrix as v2_pearson_mat,
    _spearman_corr_matrix as v2_spearman_mat,
    _mutual_info_matrix as v2_mi_mat,
    _pearson_corr as v2_pearson,
    _spearman_corr as v2_spearman,
    _fd_bins as v2_fd_bins,
    _mutual_info_discrete as v2_mi_discrete,
    _mutual_info_matrix_pairwise as v2_mi_pairwise,
    _get_pair_metric as v2_get_pair_metric,
    _compute_all_trend_similarity_matrices as v2_compute_trend,
)

from src.coherence._correlation import (
    rankdata_average,
    pearson_corr_matrix,
    spearman_corr_matrix,
    mutual_info_matrix,
    pearson_corr,
    spearman_corr,
    fd_bins,
    mutual_info_discrete,
    mutual_info_matrix_pairwise,
    get_pair_metric,
    compute_trend_similarity,
)


@pytest.fixture
def trend_data():
    rng = np.random.default_rng(55)
    return rng.standard_normal((10, 20))  # 10 rows (k), 20 columns (p)


def test_rankdata(trend_data):
    row = trend_data[0]
    assert np.allclose(rankdata_average(row), v2_rankdata(row), atol=1e-12)


def test_pearson_matrix(trend_data):
    assert np.allclose(pearson_corr_matrix(trend_data), v2_pearson_mat(trend_data), atol=1e-12)


def test_spearman_matrix(trend_data):
    assert np.allclose(spearman_corr_matrix(trend_data), v2_spearman_mat(trend_data), atol=1e-12)


def test_mi_matrix(trend_data):
    assert np.allclose(mutual_info_matrix(trend_data), v2_mi_mat(trend_data), atol=1e-10)


def test_pearson_corr(trend_data):
    a, b = trend_data[0], trend_data[1]
    assert abs(pearson_corr(a, b) - v2_pearson(a, b)) < 1e-12


def test_spearman_corr(trend_data):
    a, b = trend_data[0], trend_data[1]
    assert abs(spearman_corr(a, b) - v2_spearman(a, b)) < 1e-12


def test_fd_bins(trend_data):
    assert fd_bins(trend_data[0]) == v2_fd_bins(trend_data[0])


def test_mi_discrete(trend_data):
    a, b = trend_data[0], trend_data[1]
    assert abs(mutual_info_discrete(a, b) - v2_mi_discrete(a, b)) < 1e-10


def test_mi_pairwise(trend_data):
    assert np.allclose(
        mutual_info_matrix_pairwise(trend_data),
        v2_mi_pairwise(trend_data),
        atol=1e-10,
    )


def test_compute_trend_similarity(trend_data):
    new = compute_trend_similarity(trend_data)
    old = v2_compute_trend(trend_data)
    for key in ("pearson", "spearman", "mutual_info"):
        assert np.allclose(new[key], old[key], atol=1e-10)


def test_get_pair_metric():
    fn_new, range_new = get_pair_metric("spearman")
    fn_old, range_old = v2_get_pair_metric("spearman")
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([4.0, 3.0, 2.0, 1.0])
    assert abs(fn_new(a, b) - fn_old(a, b)) < 1e-12
    assert range_new == range_old
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
poetry run pytest tests/test_coherence/test_correlation.py -v
```

- [ ] **Step 3: Implement `src/coherence/_correlation.py`**

Copy from v2 lines 1324-1694. Functions to include:
- `rankdata_average` (from `_rankdata_average_1d`)
- `pearson_corr_matrix` (from `_pearson_corr_matrix`)
- `spearman_corr_matrix` (from `_spearman_corr_matrix`)
- `mutual_info_matrix` (from `_mutual_info_matrix`)
- `compute_trend_similarity` (from `_compute_all_trend_similarity_matrices`)
- `pearson_corr` (from `_pearson_corr`)
- `spearman_corr` (from `_spearman_corr`)
- `fd_bins` (from `_fd_bins`)
- `mutual_info_discrete` (from `_mutual_info_discrete`)
- `mutual_info_matrix_pairwise` (from `_mutual_info_matrix_pairwise`)
- `get_pair_metric` (from `_get_pair_metric`)

Add type hints. Keep logic identical.

- [ ] **Step 4: Run tests to confirm they pass**

```bash
poetry run pytest tests/test_coherence/test_correlation.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/coherence/_correlation.py tests/test_coherence/test_correlation.py
git commit -m "feat(coherence): extract correlation module from v2 sandbox"
```

---

## Task 4: Analysis module

**Files:**
- Create: `src/coherence/analysis.py`
- Test: `tests/test_coherence/test_analysis.py`

**Source lines:** `_compute_coherence.py` lines 1451-1465, 2368-2395, 2429-2543, 2581-2595, and lines 729-747 (activation extraction from compute)

- [ ] **Step 1: Write failing tests**

Create `tests/test_coherence/test_analysis.py`:

```python
"""Numerical equivalence tests: src/coherence/analysis vs v2."""
import sys
from pathlib import Path

import numpy as np
import pytest

_V2 = str(Path(__file__).resolve().parents[2] / "sandbox" / "coherence" / "kachun" / "v2")
if _V2 not in sys.path:
    sys.path.insert(0, _V2)
from _compute_coherence import (
    _baseline_correct_array as v2_baseline,
    _bh_fdr_reject as v2_bh_fdr,
    _estimate_kappa_hat as v2_kappa,
    _smooth_median as v2_smooth,
    kappa_changepoint as v2_changepoint,
    _find_first_run_ge_threshold as v2_first_run,
)

from src.coherence.analysis import (
    baseline_correct,
    bh_fdr,
    estimate_kappa,
    smooth_median,
    kappa_changepoint,
    find_first_run_above,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_baseline_correct(rng):
    arr = rng.random((5, 20, 10))
    k_list = np.array([1, 5, 10, 20, 30])
    n = 100
    new = baseline_correct(arr, k_list, n)
    old = v2_baseline(arr, k_list, n)
    assert np.allclose(new, old, atol=1e-12)


def test_bh_fdr(rng):
    pvals = rng.random(20)
    assert np.array_equal(bh_fdr(pvals, 0.05), v2_bh_fdr(pvals, 0.05))


def test_estimate_kappa(rng):
    i_med = rng.random((10, 20))
    p_list = np.linspace(0.05, 0.95, 20)
    new_k, new_info = estimate_kappa(i_med, p_list, 0.85)
    old_k, old_info = v2_kappa(i_med, p_list, 0.85)
    assert np.allclose(new_k, old_k, atol=1e-12)


def test_smooth_median(rng):
    x = rng.random(30)
    assert np.allclose(smooth_median(x, 5), v2_smooth(x, 5), atol=1e-12)


def test_kappa_changepoint(rng):
    kappa = np.sort(rng.random(20))
    k_list = np.arange(1, 21)
    new_k, _ = kappa_changepoint(kappa, k_list)
    old_k, _ = v2_changepoint(kappa, k_list)
    assert new_k == old_k


def test_find_first_run_above():
    p_list = np.linspace(0.1, 1.0, 10)
    curve = np.array([0.0, 0.0, 0.5, 0.9, 0.95, 0.97, 0.98, 0.99, 0.99, 1.0])
    new = find_first_run_above(p_list, curve, 0.9, run=2)
    old = v2_first_run(p_list, curve, 0.9, run=2)
    assert new == old
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
poetry run pytest tests/test_coherence/test_analysis.py -v
```

- [ ] **Step 3: Implement `src/coherence/analysis.py`**

Copy functions from v2. Key mapping:
- `baseline_correct` from v2 `_baseline_correct_array` (lines 1451-1465)
- `bh_fdr` from v2 `_bh_fdr_reject` (lines 2368-2395)
- `estimate_kappa` from v2 `_estimate_kappa_hat` (lines 2429-2462)
- `smooth_median` from v2 `_smooth_median` (lines 2469-2480)
- `kappa_changepoint` from v2 `kappa_changepoint` (lines 2482-2543)
- `find_first_run_above` from v2 `_find_first_run_ge_threshold` (lines 2581-2595)
- `per_component_activation` -- extract from compute.py lines 729-747 into its own function

Drop: `eps_from_signal_quantile`, `analyze_iproj_signal_alignment_v3`, `IprojAnalysisResult`, `_try_beta_cdf`, `_quantile`, `_safe_log`, `_mad`, `_find_first_p_ge_threshold`, `_default_select_k_for_curves`.

Add type hints. Keep logic identical.

- [ ] **Step 4: Run tests to confirm they pass**

```bash
poetry run pytest tests/test_coherence/test_analysis.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/coherence/analysis.py tests/test_coherence/test_analysis.py
git commit -m "feat(coherence): extract analysis module from v2 sandbox"
```

---

## Task 5: Workers module

**Files:**
- Create: `src/coherence/workers.py`

**Source lines:** `_compute_coherence.py` lines 339-425, 432-513

No separate test file for workers -- they are tested through the integration test in Task 7.

- [ ] **Step 1: Implement `src/coherence/workers.py`**

Copy from v2. Key mapping:
- `worker_one_p` from v2 `_eig_worker_one_p` (lines 432-513)
- `worker_one_boot` from v2 `_fast_iproj_worker_one_boot` (lines 339-425)

Workers import from sibling modules:
```python
from src.coherence.masking import masked_bootstrap_sample, masked_bootstrap_from_uniform
from src.coherence.eigensolve import topk_eigenvectors, topk_eigenvectors_warm, _try_import_eigsh, orthonormalize_columns
```

Keep function signatures accepting a single tuple `args` (required for `joblib.delayed`).

```python
"""Bootstrap worker functions for parallel coherence computation."""
from __future__ import annotations

import numpy as np
import numpy.linalg as la
from numpy.typing import NDArray
from tqdm import tqdm

from src.coherence.eigensolve import (
    _try_import_eigsh,
    orthonormalize_columns,
    topk_eigenvectors,
    topk_eigenvectors_warm,
)
from src.coherence.masking import (
    masked_bootstrap_from_uniform,
    masked_bootstrap_sample,
)


def worker_one_p(args: tuple) -> tuple:
    """Compute all bootstrap outputs for one masking level p."""
    # Copy from v2 lines 432-513 exactly, replacing internal function calls
    # with src.coherence imports
    (
        i, p,
        s0, w, q,
        evals_ref, u_ref_k,
        k_list,
        b_count,
        random_seed_base,
        keep_diag,
        eigsh_tol,
        eigsh_maxiter,
        collect_worker_stats,
        compute_null,
        b_null,
        null_dist,
    ) = args

    n = s0.shape[0]
    p = float(p)
    kmax = u_ref_k.shape[1]
    k_list = np.asarray(k_list, int)
    k_count = len(k_list)

    eigsh = _try_import_eigsh()
    iu = np.triu_indices(n, k=1)
    k_idx = k_list - 1
    ranks = np.arange(1, kmax + 1, dtype=float)

    iproj_boot_p = np.zeros((k_count, b_count), float)
    c_boot_p = np.zeros((k_count, b_count), float)
    i_boot_p = np.zeros((k_count, b_count), float)
    lambda_topk_boot_p = np.full((kmax, b_count), np.nan, float)

    null_boot_p = None
    if compute_null and b_null > 0:
        null_boot_p = np.zeros((k_count, b_count, b_null), float)

    stats = {"eigsh_used": int(eigsh is not None), "fail_count": 0}

    for b in range(b_count):
        rng = np.random.default_rng(
            (random_seed_base + 1000003 * i + 9176 * b) & 0xFFFFFFFF
        )

        a = masked_bootstrap_sample(s0, w, q, p, rng, keep_diag=keep_diag, iu=iu)

        try:
            vals_k, u_k = topk_eigenvectors(a, kmax, eigsh=eigsh, tol=eigsh_tol, maxiter=eigsh_maxiter)
        except Exception:
            stats["fail_count"] += 1
            vals_k, u_k = topk_eigenvectors(a, kmax, eigsh=None)

        lambda_topk_boot_p[:len(vals_k), b] = vals_k

        g = u_k.T @ u_ref_k
        g2 = g * g

        iproj_all = np.cumsum(g2, axis=0)[np.arange(kmax), np.arange(kmax)]
        c_all = np.cumsum(iproj_all) / ranks
        i_all = (ranks * c_all) - np.concatenate(([0.0], ranks[:-1] * c_all[:-1]))

        iproj_boot_p[:, b] = np.clip(iproj_all[k_idx], 0.0, 1.0)
        c_boot_p[:, b] = np.clip(c_all[k_idx], 0.0, 1.0)
        i_boot_p[:, b] = i_all[k_idx]

        if null_boot_p is not None:
            if null_dist == "gaussian":
                v = rng.standard_normal((n, b_null))
            elif null_dist == "rademacher":
                v = rng.integers(0, 2, size=(n, b_null)).astype(float)
                v[v == 0.0] = -1.0
            else:
                raise ValueError(f"Unknown null_dist={null_dist}")

            v /= np.maximum(np.sqrt(np.sum(v * v, axis=0, keepdims=True)), 1e-12)
            acoef = u_k.T @ v
            cumsq = np.cumsum(acoef * acoef, axis=0)
            null_boot_p[:, b, :] = cumsq[k_idx, :]

    worker_stats = stats if collect_worker_stats else None
    return i, p, iproj_boot_p, c_boot_p, i_boot_p, lambda_topk_boot_p, worker_stats, null_boot_p


def worker_one_boot(args: tuple) -> tuple:
    """Compute exact Iproj bootstrap curves for one replicate using warm starts."""
    # Copy from v2 lines 339-425 exactly, replacing internal function calls
    (
        b,
        p_list,
        s0,
        w,
        u_ref_k,
        k_idx,
        random_seed_base,
        keep_diag,
        solver,
        solver_tol,
        solver_maxiter,
        residual_tol,
        use_nested_masks,
        show_inner_progress,
    ) = args

    n = s0.shape[0]
    kmax = u_ref_k.shape[1]
    k_count = len(k_idx)
    p_list = np.asarray(p_list, float)
    p_count = len(p_list)
    iu = np.triu_indices(n, k=1)

    rng = np.random.default_rng((int(random_seed_base) + 9176 * int(b)) & 0xFFFFFFFF)
    shared_edge_u = rng.random(iu[0].size) if use_nested_masks else None
    shared_diag_u = (None if keep_diag else rng.random(n)) if use_nested_masks else None

    iproj_boot_b = np.zeros((k_count, p_count), float)
    x_init = np.array(u_ref_k, copy=True)
    solver_counts = {"lobpcg": 0, "eigh": 0, "none": 0}
    fallback_count = 0
    warning_count = 0
    max_residual_seen = 0.0

    p_iter = enumerate(p_list)
    if show_inner_progress:
        p_iter = enumerate(tqdm(p_list, total=p_count, desc=f"fast p-sweep b={b + 1}", leave=False))

    for j, p in p_iter:
        if use_nested_masks:
            edge_u = shared_edge_u
            diag_u = shared_diag_u
        else:
            edge_u = rng.random(iu[0].size)
            diag_u = None if keep_diag else rng.random(n)

        a = masked_bootstrap_from_uniform(
            s0, w, float(p), edge_u, iu,
            keep_diag=keep_diag, diag_u=diag_u,
        )

        _, u_k, info = topk_eigenvectors_warm(
            a, kmax, x_init=x_init,
            solver=solver, tol=solver_tol,
            maxiter=solver_maxiter, residual_tol=residual_tol,
        )

        solver_name = info.get("solver", "eigh")
        solver_counts[solver_name] = solver_counts.get(solver_name, 0) + 1
        fallback_count += int(info.get("fallback", False))
        warning_count += int(info.get("warning_count", 0))
        max_residual_seen = max(max_residual_seen, float(info.get("max_residual", 0.0)))

        x_init = np.array(u_k, copy=True)
        g = u_k.T @ u_ref_k
        g2 = g * g
        iproj_all = np.cumsum(g2, axis=0)[np.arange(kmax), np.arange(kmax)]
        iproj_boot_b[:, j] = np.clip(iproj_all[k_idx], 0.0, 1.0)

    return b, iproj_boot_b, {
        "solver_counts": solver_counts,
        "fallback_count": int(fallback_count),
        "warning_count": int(warning_count),
        "max_residual_seen": float(max_residual_seen),
    }
```

- [ ] **Step 2: Verify imports work**

```bash
poetry run python -c "from src.coherence.workers import worker_one_p, worker_one_boot; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/coherence/workers.py
git commit -m "feat(coherence): extract workers module from v2 sandbox"
```

---

## Task 6: Compute module (main orchestrator)

**Files:**
- Create: `src/coherence/compute.py`

**Source lines:** `_compute_coherence.py` lines 520-1007

- [ ] **Step 1: Implement `src/coherence/compute.py`**

Copy the main `compute_incremental_coherence_multi_k_eig_anisotropic` function and rename to `compute_coherence`. Key changes:
- Import from sibling modules instead of module-level functions
- Replace `os.cpu_count()` with a `_default_n_jobs()` helper
- Remove all Plotly visualization code (lines 770-972). Set `fig = None` always. The `visualize` and related params are kept in the signature for API compat but ignored.
- Keep all data computation, CI, tau, activation logic identical

```python
"""Main compute_coherence orchestrator."""
from __future__ import annotations

import contextlib
import os

import joblib
import numpy as np
import numpy.linalg as la
from joblib import Parallel, delayed
from numpy.typing import NDArray
from tqdm import tqdm

from src.coherence.analysis import baseline_correct
from src.coherence.eigensolve import randomized_topk_eigenspace
from src.coherence.masking import prepare_observation_mask, symmetrize_with_nan
from src.coherence.workers import worker_one_p


def _default_n_jobs(n_jobs: int | None = None) -> int:
    """Resolve n_jobs to an effective worker count."""
    cpu = os.cpu_count() or 2
    if n_jobs is None:
        return max(1, cpu - 1)
    return int(max(1, min(int(n_jobs), max(1, cpu - 1))))


def compute_coherence(
    s: NDArray[np.floating],
    k_list,
    p_list,
    b: int = 20,
    random_state: int = 0,
    keep_diag: bool = True,
    eigsh_tol: float = 1e-6,
    eigsh_maxiter: int | None = None,
    n_jobs: int | None = None,
    max_nbytes: str = "64M",
    prefer: str = "processes",
    show_progress: bool = True,
    collect_worker_stats: bool = False,
    visualize: bool = False,
    visualize_mode: str = "plotly",
    compute_null: bool = True,
    b_null: int = 30,
    null_dist: str = "gaussian",
    alpha_tau: float = 0.95,
    ci_level: float = 0.95,
    use_baseline_correction: bool = False,
    clip_baseline: bool = True,
    plot_tau: bool = True,
    mark_activation: bool = True,
    ref_oversample: int = 10,
    ref_n_iter: int = 2,
) -> dict:
    """Estimate incremental eigenspace coherence across multiple k and mask rates p.

    Returns a dict with keys: p, k_list, C_boot, C_mean, I_boot, I_mean,
    Iproj_boot, Iproj_mean, summary, fig (always None), evals_ref, U_ref_K,
    lambda_topk_boot, diagnostics.
    """
    # Copy from v2 lines 562-1007, replacing internal function calls.
    # Plotly figure generation (lines 770-972) is removed entirely.
    # See v2 source for exact logic -- kept numerically identical.
    ...  # Full implementation follows v2 exactly
```

**Implementation note:** The full body is a direct copy of v2 lines 562-1007 with these substitutions:
- `_symmetrize_with_nan` -> `symmetrize_with_nan` (from `.masking`)
- `_prepare_observation_mask` -> `prepare_observation_mask` (from `.masking`)
- `_randomized_topk_eigenspace_symmetric` -> `randomized_topk_eigenspace` (from `.eigensolve`)
- `_eig_worker_one_p` -> `worker_one_p` (from `.workers`)
- `_baseline_correct_array` -> `baseline_correct` (from `.analysis`)
- `os.cpu_count()` -> `_default_n_jobs(n_jobs)`
- Remove all plotly imports, figure construction (lines 770-972), and `fig.show()`
- `fig = None` always
- Rename parameters: `B` -> `b`, `B_null` -> `b_null` (snake_case)
- Change default `visualize=True` -> `visualize=False`

- [ ] **Step 2: Verify import works**

```bash
poetry run python -c "from src.coherence.compute import compute_coherence; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/coherence/compute.py
git commit -m "feat(coherence): extract compute orchestrator from v2 sandbox"
```

---

## Task 7: Cluster module

**Files:**
- Create: `src/coherence/cluster.py`

**Source lines:** `_compute_coherence.py` lines 1468-1577, 1696-2056

- [ ] **Step 1: Implement `src/coherence/cluster.py`**

Copy functions from v2. Key mapping:
- `compute_trend_correlation` (lines 1468-1512) -- keep name
- `compute_cumulative_trend_correlation` (lines 1515-1577) -- keep name
- `cluster_consensus_across_p` from `analyze_cluster_consensus_across_p` (lines 1696-2056) -- drop "analyze_" prefix

Changes:
- Replace internal calls with imports from `src.coherence._correlation` and `src.coherence.analysis`
- Remove all plotly visualization code from `cluster_consensus_across_p` (lines 1934-2032). The `make_plots`, `width`, `height` params are kept but ignored; `fig_consensus` is always `None`.
- Keep all data computation identical

```python
"""Cluster consensus analysis across p-prefixes."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.coherence._correlation import (
    compute_trend_similarity,
    get_pair_metric,
    mutual_info_matrix_pairwise,
    pearson_corr_matrix,
    spearman_corr_matrix,
)
from src.coherence.analysis import baseline_correct
```

Internal helpers `_similarity_matrix_rows`, `_cluster_from_similarity`, `_jaccard`, `_clusters_from_labels` are defined as module-private functions within cluster.py.

- [ ] **Step 2: Write tests for cluster functions**

Create `tests/test_coherence/test_cluster.py`:

```python
"""Numerical equivalence tests: src/coherence/cluster vs v2."""
import sys
from pathlib import Path

import numpy as np
import pytest

_V2 = str(Path(__file__).resolve().parents[2] / "sandbox" / "coherence" / "kachun" / "v2")
if _V2 not in sys.path:
    sys.path.insert(0, _V2)
from _compute_coherence import (
    compute_trend_correlation as v2_trend_corr,
    compute_cumulative_trend_correlation as v2_cumul_trend,
)

from src.coherence.cluster import (
    compute_trend_correlation,
    compute_cumulative_trend_correlation,
)


@pytest.fixture
def bootstrap_data():
    rng = np.random.default_rng(88)
    k_count, p_count, b_count = 8, 15, 5
    iproj_boot = rng.random((k_count, p_count, b_count))
    k_list = np.arange(1, k_count + 1)
    p_list = np.linspace(0.1, 0.9, p_count)
    return iproj_boot, k_list, p_list


def test_trend_correlation(bootstrap_data):
    iproj, k_list, p_list = bootstrap_data
    new = compute_trend_correlation(iproj, k_list, p_list)
    old = v2_trend_corr(iproj, k_list, p_list)
    for key in ("pearson", "spearman", "mutual_info"):
        assert np.allclose(
            new["corr_matrices"][key], old["corr_matrices"][key], atol=1e-10
        ), f"Mismatch on {key}"
    assert np.allclose(new["x_trend"], old["x_trend"], atol=1e-12)


def test_cumulative_trend_correlation(bootstrap_data):
    iproj, k_list, p_list = bootstrap_data
    new = compute_cumulative_trend_correlation(iproj, k_list, p_list)
    old = v2_cumul_trend(iproj, k_list, p_list)
    for key in ("pearson", "spearman", "mutual_info"):
        assert np.allclose(
            new["corr_matrices"][key], old["corr_matrices"][key], atol=1e-10
        ), f"Mismatch on cumulative {key}"
```

- [ ] **Step 3: Run tests**

```bash
poetry run pytest tests/test_coherence/test_cluster.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/coherence/cluster.py tests/test_coherence/test_cluster.py
git commit -m "feat(coherence): extract cluster consensus module from v2 sandbox"
```

---

## Task 8: Public API exports in `__init__.py`

**Files:**
- Modify: `src/coherence/__init__.py`

- [ ] **Step 1: Populate `__init__.py` with public API**

```python
"""Coherence analysis for symmetric matrix factorization under missing data.

Main entry points
-----------------
compute_coherence
    Estimate incremental eigenspace coherence across k and p.
cluster_consensus_across_p
    Analyze cluster stability over p-prefixes.
"""
from src.coherence.analysis import (
    baseline_correct,
    bh_fdr,
    estimate_kappa,
    find_first_run_above,
    kappa_changepoint,
    per_component_activation,
    smooth_median,
)
from src.coherence.cluster import (
    cluster_consensus_across_p,
    compute_cumulative_trend_correlation,
    compute_trend_correlation,
)
from src.coherence.compute import compute_coherence
from src.coherence.eigensolve import (
    orthonormalize_columns,
    randomized_topk_eigenspace,
    topk_eigenvectors,
    topk_eigenvectors_warm,
)
from src.coherence.masking import (
    masked_bootstrap_from_uniform,
    masked_bootstrap_sample,
    prepare_observation_mask,
    symmetrize_with_nan,
)

__all__ = [
    "compute_coherence",
    "cluster_consensus_across_p",
    "compute_trend_correlation",
    "compute_cumulative_trend_correlation",
    "symmetrize_with_nan",
    "prepare_observation_mask",
    "masked_bootstrap_sample",
    "masked_bootstrap_from_uniform",
    "topk_eigenvectors",
    "topk_eigenvectors_warm",
    "randomized_topk_eigenspace",
    "orthonormalize_columns",
    "baseline_correct",
    "bh_fdr",
    "estimate_kappa",
    "smooth_median",
    "kappa_changepoint",
    "find_first_run_above",
    "per_component_activation",
]
```

- [ ] **Step 2: Verify full package imports**

```bash
poetry run python -c "from src.coherence import compute_coherence, cluster_consensus_across_p; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/coherence/__init__.py
git commit -m "feat(coherence): populate public API exports"
```

---

## Task 9: Integration test and structure test update

**Files:**
- Create: `tests/test_coherence/test_compute.py`
- Modify: `tests/test_structure.py`

- [ ] **Step 1: Write integration test**

Create `tests/test_coherence/test_compute.py`:

```python
"""Integration test: compute_coherence produces numerically equivalent output to v2."""
import sys
from pathlib import Path

import numpy as np
import pytest

_V2 = str(Path(__file__).resolve().parents[2] / "sandbox" / "coherence" / "kachun" / "v2")
if _V2 not in sys.path:
    sys.path.insert(0, _V2)
from _compute_coherence import (
    compute_incremental_coherence_multi_k_eig_anisotropic as v2_compute,
)

from src.coherence import compute_coherence


@pytest.fixture
def small_problem():
    rng = np.random.default_rng(42)
    n = 30
    a = rng.standard_normal((n, n))
    s = a @ a.T / n
    k_list = np.array([1, 3, 5, 10])
    p_list = np.linspace(0.1, 0.9, 5)
    return s, k_list, p_list


def test_compute_coherence_equivalence(small_problem):
    """Verify new package matches v2 exactly.

    Uses atol=1e-12 because both implementations use identical RNG seeding
    and call order, so outputs should be bitwise-identical up to float noise.
    If this fails, the most likely cause is a change in function call order
    that shifts the RNG state.
    """
    s, k_list, p_list = small_problem
    kwargs = dict(
        B=3, random_state=42, n_jobs=1,
        show_progress=False, compute_null=True, B_null=5,
        visualize=False,
    )
    new = compute_coherence(s, k_list, p_list, b=3, random_state=42, n_jobs=1,
                            show_progress=False, compute_null=True, b_null=5)
    old = v2_compute(s, k_list, p_list, **kwargs)

    for key in ("Iproj_boot", "C_boot", "I_boot", "Iproj_mean", "C_mean", "I_mean"):
        assert np.allclose(new[key], old[key], atol=1e-12), f"Mismatch on {key}"

    assert np.allclose(new["evals_ref"], old["evals_ref"], atol=1e-10)
    assert np.allclose(
        new["diagnostics"]["tau_kp"],
        old["diagnostics"]["tau_kp"],
        atol=1e-10,
    )
```

- [ ] **Step 2: Add coherence import check to `tests/test_structure.py`**

Add after existing `test_similarity_imports`:

```python
def test_coherence_imports():
    """Verify coherence/ package is importable."""
    modules = [
        "coherence",
        "coherence.masking",
        "coherence.eigensolve",
        "coherence.workers",
        "coherence.compute",
        "coherence.analysis",
        "coherence.cluster",
        "coherence._correlation",
    ]
    for mod in modules:
        importlib.import_module(mod)
```

- [ ] **Step 3: Run all tests**

```bash
poetry run pytest tests/test_coherence/ tests/test_structure.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_coherence/test_compute.py tests/test_structure.py
git commit -m "test(coherence): add integration test and structure check"
```

---

## Task 10: Update sandbox imports

**Files to modify (13 files):**
- `sandbox/coherence/subspace_rank/_subspace_coherence.py`
- `sandbox/coherence/kachun/v2/run_peterson_animals.py`
- `sandbox/coherence/kachun/v2/run_peterson_comparison.py`
- `sandbox/coherence/kachun/v2/run_things_behavior.py`
- `sandbox/coherence/kachun/v2/run_things_comparison.py`
- `sandbox/coherence/kachun/v2/run_things_k200.py`
- `sandbox/coherence/subspace_rank/run_diagnostic.py`
- `sandbox/coherence/subspace_rank/run_diagnostic_large.py`
- `sandbox/coherence/subspace_rank/run_diagnostic_multi.py`
- `sandbox/coherence/subspace_rank/run_diagnostic_swow.py`
- `sandbox/swow/ppmi_coherence/run.py`

**Note:** `sandbox/samuel/coherence/run.py` and `sandbox/samuel/explore/run.py` import `impute_similarity_matrix` from a non-existent `src.tools.coherence`. This function lives in `experiments/bounds/bounds_missing.py`. These scripts are left as-is since they are already broken and unrelated to the coherence extraction.

- [ ] **Step 1: Update `_subspace_coherence.py` imports**

Replace lines 20-32:
```python
# OLD (sys.path hack + private imports)
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

With:
```python
from src.coherence.masking import (
    symmetrize_with_nan,
    prepare_observation_mask,
    masked_bootstrap_from_uniform,
)
from src.coherence.eigensolve import (
    topk_eigenvectors_warm,
    orthonormalize_columns,
    randomized_topk_eigenspace,
)
```

Then update all call sites in the file to use the new names (remove leading underscores):
- `_symmetrize_with_nan(` -> `symmetrize_with_nan(`
- `_prepare_observation_mask(` -> `prepare_observation_mask(`
- `_masked_unbiased_spd_missing_from_uniform(` -> `masked_bootstrap_from_uniform(`
- `_topk_eigenvectors_exact_warm(` -> `topk_eigenvectors_warm(`
- `_orthonormalize_columns(` -> `orthonormalize_columns(`
- `_randomized_topk_eigenspace_symmetric(` -> `randomized_topk_eigenspace(`

Also remove `import os, sys` if no longer needed.

- [ ] **Step 2: Update v2 `run_*.py` scripts (5 files)**

For each of the 5 files in `sandbox/coherence/kachun/v2/`, replace:
```python
sys.path.insert(0, os.path.dirname(__file__))
from _compute_coherence import (
    compute_incremental_coherence_multi_k_eig_anisotropic,
    analyze_cluster_consensus_across_p,  # if present
    ...
)
```
With:
```python
from src.coherence import compute_coherence
from src.coherence.cluster import cluster_consensus_across_p  # if needed
```

And update call sites: `compute_incremental_coherence_multi_k_eig_anisotropic(` -> `compute_coherence(`.

**Important:** Some scripts also import `analyze_iproj_signal_alignment_v3` which stays in sandbox. Those imports should keep the sys.path hack for that one function only. Example resulting import block (e.g. `run_things_behavior.py`):

```python
import os
import sys

# New: clean imports from src package
from src.coherence import compute_coherence
from src.coherence.cluster import cluster_consensus_across_p

# Legacy: v3 analysis stays in sandbox (not extracted)
sys.path.insert(0, os.path.dirname(__file__))
from _compute_coherence import analyze_iproj_signal_alignment_v3
```

Then replace call sites: `compute_incremental_coherence_multi_k_eig_anisotropic(` -> `compute_coherence(` and `analyze_cluster_consensus_across_p(` -> `cluster_consensus_across_p(`.

- [ ] **Step 3: Update subspace_rank `run_diagnostic*.py` scripts (4 files)**

Same pattern -- replace sys.path import of `compute_incremental_coherence_multi_k_eig_anisotropic` with `from src.coherence import compute_coherence`.

- [ ] **Step 4: Update `sandbox/swow/ppmi_coherence/run.py`**

Same pattern.

- [ ] **Step 5: Commit**

```bash
git add sandbox/coherence/ sandbox/swow/
git commit -m "refactor: update sandbox imports to use src.coherence package"
```

---

## Task 11: Final verification

- [ ] **Step 1: Run all coherence tests**

```bash
poetry run pytest tests/test_coherence/ -v
```

Expected: all tests PASS.

- [ ] **Step 2: Run full test suite**

```bash
poetry run pytest tests/ -v
```

Expected: all tests PASS (including existing tests).

- [ ] **Step 3: Verify sandbox script imports work**

```bash
poetry run python -c "
import sys; sys.path.insert(0, '.')
from src.coherence import compute_coherence, cluster_consensus_across_p
from src.coherence.masking import symmetrize_with_nan
from src.coherence.eigensolve import topk_eigenvectors_warm
print('All imports OK')
"
```

- [ ] **Step 4: Commit (if any fixes needed)**

```bash
git commit -m "fix: address any issues from final verification"
```
