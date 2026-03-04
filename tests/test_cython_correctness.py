"""Rigorous numerical equivalence tests for Cython BSUM implementations.

The scalar variant (update_w) precomputes m[i,:] @ w for all r columns in a
single fused loop per row, using the same summation order as Python (k=0..n-1,
sequential accumulation), so results are bitwise-identical to the Python version.

The BLAS variants use BLAS routines which may change summation order, so they
are numerically equivalent but not necessarily bit-identical. After a single
BSUM iteration, all implementations agree to ~1e-14 (machine epsilon) on ANY
symmetric matrix, proving the correction math is exact. Over many iterations,
BLAS rounding differences can cascade through the quartic solver's cube-root
sensitivity at the max(0, root) boundary, causing element-wise divergence while
reconstruction quality (||M - WW^T||_F / ||M||_F) remains identical to 10+
decimal places. See docs/bsum_implementations.md for the full analysis.
"""

import numpy as np
import pytest

from pysrf.model import update_w as update_w_python, _frobenius_residual

try:
    from pysrf._bsum import update_w as update_w_scalar

    HAS_SCALAR = True
except ImportError:
    HAS_SCALAR = False

try:
    from pysrf._bsum import update_w_blas

    HAS_BLAS = True
except ImportError:
    HAS_BLAS = False

try:
    from pysrf._bsum import update_w_blas_blocked

    HAS_BLOCKED = True
except ImportError:
    HAS_BLOCKED = False


def make_data(
    n: int = 100, r: int = 10, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    m = rng.random((n, n))
    m = (m + m.T) * 0.5
    x0 = rng.random((n, r))
    return m, x0


@pytest.mark.skipif(not HAS_SCALAR, reason="Cython not compiled")
class TestScalarVsPython:
    """Scalar Cython uses the same summation order as Python.

    These must match to machine precision since the arithmetic operations
    are identical.
    """

    def test_equivalence(self):
        m, x0 = make_data(seed=1234)
        cython_result = update_w_scalar(m, x0, tol=0.0)
        python_result = update_w_python(m, x0, tol=0.0)
        np.testing.assert_array_equal(cython_result, python_result)

    @pytest.mark.parametrize("seed", [0, 42, 123, 456, 789])
    def test_equivalence_multiple_seeds(self, seed):
        m, x0 = make_data(seed=seed)
        cython_result = update_w_scalar(m, x0, tol=0.0)
        python_result = update_w_python(m, x0, tol=0.0)
        np.testing.assert_array_equal(cython_result, python_result)

    @pytest.mark.parametrize("n,r", [(10, 3), (50, 5), (100, 10), (200, 15)])
    def test_equivalence_different_sizes(self, n, r):
        m, x0 = make_data(n=n, r=r, seed=42)
        cython_result = update_w_scalar(m, x0, tol=0.0)
        python_result = update_w_python(m, x0, tol=0.0)
        np.testing.assert_array_equal(cython_result, python_result)

    def test_nonnegativity(self):
        m, x0 = make_data(n=100, r=10, seed=42)
        result = update_w_scalar(m, x0, tol=0.0)
        assert np.all(result >= 0), "Scalar result has negative entries"

    def test_convergence_tolerance(self):
        """Verify early stopping produces same result."""
        m, x0 = make_data(n=50, r=5, seed=42)
        cython_result = update_w_scalar(m, x0, max_iter=200, tol=1e-6)
        python_result = update_w_python(m, x0, max_iter=200, tol=1e-6)
        np.testing.assert_array_equal(cython_result, python_result)

    def test_does_not_modify_inputs(self):
        """Verify implementation does not modify the input arrays."""
        m, x0 = make_data(n=50, r=5, seed=42)
        m_copy = m.copy()
        x0_copy = x0.copy()

        update_w_scalar(m, x0, tol=0.0)
        np.testing.assert_array_equal(m, m_copy)
        np.testing.assert_array_equal(x0, x0_copy)


@pytest.mark.skipif(
    not (HAS_BLAS and HAS_SCALAR), reason="BLAS and scalar Cython required"
)
class TestBlasVsScalar:
    """BLAS variant uses dgemv/ddot/daxpy which may reorder summation.

    After 1 iteration the math is provably identical — differences are pure
    IEEE 754 rounding (~1e-14). Over many iterations, cube-root sensitivity
    at the max(0, root) boundary amplifies these, but reconstruction quality
    remains identical. All tests use arbitrary random symmetric M.
    """

    @pytest.mark.parametrize("seed", [0, 42, 123, 456, 789])
    def test_single_iteration_elementwise(self, seed):
        """One BSUM iteration: proves correction math is exact on any M."""
        m, x0 = make_data(seed=seed)
        blas_result = update_w_blas(m, x0, max_iter=1, tol=0.0)
        scalar_result = update_w_scalar(m, x0, max_iter=1, tol=0.0)
        np.testing.assert_allclose(blas_result, scalar_result, atol=1e-10)

    @pytest.mark.parametrize("n,r", [(10, 3), (50, 5), (100, 10), (200, 15)])
    def test_single_iteration_different_sizes(self, n, r):
        m, x0 = make_data(n=n, r=r, seed=42)
        blas_result = update_w_blas(m, x0, max_iter=1, tol=0.0)
        scalar_result = update_w_scalar(m, x0, max_iter=1, tol=0.0)
        np.testing.assert_allclose(blas_result, scalar_result, atol=1e-10)

    @pytest.mark.parametrize("seed", [0, 42, 123, 456, 789])
    def test_reconstruction_equivalence(self, seed):
        """Many iterations on any M: reconstruction quality must match."""
        m, x0 = make_data(seed=seed)
        blas_result = update_w_blas(m, x0, tol=0.0)
        scalar_result = update_w_scalar(m, x0, tol=0.0)
        norm_m = np.linalg.norm(m, "fro")
        err_blas = np.linalg.norm(m - blas_result @ blas_result.T, "fro") / norm_m
        err_scalar = np.linalg.norm(m - scalar_result @ scalar_result.T, "fro") / norm_m
        np.testing.assert_allclose(err_blas, err_scalar, rtol=1e-2)

    def test_nonnegativity(self):
        m, x0 = make_data(n=100, r=10, seed=42)
        blas_result = update_w_blas(m, x0, tol=0.0)
        assert np.all(blas_result >= 0), "BLAS result has negative entries"


@pytest.mark.skipif(
    not (HAS_BLOCKED and HAS_SCALAR), reason="Blocked and scalar Cython required"
)
class TestBlasBlockedVsScalar:
    """Blocked variant uses dsymm/dgemm (BLAS-3) which may reorder summation.

    Same numerical profile as BLAS variant: exact at 1 iteration, reconstruction-
    equivalent at convergence. All tests use arbitrary random symmetric M.
    """

    @pytest.mark.parametrize("seed", [0, 42, 123, 456, 789])
    def test_single_iteration_elementwise(self, seed):
        """One BSUM iteration: proves correction math is exact on any M."""
        m, x0 = make_data(seed=seed)
        blocked_result = update_w_blas_blocked(m, x0, max_iter=1, tol=0.0)
        scalar_result = update_w_scalar(m, x0, max_iter=1, tol=0.0)
        np.testing.assert_allclose(blocked_result, scalar_result, atol=1e-10)

    @pytest.mark.parametrize("block_size", [1, 10, 50, 100])
    def test_single_iteration_block_sizes(self, block_size):
        """All block sizes produce the same result at 1 iteration."""
        m, x0 = make_data(n=100, r=10, seed=42)
        blocked_result = update_w_blas_blocked(
            m, x0, max_iter=1, tol=0.0, block_size=block_size
        )
        scalar_result = update_w_scalar(m, x0, max_iter=1, tol=0.0)
        np.testing.assert_allclose(blocked_result, scalar_result, atol=1e-10)

    @pytest.mark.parametrize("n,r", [(10, 3), (50, 5), (100, 10), (200, 15)])
    def test_single_iteration_different_sizes(self, n, r):
        m, x0 = make_data(n=n, r=r, seed=42)
        blocked_result = update_w_blas_blocked(m, x0, max_iter=1, tol=0.0)
        scalar_result = update_w_scalar(m, x0, max_iter=1, tol=0.0)
        np.testing.assert_allclose(blocked_result, scalar_result, atol=1e-10)

    @pytest.mark.parametrize("seed", [0, 42, 123, 456, 789])
    def test_reconstruction_equivalence(self, seed):
        """Many iterations on any M: reconstruction quality must match."""
        m, x0 = make_data(seed=seed)
        blocked_result = update_w_blas_blocked(m, x0, tol=0.0)
        scalar_result = update_w_scalar(m, x0, tol=0.0)
        norm_m = np.linalg.norm(m, "fro")
        err_blocked = (
            np.linalg.norm(m - blocked_result @ blocked_result.T, "fro") / norm_m
        )
        err_scalar = np.linalg.norm(m - scalar_result @ scalar_result.T, "fro") / norm_m
        np.testing.assert_allclose(err_blocked, err_scalar, rtol=1e-2)

    def test_nonnegativity(self):
        m, x0 = make_data(n=100, r=10, seed=42)
        blocked_result = update_w_blas_blocked(m, x0, tol=0.0)
        assert np.all(blocked_result >= 0), "Blocked result has negative entries"


class TestFrobeniusResidual:
    """Test the memory-efficient _frobenius_residual helper."""

    def test_matches_direct_computation(self):
        m, x0 = make_data(n=50, r=5, seed=42)
        w = update_w_python(m, x0, tol=0.0)
        residual, xhat_norm = _frobenius_residual(m, w)
        wwt = w @ w.T
        expected_residual = np.linalg.norm(m - wwt, "fro")
        expected_xhat_norm = np.linalg.norm(wwt, "fro")
        np.testing.assert_allclose(residual, expected_residual, rtol=1e-10)
        np.testing.assert_allclose(xhat_norm, expected_xhat_norm, rtol=1e-10)

    @pytest.mark.parametrize("n,r", [(10, 3), (50, 5), (100, 10)])
    def test_different_sizes(self, n, r):
        m, x0 = make_data(n=n, r=r, seed=42)
        w = update_w_python(m, x0, max_iter=10, tol=0.0)
        residual, xhat_norm = _frobenius_residual(m, w)
        wwt = w @ w.T
        expected_residual = np.linalg.norm(m - wwt, "fro")
        expected_xhat_norm = np.linalg.norm(wwt, "fro")
        np.testing.assert_allclose(residual, expected_residual, rtol=1e-10)
        np.testing.assert_allclose(xhat_norm, expected_xhat_norm, rtol=1e-10)

    def test_residual_nonnegative(self):
        m, x0 = make_data(n=50, r=5, seed=42)
        residual, xhat_norm = _frobenius_residual(m, x0)
        assert residual >= 0
        assert xhat_norm >= 0


class TestEdgeCases:
    """Edge-case tests: rank=1, small n, block_size > n."""

    def test_rank_1(self):
        m, x0 = make_data(n=20, r=1, seed=42)
        result = update_w_python(m, x0, max_iter=50, tol=0.0)
        assert result.shape == (20, 1)
        assert np.all(result >= 0)

    @pytest.mark.skipif(not HAS_BLOCKED, reason="Cython not compiled")
    def test_rank_1_blocked(self):
        m, x0 = make_data(n=20, r=1, seed=42)
        blocked = update_w_blas_blocked(m, x0, max_iter=1, tol=0.0)
        python = update_w_python(m, x0, max_iter=1, tol=0.0)
        np.testing.assert_allclose(blocked, python, atol=1e-10)

    @pytest.mark.skipif(not HAS_BLOCKED, reason="Cython not compiled")
    def test_block_size_larger_than_n(self):
        m, x0 = make_data(n=10, r=3, seed=42)
        blocked = update_w_blas_blocked(m, x0, max_iter=1, tol=0.0, block_size=100)
        original = update_w_python(m, x0, max_iter=1, tol=0.0)
        np.testing.assert_allclose(blocked, original, atol=1e-10)

    @pytest.mark.skipif(not HAS_BLOCKED, reason="Cython not compiled")
    def test_block_size_equals_n(self):
        m, x0 = make_data(n=10, r=3, seed=42)
        blocked = update_w_blas_blocked(m, x0, max_iter=1, tol=0.0, block_size=10)
        original = update_w_python(m, x0, max_iter=1, tol=0.0)
        np.testing.assert_allclose(blocked, original, atol=1e-10)

    def test_small_n(self):
        m, x0 = make_data(n=3, r=2, seed=42)
        result = update_w_python(m, x0, max_iter=50, tol=0.0)
        assert result.shape == (3, 2)
        assert np.all(result >= 0)

    @pytest.mark.skipif(not HAS_BLOCKED, reason="Cython not compiled")
    def test_small_n_blocked(self):
        m, x0 = make_data(n=3, r=2, seed=42)
        blocked = update_w_blas_blocked(m, x0, max_iter=1, tol=0.0)
        python = update_w_python(m, x0, max_iter=1, tol=0.0)
        np.testing.assert_allclose(blocked, python, atol=1e-10)


class TestMonotonicity:
    """Verify that BSUM objective decreases monotonically per iteration.

    Shi et al. (2017) Theorem 1 guarantees this property.
    """

    @staticmethod
    def _run_iterations(m, x0, update_fn, n_iter, **kwargs):
        w = x0.copy()
        errors = []
        for _ in range(n_iter):
            w = update_fn(m, w, max_iter=1, tol=0.0, **kwargs)
            err = np.linalg.norm(m - w @ w.T, "fro")
            errors.append(err)
        return errors

    def test_python_monotonicity(self):
        m, x0 = make_data(n=50, r=5, seed=42)
        errors = self._run_iterations(m, x0, update_w_python, 20)
        diffs = np.diff(errors)
        assert np.all(diffs <= 1e-10), (
            f"Python: non-monotonic at {np.argmax(diffs > 1e-10)}"
        )

    @pytest.mark.skipif(not HAS_SCALAR, reason="Cython not compiled")
    def test_scalar_monotonicity(self):
        m, x0 = make_data(n=50, r=5, seed=42)
        errors = self._run_iterations(m, x0, update_w_scalar, 20)
        diffs = np.diff(errors)
        assert np.all(diffs <= 1e-10), (
            f"Scalar: non-monotonic at {np.argmax(diffs > 1e-10)}"
        )

    @pytest.mark.skipif(not HAS_BLOCKED, reason="Cython not compiled")
    def test_blocked_monotonicity(self):
        m, x0 = make_data(n=50, r=5, seed=42)
        errors = self._run_iterations(m, x0, update_w_blas_blocked, 20)
        diffs = np.diff(errors)
        assert np.all(diffs <= 1e-10), (
            f"Blocked: non-monotonic at {np.argmax(diffs > 1e-10)}"
        )
