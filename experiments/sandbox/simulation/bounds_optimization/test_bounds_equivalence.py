"""Tests to verify numerical equivalence between bounds implementations.

These tests ensure that bounds_ultra produces identical results to the
original pysrf.bounds implementations.
"""

import numpy as np
import pytest
from pysrf.bounds import (
    pmin_bound,
    p_upper_only_k,
    lambda_bulk_dyson_raw,
    estimate_sampling_bounds,
    estimate_sampling_bounds_ultra,
    precompute_matrix_info,
    _p_upper_only_k_ultra,
    _solve_vde,
)

# Legacy imports from sandbox for backwards compatibility
from bounds_ultra import (
    pmin_bound_ultra,
    p_upper_only_k_ultra,
    lambda_bulk_dyson_ultra,
    _solve_vde_python,
    _solve_vde as _solve_vde_sandbox,
)


def generate_test_matrix(n: int = 50, rank: int = 5, seed: int = 42) -> np.ndarray:
    """Generate a low-rank symmetric positive semidefinite matrix."""
    rng = np.random.RandomState(seed)
    W = rng.rand(n, rank)
    S = W @ W.T
    return (S + S.T) / 2


class TestVDESolver:
    """Test numerical equivalence of VDE solver implementations."""

    def test_vde_numba_vs_python(self):
        """Verify numba VDE solver matches pure Python version."""
        S = generate_test_matrix(n=30, rank=5)
        p = 0.5
        V = p * (1 - p) * (S ** 2)

        for z in [0.1, 1.0, 10.0]:
            m_numba = _solve_vde_sandbox(V, z)
            m_python = _solve_vde_python(V, z)

            np.testing.assert_allclose(
                m_numba, m_python, rtol=1e-6, atol=1e-10,
                err_msg=f"VDE solver mismatch at z={z}"
            )


class TestPminBound:
    """Test pmin_bound equivalence."""

    def test_pmin_bound_equivalence(self):
        """Verify pmin_bound_ultra matches pmin_bound."""
        S = generate_test_matrix(n=50, rank=5)
        info = precompute_matrix_info(S)

        result_orig = pmin_bound(S, random_state=42)
        result_ultra = pmin_bound_ultra(S, info)

        np.testing.assert_allclose(
            result_orig[0], result_ultra[0], rtol=1e-10,
            err_msg="pmin_empirical mismatch"
        )
        np.testing.assert_allclose(
            result_orig[1], result_ultra[1], rtol=1e-10,
            err_msg="pmin_bernstein mismatch"
        )

    @pytest.mark.parametrize("n,rank", [(30, 3), (50, 5), (80, 10)])
    def test_pmin_bound_sizes(self, n: int, rank: int):
        """Test pmin equivalence across different matrix sizes."""
        S = generate_test_matrix(n=n, rank=rank)
        info = precompute_matrix_info(S)

        result_orig = pmin_bound(S, random_state=42)
        result_ultra = pmin_bound_ultra(S, info)

        np.testing.assert_allclose(
            result_orig[0], result_ultra[0], rtol=1e-10,
            err_msg=f"pmin mismatch for n={n}, rank={rank}"
        )


class TestLambdaBulkDyson:
    """Test lambda_bulk_dyson equivalence."""

    @pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_edge_equivalence(self, p: float):
        """Verify lambda_bulk_dyson_ultra matches original for various p."""
        S = generate_test_matrix(n=40, rank=5)
        info = precompute_matrix_info(S)

        edge_orig = lambda_bulk_dyson_raw(S, p)
        edge_ultra = lambda_bulk_dyson_ultra(S, p, info)

        np.testing.assert_allclose(
            edge_orig, edge_ultra, rtol=1e-6,
            err_msg=f"Edge mismatch at p={p}"
        )


class TestPUpperOnlyK:
    """Test p_upper_only_k equivalence."""

    def test_pmax_equivalence(self):
        """Verify p_upper_only_k_ultra matches original."""
        S = generate_test_matrix(n=50, rank=5)
        info = precompute_matrix_info(S)

        pmax_orig = p_upper_only_k(S, k=info.eff_dim, seed=42)
        pmax_ultra = p_upper_only_k_ultra(S, k=info.eff_dim, info=info, n_jobs=1)

        np.testing.assert_allclose(
            pmax_orig, pmax_ultra, rtol=1e-4,
            err_msg="pmax mismatch"
        )

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_pmax_different_k(self, k: int):
        """Test pmax equivalence for different k values."""
        S = generate_test_matrix(n=40, rank=5)
        info = precompute_matrix_info(S)

        pmax_orig = p_upper_only_k(S, k=k, seed=42)
        pmax_ultra = p_upper_only_k_ultra(S, k=k, info=info, n_jobs=1)

        np.testing.assert_allclose(
            pmax_orig, pmax_ultra, rtol=1e-4,
            err_msg=f"pmax mismatch for k={k}"
        )


class TestEstimateSamplingBounds:
    """Test full estimate_sampling_bounds equivalence."""

    def test_bounds_equivalence(self):
        """Verify estimate_sampling_bounds_ultra matches original."""
        S = generate_test_matrix(n=50, rank=5)

        pmin_orig, pmax_orig, _ = estimate_sampling_bounds(S, random_state=42)
        pmin_ultra, pmax_ultra, _ = estimate_sampling_bounds_ultra(
            S, random_state=42, n_jobs=1
        )

        np.testing.assert_allclose(
            pmin_orig, pmin_ultra, rtol=1e-6,
            err_msg="pmin mismatch in full bounds"
        )
        np.testing.assert_allclose(
            pmax_orig, pmax_ultra, rtol=1e-4,
            err_msg="pmax mismatch in full bounds"
        )

    @pytest.mark.parametrize("n,rank", [(30, 3), (50, 5), (80, 8)])
    def test_bounds_various_sizes(self, n: int, rank: int):
        """Test bounds equivalence across matrix sizes."""
        S = generate_test_matrix(n=n, rank=rank, seed=123)

        pmin_orig, pmax_orig, _ = estimate_sampling_bounds(S, random_state=42)
        pmin_ultra, pmax_ultra, _ = estimate_sampling_bounds_ultra(
            S, random_state=42, n_jobs=1
        )

        np.testing.assert_allclose(
            pmin_orig, pmin_ultra, rtol=1e-6,
            err_msg=f"pmin mismatch for n={n}, rank={rank}"
        )
        np.testing.assert_allclose(
            pmax_orig, pmax_ultra, rtol=1e-4,
            err_msg=f"pmax mismatch for n={n}, rank={rank}"
        )


class TestPrecomputedInfo:
    """Test that precomputed values are correct."""

    def test_s_norm_correct(self):
        """Verify s_norm matches np.linalg.norm(S, 2)."""
        S = generate_test_matrix(n=50, rank=5)
        info = precompute_matrix_info(S)

        expected = np.linalg.norm(S, 2)
        np.testing.assert_allclose(info.s_norm, expected, rtol=1e-10)

    def test_s2_max_correct(self):
        """Verify s2_max matches max(eigvalsh(S**2))."""
        S = generate_test_matrix(n=50, rank=5)
        info = precompute_matrix_info(S)

        from numpy.linalg import eigvalsh
        expected = np.max(eigvalsh(S ** 2))
        np.testing.assert_allclose(info.s2_max, expected, rtol=1e-10)

    def test_eigvals_sorted_descending(self):
        """Verify eigenvalues are sorted in descending order."""
        S = generate_test_matrix(n=50, rank=5)
        info = precompute_matrix_info(S)

        assert np.all(np.diff(info.eigvals) <= 0), "Eigenvalues not sorted descending"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
