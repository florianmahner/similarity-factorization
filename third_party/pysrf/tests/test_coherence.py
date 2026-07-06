"""Regression tests for pysrf.coherence against the original implementation."""

import numpy as np
import pytest
from pathlib import Path


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

        s = np.array([[1.0, np.nan, 3.0], [2.0, 1.0, np.nan], [np.nan, 4.0, 1.0]])
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
    def test_shape_and_order(self, test_matrix):
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
            test_matrix,
            k_max=k_max,
            p_list=p_list,
            n_boot=3,
            random_state=0,
            n_jobs=1,
        )
        assert iproj_boot.shape == (k_max, 4, 3)
        assert evals_ref.shape == (k_max,)

    def test_iproj_bounded(self, test_matrix):
        from pysrf.coherence import _bootstrap_coherence

        iproj_boot, _ = _bootstrap_coherence(
            test_matrix,
            k_max=5,
            p_list=np.linspace(0.1, 0.95, 4),
            n_boot=3,
            random_state=0,
            n_jobs=1,
        )
        assert np.all(iproj_boot >= 0.0)
        assert np.all(iproj_boot <= 1.0)

    def test_matches_reference(self, test_matrix, reference):
        from pysrf.coherence import _bootstrap_coherence

        k_list = reference["k_list"].astype(int)
        k_max = int(k_list.max())
        p_list = reference["p_list"]
        iproj_boot, _ = _bootstrap_coherence(
            test_matrix,
            k_max=k_max,
            p_list=p_list,
            n_boot=5,
            random_state=0,
            n_jobs=1,
        )
        # Select only the k indices that match k_list
        k_idx = k_list - 1
        iproj_selected = iproj_boot[k_idx]
        np.testing.assert_allclose(
            iproj_selected,
            reference["iproj_boot"],
            atol=1e-12,
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
            reference["iproj_median"],
            reference["p_list"],
            hi_quantile=0.85,
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
            test_matrix,
            k_max=5,
            p_list=np.linspace(0.1, 0.95, 5),
            n_boot=3,
            random_state=0,
            n_jobs=1,
        )
        assert "k_star" in result
        assert "p_star" in result
        assert "kappa" in result
        assert "iproj_median" in result
        assert "evals_ref" in result
        assert "k_list" in result
        assert "signal_curve" in result
        assert "noise_curve" in result
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
            result["kappa"],
            reference["kappa"],
            atol=1e-12,
        )
