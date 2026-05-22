"""Tests for SRF convergence behavior and early stopping."""

from __future__ import annotations

import numpy as np
import pytest

from pysrf import SRF

from helpers import make_symmetric_matrix, make_missing_matrix


class TestEarlyStopping:
    """Test that early stopping works correctly."""

    def test_early_stopping_triggers_before_max_iter(self):
        """Model should stop before max_outer when tolerance is met."""
        s = make_symmetric_matrix(n=30, rank=3, noise_level=0.001)
        max_outer = 100

        # Use a tolerance that allows convergence (relative to typical residuals)
        model = SRF(rank=3, max_outer=max_outer, tol=1e-2, random_state=42)
        model.fit(s)

        assert model.n_iter_ < max_outer, (
            f"Model ran all {max_outer} iterations without early stopping. "
            f"Final primal residual: {model.history_['primal_residual'][-1]:.2e}"
        )

    def test_loose_tolerance_stops_earlier(self):
        """Looser tolerance should result in fewer iterations."""
        s = make_symmetric_matrix(n=30, rank=3, noise_level=0.01)

        model_tight = SRF(rank=3, max_outer=100, tol=1e-6, random_state=42)
        model_tight.fit(s)

        model_loose = SRF(rank=3, max_outer=100, tol=1e-2, random_state=42)
        model_loose.fit(s)

        assert model_loose.n_iter_ <= model_tight.n_iter_, (
            f"Loose tolerance ({model_loose.n_iter_} iters) should not take more "
            f"iterations than tight tolerance ({model_tight.n_iter_} iters)"
        )

    def test_early_stopping_with_missing_data(self):
        """Early stopping should work with missing data."""
        s = make_symmetric_matrix(n=30, rank=3, noise_level=0.001)
        s_missing = make_missing_matrix(s, fraction=0.2)
        max_outer = 100

        model = SRF(rank=3, max_outer=max_outer, tol=1e-4, random_state=42)
        model.fit(s_missing)

        assert model.n_iter_ < max_outer, (
            f"Model with missing data ran all {max_outer} iterations. "
            f"Final primal residual: {model.history_['primal_residual'][-1]:.2e}"
        )

    def test_n_iter_matches_history_length(self):
        """n_iter_ should match the length of history arrays."""
        s = make_symmetric_matrix(n=20, rank=3)

        model = SRF(rank=3, max_outer=50, random_state=42)
        model.fit(s)

        assert model.n_iter_ == len(model.history_["rec_error"])
        assert model.n_iter_ == len(model.history_["primal_residual"])

    def test_convergence_residuals_decrease(self):
        """Primal residual should generally decrease over iterations."""
        s = make_symmetric_matrix(n=30, rank=3, noise_level=0.01)

        model = SRF(rank=3, max_outer=20, tol=1e-8, random_state=42)
        model.fit(s)

        primal_res = np.array(model.history_["primal_residual"])
        # Check that final residual is smaller than initial
        assert primal_res[-1] < primal_res[0], (
            f"Primal residual should decrease: initial={primal_res[0]:.2e}, "
            f"final={primal_res[-1]:.2e}"
        )


class TestMonotonicity:
    """Test that optimization objectives decrease monotonically."""

    def test_rec_error_decreases_complete_data(self):
        """Reconstruction error should decrease monotonically for complete data."""
        s = make_symmetric_matrix(n=30, rank=3, noise_level=0.01)

        model = SRF(rank=3, max_outer=20, tol=1e-8, random_state=42)
        model.fit(s)

        rec_error = np.array(model.history_["rec_error"])
        diffs = np.diff(rec_error)

        assert np.all(diffs <= 1e-10), (
            f"Reconstruction error should decrease monotonically. "
            f"Found {np.sum(diffs > 1e-10)} increases, max increase: {np.max(diffs):.2e}"
        )

    def test_rec_error_decreases_missing_data(self):
        """Reconstruction error should decrease monotonically with missing data."""
        s = make_symmetric_matrix(n=30, rank=3, noise_level=0.01)
        s_missing = make_missing_matrix(s, fraction=0.2)

        model = SRF(rank=3, max_outer=20, tol=1e-8, random_state=42)
        model.fit(s_missing)

        rec_error = np.array(model.history_["rec_error"])
        diffs = np.diff(rec_error)

        assert np.all(diffs <= 1e-10), (
            f"Reconstruction error should decrease monotonically with missing data. "
            f"Found {np.sum(diffs > 1e-10)} increases, max increase: {np.max(diffs):.2e}"
        )

    def test_data_fit_decreases_complete_data(self):
        """Data fit term should decrease monotonically for complete data."""
        s = make_symmetric_matrix(n=30, rank=3, noise_level=0.01)

        model = SRF(rank=3, max_outer=20, tol=1e-8, random_state=42)
        model.fit(s)

        data_fit = np.array(model.history_["primal_residual"]) ** 2
        diffs = np.diff(data_fit)

        assert np.all(diffs <= 1e-10), (
            f"Data fit should decrease monotonically. "
            f"Found {np.sum(diffs > 1e-10)} increases, max increase: {np.max(diffs):.2e}"
        )

    def test_evar_increases_complete_data(self):
        """Explained variance should increase monotonically."""
        s = make_symmetric_matrix(n=30, rank=3, noise_level=0.01)

        model = SRF(rank=3, max_outer=20, tol=1e-8, random_state=42)
        model.fit(s)

        evar = np.array(model.history_["evar"])
        diffs = np.diff(evar)

        assert np.all(diffs >= -1e-10), (
            f"Explained variance should increase monotonically. "
            f"Found {np.sum(diffs < -1e-10)} decreases, max decrease: {np.min(diffs):.2e}"
        )

    @pytest.mark.parametrize("rank", [2, 5, 10])
    def test_monotonicity_different_ranks(self, rank):
        """Monotonicity should hold for different rank values."""
        s = make_symmetric_matrix(n=50, rank=rank, noise_level=0.01)

        model = SRF(rank=rank, max_outer=15, tol=1e-8, random_state=42)
        model.fit(s)

        rec_error = np.array(model.history_["rec_error"])
        diffs = np.diff(rec_error)

        assert np.all(diffs <= 1e-10), (
            f"Monotonicity failed for rank={rank}. Max increase: {np.max(diffs):.2e}"
        )


class TestConvergenceQuality:
    """Test the quality of converged solutions."""

    def test_converged_solution_has_low_residual(self):
        """After convergence, primal residual should be small."""
        s = make_symmetric_matrix(n=30, rank=3, noise_level=0.001)

        model = SRF(rank=3, max_outer=100, tol=1e-5, random_state=42)
        model.fit(s)

        final_primal_res = model.history_["primal_residual"][-1]
        # Primal residual should be reasonably small after convergence
        assert final_primal_res < 1.0, (
            f"Final primal residual too large: {final_primal_res:.2e}"
        )

    def test_low_rank_recovery(self):
        """Model should recover good approximation of low-rank matrix."""
        s = make_symmetric_matrix(n=30, rank=3, noise_level=0.001)

        model = SRF(rank=3, max_outer=100, tol=1e-6, random_state=42)
        model.fit(s)

        final_evar = model.history_["evar"][-1]
        assert final_evar > 0.9, (
            f"Explained variance too low for low-rank matrix: {final_evar:.3f}"
        )

    def test_nonnegative_factors(self):
        """Converged factors should be non-negative."""
        s = make_symmetric_matrix(n=30, rank=3)

        model = SRF(rank=3, max_outer=50, random_state=42)
        model.fit(s)

        assert np.all(model.w_ >= 0), "Factor matrix should be non-negative"
