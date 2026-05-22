"""Tests for ``pysrf.coherence``."""

import numpy as np
import pytest

from pysrf import RankEstimate, estimate_rank
from helpers import make_missing_matrix, make_symmetric_matrix


@pytest.fixture(scope="module")
def s():
    return make_symmetric_matrix(n=60, rank=5, noise_level=0.05, seed=0)


def test_returns_rank_estimate(s):
    est = estimate_rank(s, n_bootstrap=5)
    assert isinstance(est, RankEstimate)


def test_recovers_true_rank(s):
    est = estimate_rank(s, n_bootstrap=10)
    assert abs(est.rank - 5) <= 1


def test_sampling_fraction_in_unit_interval(s):
    est = estimate_rank(s, n_bootstrap=5)
    assert 0.0 < est.sampling_fraction <= 1.0
    assert est.detectability_floor <= est.sampling_fraction


def test_diagnostic_shapes(s):
    est = estimate_rank(s, max_rank=15, n_bootstrap=5)
    assert est.eigenvalues.shape == (15,)
    assert est.leakage.shape == (15,)
    assert est.sampling_grid.shape == est.recovery_loss_raw.shape
    assert est.sampling_grid.shape == est.recovery_loss_monotone.shape
    assert est.n_features_in == s.shape[0]


def test_recovery_loss_monotone_non_increasing(s):
    est = estimate_rank(s, n_bootstrap=5)
    assert np.all(np.diff(est.recovery_loss_monotone) <= 1e-12)


def test_deterministic(s):
    a = estimate_rank(s, n_bootstrap=5, random_state=7)
    b = estimate_rank(s, n_bootstrap=5, random_state=7)
    assert a.rank == b.rank
    assert a.sampling_fraction == b.sampling_fraction
    np.testing.assert_array_equal(a.eigenvalues, b.eigenvalues)
    np.testing.assert_array_equal(a.recovery_loss_monotone, b.recovery_loss_monotone)


def test_handles_missing_entries(s):
    s_missing = make_missing_matrix(s, fraction=0.2, seed=1)
    est = estimate_rank(s_missing, n_bootstrap=5)
    assert isinstance(est, RankEstimate)
    assert est.rank >= 1


def test_frozen_dataclass(s):
    est = estimate_rank(s, n_bootstrap=5)
    with pytest.raises(AttributeError):
        est.rank = 99  # type: ignore[misc]
