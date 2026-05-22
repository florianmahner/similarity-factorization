"""Tests for ``pysrf.cross_val_score``."""

import numpy as np
import pandas as pd
import pytest

from pysrf import cross_val_score, estimate_rank
from pysrf._common import observation_mask
from pysrf.cross_validation import _cv_pool_fraction, _entry_splits
from helpers import make_symmetric_matrix


@pytest.fixture(scope="module")
def s():
    return make_symmetric_matrix(n=60, rank=5, noise_level=0.05, seed=0)


def test_returns_long_dataframe(s):
    curve = cross_val_score(s, ranks=[3, 5, 7], sampling_fraction=0.7,
                             n_folds=3, n_jobs=1)
    assert isinstance(curve, pd.DataFrame)
    assert set(curve.columns) == {"rep", "fold", "rank", "val_mse"}
    assert len(curve) == 3 * 3  # n_ranks * n_folds * n_repeats (default 1)


def test_argmin_near_true_rank(s):
    est = estimate_rank(s, n_bootstrap=10)
    curve = cross_val_score(s, ranks=[2, 4, 5, 6, 8],
                             sampling_fraction=est.sampling_fraction,
                             n_folds=5, n_jobs=1)
    mean = curve.groupby("rank")["val_mse"].mean()
    assert abs(int(mean.idxmin()) - 5) <= 1


def test_validates_sampling_fraction(s):
    with pytest.raises(ValueError, match="sampling_fraction"):
        cross_val_score(s, ranks=[3], sampling_fraction=1.5)
    with pytest.raises(ValueError, match="sampling_fraction"):
        cross_val_score(s, ranks=[3], sampling_fraction=0.0)


def test_validates_n_folds(s):
    with pytest.raises(ValueError, match="n_folds"):
        cross_val_score(s, ranks=[3], sampling_fraction=0.5, n_folds=1)


def test_deterministic(s):
    a = cross_val_score(s, ranks=[3, 5], sampling_fraction=0.7,
                         n_folds=3, random_state=11, n_jobs=1)
    b = cross_val_score(s, ranks=[3, 5], sampling_fraction=0.7,
                         n_folds=3, random_state=11, n_jobs=1)
    pd.testing.assert_frame_equal(a, b)


def test_splits_hide_only_observed_entries(s):
    s_missing = s.copy()
    s_missing[0, 1] = s_missing[1, 0] = np.nan
    observed = observation_mask(s_missing)
    splits = _entry_splits(
        observed,
        pool_fraction=_cv_pool_fraction(0.6, n_folds=3, n=s.shape[0]),
        n_folds=3,
        split_seeds=np.array([11], dtype=np.uint32),
    )
    for _, _, train_mask, validation_mask in splits:
        assert np.all(train_mask == train_mask.T)
        assert np.all(validation_mask == validation_mask.T)
        assert not np.any(train_mask & validation_mask)
        assert np.all(train_mask <= observed)
        assert np.all(validation_mask <= observed)


def test_custom_missing_value(s):
    s_missing = s.copy()
    s_missing[0, 1] = s_missing[1, 0] = -1.0
    curve = cross_val_score(
        s_missing, ranks=[3], sampling_fraction=0.7,
        n_folds=3, n_jobs=1, missing_values=-1.0,
        srf_kwargs={"max_outer": 2, "max_inner": 2},
    )
    assert len(curve) == 3


def test_cap_warning_when_inflation_exceeds_cap(s):
    with pytest.warns(RuntimeWarning, match="cap"):
        cross_val_score(s, ranks=[3], sampling_fraction=0.95,
                         n_folds=3, n_jobs=1)
