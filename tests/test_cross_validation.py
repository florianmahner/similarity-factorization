import numpy as np
import pytest
from pysrf import (
    SRF,
    cross_val_score,
    ADMMGridSearchCV,
    EntryMaskSplit,
    mask_missing_entries,
)


def generate_test_matrix(n=30, rank=5, random_state=42):
    rng = np.random.RandomState(random_state)
    w = rng.rand(n, rank)
    s = w @ w.T
    s = (s + s.T) / 2
    return s


def test_mask_missing_entries():
    np.random.seed(42)
    s = generate_test_matrix(n=20)
    rng = np.random.RandomState(42)

    mask = mask_missing_entries(s, sampling_fraction=0.8, rng=rng)

    assert mask.shape == s.shape
    assert mask.dtype == bool
    assert np.allclose(mask, mask.T)

    observed_count = np.sum(~mask)
    total_count = s.shape[0] * (s.shape[0] + 1) // 2
    assert observed_count > 0


def test_entry_mask_split():
    s = generate_test_matrix(n=20)

    cv = EntryMaskSplit(n_repeats=3, sampling_fraction=0.8, random_state=42)

    assert cv.get_n_splits() == 3

    splits = list(cv.split(s))
    assert len(splits) == 3

    for mask in splits:
        assert mask.shape == s.shape
        assert mask.dtype == bool


def test_admm_grid_search_cv():
    s = generate_test_matrix(n=20, rank=5)

    param_grid = {"rank": [3, 5, 7], "rho": [2.0, 3.0]}
    cv = EntryMaskSplit(n_repeats=2, sampling_fraction=0.8, random_state=42)

    grid = ADMMGridSearchCV(
        estimator=SRF(max_outer=5, random_state=42),
        param_grid=param_grid,
        cv=cv,
        n_jobs=1,
        verbose=0,
    )

    grid.fit(s)

    assert hasattr(grid, "best_params_")
    assert hasattr(grid, "best_score_")
    assert hasattr(grid, "cv_results_")
    assert "rank" in grid.best_params_
    assert "rho" in grid.best_params_


def test_cross_val_score():
    s = generate_test_matrix(n=20, rank=5)

    param_grid = {"rank": [3, 5]}

    result = cross_val_score(
        s,
        param_grid=param_grid,
        n_repeats=2,
        sampling_fraction=0.8,
        random_state=42,
        verbose=0,
        n_jobs=1,
    )

    assert hasattr(result, "best_params_")
    assert hasattr(result, "best_score_")
    assert "rank" in result.best_params_


def test_cross_val_score_with_missing_data():
    s = generate_test_matrix(n=20, rank=5)

    mask = np.random.rand(*s.shape) < 0.1
    mask = mask | mask.T
    s[mask] = np.nan

    param_grid = {"rank": [3, 5]}

    result = cross_val_score(
        s,
        param_grid=param_grid,
        n_repeats=2,
        sampling_fraction=0.8,
        random_state=42,
        verbose=0,
        n_jobs=1,
        missing_values=np.nan,
    )

    assert hasattr(result, "best_params_")
    assert hasattr(result, "best_score_")


def test_cross_val_score_fit_final():
    s = generate_test_matrix(n=20, rank=5)

    result = cross_val_score(
        s,
        param_grid={"rank": [5]},
        n_repeats=2,
        sampling_fraction=0.8,
        random_state=42,
        verbose=0,
        n_jobs=1,
        fit_final_estimator=True,
    )

    assert hasattr(result, "best_estimator_")
    assert hasattr(result.best_estimator_, "w_")
