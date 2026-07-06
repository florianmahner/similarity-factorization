import numpy as np
from pysrf import (
    SRF,
    cross_val_score,
    GridSearchCV,
    EntryMaskSplit,
    create_train_val_split,
)
from helpers import make_symmetric_matrix


def test_train_val_split_masks():
    np.random.seed(42)
    s = make_symmetric_matrix(n=20, noise_level=0.0)
    rng = np.random.RandomState(42)

    train_mask, val_mask = create_train_val_split(
        s, sampling_fraction=0.8, rng=rng, missing_values=np.nan
    )

    assert train_mask.shape == s.shape
    assert val_mask.shape == s.shape
    assert train_mask.dtype == bool
    assert val_mask.dtype == bool
    assert np.allclose(train_mask, train_mask.T)
    assert np.allclose(val_mask, val_mask.T)
    # No overlap between train and validation
    assert not np.any(train_mask & val_mask)
    # At least some entries in both
    assert np.sum(train_mask) > 0
    assert np.sum(val_mask) > 0


def test_entry_mask_split():
    s = make_symmetric_matrix(n=20, noise_level=0.0)

    cv = EntryMaskSplit(n_repeats=3, sampling_fraction=0.8, random_state=42)

    assert cv.get_n_splits() == 3

    splits = list(cv.split(s))
    assert len(splits) == 3

    for train_mask, val_mask in splits:
        assert train_mask.shape == s.shape
        assert val_mask.shape == s.shape
        assert train_mask.dtype == bool
        assert val_mask.dtype == bool
        assert np.allclose(train_mask, train_mask.T)
        assert np.allclose(val_mask, val_mask.T)
        assert not np.any(train_mask & val_mask)


def test_SRF_grid_search_cv():
    s = make_symmetric_matrix(n=20, rank=5, noise_level=0.0)

    param_grid = {"rank": [3, 5, 7], "rho": [2.0, 3.0]}
    cv = EntryMaskSplit(n_repeats=2, sampling_fraction=0.8, random_state=42)

    grid = GridSearchCV(
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
    s = make_symmetric_matrix(n=20, rank=5, noise_level=0.0)

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
    s = make_symmetric_matrix(n=20, rank=5, noise_level=0.0)

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
    s = make_symmetric_matrix(n=20, rank=5, noise_level=0.0)

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


def test_rank_recovery():
    s = make_symmetric_matrix(n=100, rank=5, noise_level=0.0)

    result = cross_val_score(
        s,
        estimator=SRF(rho=3.0, max_outer=50, max_inner=20, random_state=42),
        param_grid={"rank": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        n_repeats=5,
        estimate_sampling_fraction=True,
        sampling_selection="mean",
        random_state=42,
        verbose=0,
    )

    assert result.best_params_["rank"] == 5
