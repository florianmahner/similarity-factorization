import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator, ParameterGrid
from sklearn.utils import check_random_state
from joblib import Parallel, delayed
from typing import Any, Generator
from collections import defaultdict
from models.admm import ADMM


def estimate_effective_rank(similarity_matrix: np.ndarray) -> int:
    """Estimate the effective rank of the similarity matrix."""
    return np.linalg.matrix_rank(similarity_matrix)


def estimate_subsample_size(similarity_matrix: np.ndarray) -> int:
    """Estimate the subsample size for the similarity matrix."""
    # TODO this is where we should add the correct subsampling strategy
    effective_rank = estimate_effective_rank(similarity_matrix)
    n = similarity_matrix.shape[0]
    return int(n * effective_rank / n)


# def train_val_split(
#     n: int, train_ratio: float, rng: np.random.RandomState
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Create train/validation masks for symmetric matrix entries."""
#     upper = rng.random((n, n)) < train_ratio
#     upper = np.triu(upper, 1)
#     train_mask = upper + upper.T
#     val_mask = ~train_mask
#     np.fill_diagonal(train_mask, False)
#     np.fill_diagonal(val_mask, False)
#     return train_mask, val_mask


def train_val_split(
    n: int,
    train_ratio: float,
    rng: np.random.RandomState,
    observed_mask: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create train/validation masks for symmetric matrix entries."""
    if observed_mask is None:
        # Use all off-diagonal entries when no missing data
        upper = rng.random((n, n)) < train_ratio
        upper = np.triu(upper, 1)
        train_mask = upper + upper.T
        val_mask = ~train_mask
    else:
        # Only split the OBSERVED entries for train/val
        observed_upper = np.triu(observed_mask, 1)
        train_upper = rng.random(observed_upper.shape) < train_ratio
        train_upper = train_upper & observed_upper  # Only train on observed
        train_mask = train_upper + train_upper.T

        # Validation = observed entries that are NOT in training
        val_mask = observed_mask & ~train_mask

    np.fill_diagonal(train_mask, False)
    np.fill_diagonal(val_mask, False)
    return train_mask, val_mask


class EntryMaskSplit(BaseCrossValidator):
    """Cross-validator for symmetric matrices using entry-wise splits."""

    def __init__(
        self,
        n_repeats: int = 5,
        train_ratio: float = 0.8,
        random_state: int | None = None,
    ):
        self.n_repeats = n_repeats
        self.train_ratio = train_ratio
        self.random_state = random_state

    def get_n_splits(
        self, x: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None
    ) -> int:
        return self.n_repeats

    def split(
        self, x: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/validation mask pairs for the matrix."""
        if x.shape[0] != x.shape[1]:
            raise ValueError("Input matrix must be square")

        rng = check_random_state(self.random_state)
        for _ in range(self.n_repeats):
            yield train_val_split(x.shape[0], self.train_ratio, rng)


# def _fit_and_score(estimator, x, train_mask, val_mask, params):
#     """Fit estimator with parameters and return validation score."""
#     est = clone(estimator).set_params(**params)
#     est.mask = train_mask.astype(float)
#     est.fit(x)

#     val_entries = val_mask.astype(bool)

#     if not val_entries.any():
#         return {"params": params, "score": 0.0, "estimator": est}

#     mse = np.mean((x[val_entries] - est.reconstruct()[val_entries]) ** 2)
#     return {"params": params, "score": mse, "estimator": est}


def _fit_and_score(estimator, x, train_mask, val_mask, params):
    """Fit estimator with parameters and return validation score."""
    est = clone(estimator).set_params(**params)

    # Create mask for training: only use observed training entries
    observed_mask = ~np.isnan(x)
    final_train_mask = train_mask & observed_mask

    # Replace NaN with zeros for ADMM fitting (will be ignored due to mask)
    x_clean = np.nan_to_num(x, nan=0.0)
    est.mask = final_train_mask.astype(float)
    est.fit(x_clean)

    # Validation: only evaluate on observed entries that were held out
    val_entries = val_mask & observed_mask

    if not val_entries.any():
        return {"params": params, "score": 0.0, "estimator": est}

    # Evaluate only on ORIGINAL observed values (not the filled zeros)
    reconstruction = est.reconstruct()
    mse = np.mean(
        (x[val_entries] - reconstruction[val_entries]) ** 2
    )  # Use original x, not x_clean!
    return {"params": params, "score": mse, "estimator": est}


class ADMMGridSearchCV:
    """Grid search cross-validation for matrix completion with full parallelization."""

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: dict[str, list],
        cv: EntryMaskSplit,
        n_jobs: int = -1,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, x: np.ndarray) -> "ADMMGridSearchCV":
        """Fit grid search with full parallelization across all parameter x CV combinations."""
        param_grid = ParameterGrid(self.param_grid)
        cv_splits = list(self.cv.split(x))

        all_jobs = [
            (params, train_mask, val_mask)
            for params in param_grid
            for train_mask, val_mask in cv_splits
        ]

        if self.verbose:
            print(
                f"Running {len(all_jobs)} parallel jobs ({len(param_grid)} params x {len(cv_splits)} CV splits)"
            )

        all_results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_and_score)(self.estimator, x, train_mask, val_mask, params)
            for params, train_mask, val_mask in all_jobs
        )

        grouped = defaultdict(list)
        for result in all_results:
            key = tuple(sorted(result["params"].items()))
            grouped[key].append(result["score"])

        results = []
        for param_key, scores in grouped.items():
            params = dict(param_key)
            results.append(
                {
                    "params": params,
                    "mean_test_score": np.mean(scores),
                    "std_test_score": np.std(scores),
                    "split_scores": scores,
                }
            )

        results.sort(key=lambda x: x["mean_test_score"])
        for i, result in enumerate(results):
            result["rank_test_score"] = i + 1

        self.cv_results_ = results
        self.best_params_ = results[0]["params"]
        self.best_score_ = results[0]["mean_test_score"]

        # FIX: Handle NaN in final fitting - clean matrix BEFORE fitting
        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)

        # For final fit, use all observed entries
        observed_mask = ~np.isnan(x)
        np.fill_diagonal(observed_mask, False)  # Exclude diagonal

        # Replace NaN with zeros for fitting
        x_clean = np.nan_to_num(x, nan=0.0)

        self.best_estimator_.mask = observed_mask.astype(float)
        self.best_estimator_.fit(x_clean)  # Pass cleaned matrix!

        return self


class _GridSearchResults:
    """Container for grid search results with sklearn-style interface."""

    def __init__(self, grid_search: ADMMGridSearchCV):
        self._grid = grid_search

    @property
    def best_params_(self) -> dict[str, Any]:
        return self._grid.best_params_

    @property
    def best_score_(self) -> float:
        return self._grid.best_score_

    @property
    def best_estimator_(self) -> BaseEstimator:
        return self._grid.best_estimator_

    @property
    def cv_results_(self) -> pd.DataFrame:
        """Get cross-validation results as a DataFrame."""
        return pd.DataFrame(
            [
                {
                    "mean_test_score": r["mean_test_score"],
                    "std_test_score": r["std_test_score"],
                    "rank_test_score": r["rank_test_score"],
                    **{f"param_{k}": v for k, v in r["params"].items()},
                }
                for r in self._grid.cv_results_
            ]
        )


def cross_validate_admm(
    similarity_matrix: np.ndarray,
    param_grid: dict[str, list] | None = None,
    n_repeats: int = 5,
    train_ratio: float = 0.8,
    random_state: int = 0,
    verbose: int = 1,
    n_jobs: int = -1,
) -> _GridSearchResults:
    """Cross-validate ADMM parameters for matrix completion with full parallelization."""
    if param_grid is None:
        param_grid = {"rank": [5, 10, 15, 20]}

    cv = EntryMaskSplit(
        n_repeats=n_repeats, train_ratio=train_ratio, random_state=random_state
    )
    grid = ADMMGridSearchCV(
        estimator=ADMM(random_state=random_state),
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    grid.fit(similarity_matrix)

    return _GridSearchResults(grid)
