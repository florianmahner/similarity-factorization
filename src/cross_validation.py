import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator, ParameterGrid
from sklearn.utils import check_random_state
from joblib import Parallel, delayed
from typing import Generator
from models.admm import ADMM


def mask_missing_entries(
    x: np.ndarray,
    observed_fraction: float,
    rng: np.random.RandomState,
    missing_values: float | None = np.nan,
) -> np.ndarray:
    """Mask exactly observed_fraction of the observed (non-NaN) entries symmetrically."""
    observed_mask = ~np.isnan(x) if missing_values is np.nan else x != missing_values

    triu_i, triu_j = np.triu_indices_from(x)
    triu_observed = observed_mask[triu_i, triu_j]
    valid_positions = np.where(triu_observed)[0]

    n_to_keep = int(observed_fraction * len(valid_positions))
    if n_to_keep == 0:
        return np.ones_like(x, dtype=bool)

    # Subsample from valid upper triangular positions
    keep_positions = rng.choice(valid_positions, size=n_to_keep, replace=False)

    observation_mask = np.zeros_like(x, dtype=bool)
    keep_i = triu_i[keep_positions]
    keep_j = triu_j[keep_positions]
    observation_mask[keep_i, keep_j] = True
    observation_mask[keep_j, keep_i] = True

    return ~observation_mask


def fit_and_score(estimator, x, val_mask, fit_params, split_idx):
    """Fit estimator with parameters and return validation score."""
    est = clone(estimator).set_params(**fit_params)
    # Ensure missing_values is exactly np.nan so that identity checks using "is" succeed after pickling
    if hasattr(est, "missing_values"):
        est.set_params(missing_values=np.nan)
    x_copy = np.copy(x)
    x_copy[val_mask] = np.nan

    reconstruction = est.fit(x_copy).reconstruct()

    mse = np.mean((val_mask * (x - reconstruction)) ** 2)
    result = {
        "score": mse,
        "split": split_idx,
        "estimator": est,
        "params": fit_params,
    }
    return result


class EntryMaskSplit(BaseCrossValidator):
    """Cross-validator for symmetric matrices using entry-wise splits."""

    def __init__(
        self,
        n_repeats: int = 5,
        observed_fraction: float = 0.8,
        random_state: int | None = None,
        missing_values: float | None = np.nan,
    ):
        self.n_repeats = n_repeats
        self.observed_fraction = observed_fraction
        self.random_state = random_state
        self.missing_values = missing_values

    def get_n_splits(
        self, x: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None
    ) -> int:
        return self.n_repeats

    def split(
        self, x: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/validation mask pairs for the matrix."""
        rng = check_random_state(self.random_state)
        for _ in range(self.n_repeats):
            yield mask_missing_entries(
                x, self.observed_fraction, rng, self.missing_values
            )


class ADMMGridSearchCV:
    """Grid search cross-validation for matrix completion."""

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
        """Fit grid search with simple results DataFrame."""
        param_grid = ParameterGrid(self.param_grid)
        cv_splits = list(self.cv.split(x))

        all_jobs = [
            (params, val_mask, split_idx)
            for split_idx, val_mask in enumerate(cv_splits)
            for params in param_grid
        ]

        if self.verbose:
            print(
                f"Running {len(all_jobs)} jobs ({len(param_grid)} params x {len(cv_splits)} CV splits)"
            )

        all_results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(fit_and_score)(self.estimator, x, val_mask, params, split_idx)
            for params, val_mask, split_idx in all_jobs
        )

        # Find best parameters by mean score across splits (concise version)
        df = pd.DataFrame([{**r["params"], "score": r["score"]} for r in all_results])
        param_cols = [col for col in df.columns if col != "score"]
        mean_scores = df.groupby(param_cols)["score"].mean()

        best_params_idx = mean_scores.idxmin()
        self.best_params_ = dict(
            zip(
                param_cols,
                best_params_idx if len(param_cols) > 1 else [best_params_idx],
            )
        )
        self.best_score_ = mean_scores.min()

        # Create DataFrame directly from results
        self.cv_results_ = pd.DataFrame(
            [
                {
                    **{"score": result["score"], "split": result["split"]},
                    **result["params"],
                }
                for result in all_results
            ]
        )

        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
        self.best_estimator_.fit(x)

        return self


# TODO Think about if we actually want to fit the final estimator here on the best parameters
def cross_val_score(
    similarity_matrix: np.ndarray,
    param_grid: dict[str, list] | None = None,
    n_repeats: int = 5,
    observed_fraction: float = 0.8,
    random_state: int = 0,
    verbose: int = 1,
    n_jobs: int = -1,
    missing_values: float | None = np.nan,
) -> ADMMGridSearchCV:
    """Cross-validate ADMM parameters for matrix completion.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Symmetric similarity matrix to cross-validate.
    param_grid : dict[str, list] | None, optional
        Dictionary with parameter names (str) as keys and lists of values to try
        as values, or a list of such dictionaries, in which case the first
        parameter given will be used to filter the grid.
    n_repeats : int, optional
        Number of times to repeat the cross-validation.
    observed_fraction : float, optional
        Fraction of the observed entries to mask for each cross-validation split.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : int, optional
        Verbosity level.
    n_jobs : int, optional
        Number of jobs to run in parallel.
    missing_values : np.nan | float | None, optional
        Value to consider as missing.

    Returns
    -------
    ADMMGridSearchCV
        Fitted ADMMGridSearchCV object with best parameters and scores.
    """
    if param_grid is None:
        param_grid = {"rank": [5, 10, 15, 20]}

    cv = EntryMaskSplit(
        n_repeats=n_repeats,
        observed_fraction=observed_fraction,
        random_state=random_state,
        missing_values=missing_values,
    )
    grid = ADMMGridSearchCV(
        estimator=ADMM(random_state=random_state),
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    return grid.fit(similarity_matrix)
