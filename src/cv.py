"""
Sklearn-compatible cross-validation for matrix completion problems.

This module provides custom CV splitters and estimators that work with
matrix masking for similarity matrix factorization problems.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import BaseCrossValidator, ParameterGrid
from sklearn.utils.validation import check_array
from joblib import Parallel, delayed


from models.admm import ADMM

# TODO Implement the subsampling when we eg want to cross validate spose. Potentially do this when some entries are masked as NaN??


def train_val_split(
    n: int, train_ratio: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    mask_upper = rng.random((n, n)) < train_ratio
    mask_upper = np.triu(mask_upper, 1)
    train_mask = mask_upper + mask_upper.T
    val_mask = ~train_mask
    np.fill_diagonal(train_mask, False)
    np.fill_diagonal(val_mask, False)
    return train_mask, val_mask


class MatrixCompletionCV(BaseEstimator, TransformerMixin):
    """Cross-validated matrix completion estimator using parameter grid search."""

    def __init__(
        self,
        estimator=None,
        param_grid=None,
        n_repeats=10,
        train_ratio=0.8,
        n_jobs=-1,
        random_state=None,
        verbose=False,
    ):
        self.estimator = estimator
        self.param_grid = param_grid or {"rank": list(range(5, 26, 5))}
        self.n_repeats = n_repeats
        self.train_ratio = train_ratio
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _evaluate_params(self, params, X, train_mask, val_mask):
        """Evaluate a single parameter configuration on train/val split."""
        # Create ADMM directly with mask and params
        model_params = {"mask": train_mask, "random_state": self.random_state, **params}

        if self.estimator is None:
            estimator = ADMM(**model_params)
        else:
            estimator = self.estimator.__class__(**model_params)

        estimator.fit(X)

        # Score on validation entries manually
        reconstruction = estimator.reconstruct()
        val_error = np.sum(val_mask * (X - reconstruction) ** 2) / np.sum(val_mask)
        score = -val_error  # Negative because sklearn expects higher = better

        return params, score

    def _evaluate_results(self, results_list, param_combinations):
        """Evaluate the results of the repeated evaluations."""
        results_data = []

        for i, (params_dict, score) in enumerate(results_list):
            repeat_idx = i // len(param_combinations)
            results_data.append(
                {
                    "repeat": repeat_idx,
                    "score": score,
                    **{f"param_{k}": v for k, v in params_dict.items()},
                }
            )

        self.cv_results_ = pd.DataFrame(results_data)

        # Find best parameters by averaging scores across repeats
        param_cols = [
            col for col in self.cv_results_.columns if col.startswith("param_")
        ]
        mean_scores = self.cv_results_.groupby(param_cols)["score"].mean()

        best_params_row = mean_scores.idxmax()
        if isinstance(best_params_row, tuple):
            self.best_params_ = dict(
                zip([col.replace("param_", "") for col in param_cols], best_params_row)
            )
        else:
            self.best_params_ = {param_cols[0].replace("param_", ""): best_params_row}

        self.best_score_ = mean_scores.max()

    def fit(self, X, y=None):
        """Fit using repeated masked evaluations with parameter grid search."""
        X = check_array(X)
        param_combinations = list(ParameterGrid(self.param_grid))

        tasks = []
        for repeat in range(self.n_repeats):
            seed = None if self.random_state is None else self.random_state + repeat
            split_rng = np.random.default_rng(seed)
            train_mask, val_mask = train_val_split(
                X.shape[0], self.train_ratio, split_rng
            )

            for params in param_combinations:
                tasks.append(
                    delayed(self._evaluate_params)(params, X, train_mask, val_mask)
                )

        results_list = Parallel(n_jobs=self.n_jobs)(tasks)
        self._evaluate_results(results_list, param_combinations)

        # Fit final estimator on full data
        if self.estimator is None:
            self.best_estimator_ = ADMM(
                random_state=self.random_state, **self.best_params_
            )
        else:
            self.best_estimator_ = self.estimator.__class__(
                random_state=self.random_state, **self.best_params_
            )

        self.best_estimator_.fit(X)

        if self.verbose:
            print(f"Best parameters: {self.best_params_}")
            print(f"Best score: {self.best_score_:.4f}")

        return self

    # Delegate methods directly to the best estimator
    def transform(self, X):
        return self.best_estimator_.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def reconstruct(self, X=None):
        return self.best_estimator_.reconstruct(X)

    def score(self, X, y=None):
        return self.best_estimator_.score(X, y)
