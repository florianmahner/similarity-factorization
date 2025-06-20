import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, GridSearchCV
from sklearn.utils import check_random_state

from .models.admm import ADMM


def create_admm_param_grid(
    rank_range=None,
    rho_range=None,
    max_outer_range=None,
    tol_range=None,
):
    """Create parameter grid for ADMM cross-validation.

    Args:
        rank_range: List of rank values to test (default: [5, 10, 15, 20])
        rho_range: List of rho values to test (default: None, meaning not optimized)
        max_outer_range: List of max_outer values to test (default: None)
        tol_range: List of tolerance values to test (default: None)

    Returns:
        dict: Parameter grid for GridSearchCV

    Examples:
        # Only optimize rank
        grid = create_admm_param_grid()

        # Optimize rank and rho
        grid = create_admm_param_grid(rho_range=[0.1, 1.0, 10.0])

        # Custom rank range with rho optimization
        grid = create_admm_param_grid(
            rank_range=[3, 5, 7, 10],
            rho_range=[0.5, 2.0, 5.0]
        )
    """
    param_grid = {}

    # Always include rank (most important parameter)
    if rank_range is None:
        rank_range = [5, 10, 15, 20]
    param_grid["rank"] = rank_range

    # Optional parameters
    if rho_range is not None:
        param_grid["rho"] = rho_range

    if max_outer_range is not None:
        param_grid["max_outer"] = max_outer_range

    if tol_range is not None:
        param_grid["tol"] = tol_range

    return param_grid


def train_val_split(n, train_ratio, rng):
    upper = rng.random((n, n)) < train_ratio
    upper = np.triu(upper, 1)
    train_mask = upper + upper.T
    val_mask = ~train_mask
    np.fill_diagonal(train_mask, False)
    np.fill_diagonal(val_mask, False)
    return train_mask, val_mask


class EntryMaskSplit(BaseCrossValidator):

    def __init__(self, n_repeats=5, train_ratio=0.8, random_state=None):
        self.n_repeats = n_repeats
        self.train_ratio = train_ratio
        self.random_state = random_state

    def get_n_splits(self, x=None, y=None, groups=None):
        return self.n_repeats

    def split(self, x, y=None, groups=None):
        n = int(np.sqrt(len(x)))
        if n * n != len(x):
            raise ValueError("x must be a square matrix")

        rng = check_random_state(self.random_state)
        for _ in range(self.n_repeats):
            tr_mask, val_mask = train_val_split(n, self.train_ratio, rng)
            tr_mask = np.where(tr_mask.ravel())[0]
            val_mask = np.where(val_mask.ravel())[0]
            yield tr_mask, val_mask


class ADMMCompletion(ADMM):
    """ADMM wrapper for sklearn cross-validation with coordinate input format."""

    def fit(self, x, y=None):
        if y is None:
            raise ValueError("y must be provided for matrix completion")

        coords = np.asarray(x)
        values = np.asarray(y)

        n = int(coords.max()) + 1
        full = np.zeros((n, n))
        mask = np.zeros((n, n), dtype=bool)
        rows, cols = coords[:, 0], coords[:, 1]
        full[rows, cols] = values
        mask[rows, cols] = True

        full = 0.5 * (full + full.T)
        mask = mask | mask.T

        # Set the mask and fit using parent ADMM class
        self.mask = mask.astype(float)
        super().fit(full)
        return self

    def transform(self, x):
        coords = np.asarray(x)
        rec = self.reconstruct()
        return rec[coords[:, 0], coords[:, 1]]

    def predict(self, x):
        return self.transform(x)

    def score(self, x, y):
        y_pred = self.transform(x)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        return -rmse


class GridSearchResults:
    """Container for grid search results with sklearn-style interface. Since
    we minimize RMSE, we convert the negative RMSE to a positive score."""

    def __init__(self, grid_search_cv):
        self._grid = grid_search_cv

    @property
    def best_params_(self):
        return self._grid.best_params_

    @property
    def best_score_(self):
        return -self._grid.best_score_

    @property
    def best_estimator_(self):
        return self._grid.best_estimator_

    @property
    def cv_results_(self):
        return pd.DataFrame(self._grid.cv_results_)

    @property
    def grid_search_(self):
        return self._grid


def cross_validate_admm(
    similarity_matrix,
    param_grid=None,
    optimize_rho=False,
    optimize_max_outer=False,
    n_repeats=5,
    train_ratio=0.8,
    random_state=0,
    verbose=1,
    n_jobs=-1,
):
    if param_grid is None:
        # Build default parameter grid based on optimization flags
        param_grid = {"rank": [5, 10, 15, 20]}

        if optimize_rho:
            param_grid["rho"] = [0.1, 1.0, 10.0]

        if optimize_max_outer:
            param_grid["max_outer"] = [10, 15, 20]

    # bring to coordinate format, so that we can use entry-wise cross-validation
    # in a sklearn grid search cv object that expects an object level splitting
    rows, cols = np.indices(similarity_matrix.shape)
    x_coords = np.column_stack((rows.ravel(), cols.ravel()))
    y_values = similarity_matrix.ravel()

    cv = EntryMaskSplit(
        n_repeats=n_repeats, train_ratio=train_ratio, random_state=random_state
    )

    grid = GridSearchCV(
        estimator=ADMMCompletion(random_state=random_state),
        param_grid=param_grid,
        cv=cv,
        scoring=None,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    grid.fit(x_coords, y_values)

    return GridSearchResults(grid)
