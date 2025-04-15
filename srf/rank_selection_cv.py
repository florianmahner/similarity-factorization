import numpy as np
import itertools
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedKFold
from srf.models.base import BaseNMF
from scipy.optimize import nnls

Array = np.ndarray


def split_similarity_matrix_into_folds(s, fold):
    train_idx, test_idx = fold
    s_train = s[np.ix_(train_idx, train_idx)]
    s_test = s[np.ix_(test_idx, test_idx)]
    s_cross = s[np.ix_(test_idx, train_idx)]
    return s_train, s_test, s_cross


def learn_mapping(s_train: Array, w_train: Array) -> Array:
    """
    Learn a nonnegative projection mapping P from the training similarity matrix s_train to the latent space w_train.
    """
    n_train, r = w_train.shape
    p = np.zeros((n_train, r))
    for j in range(r):
        # Solve NNLS for the jth column.
        # p will be a vector of length n_train.
        p_j, _ = nnls(s_train, w_train[:, j])
        p[:, j] = p_j
    return p


def project_samples(s_cross: Array, p: Array):
    w_test = s_cross @ p
    return w_test


def trifactor_cross_validation(
    s: Array,
    fold: tuple[list[int], list[int]],
    fold_index: int,
    init_seed: int,
    rank: int,
    model: BaseNMF,
    **kwargs,
):

    s_train, s_test, s_cross = split_similarity_matrix_into_folds(s, fold)

    w_train = model.fit_transform(s_train)
    projection = learn_mapping(s_train, w_train)
    w_test = project_samples(s_cross, projection)
    s_test_hat = w_test @ w_test.T
    mse = np.mean((s_test - s_test_hat) ** 2)

    return {
        "mse": mse,
        "init_seed": init_seed,
        "rank": rank,
        "fold_index": fold_index,
    }


def create_tasks(s, candidate_ranks, n_splits, n_repeats, random_state):
    """
    Create tasks for parallel cross-validation.
    """
    rkfold = RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    n = s.shape[0]
    folds = list(rkfold.split(np.arange(n)))
    return itertools.product(enumerate(folds), candidate_ranks)


def run_cv_rank_selection(
    model: BaseNMF,
    s: Array,
    candidate_ranks: list[int],
    n_repeats: int = 4,
    random_state: int = 0,
    n_splits: int = 5,
    verbose: int = 10,
    **kwargs,
):
    tasks = create_tasks(s, candidate_ranks, n_splits, n_repeats, random_state)
    tasks = list(tasks)

    n_jobs = min(len(tasks), 100)
    print(f"Running {len(tasks)} tasks with {n_jobs} jobs")

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(trifactor_cross_validation)(
            s=s,
            fold=fold,
            fold_index=fold_idx,
            rank=candidate_rank,
            init_seed=seed,
            model=model,
        )
        for seed, ((fold_idx, fold), candidate_rank) in enumerate(tasks)
    )

    df = pd.DataFrame(results)
    # Average over folds and repeats
    df = df.groupby(["rank", "init_seed"], as_index=False)["mse"].mean()
    return df
