import pyximport
import pandas as pd
import numpy as np

pyximport.install(language_level=3, setup_args={"include_dirs": np.get_include()})
from srf.mixed.bsum_cython import update_w
from srf.mixed.cd_updates import update_v, update_lambda
from tools.rsa import compute_similarity
from collections import defaultdict
from joblib import Parallel, delayed
from srf.models.base import init_factor

# TODO rewrite loss only based on nan mask entries and then print this explained variance or so / also determine convergence.
# TODO I probably need to do the correc train/val/test split so that we can then afterwards get a true estimate on the entire matrix?
# TODO I need to actually also still solve the median splitting problem!
# TODO when we do median splitting we already hold out certain entries. this is a mask and we need to mask it furtherr!
# TODO outsource different computations again, eg rank selection etc.


class ADMM:
    def __init__(
        self,
        rank,
        rho=1.0,
        max_outer=15,
        w_inner=40,
        tol=1e-4,
        verbose=False,
        init="random_sqrt",
        random_state=None,
        eps=np.finfo(float).eps,
    ):
        self.rank = rank
        self.rho = rho
        self.max_outer = max_outer
        self.w_inner = w_inner
        self.tol = tol
        self.verbose = verbose
        self.init = init
        self.random_state = random_state
        self.eps = eps

    def fit(
        self,
        s,
        mask,
        seed=None,
        bounds=None,
    ):
        w = init_factor(s, self.rank, self.init, self.random_state, self.eps)
        lam = np.zeros_like(s)
        min_val, max_val = bounds if bounds is not None else (None, None)

        history = defaultdict(list)

        for i in range(1, self.max_outer + 1):
            v = update_v(mask, s, w, lam, self.rho, min_val, max_val)
            T = v + lam / self.rho
            w = update_w(T, w, max_iter=self.w_inner, tol=self.tol)
            lam = update_lambda(lam, v, w, self.rho)

            # Compute all three terms of the objective
            data_fit = np.linalg.norm(mask * (s - v), "fro") ** 2
            penalty = (self.rho / 2) * np.linalg.norm(v - w @ w.T, "fro") ** 2
            lagrangian = np.sum(lam * (v - w @ w.T))
            total_obj = data_fit + penalty + lagrangian

            evar = 1 - np.linalg.norm(mask * (s - w @ w.T), "fro") / np.linalg.norm(
                s * mask, "fro"
            )
            history["evar"].append(evar)
            history["data_fit"].append(data_fit)
            history["penalty"].append(penalty)
            history["lagrangian"].append(lagrangian)
            history["total_objective"].append(total_obj)
            history["rec_error"].append(np.linalg.norm((s - w @ w.T) * mask, "fro"))

            if np.linalg.norm(v - w @ w.T, "fro") < self.tol:
                break
            if self.verbose:
                print(
                    f"Iter {i}/{self.max_outer} | Obj: {total_obj:.6f} | Evar: {evar:.6f} | Recon: {data_fit:.6f} | Penalty: {penalty:.6f} | Lag: {lagrangian:.6f}",
                    end="\r",
                )

        self.s_hat_ = mask * (w @ w.T)
        self.history_ = history
        self.w_ = w
        return w

    def fit_transform(self, s, mask=None, seed=None, bounds=None):
        if mask is None:
            mask = np.ones_like(s)  # TODO Check this

        self.fit(s, mask, seed, bounds)
        return self.w_

    def fit_w(self, s):
        # TODO intergrate into the fit transform somehow!
        x0 = init_factor(s, self.rank, self.init, self.random_state, self.eps)
        self.w_ = update_w(s, x0, max_iter=self.w_inner, tol=self.tol)
        self.s_hat_ = self.w_ @ self.w_.T
        self.history_ = defaultdict(list)
        self.history_["rec_error"] = np.linalg.norm(s - self.s_hat_, "fro")
        return self.w_


def admm_symnmf_masked(
    s,
    mask,
    rank,
    rho=1.0,
    max_outer=15,
    w_inner=40,
    tol=1e-4,
    seed=None,
    bounds=None,
    verbose=False,
):
    rng = np.random.default_rng(seed)
    n = s.shape[0]
    w = rng.random((n, rank)) + 1e-3  # TODO Check tgis
    lam = np.zeros_like(s)
    min_val, max_val = bounds if bounds is not None else (None, None)

    for i in range(max_outer):
        v = update_v(mask, s, w, lam, rho, min_val, max_val)
        T = v + lam / rho
        w = update_w(T, w, max_iter=w_inner, tol=tol)
        lam = update_lambda(lam, v, w, rho)

        # TODO Check this if it is a good criterion!
        if np.linalg.norm(v - w @ w.T, "fro") < 1e-6:
            break
        if verbose:
            print(f"Iteration {i} of {max_outer} completed", end="\r")
    return w


def random_mask(n, keep_ratio, rng):
    m = rng.random((n, n)) < keep_ratio
    m = np.triu(m) + np.triu(m, 1).T
    return m.astype(float)


def train_val_test_split(n, keep_ratio, train_ratio, rng):
    # Sample upper triangle entries (without diagonal)
    triu_idx = np.triu_indices(n, k=1)
    total_edges = len(triu_idx[0])

    # Randomly keep 'keep_ratio' of possible edges
    num_keep = int(keep_ratio * total_edges)
    perm = rng.permutation(total_edges)
    keep_idx = perm[:num_keep]

    num_train = int(train_ratio * num_keep)
    train_idx = keep_idx[:num_train]
    val_idx = keep_idx[num_train:]
    mask_train = np.zeros((n, n), dtype=float)
    mask_val = np.zeros((n, n), dtype=float)

    i_train, j_train = triu_idx[0][train_idx], triu_idx[1][train_idx]
    i_val, j_val = triu_idx[0][val_idx], triu_idx[1][val_idx]

    mask_train[i_train, j_train] = 1.0
    mask_train[j_train, i_train] = 1.0  # enforce symmetry
    mask_val[i_val, j_val] = 1.0
    mask_val[j_val, i_val] = 1.0

    return mask_train, mask_val


def train_val_split(n, train_ratio, rng):
    mask_upper = rng.random((n, n)) < train_ratio
    mask_upper = np.triu(mask_upper, 1)
    train_mask = mask_upper + mask_upper.T
    val_mask = ~train_mask
    np.fill_diagonal(train_mask, False)
    np.fill_diagonal(val_mask, False)
    return train_mask, val_mask


def admm(s, rank, train_ratio=0.8, rng=None, bounds=None):
    if rng is None:
        rng = np.random.default_rng()

    # TODO Continue with the mask val!
    mask_train, mask_val = train_val_split(s.shape[0], train_ratio, rng)

    w_est = admm_symnmf_masked(
        s,
        mask_train,
        rank,
        rho=1.0,
        max_outer=15,
        w_inner=40,
        tol=1e-4,
        seed=rng,
        bounds=bounds,
    )
    return w_est


def _evaluate_rank(
    rank,
    s_full,
    mask_train,
    mask_val,
    seed,
    bounds,
    similarity_measure: str = "linear",
):

    # TODO is this correct??
    bounds_min = s_full[mask_train].min()
    bounds_max = s_full[mask_train].max()
    bounds = (bounds_min, bounds_max)

    model = ADMM(
        rank=rank,
        rho=1.0,
        max_outer=10,
        w_inner=50,
        tol=0.0,
        init="random_sqrt",
    )

    w_est = model.fit_transform(s_full, mask_train, seed, bounds)
    s_pred = compute_similarity(
        w_est, w_est, similarity_measure
    )  # NOTE this should be dot probably!!!

    val_mse = np.sum(mask_val * (s_full - s_pred) ** 2) / np.sum(mask_val)

    return rank, val_mse


def second_order_mask(mask, keep_ratio, rng):
    """
    Given an input mask (boolean or 0/1), produce:
      - mask_train: a copy of `mask` where only `keep_ratio` of the True entries are still True.
      - mask_val: a boolean mask which is True exactly on those entries that were True in `mask`
                  but got zeroed out in mask_train.
    """
    mask_bool = mask.astype(bool)
    mask_train = mask_bool.copy()
    non_zero_mask = mask_bool
    random_mask = rng.random(mask.shape) < keep_ratio
    mask_train[non_zero_mask] = random_mask[non_zero_mask]
    mask_val = non_zero_mask & (~mask_train)
    return mask_train, mask_val


def find_best_rank(
    s,
    candidate_ranks,
    train_ratio=0.8,
    rng=None,
    bounds=None,
    similarity_measure: str = "cosine",
    mask=None,
    n_repeats=1,
):
    if rng is None:
        rng = np.random.default_rng()

    tasks = []
    for repeat in range(n_repeats):
        if mask is None:
            mask_train, mask_val = train_val_split(s.shape[0], train_ratio, rng)
        else:
            mask_train, mask_val = second_order_mask(mask, train_ratio, rng)

        tasks.extend(
            [
                delayed(_evaluate_rank)(
                    rank, s, mask_train, mask_val, rng, bounds, similarity_measure
                )
                for rank in candidate_ranks
            ]
        )

    results_list = Parallel(n_jobs=-1)(tasks)
    results_df = pd.DataFrame(results_list, columns=["rank", "rmse"])
    results_df["repeat"] = np.repeat(range(n_repeats), len(candidate_ranks))

    return results_df


def run_cv_experiment(
    n=60,
    r_true=5,
    train_ratio=0.8,
    candidate_ranks=range(5, 15),
    seed=0,
    n_jobs=-1,  # -1 means use all available cores
    similarity_measure: str = "cosine",
):
    rng = np.random.default_rng(seed)
    w_true = rng.random((n, r_true))
    s_full = compute_similarity(w_true, w_true, similarity_measure)
    bounds = (s_full.min(), s_full.max())

    mask_train, mask_val = train_val_split(n, train_ratio, rng)

    tasks = [
        delayed(_evaluate_rank)(
            rank, s_full, mask_train, mask_val, seed, bounds, similarity_measure
        )
        for rank in candidate_ranks
    ]

    results_list = Parallel(n_jobs=n_jobs, verbose=10)(tasks)

    results = dict(results_list)
    return results
