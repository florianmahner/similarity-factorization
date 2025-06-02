# %%
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from numpy.random import Generator
from joblib import Parallel, delayed

# TODO rewrite loss only based on nan mask entries and then print this explained variance or so / also determine convergence.


def add_noise_with_snr(
    x: np.ndarray, snr: float, rng: Generator | int | None = None
) -> np.ndarray:
    if rng is None:
        rng_gen = np.random.default_rng()
    elif isinstance(rng, np.random.Generator):
        rng_gen = rng
    else:
        rng_gen = np.random.default_rng(rng)
    # Ensure snr is within [0, 1]
    snr = np.clip(snr + 1e-12, 1e-12, 1.0)
    # Compute the standard deviation of the signal X
    signal_std = np.std(x, ddof=1)
    # Generate noise with the same standard deviation as the signal
    noise = rng_gen.standard_normal(size=x.shape) * signal_std
    # Combine signal and noise using square-root mixing
    return np.sqrt(snr) * x + np.sqrt(1 - snr) * noise


def _quartic_root(a, b, c, d):
    """
    solve min_{x >= 0} g(x) = a/4 x^4 + b/3 x^3 + c/2 x^2 + d x (a == 4), e.g a four-order polynomial.
    See eq. (11)'
    """
    p = (3.0 * a * c - b**2) / (3.0 * a**2)
    q = (9.0 * a * b * c - 27.0 * a**2 * d - 2.0 * b**3) / (27.0 * a**3)

    # closed‑form minimiser of the 1‑D quartic upper‑bound
    if c > b**2 / (3.0 * a):  # three real roots, pick the one in (0, inf)
        delta = math.sqrt(q**2 / 4.0 + p**3 / 27.0)
        x_new = np.cbrt(q / 2.0 - delta) + np.cbrt(q / 2.0 + delta)
    else:  # one real root
        stmp = b**3 / (27.0 * a**3) - d / a
        x_new = np.cbrt(stmp)

    if x_new < 0.0:  # non neg constraint
        x_new = 0.0

    return x_new


def update_w_bsum(
    m: np.ndarray,
    x0: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
) -> np.ndarray:
    """Block successive upper bound minimization (Shi et al., 2016)."""
    x = x0.copy()
    n, r = x.shape
    xtx = x.T @ x
    diag = np.einsum("ij,ij->i", x, x)
    a = 4.0

    for it in range(max_iter):
        max_delta = 0.0
        for i in range(n):
            for j in range(r):
                old = x[i, j]
                b = 12 * old
                c = 4 * ((diag[i] - m[i, i]) + xtx[j, j] + old * old)
                d = 4 * (x[i] @ xtx[:, j]) - 4 * (m[i] @ x[:, j])

                new = _quartic_root(a, b, c, d)
                delta = new - old
                if abs(delta) > max_delta:
                    max_delta = abs(delta)

                # steps 7 - 13 in TABLE 1
                diag[i] += new * new - old * old
                update_row = delta * x[i]
                xtx[j, :] += update_row
                xtx[:, j] += update_row
                xtx[j, j] += delta * delta
                x[i, j] = new

        if verbose:
            evar = 1 - (np.linalg.norm(m - x @ x.T, "fro") / np.linalg.norm(m, "fro"))
            print(f"it {it:3d}  evar {evar:.6f}", end="\r")

        if max_delta < tol and tol > 0.0:
            break

    return x


def update_v(m, s, w, lam, rho, min_val, max_val):
    ww = w @ w.T  # current reconstruction
    v = ww - lam / rho  # default (missing-entry) formula

    obs = m.astype(bool)  # boolean mask of observed entries
    v[obs] = (s[obs] + rho * ww[obs] - lam[obs]) / (1.0 + rho)

    if min_val is not None:
        v[v < min_val] = min_val
    if max_val is not None:
        v[v > max_val] = max_val

    return v


def update_lambda(lam, v, w, rho):
    return lam + rho * (v - w @ w.T)


def admm_symnmf_masked(
    s, mask, rank, rho=1.0, max_outer=15, w_inner=40, tol=1e-4, seed=None, bounds=None
):
    rng = np.random.default_rng(seed)
    n = s.shape[0]
    w = rng.random((n, rank)) + 1e-3  # TODO Check tgis
    lam = np.zeros_like(s)
    min_val, max_val = bounds if bounds is not None else (None, None)

    for _ in range(max_outer):
        v = update_v(mask, s, w, lam, rho, min_val, max_val)
        T = v + lam / rho
        w = update_w_bsum(T, w, max_iter=w_inner, tol=tol)
        lam = update_lambda(lam, v, w, rho)

        if np.linalg.norm(v - w @ w.T, "fro") < 1e-6:
            break
    return w


def random_mask(n, keep_ratio, rng):
    m = rng.random((n, n)) < keep_ratio
    m = np.triu(m) + np.triu(m, 1).T
    return m.astype(float)


def train_val_split(n, keep_ratio, train_ratio, rng):
    # Sample upper triangle entries (without diagonal)
    triu_idx = np.triu_indices(n, k=1)
    total_edges = len(triu_idx[0])

    # Randomly keep 'keep_ratio' of possible edges
    num_keep = int(keep_ratio * total_edges)
    perm = rng.permutation(total_edges)
    keep_idx = perm[:num_keep]

    # Split into train/val
    num_train = int(train_ratio * num_keep)
    train_idx = keep_idx[:num_train]
    val_idx = keep_idx[num_train:]

    # Create train/val masks
    mask_train = np.zeros((n, n), dtype=float)
    mask_val = np.zeros((n, n), dtype=float)

    i_train, j_train = triu_idx[0][train_idx], triu_idx[1][train_idx]
    i_val, j_val = triu_idx[0][val_idx], triu_idx[1][val_idx]

    mask_train[i_train, j_train] = 1.0
    mask_train[j_train, i_train] = 1.0  # enforce symmetry
    mask_val[i_val, j_val] = 1.0
    mask_val[j_val, i_val] = 1.0

    return mask_train, mask_val


def _evaluate_rank(rank, s_full, mask_train, mask_val, seed, bounds):
    """Helper function to evaluate a single rank."""
    w_est = admm_symnmf_masked(
        s_full,
        mask_train,
        rank,
        rho=10.0,
        max_outer=20,
        w_inner=60,
        seed=seed,
        tol=1e-6,
        bounds=bounds,
    )

    # NOTE It is important to use the same kernel here!!!
    # s_pred = w_est @ w_est.T
    s_pred = cosine_similarity(w_est)

    val_rmse = np.linalg.norm(mask_val * (s_full - s_pred), "fro") / np.sqrt(
        mask_val.sum()
    )
    print(f"rank {rank:2d}  validation RMSE {val_rmse:.4f}")
    return rank, val_rmse


def run_cv_experiment(
    n=60,
    r_true=5,
    keep_ratio=0.8,
    train_ratio=0.8,
    candidate_ranks=range(5, 15),
    seed=0,
    n_jobs=-1,  # -1 means use all available cores
):
    rng = np.random.default_rng(seed)
    w_true = rng.random((n, r_true))

    # w_true = add_noise_with_snr(w_true, snr=0.3)
    s_full = cosine_similarity(w_true)

    bounds = (s_full.min(), s_full.max())

    # s_full = w_true @ w_true.T

    mask_train, mask_val = train_val_split(n, keep_ratio, train_ratio, rng)

    tasks = [
        delayed(_evaluate_rank)(rank, s_full, mask_train, mask_val, seed, bounds)
        for rank in candidate_ranks
    ]

    results_list = Parallel(n_jobs=n_jobs, verbose=10)(tasks)

    results = dict(results_list)
    return results


# %%


# run the experiment
results = run_cv_experiment(
    n=200,
    r_true=10,
    keep_ratio=0.8,
    train_ratio=0.8,
    n_jobs=50,
    candidate_ranks=range(5, 20),
)
plt.plot(results.keys(), results.values())

print("\nValidation error by rank:", results)

# look for the best (minimum) validation error
best_rank = min(results, key=results.get)
print(f"\nBest rank selected by CV: {best_rank} (true rank = 10)")
