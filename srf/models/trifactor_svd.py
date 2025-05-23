import numpy as np
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.linear_model import Lasso
from scipy.optimize import nnls
from scipy.linalg import norm


# ------------------------------------------------------------------
# helper functions
# ------------------------------------------------------------------
def issymmetric(s):
    return s.shape[0] == s.shape[1] and np.allclose(s, s.T)


def symmetric_svd(s):
    assert issymmetric(s), "matrix must be symmetric"
    u, sigma, _ = np.linalg.svd(s)
    return u, sigma


def truncated_symmetric_svd(s, rank):
    u, sigma = symmetric_svd(s)
    return u[:, :rank], sigma[:rank], sigma[:rank] / sigma.sum()


def make_trifactor_simulation(n, r, sparsity, random_state=42, snr=0.0):
    rng = np.random.RandomState(random_state)
    w = 0.5 * rng.rand(n, r)
    a = make_sparse_spd_matrix(n_dim=r, alpha=0.90, random_state=random_state)
    s = w @ a @ w.T
    noise = rng.standard_normal((n, n))
    noise = (noise + noise.T) / 2
    s = snr * s + (1 - snr) * noise
    return s, w, a


def semi_nmf_cd(
    u, alpha=1e-2, max_iter=500, tol=1e-4, random_state=None, verbose=False
):
    """
    Solve  min_{w≥0,q} 0.5‖u - w qᵀ‖_F² + alpha‖q‖₁
    by alternating NNLS (rows of w) and Lasso (columns of q).
    u : (n,k) orthonormal left singular vectors
    """
    u = np.asarray(u)
    n, k = u.shape
    rng = np.random.default_rng(random_state)

    w = np.maximum(rng.random((n, k)), 1e-8)  # avoid zeros
    q = rng.standard_normal((k, k))

    obj_path = []
    prev = np.inf

    for it in range(max_iter):
        for i in range(n):
            w[i, :], _ = nnls(q, u[i, :])

        # -------- Q-step : k independent Lasso problems ------
        for j in range(k):
            lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
            lasso.fit(w, u[:, j])
            q[j, :] = lasso.coef_

        # -------- objective & stopping -----------------------
        resid = u - w @ q.T
        obj = 0.5 * norm(resid, "fro") ** 2 + alpha * np.abs(q).sum()
        obj_path.append(obj)

        if verbose and it % 50 == 0:
            print(f"iter {it:4d}  obj = {obj:.4f}", end="\r")

    return w, q, obj_path
