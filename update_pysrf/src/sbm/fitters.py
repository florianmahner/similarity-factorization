"""SBM block-mean fitters used by the cluster-count cross-validation layer.

Implements three model families from plan_SBM_clustering.md §8:

  1. Block-mean weighted SBM (§8.1):
        S_hat_ij = B_{z_i, z_j}.
     Labels initialised by spectral clustering, refined by hard-EM style label
     updates that minimise training loss (matching MLE under MSE/Gaussian, and
     a hard-Bernoulli/Poisson cross-entropy approximation otherwise).

  2. Spectral-clustering block-mean baseline (§8.2):
        as above but labels are produced by k-means on the row-normalised
        eigenvectors of the regularised normalised Laplacian, without
        likelihood-driven label refinement.

  3. Degree-corrected weighted SBM (§8.3):
        S_hat_ij = theta_i * theta_j * B_{z_i, z_j},
     with theta blockwise-normalised. Fit by alternating updates of theta and
     B given fixed labels (labels initialised by spectral clustering).

The function ``fit_sbm`` is the user-facing entry point and conforms to the
signature stipulated in plan §13.3:

    fit_sbm(S, rank, train_mask, loss, degree_corrected, init, seed, **kwargs)

The returned object follows §13.2 ("fitter result").

Optimisation status notes (mandated by §8.1 of the plan): the block-mean
update is exact under MSE/Gaussian loss; the label update is a heuristic
single-node greedy descent, not exact global MLE. Convergence is by label
stability or max iteration count.
"""
from __future__ import annotations
import numpy as np
from typing import Optional

from .losses import _clip_prob

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _symmetrize_off_diag(M):
    """Make a (n, n) array symmetric by averaging M and M.T off the diagonal."""
    out = 0.5 * (M + M.T)
    return out


def _safe_train_matrix(S, train_mask, fill_value=0.0):
    """Return a copy of S with non-training entries replaced by ``fill_value``.

    Diagonal is set to ``fill_value`` too; the SBM fitter never uses diagonals.
    """
    S_train = np.where(train_mask, S, fill_value).astype(float)
    np.fill_diagonal(S_train, 0.0)
    return S_train


def _normalised_laplacian_embedding(W, n_components, tau=None):
    """Compute the top-``n_components`` eigenvectors of
    D_tau^{-1/2} W D_tau^{-1/2} (a.k.a. normalised adjacency), row-normalised.

    A regularisation term ``tau * I`` is added to the degree matrix to stabilise
    the embedding on sparse / under-observed similarity graphs.
    """
    n = W.shape[0]
    deg = np.asarray(W.sum(axis=1)).ravel()
    if tau is None:
        tau = float(np.mean(deg))
        tau = max(tau, 1e-6)
    d_inv_sqrt = 1.0 / np.sqrt(deg + tau)
    L = (d_inv_sqrt[:, None] * W) * d_inv_sqrt[None, :]
    L = _symmetrize_off_diag(L)
    try:
        vals, vecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(L + 1e-9 * np.eye(n))
    # take top |n_components|; vals are ascending.
    take = max(1, int(n_components))
    U = vecs[:, -take:]
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return U / norms


def _kmeans(X, k, seed=0, n_restarts=5, max_iter=200):
    """Lightweight k-means with k-means++ init, no external dependency.

    Returns ``(labels, inertia)``.
    """
    rng = np.random.default_rng(int(seed))
    n, d = X.shape
    if k <= 1:
        return np.zeros(n, dtype=int), float(np.sum(X ** 2))
    if k >= n:
        return np.arange(n) % k, 0.0

    best_labels = None
    best_inertia = np.inf
    for restart in range(n_restarts):
        # k-means++ init
        centers = np.empty((k, d))
        i0 = int(rng.integers(0, n))
        centers[0] = X[i0]
        dist2 = np.sum((X - centers[0]) ** 2, axis=1)
        for j in range(1, k):
            if dist2.sum() <= 0:
                idx = int(rng.integers(0, n))
            else:
                probs = dist2 / dist2.sum()
                idx = int(rng.choice(n, p=probs))
            centers[j] = X[idx]
            new_dist2 = np.sum((X - centers[j]) ** 2, axis=1)
            dist2 = np.minimum(dist2, new_dist2)
        # Lloyd iterations
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            d2 = np.sum(X ** 2, axis=1, keepdims=True) + np.sum(centers ** 2, axis=1)[None, :] - 2 * X @ centers.T
            new_labels = np.argmin(d2, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                in_c = labels == c
                if in_c.any():
                    centers[c] = X[in_c].mean(axis=0)
                else:
                    centers[c] = X[int(rng.integers(0, n))]
        inertia = float(np.sum(np.min(d2, axis=1)))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    return best_labels, best_inertia


# ----------------------------------------------------------------------
# Block mean / DCSBM mean estimators (given labels and training mask)
# ----------------------------------------------------------------------

def _block_mean_matrix(S, train_mask, labels, rank, shrinkage_value):
    """Return symmetric block-mean matrix B (rank x rank) and per-block counts.

    Empty training blocks fall back to ``shrinkage_value`` (typically the
    global training mean) — §3.2 / §8.1.
    """
    n = S.shape[0]
    one_hot = np.zeros((n, rank), dtype=float)
    one_hot[np.arange(n), labels] = 1.0
    M = (train_mask & ~np.eye(n, dtype=bool)).astype(float)
    counts = one_hot.T @ M @ one_hot
    sums = one_hot.T @ (M * S) @ one_hot
    with np.errstate(invalid="ignore", divide="ignore"):
        B = np.where(counts > 0, sums / np.maximum(counts, 1), shrinkage_value)
    B = 0.5 * (B + B.T)
    return B, counts


def _global_train_mean(S, train_mask, default=0.0):
    n = S.shape[0]
    off_diag = ~np.eye(n, dtype=bool)
    iu = np.triu_indices(n, k=1)
    m = (train_mask & off_diag)[iu] & np.isfinite(S[iu])
    if not m.any():
        return float(default)
    return float(np.mean(S[iu][m]))


# ----------------------------------------------------------------------
# Loss-driven label refinement (hard-EM, single-node greedy)
# ----------------------------------------------------------------------

def _label_assignment_cost(S, train_mask, labels, B, loss, theta=None, eps=1e-6):
    """For each node i and candidate label c, compute the total training loss
    contributed by node i if it were assigned to c (block means and theta held
    fixed).

    Returns an (n, rank) cost array. Lower is better.
    """
    n = S.shape[0]
    rank = B.shape[0]
    M = (train_mask & ~np.eye(n, dtype=bool)).astype(float)
    if theta is None:
        # mean structure: B[c, z_j]
        mean_for_c = B[:, labels]  # (rank, n)
    else:
        # mean structure: theta_i * theta_j * B[c, z_j]
        mean_for_c = (theta[None, :] * B[:, labels])  # (rank, n)
        # caller multiplies in theta_i later.

    # Vectorised loss evaluation. Each (i, c) pair sums over j != i.
    if loss == "gaussian_mse" or loss == "gaussian_nll":
        # contribution = sum_j M_ij * (S_ij - mean_for_c)^2
        if theta is None:
            # mean_for_c[c, j] does not depend on i. Compute per node:
            costs = np.empty((n, rank), dtype=float)
            for c in range(rank):
                resid = S - mean_for_c[c][None, :]
                costs[:, c] = np.sum(M * resid ** 2, axis=1)
            return costs
        else:
            costs = np.empty((n, rank), dtype=float)
            for c in range(rank):
                pred_no_theta_i = mean_for_c[c]
                # actual prediction = theta[i] * pred_no_theta_i[j]
                resid = S - theta[:, None] * pred_no_theta_i[None, :]
                costs[:, c] = np.sum(M * resid ** 2, axis=1)
            return costs

    if loss in ("bernoulli", "frac_bernoulli"):
        costs = np.empty((n, rank), dtype=float)
        for c in range(rank):
            if theta is None:
                pred = mean_for_c[c]
                p = _clip_prob(pred, eps)[None, :]  # (1, n)
            else:
                pred = theta[:, None] * mean_for_c[c][None, :]
                p = _clip_prob(pred, eps)
            if loss == "frac_bernoulli":
                s_use = _clip_prob(S, eps)
            else:
                s_use = S
            ll = -(s_use * np.log(p) + (1.0 - s_use) * np.log(1.0 - p))
            costs[:, c] = np.sum(M * ll, axis=1)
        return costs

    if loss == "poisson":
        costs = np.empty((n, rank), dtype=float)
        for c in range(rank):
            if theta is None:
                lam = np.maximum(mean_for_c[c], eps)[None, :]
            else:
                lam = np.maximum(theta[:, None] * mean_for_c[c][None, :], eps)
            per = lam - S * np.log(lam)
            costs[:, c] = np.sum(M * per, axis=1)
        return costs

    raise ValueError(f"unsupported loss for label refinement: {loss}")


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def fit_spectral_block_mean(S, rank, train_mask, loss="gaussian_mse",
                            seed=0, **kwargs):
    """Spectral-clustering block-mean baseline (§8.2). No label refinement."""
    return fit_sbm(S, rank=rank, train_mask=train_mask, loss=loss,
                   degree_corrected=False, init="spectral",
                   max_label_iter=0, seed=seed, **kwargs)


def fit_sbm(S, rank, train_mask, loss="gaussian_mse", degree_corrected=False,
            init="spectral", seed=0, max_label_iter=10, tol=0, n_jobs=1,
            **kwargs):
    """Fit a block-mean SBM / DCSBM and return predictions on all off-diagonal
    pairs.

    Parameters mirror plan §13.3. Extra optional kwargs are accepted but
    ignored for forward compatibility (e.g. an experiment may pass
    ``loss_extra``).

    Returns
    -------
    dict with keys (§13.2):
        S_hat        : (n, n) ndarray of predicted similarities (off-diag),
                       symmetric; diagonal set to 0.
        labels       : (n,) integer labels in [0, rank)
        block_params : (rank, rank) ndarray B (the symmetric block means)
        node_params  : (n,) ndarray theta or None
        train_loss   : float mean training loss after fitting
        fit_status   : str (``"ok"``, ``"empty_block"``, ``"diverged"``)
        n_iter       : int label refinement iterations actually performed
    """
    del kwargs  # absorbed for forward compatibility
    n = S.shape[0]
    rank = int(rank)
    rank = max(1, min(rank, n))
    rng = np.random.default_rng(int(seed))
    train_mask = train_mask & ~np.eye(n, dtype=bool)
    # Validate: any training data?
    S_train = _safe_train_matrix(S, train_mask, fill_value=0.0)

    # ---- label init ----
    if rank == 1:
        labels = np.zeros(n, dtype=int)
    else:
        if init == "spectral":
            try:
                U = _normalised_laplacian_embedding(np.abs(S_train),
                                                   n_components=rank)
                labels, _ = _kmeans(U, rank, seed=int(rng.integers(0, 2**31 - 1)))
            except Exception:
                labels = rng.integers(0, rank, size=n)
        elif init == "random":
            labels = rng.integers(0, rank, size=n)
        else:
            raise ValueError(f"unknown init {init!r}")
        # Ensure every cluster has at least one member to avoid all-empty rows.
        for c in range(rank):
            if not np.any(labels == c):
                labels[int(rng.integers(0, n))] = c

    g_mean = _global_train_mean(S, train_mask, default=0.0)
    theta = None
    if degree_corrected:
        # Initialise theta from training row sums.
        d = np.sum(np.where(train_mask, S, 0.0), axis=1)
        theta = _normalise_theta_per_block(d, labels, rank)

    # ---- iterative refinement ----
    B, _counts = _block_mean_matrix(S, train_mask, labels, rank, g_mean)
    fit_status = "ok"
    n_iter = 0
    for it in range(int(max_label_iter)):
        if rank == 1:
            break
        # label update
        costs = _label_assignment_cost(S, train_mask, labels, B, loss,
                                       theta=theta)
        new_labels = np.argmin(costs, axis=1).astype(int)
        # Avoid emptying clusters: if a cluster becomes empty, keep one node.
        for c in range(rank):
            if not np.any(new_labels == c):
                # Pick the node with smallest cost difference (cheapest move).
                gain = costs[:, c] - costs[np.arange(n), new_labels]
                idx = int(np.argmin(gain))
                new_labels[idx] = c
        if np.array_equal(new_labels, labels):
            n_iter = it + 1
            break
        labels = new_labels
        n_iter = it + 1
        # Re-estimate parameters.
        if degree_corrected:
            theta, B = _fit_dcsbm_alternating(S, train_mask, labels, rank,
                                              g_mean, theta, max_inner=5)
        else:
            B, _counts = _block_mean_matrix(S, train_mask, labels, rank, g_mean)

    if degree_corrected and theta is None:
        theta = _normalise_theta_per_block(
            np.sum(np.where(train_mask, S, 0.0), axis=1), labels, rank)

    # Final fit of parameters with the converged labels.
    if degree_corrected:
        theta, B = _fit_dcsbm_alternating(S, train_mask, labels, rank, g_mean,
                                          theta, max_inner=10)
    else:
        B, _counts = _block_mean_matrix(S, train_mask, labels, rank, g_mean)

    # Construct prediction matrix on ALL off-diagonal pairs (plan §13.2).
    S_hat = _predict_matrix(B, labels, theta=theta)

    # Training loss diagnostic (mean per-edge over training mask).
    train_loss = _evaluate_train_loss(S, S_hat, train_mask, loss)

    return {
        "S_hat": S_hat,
        "labels": labels,
        "block_params": B,
        "node_params": theta,
        "train_loss": float(train_loss),
        "fit_status": fit_status,
        "n_iter": int(n_iter),
    }


# ----------------------------------------------------------------------
# DCSBM
# ----------------------------------------------------------------------

def _normalise_theta_per_block(theta, labels, rank):
    out = np.array(theta, dtype=float, copy=True)
    out = np.maximum(out, 1e-12)
    for c in range(rank):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        m = out[idx].mean()
        if m > 0:
            out[idx] = out[idx] / m
    return out


def _fit_dcsbm_alternating(S, train_mask, labels, rank, fallback, theta_init,
                           max_inner=10, tol=1e-6):
    """Alternating LS for the Gaussian DCSBM:
        S_ij ~= theta_i * theta_j * B[z_i, z_j].

    Used for *all* DCSBM losses as a convenient mean estimator; the loss
    matters only for label assignment via _label_assignment_cost.
    """
    n = S.shape[0]
    theta = np.maximum(theta_init.astype(float).copy(), 1e-12)
    one_hot = np.zeros((n, rank), dtype=float)
    one_hot[np.arange(n), labels] = 1.0
    M = (train_mask & ~np.eye(n, dtype=bool)).astype(float)

    B = np.full((rank, rank), fallback, dtype=float)
    for _ in range(max_inner):
        # Update B given theta. For each (a, b):
        #   B_ab = sum M_ij theta_i theta_j S_ij / sum M_ij (theta_i theta_j)^2 ?
        # Actually for minimising (S_ij - theta_i theta_j B_ab)^2 over (a,b)
        # pairs: B_ab = sum_{(i,j) in T, z_i=a, z_j=b} S_ij theta_i theta_j /
        #             sum (theta_i theta_j)^2.
        tt = np.outer(theta, theta)
        W = M * tt
        num = one_hot.T @ (M * S * tt) @ one_hot
        den = one_hot.T @ (W * tt) @ one_hot
        with np.errstate(invalid="ignore", divide="ignore"):
            B_new = np.where(den > 0, num / np.maximum(den, 1e-30), fallback)
        B_new = 0.5 * (B_new + B_new.T)

        # Update theta given B. For each i:
        #   theta_i = sum_j M_ij theta_j B[z_i, z_j] S_ij /
        #             sum_j M_ij (theta_j B[z_i, z_j])^2.
        B_lab = B_new[labels][:, labels]  # (n, n) of B[z_i, z_j]
        coef_num = np.sum(M * S * theta[None, :] * B_lab, axis=1)
        coef_den = np.sum(M * (theta[None, :] * B_lab) ** 2, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            theta_new = np.where(coef_den > 0,
                                 coef_num / np.maximum(coef_den, 1e-30),
                                 theta)
        theta_new = np.maximum(theta_new, 1e-12)
        # Re-normalise per block.
        theta_new = _normalise_theta_per_block(theta_new, labels, rank)

        d_theta = np.max(np.abs(theta_new - theta))
        d_B = np.max(np.abs(B_new - B))
        theta = theta_new
        B = B_new
        if d_theta < tol and d_B < tol:
            break
    return theta, B


def _predict_matrix(B, labels, theta=None):
    n = labels.size
    pred = B[labels][:, labels]
    if theta is not None:
        pred = (theta[:, None] * pred) * theta[None, :]
    pred = 0.5 * (pred + pred.T)
    np.fill_diagonal(pred, 0.0)
    return pred


def _evaluate_train_loss(S, S_hat, train_mask, loss):
    """Mean per-edge training loss, for diagnostics only."""
    from .losses import score_sbm_predictions
    res = score_sbm_predictions(S, S_hat, train_mask, loss=loss)
    return res["loss_mean"]
