"""
Alternative representation methods with NaN-aware fitting for Experiment B.

Each fitter has signature:
    fit(S, rank, holdout_mask, seed=0) -> S_hat (n x n)

`holdout_mask` is True where entries should be treated as missing during fitting.
The same `fit_score` pattern as ADMM is used: NaN at holdout, fit, reconstruct,
score MSE on (V, M) sub-masks.

Methods:
  fit_ppca_em      : Probabilistic PCA via EM (Tipping & Bishop 1999)
  fit_softimpute   : Soft-Impute SVD via iterative imputation (Mazumder 2010)
  fit_wnmf_hals    : Weighted NMF via HALS with masked entries (custom)
  fit_robustpca    : Robust PCA via TensorLy with mask
  fit_symmnmf_admm : The existing ADMM-based SymmNMF (delegates to src.models.admm)

Each returns reconstruction S_hat. A unified `score_VM(S, S_hat, V_mask, M_mask)`
in the experiment script computes the held-out MSEs.
"""
import numpy as np
from numpy.linalg import svd


def _impute_mean(S, mask):
    """Mean-impute masked entries with the global mean of observed."""
    obs = ~mask
    mu = float(np.mean(S[obs])) if obs.any() else 0.0
    Sx = S.copy()
    Sx[mask] = mu
    return Sx, mu


# ============================================================
# 1. Probabilistic PCA (EM with missing values)
# ============================================================
def fit_ppca_em(S, rank, holdout_mask, seed=0, n_iter=30, tol=1e-5, **_):
    """PPCA via EM on Gram form with missing entries.

    Model: S = Lambda Lambda^T + sigma^2 I, with rank(Lambda) = r.
    Closed-form solution given fully-observed S: eigendecompose S = U diag(lam) U^T,
    set sigma^2 = mean(lam[r:]) and Lambda = U[:, :r] * sqrt(lam[:r] - sigma^2).
    EM with missing: at each step, impute held-out entries with the current
    Lambda Lambda^T + sigma^2 I, then redo eigendecomposition.

    Stable form: uses eigh (S is symmetric), bounds sigma^2 to avoid blow-ups.
    """
    n = S.shape[0]
    Sx, _ = _impute_mean(S, holdout_mask)
    Sx = (Sx + Sx.T) / 2

    # Initial fit via eigendecomposition
    try:
        eigvals, eigvecs = np.linalg.eigh(Sx)
    except np.linalg.LinAlgError:
        return Sx  # fall back to mean-imputed matrix
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]
    sigma2 = float(np.mean(np.maximum(eigvals[rank:], 0.0))) if rank < n else 1e-6
    sigma2 = max(min(sigma2, float(max(eigvals[rank - 1] - 1e-6, 1e-6))), 1e-6)
    Lam = eigvecs[:, :rank] * np.sqrt(np.maximum(eigvals[:rank] - sigma2, 0.0))[None, :]

    Lam_norm_init = float(np.linalg.norm(Lam)) + 1e-12
    for it in range(n_iter):
        # E-step: impute missing entries using current model
        S_hat = Lam @ Lam.T + sigma2 * np.eye(n)
        # Sanity check — bail out if model has blown up
        if not np.all(np.isfinite(S_hat)):
            break
        Sx_new = S.copy()
        Sx_new[holdout_mask] = S_hat[holdout_mask]
        Sx_new = (Sx_new + Sx_new.T) / 2

        # M-step: eigendecompose imputed matrix, update Lambda, sigma^2
        try:
            eigvals, eigvecs = np.linalg.eigh(Sx_new)
        except np.linalg.LinAlgError:
            break
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]
        sigma2_new = float(np.mean(np.maximum(eigvals[rank:], 0.0))) if rank < n else 1e-6
        sigma2_new = max(min(sigma2_new, float(max(eigvals[rank - 1] - 1e-6, 1e-6))), 1e-6)
        Lam_new = eigvecs[:, :rank] * np.sqrt(np.maximum(eigvals[:rank] - sigma2_new, 0.0))[None, :]

        delta = float(np.linalg.norm(Lam_new - Lam) / Lam_norm_init)
        Lam = Lam_new
        sigma2 = sigma2_new
        if delta < tol:
            break

    return Lam @ Lam.T + sigma2 * np.eye(n)


# ============================================================
# 2. Soft-Impute SVD (Mazumder 2010)
# ============================================================
def fit_softimpute(S, rank, holdout_mask, seed=0, n_iter=80, tol=1e-5, **_):
    """Soft-Impute SVD: iteratively (a) fill masked entries with low-rank fit,
    (b) take SVD of filled matrix, (c) hard-truncate to top-`rank` components.

    Uses HARD truncation (rank capped at `rank`) — the parameter is `rank`,
    not the soft-threshold lambda. Returns S_hat = U[:,:r] * diag(sv[:r]) * V^T[:r,:].
    """
    n = S.shape[0]
    Sx, _ = _impute_mean(S, holdout_mask)
    S_hat = Sx.copy()
    for it in range(n_iter):
        # SVD of current filled matrix (force symmetric; use eigh)
        Ssym = (S_hat + S_hat.T) / 2
        eigvals, eigvecs = np.linalg.eigh(Ssym)
        # top `rank` by absolute eigenvalue
        idx = np.argsort(np.abs(eigvals))[::-1][:rank]
        lam_top = eigvals[idx]
        U_top = eigvecs[:, idx]
        S_new = U_top @ np.diag(lam_top) @ U_top.T

        # Refill: known entries from S, unknown from S_new
        S_filled = S.copy()
        S_filled[holdout_mask] = S_new[holdout_mask]

        delta = float(np.linalg.norm(S_filled - S_hat) / max(np.linalg.norm(S_hat), 1e-12))
        S_hat = S_filled
        if delta < tol:
            break

    # Final reconstruction = rank-truncated form
    Ssym = (S_hat + S_hat.T) / 2
    eigvals, eigvecs = np.linalg.eigh(Ssym)
    idx = np.argsort(np.abs(eigvals))[::-1][:rank]
    return eigvecs[:, idx] @ np.diag(eigvals[idx]) @ eigvecs[:, idx].T


# ============================================================
# 3. Weighted (masked) NMF via HALS
# ============================================================
def fit_wnmf_hals(S, rank, holdout_mask, seed=0, n_iter=200, tol=1e-5, **_):
    """Weighted symmetric NMF via mask-aware HALS: minimize
        ||M (S - W W^T)||_F^2,  W >= 0
    where M = 1 - holdout_mask is the observed-entry indicator.

    Mask-aware rank-1 HALS update (Cichocki & Phan 2009 with weights):
        R^(k) = M * (S - sum_{j!=k} w_j w_j^T)
        w_k_new[i] = max(0, sum_j M_ij R^(k)_ij w_k_j / sum_j M_ij w_k_j^2)

    Initialization: NNDSVD-like — top-r positive eigenvectors of mean-imputed S
    (clipped to non-negative). Includes a small random fallback if a column is
    all-zero.
    """
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    obs = (~holdout_mask).astype(float)
    np.fill_diagonal(obs, 0.0)  # avoid biasing fit on diagonal

    # NNDSVD-style init from mean-imputed eigendecomposition
    Sx, _ = _impute_mean(S, holdout_mask)
    Sx = (Sx + Sx.T) / 2
    try:
        eigvals, eigvecs = np.linalg.eigh(Sx)
    except np.linalg.LinAlgError:
        eigvecs = np.eye(n); eigvals = np.ones(n)
    idx = np.argsort(eigvals)[::-1][:rank]
    W = np.maximum(eigvecs[:, idx] * np.sqrt(np.maximum(eigvals[idx], 0.0))[None, :], 0.0)
    # Random fallback for collapsed columns
    for k in range(rank):
        if np.all(W[:, k] < 1e-10):
            W[:, k] = np.abs(rng.standard_normal(n)) * 0.1

    prev_err = np.inf
    for it in range(n_iter):
        for k in range(rank):
            wk = W[:, k].copy()
            WW_others = W @ W.T - np.outer(wk, wk)
            R_k = obs * (S - WW_others)
            numer = R_k @ wk
            denom = obs @ (wk ** 2)
            wk_new = np.maximum(numer / np.maximum(denom, 1e-12), 0.0)
            # Avoid total collapse — keep last-good if all-zero
            if float(np.linalg.norm(wk_new)) > 1e-10:
                W[:, k] = wk_new
        if it % 5 == 0:
            err = float(np.linalg.norm(obs * (S - W @ W.T)))
            if abs(prev_err - err) / max(prev_err, 1e-12) < tol:
                break
            prev_err = err
    return W @ W.T


# ============================================================
# 4. Robust PCA with mask (TensorLy)
# ============================================================
def fit_robustpca(S, rank, holdout_mask, seed=0, n_iter=50, **_):
    """Robust PCA via TensorLy: S = L + S_sparse + noise, with mask of unobserved.

    Returns L (low-rank part). TensorLy's robust_pca uses ADMM internally and
    accepts a binary mask. We pass `mask=~holdout_mask` (1 at observed).

    Note: TensorLy's API may not accept mask directly; we use mean-impute then
    call robust_pca, then re-mean-subtract in case.
    """
    try:
        import tensorly as tl
        from tensorly.decomposition import robust_pca
    except ImportError:
        # Fallback: PCA-only
        return fit_softimpute(S, rank, holdout_mask, seed=seed)
    n = S.shape[0]
    Sx, _ = _impute_mean(S, holdout_mask)
    Sx_t = tl.tensor(Sx)
    try:
        L, Sp = robust_pca(Sx_t, learning_rate=1.0, mask=tl.tensor((~holdout_mask).astype(float)),
                           n_iter_max=n_iter)
        L = np.asarray(L)
    except TypeError:
        # Older TensorLy versions don't accept mask kwarg; fall back to plain RPCA
        try:
            L, Sp = robust_pca(Sx_t, learning_rate=1.0, n_iter_max=n_iter)
            L = np.asarray(L)
        except Exception:
            return fit_softimpute(S, rank, holdout_mask, seed=seed)
    # Truncate L to rank `rank` for fair comparison
    L_sym = (L + L.T) / 2
    eigvals, eigvecs = np.linalg.eigh(L_sym)
    idx = np.argsort(np.abs(eigvals))[::-1][:rank]
    return eigvecs[:, idx] @ np.diag(eigvals[idx]) @ eigvecs[:, idx].T


# ============================================================
# 5. SymmNMF via existing ADMM (the current Recipe K backend)
# ============================================================
def fit_symmnmf_admm(S, rank, holdout_mask, seed=0, **_):
    """Wrap existing ADMM SymmNMF with NaN-at-holdout."""
    from symmnmf.models.admm import ADMM
    Sx = S.copy()
    Sx[holdout_mask] = np.nan
    bounds = (float(np.nanmin(S)), float(np.nanmax(S)))
    est = ADMM(rank=rank, missing_values=np.nan, bounds=bounds, random_state=seed)
    try:
        est.fit(Sx)
        return est.reconstruct()
    except Exception:
        return np.full_like(S, np.nan)


# ============================================================
# Registry
# ============================================================

# ============================================================
# 6. Laplacian Eigenmaps with heat-kernel reconstruction (v3)
# ============================================================
def fit_laplacian_eigenmaps_heat(S, rank, holdout_mask, t_kernel, t_heat,
                                   seed=0, **_):
    """Laplacian Eigenmaps with heat-kernel reconstruction (v3 design).

    Two auxiliary hyperparameters declared by the generator (see audit §5.2.1):

      t_kernel : kernel bandwidth used by the generator to build
                 S = exp(-D^2 / t_kernel). Not used inside the fit — the
                 fitter receives S directly. Carried through for traceability
                 and so the fitter signature matches the generator's
                 declared aux dict 1-for-1.

      t_heat   : heat-kernel time in the V-objective weight
                 exp(-lambda_k * t_heat). Independent of t_kernel.
                 Normalised Laplacian eigenvalues lambda_k are bounded in
                 [0, 2], so t_heat is scale-free; default 1.0 gives
                 weights in roughly [exp(-2), 1] = [0.14, 1].

    Procedure (round-3 math §2 form, with t_kernel/t_heat decoupled):
      1. Mean-impute holdout entries to form the working matrix Sx.
      2. Compute degree vector from Sx (degrees of the imputed graph).
      3. Form symmetric normalized adjacency K = D^{-1/2} Sx D^{-1/2},
         eigendecompose. K's eigenvalues are 1 - lambda_L of the
         normalized Laplacian L_sym = I - K.
      4. Take rank-many bottom Laplacian eigenpairs INCLUDING the trivial
         lambda_L = 0 constant mode (which carries the mean component of
         S via weight exp(0) = 1). Rank counts non-trivial modes added on
         top of the constant.
      5. Heat-kernel reconstruction in normalized space:
            hat_S_norm = sum_{k=0}^r exp(-lambda_k * t_heat) phi_k phi_k^T.
      6. Un-normalize back to the original similarity scale:
            hat_S = D^{1/2} hat_S_norm D^{1/2}.

    Returns hat_S of shape (n, n). NaN-aware via mean-imputation
    (option (a) per audit §5.2.1; observed-edge sparse Laplacian
    (option (b)) is a future refinement).
    """
    n = S.shape[0]
    obs = ~holdout_mask
    if obs.sum() > 0:
        mean_obs = float(np.mean(S[obs]))
    else:
        mean_obs = 0.0
    Sx = S.copy()
    Sx[holdout_mask] = mean_obs
    Sx = (Sx + Sx.T) / 2

    degrees = Sx.sum(axis=1)
    degrees = np.maximum(degrees, 1e-12)
    D_sqrt = np.sqrt(degrees)
    Dinv_sqrt = 1.0 / D_sqrt

    K = (Dinv_sqrt[:, None] * Sx) * Dinv_sqrt[None, :]
    K = (K + K.T) / 2

    try:
        eigvals_K, eigvecs_K = np.linalg.eigh(K)
    except np.linalg.LinAlgError:
        return np.zeros_like(S)

    # K's largest eigvals = L_sym's smallest eigvals; sort descending.
    order = np.argsort(eigvals_K)[::-1]
    eigvals_K = eigvals_K[order]
    eigvecs_K = eigvecs_K[:, order]

    lambdas_L = np.maximum(1.0 - eigvals_K, 0.0)

    if rank >= n - 1:
        rank = n - 1
    # Include the trivial constant mode (lambda_L = 0, weight = exp(0) = 1)
    # as a fixed baseline so the reconstruction captures the mean component
    # of S. `rank` then counts NON-TRIVIAL Laplacian modes — for a d-manifold
    # the natural Laplace-Beltrami answer is rank = d.
    phis = eigvecs_K[:, 0:rank + 1]
    lams = lambdas_L[0:rank + 1]

    weights = np.exp(-lams * float(t_heat))
    S_hat_norm = phis @ np.diag(weights) @ phis.T
    S_hat = (D_sqrt[:, None] * S_hat_norm) * D_sqrt[None, :]
    return S_hat


# ============================================================
# 7. Laplacian Eigenmaps with embedding-distance V-objective (option C, v3)
# ============================================================
def fit_laplacian_eigenmaps_embed(S, rank, holdout_mask, t_kernel, seed=0, **_):
    """Laplacian Eigenmaps — return predicted squared distances $\\hat D^2$.

    Returns the n x n matrix $\\hat D^2_{ij} = \\| \\phi^{(r)}_i - \\phi^{(r)}_j \\|^2$
    where $\\phi^{(r)} = (\\phi_1, ..., \\phi_r)$ are the bottom-r non-trivial
    Laplacian eigenvectors of L_sym computed from the mean-imputed S.

    No heat-kernel weighting (option C in audit §5.2.1). Pairs with the
    embedding-distance V-objective in the v3 LE-embed harness. The
    auxiliary t_kernel is carried for traceability but not used inside
    this fitter (it is needed at scoring time to convert S -> D^2 if
    needed; not done here).
    """
    n = S.shape[0]
    obs = ~holdout_mask
    mean_obs = float(np.mean(S[obs])) if obs.sum() > 0 else 0.0
    Sx = S.copy()
    Sx[holdout_mask] = mean_obs
    Sx = (Sx + Sx.T) / 2

    degrees = Sx.sum(axis=1)
    degrees = np.maximum(degrees, 1e-12)
    Dinv_sqrt = 1.0 / np.sqrt(degrees)
    K = (Dinv_sqrt[:, None] * Sx) * Dinv_sqrt[None, :]
    K = (K + K.T) / 2
    try:
        eigvals_K, eigvecs_K = np.linalg.eigh(K)
    except np.linalg.LinAlgError:
        return np.zeros_like(S)

    order = np.argsort(eigvals_K)[::-1]
    eigvecs_K = eigvecs_K[:, order]

    # Skip the trivial constant mode (k=0); take next `rank` non-trivial modes
    if rank < 1:
        return np.zeros_like(S)
    if rank >= n - 1:
        rank = n - 1
    phi = eigvecs_K[:, 1:rank + 1]                   # n x r

    # Pairwise squared distances in the rank-r embedding
    sq_norms = (phi ** 2).sum(axis=1)
    D2_embed = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (phi @ phi.T)
    D2_embed = np.maximum(D2_embed, 0.0)
    return D2_embed


REPR_FITTERS = {
    "symmnmf":   fit_symmnmf_admm,
    "ppca":      fit_ppca_em,
    "softimp":   fit_softimpute,
    "wnmf":      fit_wnmf_hals,
    "rpca":      fit_robustpca,
    "le_heat":   fit_laplacian_eigenmaps_heat,
    "le_embed":  fit_laplacian_eigenmaps_embed,
}


def score_repr_VM(S, S_hat, V_mask, M_mask):
    """MSE on V and M masks. Both masks should exclude already-NaN entries."""
    if not np.all(np.isfinite(S_hat)):
        return np.nan, np.nan
    vV = V_mask & np.isfinite(S)
    vM = M_mask & np.isfinite(S)
    mse_V = float(np.mean((S[vV] - S_hat[vV]) ** 2)) if vV.any() else np.nan
    mse_M = float(np.mean((S[vM] - S_hat[vM]) ** 2)) if vM.any() else np.nan
    return mse_V, mse_M
