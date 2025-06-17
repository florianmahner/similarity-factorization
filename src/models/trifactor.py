from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from srf.models.nnls_block import nnlsm_blockpivot
from srf.models.base import BaseNMF
from scipy.stats import pearsonr, spearmanr, kendalltau

Array = np.ndarray


def rsa_score(
    rsm_pred: np.ndarray, rsm_behav: np.ndarray, method: str = "spearman"
) -> float:
    """
    Return correlation between the upper-triangle entries of two
    representational similarity matrices (square, symmetric).
    """
    iu = np.triu_indices_from(rsm_pred, k=1)
    x, y = rsm_pred[iu], rsm_behav[iu]

    if method == "pearson":
        return pearsonr(x, y)[0]
    if method == "spearman":
        return spearmanr(x, y, nan_policy="omit")[0]
    if method == "kendall":
        return kendalltau(x, y, variant="a").correlation
    raise ValueError("method must be 'pearson', 'spearman', or 'kendall'")


def update_a(s: Array, w: Array, h: Array, eps: float = 1e-12) -> Array:
    wtw = w.T @ w + eps * np.eye(w.shape[1])  # (r, r)
    hth = h.T @ h + eps * np.eye(h.shape[1])  # (r, r)
    wts = w.T @ s @ h  # (r, r)

    x = np.linalg.solve(wtw, wts)
    a = x @ np.linalg.inv(hth)
    return a


def update_a_cholesky(s: Array, w: Array, h: Array, eps: float = 1e-12) -> Array:
    # 1) form the small r×r Gram‐matrices
    wtw = w.T @ w + eps * np.eye(w.shape[1])
    hth = h.T @ h + eps * np.eye(h.shape[1])
    m = w.T @ s @ h

    # 2) Cholesky‐factor both
    c_w, lower_w = cho_factor(wtw, overwrite_a=False, check_finite=False)
    c_h, lower_h = cho_factor(hth, overwrite_a=False, check_finite=False)

    # 3) solve (WᵀW)·X = M  →  X = (WᵀW)^{-1} M
    x = cho_solve((c_w, lower_w), m, overwrite_b=False, check_finite=False)

    # 4) solve (HᵀH)·Aᵀ = Xᵀ  →  Aᵀ = (HᵀH)^{-1} Xᵀ  ⇒  A = (cho_solve on transposed)
    a = cho_solve((c_h, lower_h), x.T, overwrite_b=False, check_finite=False).T

    return a


def update_a_near_identity(
    S: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    mu: float = 1.0,  # “stay-close” weight
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Exact solution of  (WᵀW) A (HᵀH) + μA = Wᵀ S H + μI .
    Setting μ→0 reproduces the free update; large μ pins A≈I.
    """
    r = W.shape[1]

    # Gram matrices -----------------------------------------------------------
    K = W.T @ W + eps * np.eye(r)
    L = H.T @ H + eps * np.eye(r)
    M = W.T @ S @ H

    # Eigendecompositions -----------------------------------------------------
    d_k, U = np.linalg.eigh(K)  # K = U diag(d_k) Uᵀ
    d_l, V = np.linalg.eigh(L)  # L = V diag(d_l) Vᵀ

    # Transform RHS -----------------------------------------------------------
    Mt = U.T @ M @ V  # Uᵀ M V
    UtV = U.T @ V  #   Uᵀ I V  (dense!)

    # Element-wise solve ------------------------------------------------------
    denom = d_k[:, None] * d_l[None, :] + mu  # r×r
    Y = (Mt + mu * UtV) / denom  # r×r

    # Back-transform ----------------------------------------------------------
    A = U @ Y @ V.T
    return A


def update_a_diag(
    S: np.ndarray, W: np.ndarray, H: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """
    Exact least-squares update of **A = diag(d)** solving
        (WᵀW)·diag(d)·(HᵀH) = Wᵀ S H
    with small Tikhonov eps for numerical stability.
    """
    r = W.shape[1]
    K = W.T @ W + eps * np.eye(r)  # SPD
    L = H.T @ H + eps * np.eye(r)  # SPD
    M = W.T @ S @ H  # off-line once

    G = K * L.T  # Hadamard  (r×r)
    rhs = np.diag(M)  # (r,)

    # Solve (G) d = rhs   (SPD, size r×r)
    c, lower = cho_factor(G, check_finite=False)
    d = cho_solve((c, lower), rhs, check_finite=False)

    return np.diag(d)


def update_a_diag_ridge(
    S: np.ndarray, W: np.ndarray, H: np.ndarray, lam: float = 1e-2, eps: float = 1e-12
) -> np.ndarray:
    """
    Ridge-regularised variant solving
        (WᵀW) diag(d) (HᵀH) + λ diag(d) = Wᵀ S H  .
    """
    r = W.shape[1]
    K = W.T @ W + eps * np.eye(r)
    L = H.T @ H + eps * np.eye(r)
    M = W.T @ S @ H

    G = K * L.T
    G[np.diag_indices_from(G)] += lam  # ==> G + λI_r
    rhs = np.diag(M)

    c, lower = cho_factor(G, check_finite=False)
    d = cho_solve((c, lower), rhs, check_finite=False)

    return np.diag(d)


def update_w(
    s: np.ndarray, w: np.ndarray, h: np.ndarray, a: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Update step for W using stacked normal equations and NNLS.

    Solves: min_W ||S - WAH^T||_F^2 + alpha * ||W - H||_F^2
    """
    sqrt_alpha = np.sqrt(alpha)
    r = h.shape[1]

    # Design matrix shape (n + r, r)
    A_aug = np.vstack([h @ a.T, sqrt_alpha * np.eye(r)])  # (n, r)  # (r, r)

    # Target matrix shape (n + r, n)
    B_aug = np.vstack([s.T, sqrt_alpha * h.T])  # (n, n)  # (r, n)

    # Normal equations
    left = A_aug.T @ A_aug  # (r x r)
    right = A_aug.T @ B_aug  # (r x n)

    x_t = nnlsm_blockpivot(left, right, is_input_prod=True, init=w.T)[0]
    return x_t.T


def update_h(
    s: np.ndarray, w: np.ndarray, h: np.ndarray, a: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Update step for H using stacked normal equations and NNLS.

    Solves: min_H ||S - WAH^T||_F^2 + alpha * ||H - W||_F^2
    """
    sqrt_alpha = np.sqrt(alpha)
    r = w.shape[1]

    # Design matrix C and target A for stacked system
    A_aug = np.vstack([w @ a, sqrt_alpha * np.eye(r)])  # (n, r)  # (r, r)
    B_aug = np.vstack([s.T, sqrt_alpha * w.T])  # (n, n)  # (r, n)
    # Normal equations
    left = A_aug.T @ A_aug  # (r x r)
    right = A_aug.T @ B_aug  # (r x n)

    x_t = nnlsm_blockpivot(left, right, is_input_prod=True, init=h.T)[0]
    return x_t.T


def normalize_factors_(w: Array, h: Array, a: Array) -> tuple[Array, Array, Array]:
    d = np.sqrt(np.clip(np.diag(a), 1e-12, None))
    a /= d[:, None] * d[None, :]  #  (C-1)
    w *= d[None, :]
    h *= d[None, :]


def update_a_identity(
    s: np.ndarray, w: np.ndarray, h: np.ndarray, lam: float = 1.0, eps: float = 1e-12
):
    r = w.shape[1]
    k = w.T @ w + eps * np.eye(r)
    l = h.T @ h + eps * np.eye(r)
    m = w.T @ s @ h
    d_k, u = np.linalg.eigh(k)
    d_l, v = np.linalg.eigh(l)
    mt = u.T @ m @ v
    utv = u.T @ v
    denom = d_k[:, None] * d_l[None, :] + lam
    y = (mt + lam * utv) / denom
    return u @ y @ v.T


def soft_threshold_a(A_raw: np.ndarray, lam: float) -> np.ndarray:
    """
    Soft-threshold the off-diagonals of (A_raw - I) at level lam.
    Returns A_new = I + soft(A_raw - I, lam).
    """
    r = A_raw.shape[0]
    # Compute delta = A_raw - I
    Delta = A_raw.copy()
    np.fill_diagonal(Delta, 0)
    Delta = Delta  # now Delta[i,i]=0

    # Soft-threshold off-diagonals
    # sign(x)*max(|x|-lam, 0)
    Delta = np.sign(Delta) * np.maximum(np.abs(Delta) - lam, 0.0)

    # Reassemble A_new
    A_new = Delta
    np.fill_diagonal(A_new, 1.0)
    return A_new


@dataclass(kw_only=True)
class TriFactor(BaseNMF):
    rank: int = 10
    alpha: float = 1.0
    init: str = "random_sqrt"
    max_iter: int = 300
    verbose: bool = False
    random_state: int | None = None
    lam: float = 0.01
    update_a: bool = True
    # runtime fields (initialised at fit)
    w_: Array | None = field(init=False, default=None)
    h_: Array | None = field(init=False, default=None)
    a_: Array | None = field(init=False, default=None)
    history: dict[str, list[float]] = field(init=False, default_factory=dict)

    def fit(self, s: Array) -> "TriFactor":
        self.w_ = self.init_factor(s)
        self.h_ = self.w_.copy()
        self.a_ = np.eye(self.rank)

        self.history = {"obj": [], "rsa": []}

        for it in range(self.max_iter):

            self.w_ = update_w(s, self.w_, self.h_, self.a_, self.alpha)
            self.h_ = update_h(s, self.w_, self.h_, self.a_, self.alpha)

            if self.update_a:
                # NOTE replaced this
                self.a_ = update_a_cholesky(s, self.w_, self.h_, self.eps)
                # self.a_ = update_a_diag_ridge(s, self.w_, self.h_, self.lam, self.eps)
                # self.a_ = update_a_diag(s, self.w_, self.h_, self.eps)

                # self.a_ = update_a_identity(s, self.w_, self.h_, self.lam, self.eps)
                # self.a_ = update_a(s, self.w_, self.h_, self.eps)

                # # 2) Choose lam by percentile heuristic:
                # offdiag = np.abs(self.a_ - np.eye(self.rank))[
                #     ~np.eye(self.rank, dtype=bool)
                # ]
                # lam = np.percentile(offdiag, 90)  # keep top 10% of interactions

                # # 3) Apply soft‐threshold:
                # self.a_ = soft_threshold_a(self.a_, lam)

            self.s_hat_ = self.w_ @ self.a_ @ self.h_.T
            rec_error = np.linalg.norm(s - self.s_hat_, "fro") ** 2
            penalty = self.alpha * np.linalg.norm(self.w_ - self.h_, "fro") ** 2
            l1_offdiag = self.lam * np.sum(np.abs(self.a_ - np.eye(self.rank)))
            obj = rec_error + penalty + l1_offdiag
            self.history["obj"].append(obj)
            self.history["rsa"].append(rsa_score(self.s_hat_, s))

            # TODO make this dependent on freeze a and more elegant
            normalize_factors_(self.w_, self.h_, self.a_)

            if not self.update_a:
                self.a_ = np.eye(self.rank)

            if self.verbose:
                print(f"Iter {it:4d}/{self.max_iter:4d}  Obj={obj:.2f}", end="\r")

        return self

    def fit_transform(self, s: Array) -> Array:
        self.fit(s)
        return self.w_
