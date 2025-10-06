from dataclasses import dataclass, field
import numpy as np
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


def update_w_prox(
    S: Array,
    W_prev: Array,  # ← last iterate, acts as the RHS
    A: Array,
    eps: float = 1e-8,
) -> Array:
    """
    Solve in closed form by NNLS-block-pivot

        argmin_{W ≥ 0}
            ½‖S - W A W_prevᵀ‖_F²
          + ½‖W - W_prev‖_F²            (α fixed to 1)

    The second term is *only* a numerical prox term; it vanishes at
    convergence and is not exposed to the user.
    """
    r = A.shape[0]
    sqrt_alpha = 1.0  # fixed
    # design / target for stacked normal eqns
    A_aug = np.vstack([W_prev @ A.T, sqrt_alpha * np.eye(r)])
    B_aug = np.vstack([S.T, sqrt_alpha * W_prev.T])

    left = A_aug.T @ A_aug + eps * np.eye(r)
    right = A_aug.T @ B_aug

    W_new_T = nnlsm_blockpivot(left, right, is_input_prod=True, init=W_prev.T)[0]
    return W_new_T.T


def _soft_threshold_off(mat: Array, tau: float) -> Array:
    """Soft-threshold off-diagonal elements of a square matrix."""
    out = mat.copy()
    dmask = np.eye(out.shape[0], dtype=bool)
    out[~dmask] = np.sign(out[~dmask]) * np.maximum(np.abs(out[~dmask]) - tau, 0.0)
    out[dmask] = 0.0  # keep diag = 0
    return out


def _update_b_admm(
    S: Array, W: Array, Z: Array, U: Array, rho: float, eps: float = 1e-12
) -> Array:
    """
    Solve for B in  P B P + rho B = M + rho (Z - U),  diag(B)=0,
    with  P = WᵀW  and  M = Wᵀ(S - W Wᵀ)W .
    Closed form via eigen-decomposition of P.
    """
    r = W.shape[1]
    P = W.T @ W + eps * np.eye(r)
    M = W.T @ (S - W @ W.T) @ W  # r×r
    RHS = M + rho * (Z - U)  # r×r

    d, U_eig = np.linalg.eigh(P)  # P = U diag(d) Uᵀ
    Rt = U_eig.T @ RHS @ U_eig  # transform
    denom = d[:, None] * d[None, :] + rho
    Bt = Rt / denom  # elementwise
    B = U_eig @ Bt @ U_eig.T
    B = 0.5 * (B + B.T)  # symmetrise
    np.fill_diagonal(B, 0.0)
    return B


@dataclass(kw_only=True)
class SingleFactorADMM(BaseNMF):
    """Non-negative SymNMF  S ≈ W A Wᵀ  with ADMM-regularised A."""

    rank: int = 10
    max_iter: int = 300  # outer alternations
    admm_iter: int = 20  # inner ADMM steps per outer iter
    alpha: float = 1e-6  # tiny tie-term required by update_w
    lam: float = 1e-2  # L₁ weight on off-diagonals
    rho: float = 1.0  # ADMM penalty
    verbose: bool = False
    random_state: int | None = None

    # learned parameters
    w_: Array | None = field(init=False, default=None)
    a_: Array | None = field(init=False, default=None)
    history: dict[str, list[float]] = field(init=False, default_factory=dict)

    # ────────────────────────────────────────
    def fit(self, S: Array) -> "SingleFactorADMM":
        n, m = S.shape
        assert n == m, "S must be square / symmetric"
        rng = np.random.default_rng(self.random_state)
        # self.w_ = np.maximum(rng.standard_normal((n, self.rank)), 1e-4)
        self.w_ = 0.01 * np.random.rand(n, self.rank)
        B = np.zeros((self.rank, self.rank))
        Z = np.zeros_like(B)
        U = np.zeros_like(B)

        self.history = dict(obj=[], rsa=[])

        for it in range(self.max_iter):
            # --------- 1. update W (NNLS) ----------

            W_new = update_w_prox(S, self.w_, np.eye(self.rank) + B)
            self.w_ = W_new
            # --------- 2. ADMM loop for A ----------
            for _ in range(self.admm_iter):
                # 2a. B-update (closed form)
                B = _update_b_admm(S, self.w_, Z, U, self.rho)

                # 2b. Z-update  (soft thresh)
                Z = _soft_threshold_off(B + U, self.lam / self.rho)

                # 2c. dual
                U += B - Z

            self.a_ = np.eye(self.rank) + B

            # --------- bookkeeping ----------
            S_hat = self.w_ @ self.a_ @ self.w_.T
            rec = np.linalg.norm(S - S_hat, "fro") ** 2
            sparsity = self.lam * np.sum(np.abs(B))
            obj = rec + sparsity
            self.history["obj"].append(obj)
            self.history["rsa"].append(rsa_score(S_hat, S))

            if self.verbose:
                print(
                    f"it {it:4d}/{self.max_iter}  obj={obj:.3f}  "
                    f"‖B‖₁={np.sum(np.abs(B)):.2f}  ",
                    end="\r",
                )

        self.s_hat_ = self.w_ @ self.a_ @ self.w_.T
        if self.verbose:  # finish line
            print()

        return self

    # convenient alias
    def fit_transform(self, S: Array) -> Array:
        return self.fit(S).w_


# ───────────────────────────────── helpers ───────────────────────────────────
def _soft_threshold_off(M: Array, tau: float) -> Array:
    """Soft-threshold off-diagonal entries of a square matrix; keep diag 0."""
    out = M.copy()
    dmask = np.eye(out.shape[0], dtype=bool)
    off = ~dmask
    out[off] = np.sign(out[off]) * np.maximum(np.abs(out[off]) - tau, 0.0)
    out[dmask] = 0.0
    return out


# ───────────────────────────── NNLS-prox updates ─────────────────────────────
def update_w_prox(
    S: Array, W_prev: Array, H: Array, A: Array, eps: float = 1e-8
) -> Array:
    """
    NNLS block-pivot update of W:
        argmin_{W≥0} ½‖S − W A Hᵀ‖_F² + ½‖W − W_prev‖_F²
    """
    r = A.shape[0]  # rank
    sqrt_alpha = 1.0  # fixed prox weight
    G = A @ H.T  # (r × n)
    A_aug = np.vstack([G.T, sqrt_alpha * np.eye(r)])  # (n+r)×r
    B_aug = np.vstack([S.T, sqrt_alpha * W_prev.T])  # (n+r)×n

    left = A_aug.T @ A_aug + eps * np.eye(r)  # r×r
    right = A_aug.T @ B_aug  # r×n

    W_new_T = nnlsm_blockpivot(left, right, is_input_prod=True, init=W_prev.T)[0]
    return W_new_T.T


def update_h_prox(S, H_prev, W, A, eps=1e-8):
    """
    NNLS-prox update of H:
      argmin_{H≥0} ½‖Sᵀ - H Aᵀ Wᵀ‖_F² + ½‖H - H_prev‖_F²
    """
    r = A.shape[0]
    sqrt_alpha = 1.0

    # primary block: n×r
    C = W @ A

    # build (n+r)×r design and (n+r)×n targets
    A_aug = np.vstack([C, sqrt_alpha * np.eye(r)])  # → (n+r)×r
    B_aug = np.vstack([S.T, sqrt_alpha * H_prev.T])  # → (n+r)×n

    # normal equations (with tiny ridge eps)
    left = A_aug.T @ A_aug + eps * np.eye(r)  # r×r
    right = A_aug.T @ B_aug  # r×n

    # block-pivot NNLS solves for H_newᵀ (r×n), then transpose back
    H_new_T = nnlsm_blockpivot(left, right, is_input_prod=True, init=H_prev.T)[0]
    return H_new_T.T


# ─────────────────────────────  ADMM for B  ──────────────────────────────────
def _update_b_admm(
    S: Array, W: Array, H: Array, Z: Array, U: Array, rho: float, eps: float = 1e-12
) -> Array:
    """
    Solve  (P B Q  +  rho B) = R,   diag B = 0,
    with   P = WᵀW,  Q = HᵀH,
           R = Wᵀ(S − W Hᵀ)H  +  rho(Z − U).
    Closed form via *two* eigen-bases => element-wise division.
    """
    r = W.shape[1]
    P = W.T @ W + eps * np.eye(r)
    Q = H.T @ H + eps * np.eye(r)
    R_const = W.T @ (S - W @ H.T) @ H  # RHS part not involving Z/U

    # eigen-decompose once
    d, Up = np.linalg.eigh(P)  # Up diag(d) Upᵀ = P
    e, Uh = np.linalg.eigh(Q)  # Uh diag(e) Uhᵀ = Q

    # transform RHS into eigen space each ADMM sweep
    Rt = Up.T @ (R_const + rho * (Z - U)) @ Uh
    denom = d[:, None] * e[None, :] + rho  # outer sum, r×r
    Bt = Rt / denom  # element-wise solve
    B = Up @ Bt @ Uh.T
    B = 0.5 * (B + B.T)  # harmless symmetrisation
    np.fill_diagonal(B, 0.0)
    return B


# ──────────────────────────  two-factor ADMM class  ──────────────────────────
@dataclass(kw_only=True)
class TwoFactorADMM:
    """ADMM-regularised tri-factor model  S ≈ W (I+B) Hᵀ ,  diag B=0."""

    rank: int = 10
    max_iter: int = 300  # outer alternations
    admm_iter: int = 20  # inner ADMM sweeps
    lam: float = 1e-2  # L₁ on off-diagonals
    rho: float = 1.0  # ADMM penalty
    prox_eps: float = 1e-8  # stab. diag in proximal NNLS
    verbose: bool = False
    random_state: int | None = None

    # learned params
    w_: Array | None = field(init=False, default=None)
    h_: Array | None = field(init=False, default=None)
    a_: Array | None = field(init=False, default=None)
    history: dict[str, list[float]] = field(init=False, default_factory=dict)

    # ────────────────────────────────────────────────────────────────────────
    def fit(self, S: Array) -> "TwoFactorADMM":
        n, m = S.shape
        assert n == m, "S must be square (n = m) for this symmetric loss."
        rng = np.random.default_rng(self.random_state)
        self.w_ = 0.01 * np.random.rand(n, self.rank)
        self.h_ = 0.01 * np.random.rand(n, self.rank)

        B = np.zeros((self.rank, self.rank))  # off-diagonal part
        Z = np.zeros_like(B)  # ADMM auxiliary
        U = np.zeros_like(B)  # ADMM dual

        self.history = dict(obj=[])

        eye_r = np.eye(self.rank)

        for it in range(self.max_iter):
            # ── 1. NNLS-prox updates ───────────────────────────────────────
            self.w_ = update_w_prox(S, self.w_, self.h_, eye_r + B, eps=self.prox_eps)
            self.h_ = update_h_prox(S, self.h_, self.w_, eye_r + B, eps=self.prox_eps)

            # ── 2. ADMM for B (lam on off-diag) ────────────────────────────
            for _ in range(self.admm_iter):
                B = _update_b_admm(S, self.w_, self.h_, Z, U, self.rho)
                Z = _soft_threshold_off(B + U, self.lam / self.rho)
                U += B - Z

            # store full A = I + B
            self.a_ = eye_r + B

            # ── 3. objective trace (optional) ──────────────────────────────
            R = S - self.w_ @ self.a_ @ self.h_.T
            obj = np.linalg.norm(R, "fro") ** 2 + self.lam * np.abs(B).sum()
            self.history.setdefault("obj", []).append(obj)
            if self.verbose:
                print(
                    f"it {it:3d}  obj = {obj:>11.3f}, ||A||₁ = {np.sum(np.abs(self.a_)):.3f}",
                    end="\r",
                )

            # naive stop: break if objective stopped improving
            if it > 1 and abs(self.history["obj"][-2] - obj) < 1e-6 * obj:
                break

        return self

    def fit_transform(self, S: Array) -> Array:
        return self.fit(S).w_
