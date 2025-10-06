import numpy as np
from dataclasses import dataclass, field
from srf.models.nnls_block import nnlsm_blockpivot
from srf.models.base import BaseNMF
from typing import Literal
from collections import defaultdict
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import Lasso

Array = np.ndarray


def get_a_update_func(str: Literal["cd", "fista", "ista", "admm"]):
    if str == "cd":
        return update_a_cd
    elif str == "fista":
        return fista_update_a
    elif str == "ista":
        return ista_update_a
    elif str == "admm":
        return admm_update
    else:
        raise ValueError(f"Invalid update method: {str}")


def soft_threshold(x, thresh):
    """Applies element-wise soft-thresholding"""
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def soft_threshold_offdiag(x, thresh, diag_value=1.0):
    """Soft-threshold off-diagonal only; fix diagonal to diag_value"""
    a = x.copy()
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if i != j:
                a[i, j] = np.sign(x[i, j]) * max(abs(x[i, j]) - thresh, 0.0)
            else:
                a[i, j] = diag_value
    return a


def soft_threshold_offdiag_elastic_net(
    g: np.ndarray,
    lambda1: float,
    lambda2: float,
    step_size: float,
    diag_value: float = 1.0,
) -> np.ndarray:
    """
    Prox for   lambda1 * |A_off|_1  +  (lambda2/2) * ||A||_F^2
    applied entryise to g, fixing diag to diag_value.
    """
    k = g.shape[0]
    a = np.zeros_like(g)
    t = step_size
    for i in range(k):
        for j in range(k):
            if i == j:
                # keep diagonal at a prescribed value
                a[i, i] = diag_value
            else:
                # soft-threshold then ridge shrink
                #  shrink = max(abs(g)-lambda1 t, 0)
                #  a = sign(g) * shrink / (1 + lambda2 t)
                gi = g[i, j]
                shr = max(abs(gi) - lambda1 * t, 0.0)
                a[i, j] = np.sign(gi) * shr / (1.0 + lambda2 * t)
    return a


def fista_update_a(s, w, h, lambda_reg, a_init=None, num_iters=100, delta=1e-6):
    """
    FISTA update for A in the problem:
        min_A ||S - W A H.T||_F^2 + lambda * ||A||_1
    """
    n, m = s.shape
    k = w.shape[1]
    if a_init is None:
        a = np.zeros((k, k))
    else:
        a = a_init.copy()
    y = a.copy()
    t = 1.0  # step size

    # Precompute constant matrices
    wtw = w.T @ w
    hth = h.T @ h
    wts_h = w.T @ s @ h

    # Lipschitz constant can be computed from the spectral norm of the matrices
    l_w = np.linalg.norm(wtw, 2)
    l_h = np.linalg.norm(hth, 2)
    lipschitz_const = 2 * l_w * l_h
    step_size = 1.0 / lipschitz_const
    for i in range(num_iters):
        a_old = a.copy()

        # Gradient of reconstruction loss at y
        grad = 2 * (wtw @ y @ hth - wts_h)

        a = soft_threshold_offdiag(y - step_size * grad, lambda_reg * step_size)

        # FISTA momentum
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = a + ((t - 1) / t_new) * (a - a_old)
        t = t_new

        if np.linalg.norm(a - a_old) < delta:
            break

    return a


def ista_update_a(s, w, h, lam, a_init=None, num_iters=100, delta=1e-6):
    """
    ISTA (proximal gradient) for:
        min_A ||S - W A H.T||_F^2 + lambda * ||A||_1
    """
    k = w.shape[1]
    if a_init is None:
        a = np.zeros((k, k))
    else:
        a = a_init.copy()
    wtw = w.T @ w
    hth = h.T @ h
    wts_h = w.T @ s @ h

    # Lipschitz constant (spectral norm)
    l_w = np.linalg.norm(wtw, 2)
    l_h = np.linalg.norm(hth, 2)
    lipschitz_const = 2 * l_w * l_h
    step_size = 1.0 / lipschitz_const
    # FIXME hard coded this to see if this makes it smoother
    # step_size = 0.001
    for it in range(num_iters):
        a_old = a.copy()

        # Gradient
        grad = 2 * (wtw @ a @ hth - wts_h)

        # Gradient step + soft-thresholding
        g = a - step_size * grad

        # FIXME hard coded this to see if this makes it smoother
        a = soft_threshold_offdiag(g, lam * step_size)

        if np.linalg.norm(a - a_old) < delta:
            break

    return a


def admm_update_a(
    s: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
    rho: float,
) -> np.ndarray:
    # build mats
    m_mat = 2 * w.T @ w  # (k×k)
    n_mat = h.T @ h  # (k×k)
    rhs = 2 * w.T @ s @ h + rho * (z - u)

    # eigendecompositions
    lam_m, q_m = np.linalg.eigh(m_mat)
    lam_n, q_n = np.linalg.eigh(n_mat)

    # rotate into eigen‐bases
    y = q_m.T @ rhs @ q_n

    # element‐wise division
    denom = lam_m[:, None] * lam_n[None, :] + rho
    x = y / denom

    # back‐transform
    return q_m @ x @ q_n.T


def admm_update_z(a, u, lam, rho):
    """
    Z-update in ADMM via soft-thresholding.
    """
    thresh = lam / rho
    z = np.sign(a + u) * np.maximum(np.abs(a + u) - thresh, 0.0)
    np.fill_diagonal(z, np.diag(a))
    return z


def admm_update_u(u, a, z):
    """
    Dual update.
    """
    return u + a - z


def admm_update(
    s,
    w,
    h,
    a_init=None,
    z_init=None,
    u_init=None,
    lam=0.01,
    rho=1.0,
    num_iters=100,
    tol=1e-6,
):
    k = w.shape[1]
    a = np.zeros((k, k)) if a_init is None else a_init
    z = np.zeros((k, k)) if z_init is None else z_init
    u = np.zeros((k, k)) if u_init is None else u_init

    for it in range(num_iters):
        a_old = a.copy()

        # A-update
        a = admm_update_a(s, w, h, z, u, rho)

        # Z-update
        z = admm_update_z(a, u, lam, rho)

        # U-update
        u = admm_update_u(u, a, z)

        # Convergence check (optional)
        if np.linalg.norm(a - a_old) < tol:
            break

    return a, z, u


def update_a_cd(
    s: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    a: np.ndarray,
    lam: float,
    cd_inner: int = 100,
    eps: float = 1e-12,
) -> np.ndarray:
    # FIXME: Coordinate descent update needs fixing - currently not converging properly

    k = a.shape[0]
    # pre‑compute static matrices
    m_mat = w.T @ s @ h  # (k×k)
    g_w = w.T @ w  # (k×k)
    g_h = h.T @ h  # (k×k)

    for _ in range(cd_inner):
        # current residual term G_w·A·G_h
        gac = g_w @ a @ g_h
        for theta in range(k):
            # scalar for denom component from W
            denom_w = g_w[theta, theta]
            for ell in range(k):
                denom_h = g_h[ell, ell]
                denom = denom_w * denom_h + eps
                # gradient-like residual
                resid = m_mat[theta, ell] - gac[theta, ell]
                # coordinate update with sparsity bias
                z = a[theta, ell] + resid / denom
                a[theta, ell] = np.sign(z) * max(abs(z) - lam / denom, 0)
    return a


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


def update_w_l1(
    s: np.ndarray,
    h: np.ndarray,
    a: np.ndarray,
    alpha: float,
    beta: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Coordinate-descent update for W with an L1 penalty (β‖W‖₁) and
    non-negativity constraint, re-using the stacked design matrix.

    Returns an (n × k) matrix.
    """
    k = h.shape[1]  # rank
    sqrt_alpha = np.sqrt(alpha)

    # stacked design & target
    x_mat = np.vstack([h @ a.T, sqrt_alpha * np.eye(k)])  # ((n+k) × k)
    y_mat = np.vstack([s.T, sqrt_alpha * h.T])  # ((n+k) × n)

    n_samples = x_mat.shape[0]
    lasso_alpha = beta / (2 * n_samples)  # << correct scaling

    w_new_t = np.empty((k, s.shape[0]))  # fill column-wise

    lasso = Lasso(
        alpha=lasso_alpha,
        fit_intercept=False,
        positive=True,
        max_iter=max_iter,
        tol=tol,
    )

    for j in range(s.shape[0]):  # each column is an independent problem
        lasso.fit(x_mat, y_mat[:, j])
        w_new_t[:, j] = lasso.coef_

    return w_new_t.T  # shape (n × k)


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


@dataclass(kw_only=True)
class TriFactor(BaseNMF):
    rank: int = 10
    alpha: float = 1.0
    beta: float = 0.0
    lam: float = 0.01
    init: str = "random_sqrt"
    max_iter: int = 300
    verbose: bool = False
    random_state: int | None = None
    cd_inner: int = 100
    update_a: bool = True
    update_w: bool = True
    update_h: bool = True
    a_method: Literal["cd", "fista", "ista", "admm"] = "cd"
    freeze_a: int = 0
    rho: float = 1.0
    tol: float = 1e-6
    w_true: Array | None = None
    h_true: Array | None = None
    a_true: Array | None = None

    # runtime fields (initialised at fit)
    w_: Array | None = field(init=False, default=None)
    h_: Array | None = field(init=False, default=None)
    a_: Array | None = field(init=False, default=None)
    z_: Array | None = field(init=False, default=None)
    u_: Array | None = field(init=False, default=None)
    history: dict[str, list[float]] = field(init=False, default_factory=dict)

    def fit(self, s: Array) -> "TriFactor":
        self.w_ = self.init_factor(s)
        self.h_ = self.init_factor(s)

        self.a_ = np.eye(self.rank)
        self.z_ = self.a_.copy()
        self.u_ = np.zeros_like(self.a_)
        self.history = defaultdict(list)
        for it in range(self.max_iter):
            if self.update_w and self.beta == 0.0:
                self.w_ = update_w(s, self.w_, self.h_, self.a_, self.alpha)
            elif self.update_w and self.beta > 0.0:
                self.w_ = update_w_l1(s, self.h_, self.a_, self.alpha, self.beta)
            else:
                self.w_ = self.w_true
            if self.update_h:
                self.h_ = update_h(s, self.w_, self.h_, self.a_, self.alpha)
            else:
                self.h_ = self.w_

            if self.update_a and it > self.freeze_a:
                if self.a_method == "fixed" and self.a_true is not None:
                    self.a_ = self.a_true

                elif self.a_method == "cd":
                    self.a_ = update_a_cd(
                        s,
                        self.w_,
                        self.h_,
                        self.a_,
                        self.lam,
                    )
                elif self.a_method == "ista":
                    self.a_ = ista_update_a(
                        s,
                        self.w_,
                        self.h_,
                        self.lam,
                        a_init=self.a_,
                        num_iters=self.cd_inner,
                    )
                elif self.a_method == "fista":
                    self.a_ = fista_update_a(
                        s,
                        self.w_,
                        self.h_,
                        self.lam,
                        a_init=self.a_,
                        num_iters=self.cd_inner,
                    )

                elif self.a_method == "admm":
                    self.a_, self.z_, self.u_ = admm_update(
                        s,
                        self.w_,
                        self.h_,
                        a_init=self.a_,
                        z_init=self.z_,
                        u_init=self.u_,
                        lam=self.lam,
                        rho=self.rho,
                        num_iters=self.cd_inner,
                        tol=self.tol,
                    )

                self.a_ = (self.a_ + self.a_.T) / 2

            self.s_hat_ = self.w_ @ self.a_ @ self.h_.T
            rec_error = np.linalg.norm(s - self.s_hat_, "fro") ** 2
            alpha_penalty = self.alpha * np.linalg.norm(self.w_ - self.h_, "fro") ** 2

            l1_offdiag = self.lam * np.sum(np.abs(self.a_ - np.eye(self.rank)))
            obj = rec_error + alpha_penalty + l1_offdiag
            self.history["obj"].append(obj)
            self.history["rec_error"].append(rec_error)
            self.history["alpha_penalty"].append(alpha_penalty)
            self.history["l1_offdiag"].append(l1_offdiag)

            if self.w_true is not None:
                self.history["w_diff"].append(
                    np.linalg.norm(self.w_ - self.w_true, "fro")
                )
            if self.h_true is not None:
                self.history["h_diff"].append(
                    np.linalg.norm(self.h_ - self.h_true, "fro")
                )
            if self.a_true is not None:
                self.history["a_diff"].append(
                    np.linalg.norm(self.a_ - self.a_true, "fro")
                )
            if self.verbose:
                print(
                    f"Iter {it:4d}/{self.max_iter:4d}  Obj={obj:.2f} Rec={rec_error:.2f} Alpha={alpha_penalty:.2f} L1={l1_offdiag:.2f}",
                    end="\r",
                )

        return self

    def fit_transform(self, s: Array) -> Array:
        self.fit(s)
        return self.w_
