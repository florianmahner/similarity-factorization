from dataclasses import dataclass, field
import numpy as np
from .base import BaseNMF
from .nnls_block import nnlsm_blockpivot
from .metrics import frobenius_norm
from collections import defaultdict
from scipy.optimize import lsq_linear
from ..helpers import median_matrix_split, evar, reconstruct_matrix

Array = np.ndarray


EPS = 1e-12  # global tiny margin


def bounded_row_ls(A, b, upper, lb_zero=True, eps=EPS):
    """
        Solve   min_x ||A x - b||²   s.t. 0 ≤ x ≤ upper (element-wise).
    `
        Any upper_k ≤ eps is replaced by eps to satisfy lb < ub.
    """
    # SciPy wants a 1-D array for bounds
    upper_safe = np.asarray(upper, dtype=float).copy()
    # enforce strict inequality
    upper_safe[upper_safe <= eps] = eps

    res = lsq_linear(
        A, b, bounds=(0.0 if lb_zero else -np.inf, upper_safe), method="trf"
    )
    return res.x


def solve_stacked_normal_eqs(
    s: Array, fixed: Array, alpha: float, update: Array
) -> Array:
    """
    We stack the matrices and rearrange the normal equations.
    see eq. 16 and 17 in Kuang et al. (2014)
    """
    sqrt_alpha = np.sqrt(alpha)
    rank = fixed.shape[1]
    # Build the stacked matrices:
    # a_stack has A^T on top and sqrt(alpha)*fixed.T on the bottom.
    a_stack = np.vstack([s.T, sqrt_alpha * fixed.T])
    # c_stack has fixed on top and sqrt(alpha)*I on the bottom.
    c_stack = np.vstack([fixed, sqrt_alpha * np.eye(rank)])

    # solve for x (which approximates the updated factor's transpose) via NNLS.
    # this basically solves the normal equations for the stacked matrices
    left = c_stack.T @ c_stack
    right = c_stack.T @ a_stack
    x_t = nnlsm_blockpivot(left, right, True, update.T)[0]
    # transpose back to get the updated factor.
    return x_t.T


def _update_block_symmetric(
    z: np.ndarray, w: np.ndarray, h: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Update a block (W, H) for symmetric mixed NMF.

    Updates W given H, then updates H given the new W.
    """
    w = solve_stacked_normal_eqs(z, fixed=h, alpha=alpha, update=w)
    h = solve_stacked_normal_eqs(z.T, fixed=w, alpha=alpha, update=h)
    return w, h


def _project_onto_feasible_set(pos: Array, neg: Array) -> None:
    """
    Project (pos, neg) onto the set { (pos, neg): pos >= neg >= 0 } elementwise, in-place.
    """
    pos[pos < 0] = 0
    neg[neg < 0] = 0
    mask = neg > pos
    neg[mask] = pos[mask]


def _normalize_factors_column(w: Array, h: Array) -> tuple[Array, Array]:
    """
    Column normalization: for matrices where both W and H are (n x r),
    normalize each column by scaling them using the geometric mean of the
    column norms.

    This ensures that the product of the column norms is preserved.
    """
    norms_w = np.linalg.norm(w, axis=0)
    norms_h = np.linalg.norm(h, axis=0)
    norms = np.sqrt(norms_w * norms_h)
    # Avoid division by zero
    norms_w = np.where(norms_w == 0, 1, norms_w)
    norms_h = np.where(norms_h == 0, 1, norms_h)
    w = w * (norms / norms_w)
    h = h * (norms / norms_h)
    return w, h


@dataclass(kw_only=True)
class SymmetricMixed(BaseNMF):

    # Parameters (with defaults)
    rank: int = 10
    alpha: float | None = None  # Default set in fit if None
    init: str = "random_sqrt"
    max_iter: int = 300
    tol: float = 1e-6
    verbose: bool = False
    random_state: int | None = None
    normalize_factors: bool = False  # Geometric mean normalization per iteration

    # Runtime fields (initialized in fit)
    w_pos_: Array | None = field(init=False, default=None)
    h_pos_: Array | None = field(init=False, default=None)
    w_neg_: Array | None = field(init=False, default=None)
    h_neg_: Array | None = field(init=False, default=None)
    x_hat_: Array | None = field(init=False, default=None)
    objective_: float | None = field(init=False, default=None)
    iter_: int | None = field(init=False, default=None)
    projection_step: bool = False  # project onto feasible set insteaf of bounded NNLS
    history: dict[str, list[float]] = field(
        init=False, default_factory=dict
    )  # Added history like TriFactor

    def fit(self, s: Array) -> "SymmetricMixed":
        self.init_progress_bar(self.max_iter)  # Keep progress bar for now

        alpha = self.alpha if self.alpha is not None else np.max(s) ** 2
        n = s.shape[0]
        r = self.rank
        sqrt_alpha = np.sqrt(alpha)

        # Initialize factors. W, H are (n x r) for an (n x n) matrix X
        w_pos = self.init_factor(s)
        h_pos = w_pos.copy()
        w_neg = self.init_factor(s)
        h_neg = w_neg.copy()

        s_hat = w_pos @ h_pos.T - w_neg @ h_neg.T
        error_init = frobenius_norm(s, s_hat)
        previous_error = error_init

        self.history = defaultdict(list)

        for it in range(1, self.max_iter + 1):

            z_pos = s + w_neg @ h_neg.T
            w_pos, h_pos = _update_block_symmetric(z_pos, w_pos, h_pos, alpha)

            if self.projection_step:
                z_neg = w_pos @ h_pos.T - s
                w_neg, h_neg = _update_block_symmetric(z_neg, w_neg, h_neg, alpha)
                _project_onto_feasible_set(w_pos, w_neg)
                _project_onto_feasible_set(h_pos, h_neg)

            else:
                z_neg = w_pos @ h_pos.T - s

                A_hn = np.vstack([h_neg, sqrt_alpha * np.eye(r)])  # fixed across rows
                for i in range(n):
                    b_i = np.concatenate([z_neg[i], sqrt_alpha * h_neg[i]])
                    w_neg[i] = bounded_row_ls(A_hn, b_i, upper=w_pos[i])

                z_neg_t = z_neg.T
                A_wn = np.vstack([w_neg, sqrt_alpha * np.eye(r)])
                for i in range(n):
                    b_i = np.concatenate([z_neg_t[i], sqrt_alpha * h_neg[i]])
                    h_neg[i] = bounded_row_ls(A_wn, b_i, upper=h_pos[i])

            if self.normalize_factors:
                w_pos, h_pos = _normalize_factors_column(w_pos, h_pos)
                w_neg, h_neg = _normalize_factors_column(w_neg, h_neg)

            s_hat = w_pos @ h_pos.T - w_neg @ h_neg.T
            error = frobenius_norm(s, s_hat)
            diff = (previous_error - error) / error_init if error_init > 0 else 0.0

            self.history["rec_error"].append(error)
            self.history["diff"].append(diff)
            self.history["neg_mass"].append(np.linalg.norm(w_neg + h_neg, "fro"))
            self.history["pos_mass"].append(np.linalg.norm(w_pos + h_pos, "fro"))
            self.history["z_neg"].append(np.linalg.norm(z_neg, "fro"))
            self.history["z_pos"].append(np.linalg.norm(z_pos, "fro"))

            if self.verbose:

                print(
                    f"Iter {it:4d}/{self.max_iter:4d}: RecErr={error:.4f}",
                    end="\r",
                )
                # Could replace print with self.update_progress_bar if preferred

            if diff < self.tol and self.tol > 0:
                if self.verbose:
                    print()  # Newline after final status
                break
            previous_error = error

        # Final optional normalization if alpha was 0.
        if alpha == 0:
            w_pos, h_pos = _normalize_factors_column(w_pos, h_pos)
            w_neg, h_neg = _normalize_factors_column(w_neg, h_neg)

        # Save final factors and related metrics.
        self.w_pos_ = w_pos
        self.h_pos_ = h_pos
        self.w_neg_ = w_neg
        self.h_neg_ = h_neg
        self.s_hat_ = s_hat
        self.objective_ = error
        self.iter_ = it

        self.close_progress_bar()
        return self

    def fit_transform(self, x: Array) -> Array:
        """Fit the model and return the combined factor W = W⁺ - W⁻."""
        self.fit(x)
        # Ensure factors are computed before returning
        if self.w_pos_ is None or self.w_neg_ is None:
            raise RuntimeError("Fit method must be called before fit_transform.")
        return self.w_pos_ - self.w_neg_

    # Added property for consistency with TriFactor attribute access, though not strictly necessary
    @property
    def w_(self) -> Array | None:
        """Return the combined factor W = W⁺ - W⁻."""
        if self.w_pos_ is None or self.w_neg_ is None:
            return None
        return self.w_pos_ - self.w_neg_


def solve_stacked_normal_eqs(
    s: Array, fixed: Array, alpha: float, update: Array
) -> Array:
    """
    We stack the matrices and rearrange the normal equations.
    see eq. 16 and 17 in Kuang et al. (2014)
    """
    sqrt_alpha = np.sqrt(alpha)
    rank = fixed.shape[1]
    # Build the stacked matrices:
    # a_stack has A^T on top and sqrt(alpha)*fixed.T on the bottom.
    a_stack = np.vstack([s.T, sqrt_alpha * fixed.T])
    # c_stack has fixed on top and sqrt(alpha)*I on the bottom.
    c_stack = np.vstack([fixed, sqrt_alpha * np.eye(rank)])

    # solve for x (which approximates the updated factor's transpose) via NNLS.
    # this basically solves the normal equations for the stacked matrices
    left = c_stack.T @ c_stack
    right = c_stack.T @ a_stack
    x_t = nnlsm_blockpivot(left, right, True, update.T)[0]
    # transpose back to get the updated factor.
    return x_t.T


@dataclass(kw_only=True)
class SymmetricMixedResFree(BaseNMF):

    # Parameters (with defaults)
    rank: int = 10
    alpha: float | None = None  # Default set in fit if None
    init: str = "random_sqrt"
    max_iter: int = 300
    tol: float = 1e-6
    verbose: bool = False
    random_state: int | None = None
    normalize_factors: bool = False  # Geometric mean normalization per iteration

    # Runtime fields (initialized in fit)
    w_pos_: Array | None = field(init=False, default=None)
    h_pos_: Array | None = field(init=False, default=None)
    w_neg_: Array | None = field(init=False, default=None)
    h_neg_: Array | None = field(init=False, default=None)
    x_hat_: Array | None = field(init=False, default=None)
    objective_: float | None = field(init=False, default=None)
    iter_: int | None = field(init=False, default=None)
    projection_step: bool = True  # project onto feasible set insteaf of bounded NNLS
    history: dict[str, list[float]] = field(
        init=False, default_factory=dict
    )  # Added history like TriFactor

    def fit(self, s: Array) -> "SymmetricMixed":
        self.init_progress_bar(self.max_iter)  # Keep progress bar for now

        alpha = self.alpha if self.alpha is not None else np.max(s) ** 2

        # Initialize factors. W, H are (n x r) for an (n x n) matrix X
        w_pos = self.init_factor(s)
        h_pos = w_pos.copy()
        w_neg = self.init_factor(s)
        h_neg = w_neg.copy()

        s_hat = w_pos @ h_pos.T - w_neg @ h_neg.T
        error_init = frobenius_norm(s, s_hat)
        previous_error = error_init

        self.history = defaultdict(list)

        s_plus, s_minus, thresh = median_matrix_split(s)

        # TODO recerror not correctly defined.

        for it in range(1, self.max_iter + 1):

            w_pos = solve_stacked_normal_eqs(
                s_plus, fixed=h_pos, alpha=self.alpha, update=w_pos
            )
            h_pos = solve_stacked_normal_eqs(
                s_plus, fixed=w_pos, alpha=self.alpha, update=h_pos
            )

            w_neg = solve_stacked_normal_eqs(
                s_minus, fixed=h_neg, alpha=self.alpha, update=w_neg
            )
            h_neg = solve_stacked_normal_eqs(
                s_minus, fixed=w_neg, alpha=self.alpha, update=h_neg
            )

            if self.normalize_factors:
                w_pos, h_pos = _normalize_factors_column(w_pos, h_pos)
                w_neg, h_neg = _normalize_factors_column(w_neg, h_neg)

            s_hat = reconstruct_matrix(w_pos, h_pos, w_neg, h_neg, thresh)
            s_minus_hat = w_neg @ h_neg.T
            s_plus_hat = w_pos @ h_pos.T
            error = frobenius_norm(s, s_hat)
            diff = (previous_error - error) / error_init if error_init > 0 else 0.0

            self.history["rec_error"].append(error)
            self.history["diff"].append(diff)
            self.history["neg_mass"].append(np.linalg.norm(w_neg + h_neg, "fro"))
            self.history["pos_mass"].append(np.linalg.norm(w_pos + h_pos, "fro"))
            self.history["s_plus_evar"].append(evar(s_plus, s_plus_hat))
            self.history["s_minus_evar"].append(evar(s_minus, s_minus_hat))

            if self.verbose:

                print(
                    f"Iter {it:4d}/{self.max_iter:4d}: RecErr={error:.4f}",
                    end="\r",
                )
                # Could replace print with self.update_progress_bar if preferred

            if diff < self.tol and self.tol > 0:
                if self.verbose:
                    print()  # Newline after final status
                break
            previous_error = error

        # Save final factors and related metrics.
        self.w_pos_ = w_pos
        self.h_pos_ = h_pos
        self.w_neg_ = w_neg
        self.h_neg_ = h_neg
        self.s_hat_ = s_hat
        self.objective_ = error
        self.iter_ = it
        self.s_plus_ = s_plus
        self.s_minus_ = s_minus

        self.close_progress_bar()
        return self

    def fit_transform(self, x: Array) -> Array:
        """Fit the model and return the combined factor W = W⁺ - W⁻."""
        self.fit(x)
        # Ensure factors are computed before returning
        if self.w_pos_ is None or self.w_neg_ is None:
            raise RuntimeError("Fit method must be called before fit_transform.")
        return self.w_pos_ - self.w_neg_

    # Added property for consistency with TriFactor attribute access, though not strictly necessary
    @property
    def w_(self) -> Array | None:
        """Return the combined factor W = W⁺ - W⁻."""
        if self.w_pos_ is None or self.w_neg_ is None:
            return None
        return self.w_pos_ - self.w_neg_
