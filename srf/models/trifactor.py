import numpy as np
from .base import BaseNMF
from .normal_eqs import regularized_normal_eqs

Array = np.ndarray


def update_a_least_squares(s: Array, w: Array, h: Array):
    """
    Unconstrained least squares update for S,
    solving min_S ||X - U * S * H^T||_F^2.

    Uses the Moore-Penrose pseudoinverse:
        S = U^+ * X * (H^+)^T.
    """
    w_pinv = np.linalg.pinv(w)
    h_pinv = np.linalg.pinv(h)
    return w_pinv @ s @ h_pinv.T


class TrifactorANLS(BaseNMF):
    """
    Tri-factor Optimization of the form  min ||S - W A H.T||_F^2 + alpha * ||W - H||_F^2
    """

    def __init__(
        self,
        rank: int,
        alpha: float | None = None,
        max_iter: int = 1000,
        tol: float = 1e-5,
        random_state: int | None = None,
        init: str = "random_sqrt",
        verbose: bool = False,
        eval_every: int = 100,
        eps: float = np.finfo(float).eps,
        **kwargs,
    ):
        super().__init__(
            rank, max_iter, tol, random_state, init, verbose, eval_every, eps
        )
        self.alpha = alpha

    def init_mixing(self) -> Array:
        """Initialize a symmetric mixing matrix with small random values"""
        a = 0.01 * np.random.randn(self.rank, self.rank)
        a = 0.5 * (a + a.T)
        return a

    def fit(self, s: np.ndarray):

        if self.alpha is None:
            self.alpha = np.max(s) ** 2

        # Initialize U using the base initializer (n x r).
        w = self.init_factor(s)
        h = w.copy()
        a = self.init_mixing()

        if self.alpha is None:
            self.alpha = np.max(s) ** 2

        for it in range(1, self.max_iter + 1):
            w = regularized_normal_eqs(s, fixed=h, alpha=self.alpha, update=w)
            h = regularized_normal_eqs(s, fixed=w, alpha=self.alpha, update=h)
            a = update_a_least_squares(s, w, h)

            # Compute the reconstruction error.
            s_hat = w @ a @ h.T
            obj = np.linalg.norm(s - s_hat, "fro") ** 2

            if self.verbose:
                s_upper = s[np.triu_indices(s.shape[0], k=1)]
                s_hat_upper = s_hat[np.triu_indices(s_hat.shape[0], k=1)]

                var_s = np.var(s_upper)
                var_residuals = np.var(s_upper - s_hat_upper)

                # Compute the explained variance
                explained_var = 1 - (var_residuals / var_s)

                print(
                    f"Iteration {it}, objective: {obj:.3f}, explained variance: {explained_var:.3f}",
                    end="\r",
                )
            if obj < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {it} with objective {obj:.4e}")
                break

        self.w_ = w
        self.h_ = h
        self.a_ = a
        self.s_hat_ = s_hat
        self.iter_ = it
        return self

    def fit_transform(self, s: np.ndarray) -> np.ndarray:
        self.fit(s)
        return self.w_
