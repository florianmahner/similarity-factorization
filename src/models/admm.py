import pyximport
import numpy as np

pyximport.install(language_level=3, setup_args={"include_dirs": np.get_include()})
from models.bsum_cython import update_w
from models.cd_updates import update_v, update_lambda
from collections import defaultdict
from models.base import init_factor

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
        max_inner=40,
        tol=1e-4,
        verbose=False,
        init="random_sqrt",
        random_state=None,
        eps=np.finfo(float).eps,
    ):
        self.rank = rank
        self.rho = rho
        self.max_outer = max_outer
        self.max_inner = max_inner
        self.tol = tol
        self.verbose = verbose
        self.init = init
        self.random_state = random_state
        self.eps = eps

    def fit(
        self,
        s,
        mask,
        bounds=None,
    ):
        w = init_factor(s, self.rank, self.init, self.random_state, self.eps)
        lam = np.zeros_like(s)
        min_val, max_val = bounds if bounds is not None else (None, None)

        history = defaultdict(list)

        for i in range(1, self.max_outer + 1):
            v = update_v(mask, s, w, lam, self.rho, min_val, max_val)
            T = v + lam / self.rho
            w = update_w(T, w, max_iter=self.max_inner, tol=self.tol)
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

            if np.linalg.norm(v - w @ w.T, "fro") < self.tol and self.tol > 0.0:
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

    def fit_transform(self, s, mask=None, bounds=None):
        if mask is None:
            mask = np.ones_like(s)  # TODO Check this

        self.fit(s, mask, bounds)
        return self.w_

    def fit_w(self, s):
        # TODO intergrate into the fit transform somehow!
        x0 = init_factor(s, self.rank, self.init, self.random_state, self.eps)
        self.w_ = update_w(s, x0, max_iter=self.max_inner, tol=self.tol)
        self.s_hat_ = self.w_ @ self.w_.T
        self.history_ = defaultdict(list)
        self.history_["rec_error"] = np.linalg.norm(s - self.s_hat_, "fro")
        return self.w_

