# %%
import pyximport

pyximport.install()
import numpy as np
from srf._cdnmf_fast import _update_cdnmf_fast_per_entry_bounds
from sklearn.utils.extmath import safe_sparse_dot
from collections import defaultdict
import matplotlib.pyplot as plt

from srf.helpers import (
    best_pairwise_match,
    safe_cross_corr,
)


def _update_coordinate_descent(
    x: np.ndarray,
    w: np.ndarray,
    ht: np.ndarray,
    regularizer: int,
    beta: float,
    upperbd: np.ndarray,
    shuffle: bool,
    random_state,
) -> float:
    n_comp = ht.shape[1]
    hht = ht.T @ ht
    xht = x @ ht

    if regularizer == 2:
        hht.flat[:: n_comp + 1] += beta
    elif regularizer == 1:
        xht -= beta

    perm = random_state.permutation(n_comp) if shuffle else np.arange(n_comp)
    perm = perm.astype(np.intp, copy=False)

    return _update_cdnmf_fast_per_entry_bounds(w, hht, xht, upperbd, perm)


def update_block(
    z,
    w,
    h,
    *,
    beta=0.0,
    regularizer=2,  # ⟵ new
    upperbd=None,
    shuffle=True,
    random_state=None,
):
    if upperbd is None:
        ub = np.full_like(w, np.inf, order="C")
    else:
        ub = np.asarray(upperbd, dtype=w.dtype, order="C")

    _update_coordinate_descent(
        x=z,
        w=w,
        ht=h,
        regularizer=regularizer,  # <── L1 or L2
        beta=beta,  # strength
        upperbd=ub,
        shuffle=shuffle,
        random_state=random_state or np.random,
    )
    return w


def mixed_sign_nmf_cd_with_net_nonneg(
    s: np.ndarray, k: int, n_iter: int = 50, alpha: float = 0.0, random_state=None
):
    """
    Mixed-sign SymNMF: decompose S ≈ W+H+.T - W-H-.T
    enforce (W+ - W-) ≥ 0, (H+ - H-) ≥ 0 by post-update clipping.
    """
    rng = np.random.RandomState(random_state)
    n = s.shape[0]

    w_plus = 0.1 * rng.rand(n, k)
    h_plus = 0.1 * rng.rand(n, k)
    w_minus = 0.1 * rng.rand(n, k)
    h_minus = 0.1 * rng.rand(n, k)

    history = defaultdict(list)

    for it in range(n_iter):
        # positive block
        zp = s + w_minus @ h_minus.T
        w_plus = update_block(
            zp, w_plus, h_plus, beta=alpha, upperbd=None, shuffle=True, random_state=rng
        )
        h_plus = update_block(
            zp.T,
            h_plus,
            w_plus,
            beta=alpha,
            upperbd=None,
            shuffle=True,
            random_state=rng,
        )

        # – block with elementwise bounds
        Zn = w_plus @ h_plus.T - s
        w_minus = update_block(
            Zn,
            w_minus,
            h_minus,
            beta=alpha,
            upperbd=w_plus,
            shuffle=True,
            random_state=rng,
        )
        h_minus = update_block(
            Zn.T,
            h_minus,
            w_minus,
            beta=alpha,
            upperbd=h_plus,
            shuffle=True,
            random_state=rng,
        )

        # Compute metrics
        net_recon = w_plus @ h_plus.T - w_minus @ h_minus.T
        resid_norm = np.linalg.norm(S - net_recon, "fro")
        norm_wp = np.linalg.norm(w_plus, "fro")
        norm_hp = np.linalg.norm(h_plus, "fro")
        norm_wm = np.linalg.norm(w_minus, "fro")

        norm_hm = np.linalg.norm(h_minus, "fro")
        norm_netw = np.linalg.norm(w_plus - w_minus, "fro")
        norm_neth = np.linalg.norm(h_plus - h_minus, "fro")
        objective = resid_norm**2 + alpha * (
            norm_wp**2 + norm_hp**2 + norm_wm**2 + norm_hm**2
        )

        # Log history
        history["iter"].append(it)
        history["residual_fro"].append(resid_norm)
        history["obj"].append(objective)
        history["norm_wp"].append(norm_wp)
        history["norm_hp"].append(norm_hp)
        history["norm_wm"].append(norm_wm)
        history["norm_hm"].append(norm_hm)
        history["norm_netw"].append(norm_netw)
        history["norm_neth"].append(norm_neth)

        print(f"Iteration {it} objective: {objective:.3f}", end="\r")

    return w_plus, h_plus, w_minus, h_minus, history

    # quick smoke test


rng = np.random.RandomState(0)

# A = 0.1 * rng.rand(300, 50)
# S = A @ A.T
A = np.loadtxt("/LOCAL/fmahner/srf/data/misc/spose_embedding_66d.txt")
S = A @ A.T
S = S / np.max(S)

wp, hp, wm, hm, history = mixed_sign_nmf_cd_with_net_nonneg(
    S, k=66, n_iter=1000, alpha=1.0, random_state=42
)
print("shapes:", wp.shape, hp.shape, wm.shape, hm.shape)
assert np.all(wp - wm >= -1e-8)
assert np.all(hp - hm >= -1e-8)
print("success: net factors are elementwise nonnegative!")


# %%

# plot all the history
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Norms of Matrix Factors Over Time")

axes[0, 0].plot(history["norm_wp"])
axes[0, 0].set_title("norm_wp")
axes[0, 0].set_xlabel("Iteration")
axes[0, 0].set_ylabel("Norm")


axes[0, 1].plot(history["norm_hp"])
axes[0, 1].set_title("norm_hp")
axes[0, 1].set_xlabel("Iteration")
axes[0, 1].set_ylabel("Norm")


axes[1, 0].plot(history["norm_wm"])
axes[1, 0].set_title("norm_wm")
axes[1, 0].set_xlabel("Iteration")
axes[1, 0].set_ylabel("Norm")

axes[1, 1].plot(history["norm_hm"])
axes[1, 1].set_title("norm_hm")
axes[1, 1].set_xlabel("Iteration")
axes[1, 1].set_ylabel("Norm")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 1, figsize=(6, 3))

axes.plot(history["obj"])
axes.set_title("Objective")
axes.set_xlabel("Iteration")
axes.set_ylabel("Objective")
plt.show()


w = wp - wm
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
corrs = best_pairwise_match(A, w)
ax.plot(corrs)
ax.set_title("Correlation between Original and Reconstructed Features")
ax.set_xlabel("Feature Index")
ax.set_ylabel("Correlation")
plt.tight_layout()
plt.show()

# %%
