"""Figure helpers for the Recipe-K + SBM experiment.

Required panels (plan §13.8):

  1. eigenvalue scree plot with k_cut and K_true.
  2. kappa_hat_r curve with k_cut.
  3. CV loss curve with argmin and 1-SE rank.
  4. ARI/NMI vs r when labels are known.
  5. UMAP panels coloured by true and predicted labels.
  6. co-assignment stability heatmap across folds / repetitions.

The functions are intentionally small and standalone — each panel saves a
single PNG so that figures can be inspected piecewise.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_scree(spec_out, K_true, save_path):
    evals = np.asarray(spec_out["evals_ref"], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(1, len(evals) + 1)
    ax.plot(x, evals, "o-", ms=3, lw=1)
    if "k_cut" in spec_out and spec_out["k_cut"] is not None:
        kc = int(spec_out["k_cut"])
        ax.axvline(kc, color="C0", ls="--",
                   label=f"recipe-K k_cut = {kc}")
    if K_true is not None:
        ax.axvline(int(K_true), color="C3", ls=":",
                   label=f"K_true = {int(K_true)}")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("index r")
    ax.set_ylabel("eigenvalue lambda_r(S)")
    ax.set_title("scree")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_kappa(spec_out, K_true, save_path):
    kappa = np.asarray(spec_out["kappa_hat"], dtype=float)
    k_list = np.asarray(spec_out["k_list"], dtype=int)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_list, kappa, "o-", ms=3, lw=1)
    if "k_cut" in spec_out and spec_out["k_cut"] is not None:
        ax.axvline(int(spec_out["k_cut"]), color="C0", ls="--",
                   label=f"k_cut = {int(spec_out['k_cut'])}")
    if K_true is not None:
        ax.axvline(int(K_true), color="C3", ls=":",
                   label=f"K_true = {int(K_true)}")
    ax.set_xlabel("rank r")
    ax.set_ylabel("kappa_hat_r")
    ax.set_yscale("log")
    ax.set_title("Recipe-K leakage profile")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_cv_curve(cv_out, K_true, save_path, title=None):
    ranks = np.asarray(cv_out["ranks"], dtype=int)
    mean = np.asarray(cv_out["cv_loss"], dtype=float)
    sem = np.asarray(cv_out["cv_sem"], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(ranks, mean, yerr=sem, fmt="o-", ms=3, lw=1, capsize=2)
    if "argmin" in cv_out and cv_out["argmin"] > 0:
        ax.axvline(int(cv_out["argmin"]), color="C2", ls="--",
                   label=f"argmin = {int(cv_out['argmin'])}")
    if "one_se" in cv_out and cv_out["one_se"] > 0:
        ax.axvline(int(cv_out["one_se"]), color="C1", ls="-.",
                   label=f"1-SE  = {int(cv_out['one_se'])}")
    if K_true is not None:
        ax.axvline(int(K_true), color="C3", ls=":",
                   label=f"K_true = {int(K_true)}")
    ax.set_xlabel("candidate rank r")
    ax.set_ylabel("held-out predictive loss")
    ax.set_title(title or "CV loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_ari_nmi(cv_out, labels_true, save_path):
    from .metrics import adjusted_rand_index, normalized_mutual_info
    if labels_true is None:
        return
    ranks = list(cv_out["ranks"])
    aris, nmis = [], []
    for r in ranks:
        lab = cv_out["labels_by_rank"].get(int(r))
        if lab is None:
            aris.append(np.nan); nmis.append(np.nan)
            continue
        aris.append(adjusted_rand_index(labels_true, lab))
        nmis.append(normalized_mutual_info(labels_true, lab))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ranks, aris, "o-", ms=3, lw=1, label="ARI")
    ax.plot(ranks, nmis, "s-", ms=3, lw=1, label="NMI")
    if "argmin" in cv_out and cv_out["argmin"] > 0:
        ax.axvline(int(cv_out["argmin"]), color="C2", ls="--",
                   label=f"argmin = {int(cv_out['argmin'])}")
    ax.set_xlabel("candidate rank r")
    ax.set_ylabel("ARI / NMI")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("partition quality vs rank")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_umap_panels(S, labels_true, labels_pred, save_path, seed=0):
    """UMAP coloured by true labels and by predicted labels (§9 / §11).

    UMAP runs on the squared-Euclidean kernel distance D_ij^2 = S_ii + S_jj - 2 S_ij
    when S is PSD-like; otherwise it falls back to ``1 - S/S.max()``.
    """
    try:
        import umap
    except Exception:
        return  # UMAP unavailable; silently skip.
    n = S.shape[0]
    diag = np.diag(S)
    # Try PSD-style kernel distance; if it produces negative values, fall back.
    D2 = diag[:, None] + diag[None, :] - 2.0 * S
    if np.any(D2 < -1e-6):
        max_s = float(np.max(S)) if np.max(S) > 0 else 1.0
        D = 1.0 - S / max_s
    else:
        D = np.sqrt(np.clip(D2, 0.0, None))
    np.fill_diagonal(D, 0.0)
    try:
        emb = umap.UMAP(metric="precomputed", random_state=int(seed),
                        n_neighbors=min(15, max(2, n // 10)),
                        min_dist=0.1).fit_transform(D)
    except Exception:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, lab, title in zip(
        axes, [labels_true, labels_pred],
        ["UMAP — true labels", "UMAP — predicted labels (argmin)"]
    ):
        if lab is None:
            ax.scatter(emb[:, 0], emb[:, 1], s=8, c="0.5")
        else:
            ax.scatter(emb[:, 0], emb[:, 1], s=8, c=lab, cmap="tab20")
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_coassignment(cv_out, labels_true, save_path):
    """Co-assignment frequency across folds / repetitions (panel 6 of §13.8).

    For each rank we have at most one labelling (the rep=0 fold=0 fit),
    so this panel uses the predicted labelling at ``argmin`` and contrasts it
    with the true labelling by computing the symmetric co-assignment indicator
    matrix.
    """
    ranks = list(cv_out["ranks"])
    am = int(cv_out.get("argmin", -1))
    if am <= 0:
        return
    lab_pred = cv_out["labels_by_rank"].get(am)
    if lab_pred is None:
        return
    n = lab_pred.size
    pred_mat = (lab_pred[:, None] == lab_pred[None, :]).astype(float)
    fig, axes = plt.subplots(1, 2 if labels_true is not None else 1,
                              figsize=(11, 5) if labels_true is not None
                              else (6, 5))
    axes = np.atleast_1d(axes)
    order = np.argsort(lab_pred)
    axes[0].imshow(pred_mat[np.ix_(order, order)], cmap="Greys",
                    interpolation="nearest", aspect="auto")
    axes[0].set_title(f"predicted co-assignment (r={am})")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    if labels_true is not None:
        true_mat = (labels_true[:, None] == labels_true[None, :]).astype(float)
        order_t = np.argsort(labels_true)
        axes[1].imshow(true_mat[np.ix_(order_t, order_t)], cmap="Greys",
                        interpolation="nearest", aspect="auto")
        axes[1].set_title("true co-assignment")
        axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
