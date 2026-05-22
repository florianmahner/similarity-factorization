"""1:1 reproduction of ``update_pysrf/src/experiment_cvrank_at_pstar.py``.

Same synthetic (clean weighted SBM), same masking primitives
(``mask_missing_entries`` + ``split_observed_into_folds``), same scoring
(``mse_on_mask`` on the off-diagonal upper triangle), same ADMM args
(``rho=3.0, max_inner=30, tol=0.0``), same outer reps (3), same rank grid
(2..30), same ``max_outer_grid = [5, 10, 30, 100]``.

The only thing that is not 1:1 is the **fitter class**: the reference
imports ``fit_symmnmf`` from a ``symmnmf_cv`` module that no longer exists
in the tree. We use ``pysrf.SRF`` instead — we already verified (sandbox
``compare_fitters.py``) it produces identical V-MSE to ``symmnmf.ADMM``
at matched iteration budget.

Diagnostics ``p_star``, ``p_cv``, ``k_cut`` come from our ``RankEstimator``
(same algorithm as Recipe-K with cleaner naming): ``rank_`` is ``k_cut``,
``sampling_fraction_`` is ``p_star``, ``cv_sampling_fraction(k_cv)`` is
``p_cv``.

Expected result (from the reference plot): argmin rank is essentially
insensitive to ``max_outer`` because the SBM has a hard rank-K floor.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

from pysrf import SRF
from src.utils import get_output_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "update_pysrf" / "src"))

from coherence import RankEstimator  # noqa: E402
from sbm.datasets import clean_weighted_sbm  # noqa: E402
from symmnmf.cross_validation import mask_missing_entries  # noqa: E402


log = logging.getLogger(__name__)
OUTPUT_DIR = get_output_dir()


# -------------------------------------------------------------------------
# Mask helpers — names match the reference (experiment_cvrank_at_pstar.py).
# -------------------------------------------------------------------------

def split_observed_into_folds(M_outer, k_inner, rng):
    """Symmetric k-fold split of observed (= not pre-masked) off-diag pairs."""
    n = M_outer.shape[0]
    iu = np.triu_indices(n, k=1)
    observed = ~M_outer[iu]
    valid = np.where(observed)[0]
    if len(valid) < k_inner:
        return None
    perm = rng.permutation(len(valid))
    groups = np.array_split(perm, k_inner)
    folds = []
    for g in groups:
        m = np.zeros((n, n), dtype=bool)
        ii, jj = iu[0][valid[g]], iu[1][valid[g]]
        m[ii, jj] = True
        m[jj, ii] = True
        folds.append(m)
    return folds


def mse_on_mask(S, S_hat, mask):
    """V-MSE on off-diagonal upper-triangle entries inside ``mask``."""
    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    m = mask[iu] & np.isfinite(S[iu]) & np.isfinite(S_hat[iu])
    if not m.any():
        return float("nan"), 0
    return float(np.mean((S[iu][m] - S_hat[iu][m]) ** 2)), int(m.sum())


def fit_symmnmf(S, rank, train_mask, seed, rho, max_outer, max_inner, tol, bounds):
    """Stand-in for the reference's ``fit_symmnmf`` (using pysrf.SRF backend)."""
    x_train = np.full_like(S, np.nan)
    x_train[train_mask] = S[train_mask]
    est = SRF(
        rank=rank, missing_values=np.nan, bounds=bounds, random_state=seed,
        rho=rho, max_outer=int(max_outer), max_inner=int(max_inner), tol=float(tol),
    )
    est.fit(x_train)
    return {"S_hat": est.reconstruct()}


# -------------------------------------------------------------------------
# Per-job worker (flat list for joblib).
# -------------------------------------------------------------------------

def _one_fit(S, rank, max_outer, max_inner, rho, tol, train_mask, val_mask, bounds, seed):
    out = fit_symmnmf(
        S, rank=rank, train_mask=train_mask, seed=seed,
        rho=rho, max_outer=max_outer, max_inner=max_inner, tol=tol, bounds=bounds,
    )
    mse, nv = mse_on_mask(S, out["S_hat"], val_mask)
    return mse, nv


# -------------------------------------------------------------------------
# Main.
# -------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    sns.set_theme(style="ticks", context="paper")

    # Reference settings (experiment_cvrank_at_pstar.py:78-115).
    n, K, sigma = 200, 10, 0.3
    k_cv = 5
    data_seed = 42

    ranks = list(range(2, 31, 1))
    max_outer_grid = [5, 10, 30, 100]
    rep_seeds = [0, 1, 2]

    rho = 3.0
    tol = 0.0      # forces full inner-w iterations (no early stopping)
    max_inner = 30

    # Build the SBM, estimate p_star / p_cv / k_cut.
    data = clean_weighted_sbm(n=n, K=K, mu_in=1.0, mu_out=0.0, sigma=sigma, seed=data_seed)
    S = data["S"]
    bounds = (float(S.min()), float(S.max()))
    off_diag = ~np.eye(n, dtype=bool)

    est = RankEstimator(recovery_tolerance=0.10, n_bootstrap=20, random_state=0).fit(S)
    k_cut = int(est.rank_)
    p_star = float(est.sampling_fraction_)
    p_cv = float(est.cv_sampling_fraction(k_cv))
    log.info("k_cut=%d  p_star=%.4f  p_cv=%.4f  (k_cv=%d)", k_cut, p_star, p_cv, k_cv)
    log.info("max_outer_grid=%s  ranks=%d..%d  rho=%.1f tol=%.1f max_inner=%d",
             max_outer_grid, ranks[0], ranks[-1], rho, tol, max_inner)

    # Build all jobs across (rep, fold, rank, max_outer).
    jobs = []
    for rep_idx, rep_seed in enumerate(rep_seeds):
        rng = np.random.default_rng(2_000_000 + rep_seed)
        M_outer = mask_missing_entries(S, p_cv, rng, missing_values=np.nan)
        val_folds = split_observed_into_folds(M_outer, k_cv, rng)
        if val_folds is None:
            log.warning("rep %d: insufficient observed pool, skipping", rep_idx)
            continue
        for fold_idx, V in enumerate(val_folds):
            holdout = V | M_outer
            train_mask = (~holdout) & off_diag
            val_mask = V & off_diag
            for r in ranks:
                for mo in max_outer_grid:
                    fit_seed = 7919 * (rep_idx + 1) + 101 * fold_idx + 13 * r + mo
                    jobs.append((rep_idx, fold_idx, r, mo, train_mask, val_mask, fit_seed))

    log.info("dispatching %d fits", len(jobs))
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_one_fit)(
            S, r, mo, max_inner, rho, tol, tm, vm, bounds, sd,
        )
        for (_, _, r, mo, tm, vm, sd) in jobs
    )

    records = []
    for (rep_idx, fold_idx, r, mo, *_), (mse, nv) in zip(jobs, results):
        records.append(dict(rep=rep_idx, fold=fold_idx, rank=r, max_outer=mo,
                            val_mse=mse, n_val=nv))
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "reproduce_sbm_records.csv", index=False)
    log.info("saved %s", OUTPUT_DIR / "reproduce_sbm_records.csv")

    # Aggregate.
    summary_rows = []
    for mo in max_outer_grid:
        sub = df[df["max_outer"] == mo]
        means = sub.groupby("rank")["val_mse"].mean().reindex(ranks)
        counts = sub.groupby("rank")["val_mse"].apply(lambda x: int(np.isfinite(x).sum())).reindex(ranks)
        stds = sub.groupby("rank")["val_mse"].std().reindex(ranks)
        sems = stds / np.maximum(np.sqrt(counts), 1)
        i_argmin = int(np.argmin(means.values))
        r_argmin = ranks[i_argmin]
        thresh = means.values[i_argmin] + sems.values[i_argmin]
        i_1se = int(np.argmax(means.values <= thresh))
        r_1se = ranks[i_1se]
        summary_rows.append(dict(max_outer=mo, r_argmin=r_argmin, r_1se=r_1se,
                                  mean=means.values, sem=sems.values))
        log.info("max_outer=%3d  argmin=%d  1-SE=%d  V-MSE@argmin=%.4e",
                 mo, r_argmin, r_1se, means.values[i_argmin])

    # Plot: top combined panel + per-max_outer panels (matches reference layout).
    palette = sns.color_palette("rocket_r", n_colors=len(max_outer_grid))
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, len(max_outer_grid), height_ratios=[1.3, 1.0],
                          hspace=0.45, wspace=0.30)

    ax_top = fig.add_subplot(gs[0, :])
    for row, color in zip(summary_rows, palette):
        ax_top.plot(ranks, row["mean"], "-o", color=color, lw=1.8, ms=4,
                    label=f"max_outer={row['max_outer']:>3}  argmin={row['r_argmin']}  1-SE={row['r_1se']}")
        ax_top.fill_between(ranks, row["mean"] - row["sem"], row["mean"] + row["sem"],
                            alpha=0.15, color=color)
        ax_top.scatter([row["r_argmin"]], [row["mean"][ranks.index(row["r_argmin"])]],
                       marker="*", s=180, color=color, edgecolors="k", linewidths=1.0,
                       zorder=5)
    ax_top.axvline(k_cut, color="k", lw=0.8, ls="--", label=f"k_cut={k_cut}")
    ax_top.axvline(K, color="0.5", lw=0.8, ls=":", label=f"K_true={K}")
    ax_top.set_xlabel("Rank")
    ax_top.set_ylabel("5-fold CV-MSE (mean ± SE)")
    ax_top.set_title(
        f"SBM reproduction  |  n={n}, K_true={K}, sigma={sigma}, "
        f"p_star={p_star:.3f}, p_cv={p_cv:.3f}, rho={rho}, tol={tol}, max_inner={max_inner}",
        fontsize=11,
    )
    ax_top.legend(fontsize=8, loc="upper right", frameon=False)
    sns.despine(ax=ax_top)

    for col, (row, color) in enumerate(zip(summary_rows, palette)):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(ranks, row["mean"], "-o", color=color, lw=1.6, ms=4)
        ax.fill_between(ranks, row["mean"] - row["sem"], row["mean"] + row["sem"],
                        alpha=0.18, color=color)
        ax.scatter([row["r_argmin"]], [row["mean"][ranks.index(row["r_argmin"])]],
                   marker="*", s=180, color=color, edgecolors="k", linewidths=1.0,
                   zorder=5, label=f"argmin={row['r_argmin']}")
        ax.axvline(k_cut, color="k", lw=0.8, ls="--")
        ax.axvline(K, color="0.5", lw=0.8, ls=":")
        ax.set_xlabel("Rank")
        if col == 0:
            ax.set_ylabel("CV-MSE")
        ax.set_title(f"max_outer = {row['max_outer']}")
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        sns.despine(ax=ax)

    fig.suptitle(
        "CV-MSE vs rank on clean weighted SBM, by ADMM iteration budget (reference port)",
        fontsize=12, y=0.995,
    )
    out = OUTPUT_DIR / "reproduce_sbm.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("saved %s", out)

    lines = [
        f"# Reference reproduction (SBM): n={n}, K_true={K}, sigma={sigma}, seed={data_seed}",
        f"# p_star={p_star:.4f}, p_cv={p_cv:.4f}, k_cut={k_cut}",
        f"# rho={rho}, tol={tol}, max_inner={max_inner}",
        "",
        "argmin and 1-SE rank per max_outer:",
    ]
    for row in summary_rows:
        lines.append(f"  max_outer={row['max_outer']:4d}   argmin={row['r_argmin']:>2}   1-SE={row['r_1se']:>2}")
    (OUTPUT_DIR / "reproduce_sbm_summary.txt").write_text("\n".join(lines) + "\n")
    log.info("\n%s", "\n".join(lines))


if __name__ == "__main__":
    main()
