"""Same entrywise CV reproduction protocol, applied to real RSMs.

Loads cached similarity matrices for Mur (n=92) and Peterson Animals /
Various (n=120). Runs the same pipeline as
``reproduce_reference_sbm.py``: ``mask_missing_entries`` →
``split_observed_into_folds`` → ``fit_symmnmf`` with
``rho=3.0, max_inner=30, tol=0.0`` → ``mse_on_mask`` on off-diagonal
upper triangle. Diagnostics (``k_cut``, ``p_star``, ``p_cv``) from our
``RankEstimator``.

For each dataset we sweep ``max_outer ∈ {5, 10, 30, 100, 200}`` and
report argmin + 1-SE rank.

Cached RSMs come from ``experiments/figures/plot_embeddings/outputs/.cache``.
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
from symmnmf.cross_validation import mask_missing_entries  # noqa: E402


log = logging.getLogger(__name__)
OUTPUT_DIR = get_output_dir()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RSM_CACHE = PROJECT_ROOT / "experiments" / "figures" / "plot_embeddings" / "outputs" / ".cache"

DATASETS = [
    ("mur92", RSM_CACHE / "sim_mur92.npy"),
    ("peterson-animals", RSM_CACHE / "sim_peterson-animals.npy"),
    ("peterson-various", RSM_CACHE / "sim_peterson-various.npy"),
]


def split_observed_into_folds(M_outer, k_inner, rng):
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
    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    m = mask[iu] & np.isfinite(S[iu]) & np.isfinite(S_hat[iu])
    if not m.any():
        return float("nan"), 0
    return float(np.mean((S[iu][m] - S_hat[iu][m]) ** 2)), int(m.sum())


def fit_symmnmf(S, rank, train_mask, seed, rho, max_outer, max_inner, tol, bounds):
    x_train = np.full_like(S, np.nan)
    x_train[train_mask] = S[train_mask]
    est = SRF(
        rank=rank, missing_values=np.nan, bounds=bounds, random_state=seed,
        rho=rho, max_outer=int(max_outer), max_inner=int(max_inner), tol=float(tol),
    )
    est.fit(x_train)
    return {"S_hat": est.reconstruct()}


def _one_fit(S, rank, max_outer, max_inner, rho, tol, train_mask, val_mask, bounds, seed):
    out = fit_symmnmf(
        S, rank=rank, train_mask=train_mask, seed=seed,
        rho=rho, max_outer=max_outer, max_inner=max_inner, tol=tol, bounds=bounds,
    )
    mse, nv = mse_on_mask(S, out["S_hat"], val_mask)
    return mse, nv


def run_one_dataset(name, S, max_outer_grid, ranks, rep_seeds, k_cv,
                     rho, tol, max_inner):
    n = S.shape[0]
    bounds = (float(S.min()), float(S.max()))
    off_diag = ~np.eye(n, dtype=bool)

    est = RankEstimator(recovery_tolerance=0.10, n_bootstrap=20, random_state=0).fit(S)
    k_cut = int(est.rank_)
    p_star = float(est.sampling_fraction_)
    p_cv = float(est.cv_sampling_fraction(k_cv))
    log.info("[%s] n=%d  k_cut=%d  p_star=%.4f  p_cv=%.4f",
             name, n, k_cut, p_star, p_cv)

    jobs = []
    for rep_idx, rep_seed in enumerate(rep_seeds):
        rng = np.random.default_rng(6_000_000 + rep_seed)
        M_outer = mask_missing_entries(S, p_cv, rng, missing_values=np.nan)
        val_folds = split_observed_into_folds(M_outer, k_cv, rng)
        if val_folds is None:
            log.warning("[%s] rep %d: insufficient observed pool, skipping", name, rep_idx)
            continue
        for fold_idx, V in enumerate(val_folds):
            holdout = V | M_outer
            train_mask = (~holdout) & off_diag
            val_mask = V & off_diag
            for r in ranks:
                for mo in max_outer_grid:
                    fit_seed = 7919 * (rep_idx + 1) + 101 * fold_idx + 13 * r + mo
                    jobs.append((rep_idx, fold_idx, r, mo, train_mask, val_mask, fit_seed))

    log.info("[%s] dispatching %d fits", name, len(jobs))
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
    return df, dict(n=n, k_cut=k_cut, p_star=p_star, p_cv=p_cv)


def aggregate(df, ranks, max_outer_grid):
    rows = []
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
        rows.append(dict(max_outer=mo, r_argmin=r_argmin, r_1se=r_1se,
                          mean=means.values, sem=sems.values))
    return rows


def plot_dataset(name, info, summary_rows, ranks, max_outer_grid, rho, tol, max_inner, out_path):
    sns.set_theme(style="ticks", context="paper")
    palette = sns.color_palette("rocket_r", n_colors=len(max_outer_grid))
    fig = plt.figure(figsize=(4.0 * len(max_outer_grid), 8.0))
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
    ax_top.axvline(info["k_cut"], color="k", lw=0.8, ls="--", label=f"k_cut={info['k_cut']}")
    ax_top.set_xlabel("Rank")
    ax_top.set_ylabel("5-fold CV-MSE (mean ± SE)")
    ax_top.set_yscale("log")
    ax_top.set_title(
        f"{name}  |  n={info['n']}  |  p_star={info['p_star']:.3f}, p_cv={info['p_cv']:.3f}, "
        f"rho={rho}, tol={tol}, max_inner={max_inner}",
        fontsize=10,
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
        ax.axvline(info["k_cut"], color="k", lw=0.8, ls="--")
        ax.set_xlabel("Rank")
        ax.set_yscale("log")
        if col == 0:
            ax.set_ylabel("CV-MSE")
        ax.set_title(f"max_outer = {row['max_outer']}")
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        sns.despine(ax=ax)

    fig.suptitle(f"Entrywise CV-MSE vs rank on {name} (real RSM)", fontsize=12, y=0.995)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    max_outer_grid = [5, 10, 30, 100, 200]
    rep_seeds = [0, 1, 2]
    k_cv = 5
    rho = 3.0
    tol = 0.0
    max_inner = 30

    summary_lines = ["# Real-data entrywise CV  |  rho=3.0, tol=0.0, max_inner=30, 3 outer reps", ""]

    for name, path in DATASETS:
        if not path.exists():
            log.warning("missing cache for %s at %s — skipping", name, path)
            continue
        S = np.load(path)
        # Rank grid scales with n. Use [2, min(n-1, 40)].
        max_rank = min(S.shape[0] - 1, 40)
        ranks = list(range(2, max_rank + 1))

        df, info = run_one_dataset(name, S, max_outer_grid, ranks, rep_seeds, k_cv,
                                    rho, tol, max_inner)
        df.to_csv(OUTPUT_DIR / f"real_{name}_records.csv", index=False)

        summary_rows = aggregate(df, ranks, max_outer_grid)
        for row in summary_rows:
            log.info("[%s] max_outer=%3d  argmin=%d  1-SE=%d  V-MSE@argmin=%.4e",
                     name, row["max_outer"], row["r_argmin"], row["r_1se"],
                     row["mean"][ranks.index(row["r_argmin"])])

        plot_path = OUTPUT_DIR / f"real_{name}.png"
        plot_dataset(name, info, summary_rows, ranks, max_outer_grid, rho, tol, max_inner, plot_path)
        log.info("[%s] saved %s", name, plot_path)

        summary_lines.append(
            f"{name}  n={info['n']}  k_cut={info['k_cut']}  p_star={info['p_star']:.3f}  p_cv={info['p_cv']:.3f}"
        )
        for row in summary_rows:
            summary_lines.append(f"  max_outer={row['max_outer']:4d}   argmin={row['r_argmin']:>2}   1-SE={row['r_1se']:>2}")
        summary_lines.append("")

    (OUTPUT_DIR / "real_summary.txt").write_text("\n".join(summary_lines) + "\n")
    log.info("\n%s", "\n".join(summary_lines))


if __name__ == "__main__":
    main()
