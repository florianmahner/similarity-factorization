"""Recipe-K-correct CV: pre-mask at p_cv (= p_star * k_cv/(k_cv-1)),
5-fold split, sweep rank, do this for several max_outer budgets, report
argmin rank AND 1-SE rank per budget.

Pipeline
--------
1. Estimate p_star via Recipe K on the *same* 10-block n=200 SBM.
2. Pre-mask at p_cv with k_cv=5 → per-fold training fraction = p_star.
3. For each (max_outer, rank): fit SymmNMF, score val MSE per fold.
   Repeat with n_reps independent pre-masks.
4. Aggregate: CV-MSE = mean across (rep, fold); SE = std / sqrt(n_obs).
5. Per max_outer:
     argmin rank = arg min CV-MSE
     1-SE rank   = smallest rank with CV-MSE <= min + SE_at_min
6. Plot all max_outer curves on one panel + 4 per-budget panels.

Outputs
-------
- ``output/cvrank_at_pstar/records.csv``
- ``output/cvrank_at_pstar/cvrank_curves.png``
"""
from __future__ import annotations
import os
import sys
import csv

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from sbm.datasets import clean_weighted_sbm
from _common import spectral_pass, recipe_K
from symmnmf.cross_validation import mask_missing_entries
from symmnmf_cv import fit_symmnmf

OUT_DIR = os.path.join(ROOT, "output", "cvrank_at_pstar")
os.makedirs(OUT_DIR, exist_ok=True)


def split_observed_into_folds(M_outer, k_inner, rng):
    """Symmetric k-fold split of observed (= not pre-masked) off-diag pairs.
    Returns a list of bool ``val_mask`` arrays (True = val for this fold)."""
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


def main():
    n, K, sigma = 200, 10, 0.3
    k_cv = 5

    data = clean_weighted_sbm(n=n, K=K, mu_in=1.0, mu_out=0.0, sigma=sigma, seed=42)
    S = data["S"]

    print("=" * 70)
    print("Step 1: estimate p_star via Recipe K (spectral pass).")
    print("=" * 70)
    sp = spectral_pass(S, B=20, show_progress=False)
    rk = recipe_K(sp, delta=0.10, k_cv=k_cv, p_floor=0.5)
    p_star = float(rk["p_star"])
    p_cv = float(rk["p_cv"])
    k_cut = int(rk["k_cut"])
    print(f"  k_cut       = {k_cut}")
    print(f"  p_star      = {p_star:.3f}")
    print(f"  p_cv (k_cv={k_cv}) = {p_cv:.3f}  → per-fold training fraction "
          f"= (k_cv-1)/k_cv * p_cv = {p_cv * (k_cv - 1) / k_cv:.3f}")
    print(f"  recipe_K status = {rk['status']}")

    ranks = list(range(2, 31, 1))         # 2..30
    max_outer_grid = [5, 10, 30, 100]
    rep_seeds = [0, 1, 2]                  # 3 outer pre-mask repeats

    print()
    print("=" * 70)
    print("Step 2: sweep rank × max_outer with proper 5-fold CV at p_cv.")
    print(f"  ranks = {ranks}")
    print(f"  max_outer = {max_outer_grid}")
    print(f"  outer reps (pre-mask seeds) = {len(rep_seeds)}")
    print("=" * 70)

    csv_path = os.path.join(OUT_DIR, "records.csv")
    fh = open(csv_path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(
        ["rank", "max_outer", "rep", "fold", "val_mse", "n_val"]
    )

    records = []
    bounds = (float(S.min()), float(S.max()))
    off_diag = ~np.eye(n, dtype=bool)

    for rep_idx, rep_seed in enumerate(rep_seeds):
        rng = np.random.default_rng(2_000_000 + rep_seed)
        M_outer = mask_missing_entries(S, p_cv, rng, missing_values=np.nan)
        val_folds = split_observed_into_folds(M_outer, k_cv, rng)
        if val_folds is None:
            print(f"  rep {rep_idx}: insufficient observed pool, skipping")
            continue

        for fold_idx, V in enumerate(val_folds):
            holdout = V | M_outer
            train_mask = (~holdout) & off_diag
            val_mask = V & off_diag

            for r in ranks:
                for mo in max_outer_grid:
                    fit_seed = (
                        7919 * (rep_idx + 1) + 101 * fold_idx + 13 * r + mo
                    )
                    out = fit_symmnmf(
                        S, rank=r, train_mask=train_mask,
                        seed=fit_seed, rho=3.0,
                        max_outer=int(mo), max_inner=30, tol=0.0,
                        bounds=bounds,
                    )
                    mse, nv = mse_on_mask(S, out["S_hat"], val_mask)
                    records.append(dict(
                        rank=r, max_outer=mo, rep=rep_idx, fold=fold_idx,
                        val_mse=mse, n_val=nv,
                    ))
                    writer.writerow([r, mo, rep_idx, fold_idx, mse, nv])
            print(f"  rep {rep_idx}  fold {fold_idx+1}/{k_cv}  done "
                  f"(train pairs ≈ {int(train_mask[np.triu_indices(n, 1)].sum())})")

    fh.close()
    print(f"\nSaved records: {csv_path}")

    print()
    print("=" * 70)
    print("Step 3: aggregate and pick argmin + 1-SE rank per max_outer.")
    print("=" * 70)
    rows = []
    for mo in max_outer_grid:
        sub = [x for x in records if x["max_outer"] == mo]
        mean = np.array([
            np.nanmean([x["val_mse"] for x in sub if x["rank"] == r])
            for r in ranks
        ])
        # SE across (rep, fold) cells per rank
        counts = np.array([
            sum(1 for x in sub if x["rank"] == r and np.isfinite(x["val_mse"]))
            for r in ranks
        ])
        std = np.array([
            np.nanstd([x["val_mse"] for x in sub if x["rank"] == r])
            for r in ranks
        ])
        sem = std / np.maximum(np.sqrt(counts), 1)

        i_argmin = int(np.argmin(mean))
        r_argmin = ranks[i_argmin]
        thresh = mean[i_argmin] + sem[i_argmin]
        # smallest rank with mean <= thresh (1-SE rule for rank)
        i_1se = int(np.argmax(mean <= thresh))
        r_1se = ranks[i_1se]
        rows.append(dict(mo=mo, mean=mean, sem=sem,
                          r_argmin=r_argmin, r_1se=r_1se))
        print(f"  max_outer = {mo:>3}  argmin rank = {r_argmin:>2}  "
              f"1-SE rank = {r_1se:>2}  CV-MSE@argmin = {mean[i_argmin]:.4f} "
              f"(SE {sem[i_argmin]:.4f})")

    # -------------------------------------------------------------
    # Figure: combined panel (top) + per-max_outer panels (bottom 4)
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.3, 1.0], hspace=0.45,
                           wspace=0.30)

    ax_top = fig.add_subplot(gs[0, :])
    colors = ["C0", "C1", "C2", "C3"]
    for row, c in zip(rows, colors):
        ax_top.plot(ranks, row["mean"], "-o", color=c, lw=2,
                    label=(f"max_outer={row['mo']:>3}  "
                           f"argmin rank={row['r_argmin']}, "
                           f"1-SE rank={row['r_1se']}"))
        ax_top.fill_between(ranks, row["mean"] - row["sem"],
                             row["mean"] + row["sem"], alpha=0.15, color=c)
        ax_top.scatter([row["r_argmin"]],
                        [row["mean"][ranks.index(row["r_argmin"])]],
                        marker="*", s=240, color=c, edgecolors="k",
                        linewidths=1.2, zorder=5)
        ax_top.scatter([row["r_1se"]],
                        [row["mean"][ranks.index(row["r_1se"])]],
                        marker="o", s=160, color="white", edgecolors=c,
                        linewidths=2.4, zorder=5)
    ax_top.axvline(k_cut, color="k", lw=0.8, ls="--",
                    label=f"k_cut = {k_cut}")
    ax_top.axvline(K, color="0.5", lw=0.8, ls=":",
                    label=f"K_true = {K}")
    ax_top.set_xticks(ranks)
    ax_top.tick_params(axis="x", labelsize=9)
    ax_top.set_xlabel("rank r")
    ax_top.set_ylabel("5-fold CV-MSE (mean ± SE)")
    ax_top.set_title(
        f"SymmNMF 5-fold CV at Recipe-K p_star={p_star:.3f} (p_cv={p_cv:.3f}); "
        f"n={n}, K_true={K}, 3 outer reps.   "
        f"★ = argmin rank,  ○ = 1-SE rank",
        fontsize=11,
    )
    ax_top.legend(fontsize=8, loc="upper right")
    ax_top.grid(alpha=0.3)

    for col, (row, c) in enumerate(zip(rows, colors)):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(ranks, row["mean"], "-o", color=c, lw=1.8)
        ax.fill_between(ranks, row["mean"] - row["sem"],
                          row["mean"] + row["sem"], alpha=0.20, color=c)
        ax.scatter([row["r_argmin"]],
                    [row["mean"][ranks.index(row["r_argmin"])]],
                    marker="*", s=240, color=c, edgecolors="k",
                    linewidths=1.2, zorder=5,
                    label=f"argmin = {row['r_argmin']}")
        ax.scatter([row["r_1se"]],
                    [row["mean"][ranks.index(row["r_1se"])]],
                    marker="o", s=160, color="white", edgecolors=c,
                    linewidths=2.4, zorder=5,
                    label=f"1-SE = {row['r_1se']}")
        # show the SE threshold band
        y_min = row["mean"].min()
        y_thresh = y_min + row["sem"][np.argmin(row["mean"])]
        ax.axhline(y_thresh, color=c, lw=0.8, ls=":",
                    label=f"min + 1 SE")
        ax.axvline(k_cut, color="k", lw=0.8, ls="--")
        ax.axvline(K, color="0.5", lw=0.8, ls=":")
        ax.set_xticks(ranks[::2])
        ax.set_xlabel("rank r")
        if col == 0:
            ax.set_ylabel("5-fold CV-MSE")
        ax.set_title(f"max_outer = {row['mo']}")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle(
        "CV-MSE vs rank at Recipe-K-prescribed p_star, by ADMM iteration budget",
        fontsize=13, y=0.995,
    )
    fig_path = os.path.join(OUT_DIR, "cvrank_curves.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {fig_path}")


if __name__ == "__main__":
    main()
