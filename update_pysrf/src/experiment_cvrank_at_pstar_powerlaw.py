"""Power-law-spectrum version of ``experiment_cvrank_at_pstar.py``.

Dataset: A1_power_law (Experiment A) shifted to be non-negative.
  - ``S_raw, _ = make_power_law(n=400, k_signal=10, alpha=1.0, snr=5.0, seed=42)``
  - ``S = S_raw - S_raw.min() + 0.01``  (shift adds a rank-1 spike of
    magnitude ~shift*n ≈ 57 to the spectrum; original power-law signal is
    preserved beneath it.)

Why this dataset
----------------
A1's power-law spectrum is the canonical "heavy-tail / smooth-spectrum"
challenge cited in §7.3 of the thesis. Detectable rank is < k_signal = 10
because lambda_5..lambda_10 sit near or below the Wigner bulk edge
(~1.18 at n=400). Compared to a clean 10-block SBM, the CV-MSE-vs-rank
curve should:
  - have a much flatter valley (less curvature → 1-SE has more work to do)
  - be more sensitive to iteration budget (over-rank arm decays slower)

Outputs
-------
- ``output/cvrank_at_pstar_powerlaw/records.csv``
- ``output/cvrank_at_pstar_powerlaw/cvrank_curves.png``
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

from _synthetic_extras import make_power_law
from _common import spectral_pass, recipe_K
from symmnmf.cross_validation import mask_missing_entries
from symmnmf_cv import fit_symmnmf

OUT_DIR = os.path.join(ROOT, "output", "cvrank_at_pstar_powerlaw")
os.makedirs(OUT_DIR, exist_ok=True)


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


def main():
    n = 400
    k_signal = 10
    alpha = 1.0
    snr = 5.0
    data_seed = 42
    k_cv = 5

    S_raw, _ = make_power_law(n=n, k_signal=k_signal, alpha=alpha, snr=snr,
                                seed=data_seed)
    shift = -S_raw.min() + 0.01
    S = S_raw + shift

    print("=" * 70)
    print(f"Dataset: A1_power_law (n={n}, k_signal={k_signal}, alpha={alpha}, "
          f"snr={snr}), shifted by +{shift:.4f}.")
    print(f"S range: [{S.min():.4f}, {S.max():.4f}]")
    print("=" * 70)

    ev = np.linalg.eigvalsh(S)[::-1]
    wigner_edge = (2 * np.std(S_raw - np.diag(np.diag(S_raw)))
                   * np.sqrt(n))
    print(f"Top 12 eigenvalues of shifted S: {ev[:12].round(3)}")
    print(f"Wigner-bulk edge ~ {wigner_edge:.3f}")
    print("=" * 70)

    print("\nStep 1: estimate p_star via Recipe K spectral pass.")
    sp = spectral_pass(S, B=20, show_progress=False)
    rk = recipe_K(sp, delta=0.10, k_cv=k_cv, p_floor=0.5)
    p_star = float(rk["p_star"])
    p_cv = float(rk["p_cv"])
    k_cut = int(rk["k_cut"])
    print(f"  k_cut = {k_cut}, p_star = {p_star:.3f}, p_cv = {p_cv:.3f}, "
          f"status = {rk['status']}")
    print(f"  Detectable rank vs structural rank: k_cut={k_cut}, k_signal={k_signal}")

    ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 25, 30]
    max_outer_grid = [5, 10, 30, 100, 200]
    rep_seeds = [0, 1]

    print()
    print("=" * 70)
    print(f"Step 2: 5-fold CV at p_cv={p_cv:.3f} × max_outer={max_outer_grid}")
    print(f"  ranks = {ranks}")
    print(f"  outer reps = {len(rep_seeds)}, folds = {k_cv}")
    n_fits = len(max_outer_grid) * len(ranks) * len(rep_seeds) * k_cv
    print(f"  total fits = {n_fits}")
    print("=" * 70)

    csv_path = os.path.join(OUT_DIR, "records.csv")
    fh = open(csv_path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(["rank", "max_outer", "rep", "fold", "val_mse", "n_val"])

    records = []
    bounds = (float(S.min()), float(S.max()))
    off_diag = ~np.eye(n, dtype=bool)

    for rep_idx, rep_seed in enumerate(rep_seeds):
        rng = np.random.default_rng(3_000_000 + rep_seed)
        M_outer = mask_missing_entries(S, p_cv, rng, missing_values=np.nan)
        val_folds = split_observed_into_folds(M_outer, k_cv, rng)
        if val_folds is None:
            print(f"  rep {rep_idx}: insufficient pool, skipping")
            continue
        for fold_idx, V in enumerate(val_folds):
            holdout = V | M_outer
            train_mask = (~holdout) & off_diag
            val_mask = V & off_diag
            for r in ranks:
                for mo in max_outer_grid:
                    fit_seed = 7919 * (rep_idx + 1) + 101 * fold_idx + 13 * r + mo
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
            print(f"  rep {rep_idx}  fold {fold_idx + 1}/{k_cv}  done", flush=True)

    fh.close()
    print(f"\nSaved records: {csv_path}")

    print()
    print("=" * 70)
    print("Step 3: aggregate, find argmin + 1-SE rank per max_outer.")
    print("=" * 70)

    rows = []
    for mo in max_outer_grid:
        sub = [x for x in records if x["max_outer"] == mo]
        mean = np.array([
            np.nanmean([x["val_mse"] for x in sub if x["rank"] == r])
            for r in ranks
        ])
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
        i_1se = int(np.argmax(mean <= thresh))
        r_1se = ranks[i_1se]
        rows.append(dict(mo=mo, mean=mean, sem=sem,
                          r_argmin=r_argmin, r_1se=r_1se,
                          i_argmin=i_argmin, i_1se=i_1se))
        print(f"  max_outer = {mo:>3}  argmin rank = {r_argmin:>2}  "
              f"1-SE rank = {r_1se:>2}  CV-MSE@argmin = {mean[i_argmin]:.4f} "
              f"(SE {sem[i_argmin]:.4f})")

    n_panels = len(max_outer_grid)
    fig = plt.figure(figsize=(4.0 * n_panels, 9))
    gs = fig.add_gridspec(2, n_panels, height_ratios=[1.3, 1.0], hspace=0.45,
                           wspace=0.30)
    ax_top = fig.add_subplot(gs[0, :])
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, n_panels))
    for row, c in zip(rows, colors):
        ax_top.plot(ranks, row["mean"], "-o", color=c, lw=2,
                    label=(f"max_outer={row['mo']:>3}  "
                           f"argmin rank={row['r_argmin']}, "
                           f"1-SE rank={row['r_1se']}"))
        ax_top.fill_between(ranks, row["mean"] - row["sem"],
                             row["mean"] + row["sem"], alpha=0.15, color=c)
        ax_top.scatter([row["r_argmin"]], [row["mean"][row["i_argmin"]]],
                        marker="*", s=240, color=c, edgecolors="k",
                        linewidths=1.2, zorder=5)
        ax_top.scatter([row["r_1se"]], [row["mean"][row["i_1se"]]],
                        marker="o", s=160, color="white", edgecolors=c,
                        linewidths=2.4, zorder=5)
    ax_top.axvline(k_cut, color="k", lw=0.8, ls="--",
                    label=f"k_cut = {k_cut} (Recipe K)")
    ax_top.axvline(k_signal, color="0.5", lw=0.8, ls=":",
                    label=f"k_signal = {k_signal} (structural)")
    ax_top.set_xticks(ranks)
    ax_top.tick_params(axis="x", labelsize=9)
    ax_top.set_xlabel("rank r")
    ax_top.set_ylabel("5-fold CV-MSE (mean ± SE)")
    ax_top.set_title(
        f"SymmNMF 5-fold CV on shifted A1_power_law (n={n}, k_signal={k_signal}, "
        f"alpha={alpha}, snr={snr})  |  p_star={p_star:.3f}, p_cv={p_cv:.3f}, "
        f"k_cut={k_cut}, {len(rep_seeds)} outer reps.  ★ = argmin rank,  ○ = 1-SE rank",
        fontsize=10,
    )
    ax_top.legend(fontsize=8, loc="best")
    ax_top.grid(alpha=0.3)

    for col, (row, c) in enumerate(zip(rows, colors)):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(ranks, row["mean"], "-o", color=c, lw=1.8)
        ax.fill_between(ranks, row["mean"] - row["sem"],
                          row["mean"] + row["sem"], alpha=0.20, color=c)
        ax.scatter([row["r_argmin"]], [row["mean"][row["i_argmin"]]],
                    marker="*", s=240, color=c, edgecolors="k",
                    linewidths=1.2, zorder=5,
                    label=f"argmin = {row['r_argmin']}")
        ax.scatter([row["r_1se"]], [row["mean"][row["i_1se"]]],
                    marker="o", s=160, color="white", edgecolors=c,
                    linewidths=2.4, zorder=5,
                    label=f"1-SE = {row['r_1se']}")
        y_thresh = row["mean"][row["i_argmin"]] + row["sem"][row["i_argmin"]]
        ax.axhline(y_thresh, color=c, lw=0.8, ls=":", label="min + 1 SE")
        ax.axvline(k_cut, color="k", lw=0.8, ls="--")
        ax.axvline(k_signal, color="0.5", lw=0.8, ls=":")
        ax.set_xticks(ranks)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("rank r")
        if col == 0:
            ax.set_ylabel("5-fold CV-MSE")
        ax.set_title(f"max_outer = {row['mo']}")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    fig.suptitle(
        "CV-MSE vs rank on power-law spectrum (A1, shifted), by ADMM iteration budget",
        fontsize=13, y=0.995,
    )
    fig_path = os.path.join(OUT_DIR, "cvrank_curves.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

    print()
    print("=" * 70)
    print("Convergence check: CV-MSE @ argmin vs max_outer")
    print("=" * 70)
    print(f"  {'max_outer':>10}  {'argmin rank':>11}  {'1-SE rank':>9}  "
          f"{'CV-MSE@argmin':>13}  {'Δ from prev':>11}")
    prev = None
    for row in rows:
        msa = row["mean"][row["i_argmin"]]
        delta = f"{msa - prev:+.5f}" if prev is not None else "        —"
        print(f"  {row['mo']:>10}  {row['r_argmin']:>11}  "
              f"{row['r_1se']:>9}  {msa:>13.5f}  {delta:>11}")
        prev = msa


if __name__ == "__main__":
    main()
