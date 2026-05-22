"""One figure to convey *why* the CV protocol works on some datasets and not
others.

Six rows (one per dataset) × two columns:

- **Left:** top-30 eigenvalues of S (log-y), with ``k_cut`` marked. Shows
  whether the spectrum has a sharp cliff or a smooth tail.
- **Right:** 5-fold CV-MSE vs rank at the calibrated ``p_cv``, overlaid for
  two ADMM iteration budgets (``max_outer ∈ {50, 200}``). Same ``k_cut``
  marker. If the curves overlap and bottom out at ``k_cut``, the protocol
  is doing its job; if the converged (200) curve keeps falling below the
  short-budget (50) curve, the data has structure beyond ``k_cut``.

The intuition: shape of the *spectrum* predicts the shape of the *CV
curve*. Cliff ⇒ stable CV. Smooth tail ⇒ drifting CV. The "failure" is
diagnosable from S alone, before any optimisation.
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
from scipy.linalg import eigh
from scipy.spatial.distance import cdist

from pysrf import SRF
from src.utils import get_output_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "update_pysrf" / "src"))

from coherence import RankEstimator  # noqa: E402
from sbm.datasets import clean_weighted_sbm  # noqa: E402
from _synthetic_extras import make_power_law  # noqa: E402
from symmnmf.cross_validation import mask_missing_entries  # noqa: E402


log = logging.getLogger(__name__)
OUTPUT_DIR = get_output_dir()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RSM_CACHE = PROJECT_ROOT / "experiments" / "figures" / "plot_embeddings" / "outputs" / ".cache"

K_CV = 5
REP_SEEDS = [0, 1, 2, 3, 4]
MAX_OUTER_VALUES = [100, 1000]
MAX_INNER = 30
RHO = 3.0
TOL = 0.0
CAP = 0.95
N_EIGENVALUES_SHOWN = 30


# -------------------------------------------------------------------------
# Datasets.
# -------------------------------------------------------------------------

def _rbf_block_similarity(n, k_signal, bandwidth=1.0, seed=0):
    rng = np.random.default_rng(seed)
    overlap = max(1, n // 40)
    d = np.zeros((n, k_signal), dtype=float)
    block = n // k_signal
    for k in range(k_signal):
        start = k * block
        end = start + block + overlap
        for j in range(n):
            if start <= j <= end:
                d[j, k] = rng.uniform(0.5, 1.0) * rng.binomial(1, 0.9)
    d = np.clip(d, 0.0, 1.0)
    d = d + rng.random((n, k_signal)) * 0.5
    return np.exp(-cdist(d, d, "sqeuclidean") / (2.0 * bandwidth * bandwidth))


def _load_sbm():
    return clean_weighted_sbm(n=200, K=10, mu_in=1.0, mu_out=0.0,
                               sigma=0.3, seed=42)["S"]


def _load_powerlaw():
    s_raw, _ = make_power_law(n=400, k_signal=10, alpha=1.0, snr=5.0, seed=42)
    return s_raw - s_raw.min() + 0.01


def _load_rbf():
    return _rbf_block_similarity(n=300, k_signal=10, bandwidth=1.0, seed=42)


def _load_cached(name):
    path = RSM_CACHE / f"sim_{name}.npy"
    if not path.exists():
        return None
    return np.load(path)


DATASETS = [
    ("SBM",                "sbm",                _load_sbm,                              "synthetic"),
    ("Powerlaw",           "powerlaw",           _load_powerlaw,                         "synthetic"),
    ("Mur92",              "mur92",              lambda: _load_cached("mur92"),          "real"),
    ("Peterson-animals",   "peterson-animals",   lambda: _load_cached("peterson-animals"), "real"),
    ("Peterson-various",   "peterson-various",   lambda: _load_cached("peterson-various"), "real"),
]


# -------------------------------------------------------------------------
# Computation.
# -------------------------------------------------------------------------

def split_observed_into_folds(M_outer, k_inner, rng):
    n = M_outer.shape[0]
    iu = np.triu_indices(n, k=1)
    observed = ~M_outer[iu]
    valid = np.where(observed)[0]
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
        return float("nan")
    return float(np.mean((S[iu][m] - S_hat[iu][m]) ** 2))


def _one_fit(S, rank, max_outer, train_mask, val_mask, bounds, seed):
    x_train = np.full_like(S, np.nan)
    x_train[train_mask] = S[train_mask]
    est = SRF(rank=rank, missing_values=np.nan, bounds=bounds, random_state=seed,
              rho=RHO, max_outer=int(max_outer), max_inner=MAX_INNER, tol=TOL)
    est.fit(x_train)
    return mse_on_mask(S, est.reconstruct(), val_mask)


def cv_curves_all_max_outer(S, p_outer, max_outer_values, ranks):
    """One flat Parallel call across every (max_outer, rep, fold, rank) fit.

    The heavy max_outer=1000 fits mix with cheap max_outer=100 fits in one
    pool so cores stay full; otherwise the long-tail of 1000-iter fits would
    leave most cores idle at the end of a dispatch batch.
    """
    n = S.shape[0]
    bounds = (float(S.min()), float(S.max()))
    off_diag = ~np.eye(n, dtype=bool)

    # Build the outer mask + folds ONCE per rep (shared across max_outer and rank).
    per_rep = []
    for rep_idx, rep_seed in enumerate(REP_SEEDS):
        rng = np.random.default_rng(6_000_000 + rep_seed)
        M_outer = mask_missing_entries(S, p_outer, rng, missing_values=np.nan)
        val_folds = split_observed_into_folds(M_outer, K_CV, rng)
        folds = []
        for V in val_folds:
            holdout = V | M_outer
            folds.append(((~holdout) & off_diag, V & off_diag))
        per_rep.append(folds)

    jobs = []
    for rep_idx, folds in enumerate(per_rep):
        for fold_idx, (train_mask, val_mask) in enumerate(folds):
            for max_outer in max_outer_values:
                for r in ranks:
                    seed = 7919 * (rep_idx + 1) + 101 * fold_idx + 13 * r + max_outer
                    jobs.append((max_outer, r, train_mask, val_mask, seed))

    scores = Parallel(n_jobs=-1)(
        delayed(_one_fit)(S, r, mo, tm, vm, bounds, sd)
        for (mo, r, tm, vm, sd) in jobs
    )
    df = pd.DataFrame({
        "max_outer": [j[0] for j in jobs],
        "rank": [j[1] for j in jobs],
        "val_mse": scores,
    })
    out = {}
    for mo in max_outer_values:
        grouped = df[df["max_outer"] == mo].groupby("rank")["val_mse"]
        means = grouped.mean().reindex(ranks).values
        counts = grouped.apply(lambda x: int(np.isfinite(x).sum())).reindex(ranks).values
        sems = grouped.std().reindex(ranks).values / np.maximum(np.sqrt(counts), 1)
        out[mo] = (means, sems)
    return out, df


def analyse_one(label, key, loader, kind):
    S = loader()
    if S is None:
        log.warning("[%s] cache missing — skipping", label)
        return None
    n = S.shape[0]
    eigvals = np.sort(eigh(0.5 * (S + S.T))[0])[::-1]
    est = RankEstimator(recovery_tolerance=0.10, n_bootstrap=20, random_state=0).fit(S)
    k_cut = int(est.rank_)
    p_star = float(est.sampling_fraction_)
    p_cv = min(p_star * K_CV / (K_CV - 1), CAP)
    log.info("[%s] n=%d  k_cut=%d  p_star=%.3f  p_cv=%.3f", label, n, k_cut, p_star, p_cv)
    ranks = list(range(2, min(n - 1, N_EIGENVALUES_SHOWN + 1)))
    curves, records = cv_curves_all_max_outer(S, p_cv, MAX_OUTER_VALUES, ranks)
    records["dataset"] = key
    return dict(
        label=label, key=key, kind=kind, n=n,
        k_cut=k_cut, p_star=p_star, p_cv=p_cv,
        eigvals_top=eigvals[:N_EIGENVALUES_SHOWN],
        ranks=ranks, curves=curves, records=records,
    )


# -------------------------------------------------------------------------
# Plot.
# -------------------------------------------------------------------------

def build_figure(results, out_path):
    sns.set_theme(
        style="ticks", context="paper",
        rc={"axes.labelsize": 11, "axes.titlesize": 12,
            "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5, "axes.linewidth": 0.9,
            "xtick.major.width": 0.9, "ytick.major.width": 0.9,
            "xtick.major.size": 3.5, "ytick.major.size": 3.5,
            "axes.spines.top": False, "axes.spines.right": False,
            "font.family": "DejaVu Sans"},
    )
    n_datasets = len(results)
    fig, axes = plt.subplots(
        n_datasets, 2,
        figsize=(11.5, 2.6 * n_datasets),
        squeeze=False, sharex="col",
        gridspec_kw=dict(wspace=0.22, hspace=0.32, width_ratios=[1.0, 1.25]),
    )

    spectrum_color = "#2e5275"
    kcut_color = "#444444"
    cv_palette = {100: "#f0a868", 1000: "#9c2a2a"}

    for row, res in enumerate(results):
        ax_spec = axes[row, 0]
        ax_cv = axes[row, 1]

        # ----- Left: spectrum -----
        x = np.arange(1, len(res["eigvals_top"]) + 1)
        ax_spec.plot(x, res["eigvals_top"], "-",
                     color=spectrum_color, lw=1.6, alpha=0.95)
        ax_spec.plot(x, res["eigvals_top"], "o",
                     color=spectrum_color, ms=3.0,
                     markeredgecolor="white", markeredgewidth=0.4)
        ax_spec.axvline(res["k_cut"], color=kcut_color, lw=0.9,
                        ls=(0, (4, 3)), alpha=0.7)
        ax_spec.set_yscale("log")
        ax_spec.set_xlim(0, len(x) + 1)
        ax_spec.grid(True, axis="y", which="major", alpha=0.18, linewidth=0.5)
        ax_spec.set_axisbelow(True)
        sns.despine(ax=ax_spec)

        # Dataset name + meta as left-side y-axis label (multiline, left-aligned).
        kind_tag = "synthetic" if res["kind"] == "synthetic" else "real RSM"
        ax_spec.set_ylabel(
            f"{res['label']}\n"
            f"({kind_tag}, n={res['n']})\n"
            f"$p^\\star={res['p_star']:.2f}$,  "
            f"$p_{{\\rm cv}}={res['p_cv']:.2f}$",
            fontsize=10.5, rotation=0, ha="right", va="center",
            labelpad=24, linespacing=1.6,
        )

        if row == 0:
            ax_spec.set_title("Spectrum of $S$", fontsize=12, pad=12,
                               loc="left", fontweight="bold")
        if row == n_datasets - 1:
            ax_spec.set_xlabel("Eigenvalue index $r$")
        ax_spec.text(
            0.97, 0.94, f"$k_{{\\rm cut}}={res['k_cut']}$",
            transform=ax_spec.transAxes, ha="right", va="top",
            fontsize=9.5, color=kcut_color, fontweight="bold",
        )

        # ----- Right: CV curves with SEM bands -----
        for mo in MAX_OUTER_VALUES:
            means, sems = res["curves"][mo]
            # Clip the lower band so log-y rendering doesn't drop it; visible alpha.
            lower = np.maximum(means - sems, np.nanmin(means) * 1e-3)
            upper = means + sems
            ax_cv.fill_between(res["ranks"], lower, upper,
                                color=cv_palette[mo], alpha=0.30, linewidth=0)
            ax_cv.plot(res["ranks"], means, "-",
                       color=cv_palette[mo], lw=2.0,
                       label=f"max_outer = {mo:,}")
            i_argmin = int(np.argmin(means))
            ax_cv.scatter([res["ranks"][i_argmin]], [means[i_argmin]],
                          marker="*", s=170, color=cv_palette[mo],
                          edgecolors="white", linewidths=1.4, zorder=10)
            ax_cv.scatter([res["ranks"][i_argmin]], [means[i_argmin]],
                          marker="*", s=170, facecolor="none",
                          edgecolors=kcut_color, linewidths=0.6, zorder=11)
        ax_cv.axvline(res["k_cut"], color=kcut_color, lw=0.9,
                      ls=(0, (4, 3)), alpha=0.7)
        ax_cv.set_yscale("log")
        ax_cv.set_xlim(0, max(res["ranks"]) + 1)
        ax_cv.grid(True, axis="y", which="major", alpha=0.18, linewidth=0.5)
        ax_cv.set_axisbelow(True)
        sns.despine(ax=ax_cv)

        if row == 0:
            ax_cv.set_title(
                "5-fold CV-MSE at calibrated $p_{\\rm cv}$",
                fontsize=12, pad=12, loc="left", fontweight="bold",
            )
            ax_cv.legend(
                loc="upper right", frameon=False, fontsize=10,
                labelspacing=0.45, handlelength=1.8, borderaxespad=0.4,
            )
        if row == n_datasets - 1:
            ax_cv.set_xlabel("Rank $r$")
        ax_cv.set_ylabel("Validation MSE")

    fig.suptitle(
        "Spectrum of $S$ vs. cross-validation curve at the calibrated operating point",
        fontsize=13.5, y=0.995, fontweight="bold",
    )
    fig.text(
        0.5, 0.972,
        f"Mean ± 1 SEM across {len(REP_SEEDS)} outer reps × {K_CV} folds  "
        f"|  ★ = argmin per max_outer  |  dashed line = $k_{{\\rm cut}}$ from spectral pass",
        ha="center", va="top", fontsize=10.5, color="#444",
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s | %(levelname)s | %(message)s")

    results = []
    for label, key, loader, kind in DATASETS:
        res = analyse_one(label, key, loader, kind)
        if res is not None:
            results.append(res)

    out_path = OUTPUT_DIR / "spectrum_vs_cv.png"
    build_figure(results, out_path)
    log.info("saved %s", out_path)

    # Save raw records so we can re-plot cheaply (no re-fit).
    all_records = pd.concat([res["records"] for res in results], ignore_index=True)
    all_records.to_csv(OUTPUT_DIR / "records.csv", index=False)

    summary = [
        f"# spectrum_vs_cv  |  k_cv={K_CV}, max_outer ∈ {MAX_OUTER_VALUES}, "
        f"rho={RHO}, max_inner={MAX_INNER}, tol={TOL}, "
        f"{len(REP_SEEDS)} outer reps",
        "",
    ]
    for res in results:
        argmins = {mo: res["ranks"][int(np.argmin(res["curves"][mo][0]))]
                    for mo in MAX_OUTER_VALUES}
        line = (
            f"{res['label']:<22s} n={res['n']:>4d}  k_cut={res['k_cut']:>2d}  "
            f"p_star={res['p_star']:.3f}  p_cv={res['p_cv']:.3f}  argmin: "
            + "  ".join(f"mo={mo}:{argmins[mo]:>2d}" for mo in MAX_OUTER_VALUES)
        )
        summary.append(line)
    (OUTPUT_DIR / "spectrum_vs_cv_summary.txt").write_text("\n".join(summary) + "\n")
    log.info("\n%s", "\n".join(summary))


if __name__ == "__main__":
    main()
