"""Reproduce the sandbox spectrum-vs-CV plot using the new pysrf public API.

If the figure matches
``sandbox/rank_estimation/cv_diagnostics/spectrum_vs_cv/outputs/spectrum_vs_cv.png``,
the integration of ``estimate_rank`` and ``cross_val_score`` into
``third_party/pysrf`` is verified end-to-end.
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
from scipy.linalg import eigh
from scipy.spatial.distance import cdist

from pysrf import cross_val_score, estimate_rank
from src.utils import get_output_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "update_pysrf" / "src"))
from sbm.datasets import clean_weighted_sbm  # noqa: E402
from _synthetic_extras import make_power_law  # noqa: E402


log = logging.getLogger(__name__)
OUTPUT_DIR = get_output_dir()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RSM_CACHE = PROJECT_ROOT / "experiments" / "figures" / "plot_embeddings" / "outputs" / ".cache"

K_CV = 5
N_REPEATS = 5
MAX_OUTER_VALUES = [100, 1000]
N_EIGENVALUES_SHOWN = 30
SRF_KWARGS = {"rho": 3.0, "max_inner": 30, "tol": 0.0}


# -------------------------------------------------------------------------
# Datasets (identical to the original sandbox script).
# -------------------------------------------------------------------------

def _load_sbm():
    return clean_weighted_sbm(n=200, K=10, mu_in=1.0, mu_out=0.0,
                               sigma=0.3, seed=42)["S"]


def _load_powerlaw():
    s_raw, _ = make_power_law(n=400, k_signal=10, alpha=1.0, snr=5.0, seed=42)
    return s_raw - s_raw.min() + 0.01


def _load_cached(name):
    path = RSM_CACHE / f"sim_{name}.npy"
    return np.load(path) if path.exists() else None


DATASETS = [
    ("SBM",                "sbm",                _load_sbm,                                "synthetic"),
    ("Powerlaw",           "powerlaw",           _load_powerlaw,                           "synthetic"),
    ("Mur92",              "mur92",              lambda: _load_cached("mur92"),            "real"),
    ("Peterson-animals",   "peterson-animals",   lambda: _load_cached("peterson-animals"), "real"),
    ("Peterson-various",   "peterson-various",   lambda: _load_cached("peterson-various"), "real"),
]


# -------------------------------------------------------------------------
# Per-dataset run.
# -------------------------------------------------------------------------

def analyse_one(label, key, loader, kind):
    s = loader()
    if s is None:
        log.warning("[%s] cache missing — skipping", label)
        return None

    eigvals = np.sort(eigh(0.5 * (s + s.T))[0])[::-1][:N_EIGENVALUES_SHOWN]

    est = estimate_rank(s)
    k_cut = est.rank
    p_star = est.sampling_fraction
    p_cv_for_label = min(p_star * K_CV / (K_CV - 1), 0.95)
    log.info("[%s] n=%d  k_cut=%d  p_star=%.3f  p_cv(label)=%.3f",
             label, s.shape[0], k_cut, p_star, p_cv_for_label)

    ranks = list(range(2, min(s.shape[0] - 1, N_EIGENVALUES_SHOWN + 1)))
    curves = {}
    records = []
    for mo in MAX_OUTER_VALUES:
        # Vary random_state per max_outer so the two budgets get independent
        # SRF initializations. n_repeats=5 matches the original sandbox's
        # 5-rep × 5-fold = 25 cells per (rank, max_outer).
        curve = cross_val_score(
            s, ranks=ranks, sampling_fraction=p_star,
            n_folds=K_CV, n_repeats=N_REPEATS,
            random_state=mo, srf_kwargs={**SRF_KWARGS, "max_outer": mo},
        )
        curve = curve.assign(dataset=key, max_outer=mo)
        records.append(curve)
        grouped = curve.groupby("rank")["val_mse"]
        means = grouped.mean().reindex(ranks).values
        sems = grouped.std().reindex(ranks).values / np.maximum(
            np.sqrt(grouped.count().reindex(ranks).values), 1
        )
        curves[mo] = (means, sems)
        log.info("   max_outer=%-4d  argmin=%d  V-MSE@min=%.4e",
                 mo, ranks[int(np.argmin(means))], float(np.min(means)))

    return dict(label=label, key=key, kind=kind, n=s.shape[0],
                k_cut=k_cut, p_star=p_star, p_cv=p_cv_for_label,
                eigvals_top=eigvals, ranks=ranks, curves=curves,
                records=pd.concat(records, ignore_index=True))


# -------------------------------------------------------------------------
# Plot (matches the original sandbox layout).
# -------------------------------------------------------------------------

def build_figure(results, out_path):
    sns.set_theme(
        style="ticks", context="paper",
        rc={"axes.labelsize": 11, "axes.titlesize": 12,
            "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5, "axes.linewidth": 0.9,
            "xtick.major.width": 0.9, "ytick.major.width": 0.9,
            "axes.spines.top": False, "axes.spines.right": False,
            "font.family": "DejaVu Sans"},
    )
    n_datasets = len(results)
    fig, axes = plt.subplots(
        n_datasets, 2, figsize=(11.5, 2.6 * n_datasets),
        squeeze=False, sharex="col",
        gridspec_kw=dict(wspace=0.22, hspace=0.32, width_ratios=[1.0, 1.25]),
    )

    spectrum_color = "#2e5275"
    kcut_color = "#444444"
    cv_palette = {100: "#f0a868", 1000: "#9c2a2a"}

    for row, res in enumerate(results):
        ax_spec, ax_cv = axes[row, 0], axes[row, 1]

        # Left: spectrum.
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

        kind_tag = "synthetic" if res["kind"] == "synthetic" else "real RSM"
        ax_spec.set_ylabel(
            f"{res['label']}\n({kind_tag}, n={res['n']})\n"
            f"$p^\\star={res['p_star']:.2f}$,  $p_{{\\rm cv}}={res['p_cv']:.2f}$",
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

        # Right: CV curves with SEM bands.
        for mo in MAX_OUTER_VALUES:
            means, sems = res["curves"][mo]
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
            ax_cv.set_title("5-fold CV-MSE at calibrated $p_{\\rm cv}$",
                             fontsize=12, pad=12, loc="left", fontweight="bold")
            ax_cv.legend(loc="upper right", frameon=False, fontsize=10,
                          labelspacing=0.45, handlelength=1.8, borderaxespad=0.4)
        if row == n_datasets - 1:
            ax_cv.set_xlabel("Rank $r$")
        ax_cv.set_ylabel("Validation MSE")

    fig.suptitle(
        "Reproduction via pysrf.estimate_rank + pysrf.cross_val_score",
        fontsize=13.5, y=0.995, fontweight="bold",
    )
    fig.text(
        0.5, 0.972,
        f"Mean ± 1 SEM across {K_CV} folds  "
        f"|  ★ = argmin per max_outer  |  dashed line = $k_{{\\rm cut}}$",
        ha="center", va="top", fontsize=10.5, color="#444",
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s | %(levelname)s | %(message)s")

    results = [r for r in (analyse_one(*d) for d in DATASETS) if r is not None]

    out_path = OUTPUT_DIR / "spectrum_vs_cv.png"
    build_figure(results, out_path)
    log.info("saved %s", out_path)

    all_records = pd.concat([r["records"] for r in results], ignore_index=True)
    all_records.to_csv(OUTPUT_DIR / "records.csv", index=False)
    log.info("saved %s", OUTPUT_DIR / "records.csv")

    lines = [f"# pysrf reproduction  |  k_cv={K_CV}, max_outer ∈ {MAX_OUTER_VALUES}, "
              f"srf_kwargs={SRF_KWARGS}", ""]
    for res in results:
        argmins = {mo: res["ranks"][int(np.argmin(res["curves"][mo][0]))]
                    for mo in MAX_OUTER_VALUES}
        lines.append(
            f"{res['label']:<22s} n={res['n']:>4d}  k_cut={res['k_cut']:>2d}  "
            f"p_star={res['p_star']:.3f}  argmin: "
            + "  ".join(f"mo={mo}:{argmins[mo]:>2d}" for mo in MAX_OUTER_VALUES)
        )
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n")
    log.info("\n%s", "\n".join(lines))


if __name__ == "__main__":
    main()
