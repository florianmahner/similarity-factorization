"""Simulation figure (Fig. 2), 180mm Nature double-column width.

Layout (2x2, left=1/3, right=2/3):
  Row 1: a) Generative model (placeholder)  |  b) Reconstruction + Factor alignment (2 panels)
  Row 2: c) Imputation + Factor recovery    |  d) Rank detection 2x3 grid

Data sources (experiments/analyses/simulation/ only):
  - imputation/outputs/imputation.csv
  - interpretability/outputs/interpretability.csv
  - rank_detection/outputs/rank_detection.csv

Usage:
    poetry run python experiments/figures/plot_simulation/plot.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator
from scipy.interpolate import make_interp_spline

from src.colors import (
    GRAY, GRAY_LIGHT, GRAY_PALE, INDIGO, ROSE, TEAL, lighten, setup_style,
)
from src.utils.figure_theme import despine

from experiments.analyses.simulation.rank_detection.plot import (
    METHODS,
    _draw_panel,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SIM_DIR = PROJECT_ROOT / "experiments" / "analyses" / "simulation"
OUTPUT = Path(__file__).resolve().parent / "outputs"

FIG_WIDTH_MM = 180
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4


def _nature_rc(font_size: float) -> dict:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "lines.linewidth": 0.8,
        "lines.markersize": 3,
        "lines.markeredgewidth": 0.3,
    }


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / f"{name}.pdf"
    plt.rcParams["savefig.bbox"] = "standard"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_imputation() -> pd.DataFrame:
    return pd.read_csv(SIM_DIR / "imputation" / "outputs" / "imputation.csv")


def _load_interpretability() -> pd.DataFrame:
    return pd.read_csv(SIM_DIR / "interpretability" / "outputs" / "interpretability.csv")


def _load_rank_detection() -> pd.DataFrame:
    return pd.read_csv(SIM_DIR / "rank_detection" / "outputs" / "rank_detection.csv")


# ---------------------------------------------------------------------------
# Panel a: Generative model (placeholder for manual illustration)
# ---------------------------------------------------------------------------

def _panel_generative(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.text(5, 0.5, "(Generative model\nillustration)", ha="center", va="center",
            fontsize=5, color=GRAY_LIGHT, style="italic")


# ---------------------------------------------------------------------------
# Panel b left: Reconstruction R2 vs complexity (dual-axis with factor alignment)
# ---------------------------------------------------------------------------

K_THEORY = 30
N_THEORY = 300
ALPHAS_THEORY = np.logspace(-1, 3, 150)


def _panel_theory_left(ax: plt.Axes, fs: float) -> None:
    """Entropy + sparsity dual-axis plot."""
    n_samples = 50
    rng = np.random.default_rng(42)
    max_entropy = np.log(K_THEORY)

    entropy_mean = np.zeros(len(ALPHAS_THEORY))
    sparsity_mean = np.zeros(len(ALPHAS_THEORY))

    for i, alpha in enumerate(ALPHAS_THEORY):
        w = rng.dirichlet([alpha] * K_THEORY, size=n_samples * N_THEORY)
        ent = -np.sum(w * np.log(w + 1e-10), axis=1) / max_entropy
        entropy_mean[i] = ent.reshape(n_samples, N_THEORY).mean(axis=1).mean()
        l1 = np.sum(np.abs(w), axis=1)
        l2 = np.sqrt(np.sum(w**2, axis=1))
        hoyer = (np.sqrt(K_THEORY) - l1 / (l2 + 1e-10)) / (np.sqrt(K_THEORY) - 1)
        sparsity_mean[i] = hoyer.reshape(n_samples, N_THEORY).mean(axis=1).mean()

    ax.axhline(1.0, color=GRAY_LIGHT, linestyle="--", linewidth=0.5, zorder=0)
    ax.plot(ALPHAS_THEORY, entropy_mean, "-", color=ROSE, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"Complexity $\alpha$")
    ax.set_ylabel("Normalized entropy", color=ROSE)
    ax.set_xlim(0.08, 1200)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(ALPHAS_THEORY, sparsity_mean, "-", color=TEAL, linewidth=0.8)
    ax2.set_ylabel("Hoyer sparsity", color=TEAL)
    ax2.set_ylim(0, 1.08)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.spines["top"].set_visible(False)


def _load_srf_performance() -> pd.DataFrame:
    return pd.read_csv(SIM_DIR / "interpretability" / "outputs" / "srf_performance.csv")


def _panel_factor_alignment(ax: plt.Axes, df_perf: pd.DataFrame, fs: float) -> None:
    """Factor alignment (r) vs complexity at multiple SNR levels."""
    colors = {1.0: ROSE, 0.7: TEAL, 0.5: INDIGO, 0.3: GRAY}

    for snr in sorted(df_perf["snr"].unique(), reverse=True):
        sub = df_perf[df_perf["snr"] == snr].sort_values("alpha")
        label = "No noise" if snr == 1.0 else f"SNR = {snr}"
        ax.plot(sub["alpha"], sub["mean"], "-", color=colors.get(snr, GRAY),
                linewidth=0.8, label=label)
        ax.fill_between(sub["alpha"],
                        sub["mean"] - sub["sem"], sub["mean"] + sub["sem"],
                        color=colors.get(snr, GRAY), alpha=0.15, linewidth=0)

    ax.set_xscale("log")
    ax.set_xlabel(r"Complexity $\alpha$")
    ax.set_ylabel("Factor alignment (r)")
    ax.set_xlim(0.08, 1200)
    ax.set_ylim(-0.02, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    despine(ax)
    ax.legend(loc="lower left", frameon=False, fontsize=fs - 1)


# ---------------------------------------------------------------------------
# Panel c: Imputation (top) + Factor recovery (bottom)
# ---------------------------------------------------------------------------

def _panel_imputation(ax: plt.Axes, df: pd.DataFrame, fs: float) -> None:
    df_plot = df[df["method"] != "Mean"].copy()
    ann = fs - 0.5

    agg = (
        df_plot.groupby(["obs_per_dof", "method"])["r2"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    agg = agg.dropna()

    colors = {"SRF": ROSE, "KNN": TEAL, "Median": INDIGO}

    ax.axvspan(0, 1.0, color=GRAY_PALE, zorder=0)
    ax.axvline(x=1.0, color=GRAY_LIGHT, linestyle="--", linewidth=0.5, zorder=1)
    ax.text(0.9, 50, "Under-\ndetermined", rotation=90, va="center", ha="center",
            fontsize=ann, color=GRAY)

    for method in ["SRF", "KNN", "Median"]:
        data = agg[agg["method"] == method].sort_values("obs_per_dof")
        if data.empty:
            continue
        c = colors[method]
        x = data["obs_per_dof"].values
        y = data["mean"].values * 100
        y_lo = (data["mean"] - data["sem"]).values * 100
        y_hi = (data["mean"] + data["sem"]).values * 100

        x_log = np.log(x)
        x_smooth_log = np.linspace(x_log.min(), x_log.max(), 200)
        x_smooth = np.exp(x_smooth_log)

        spl = make_interp_spline(x_log, y, k=3)
        spl_lo = make_interp_spline(x_log, y_lo, k=3)
        spl_hi = make_interp_spline(x_log, y_hi, k=3)

        ax.plot(x_smooth, spl(x_smooth_log), label=method, color=c, zorder=3)
        ax.fill_between(x_smooth, spl_lo(x_smooth_log), spl_hi(x_smooth_log),
                        color=c, alpha=0.2, linewidth=0, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("Samples per model parameter")
    ax.set_ylabel("Held-out variance explained (%)")
    ax.set_xlim(0.7, 25)
    ax.set_ylim(0, 105)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1", "2", "5", "10", "20"])
    despine(ax)
    ax.legend(loc="lower right", frameon=False)


def _load_imputation_recovery() -> pd.DataFrame:
    return pd.read_csv(SIM_DIR / "interpretability" / "outputs" / "imputation_recovery.csv")


def _panel_factor_recovery_missing(ax: plt.Axes, df: pd.DataFrame, fs: float) -> None:
    """Factor recovery (r) vs samples per model parameter."""
    ann = fs - 0.5
    ann = fs - 0.5
    colors = {"SRF": ROSE, "KNN": TEAL, "Median": INDIGO}

    ax.axvspan(0, 1.0, color=GRAY_PALE, zorder=0)
    ax.axvline(x=1.0, color=GRAY_LIGHT, linestyle="--", linewidth=0.5, zorder=1)

    for method in ["SRF", "KNN", "Median"]:
        data = df[df["method"] == method].sort_values("obs_per_dof")
        if data.empty:
            continue
        c = colors[method]
        x = data["obs_per_dof"].values
        y = data["mean"].values
        y_lo = (data["mean"] - data["sem"]).values
        y_hi = (data["mean"] + data["sem"]).values

        x_log = np.log(x)
        x_smooth_log = np.linspace(x_log.min(), x_log.max(), 200)
        x_smooth = np.exp(x_smooth_log)

        spl = make_interp_spline(x_log, y, k=3)
        spl_lo = make_interp_spline(x_log, y_lo, k=3)
        spl_hi = make_interp_spline(x_log, y_hi, k=3)

        ax.plot(x_smooth, spl(x_smooth_log), label=method, color=c, zorder=3)
        ax.fill_between(x_smooth, spl_lo(x_smooth_log), spl_hi(x_smooth_log),
                        color=c, alpha=0.2, linewidth=0, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("Samples per model parameter")
    ax.set_ylabel("Factor alignment (r)")
    ax.set_xlim(0.7, 25)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1", "2", "5", "10", "20"])
    despine(ax)
    ax.legend(loc="lower right", frameon=False)


# ---------------------------------------------------------------------------
# Panel d: Rank detection 2x3 grid
# ---------------------------------------------------------------------------

def _panel_rank_detection(
    axes_top, axes_bot, df: pd.DataFrame, fs: float,
    cbar_ax_alpha=None, cbar_ax_snr=None,
) -> None:
    alphas = sorted(df["alpha"].unique())
    snrs = sorted(df["snr"].unique())

    cmap_alpha = mcolors.LinearSegmentedColormap.from_list(
        "complexity", [lighten(ROSE, 0.7), ROSE, INDIGO], N=256,
    )
    norm_alpha = mcolors.LogNorm(vmin=min(alphas), vmax=max(alphas))

    for i, (label, col) in enumerate(METHODS):
        _draw_panel(axes_top[i], df, col, df["alpha"].values, cmap_alpha, norm_alpha,
                    label, show_ylabel=(i == 0), jitter_seed=42)
        axes_top[i].set_title(label, fontsize=fs + 1, pad=3)

    if cbar_ax_alpha is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap_alpha, norm=norm_alpha)
        cb = plt.colorbar(sm, cax=cbar_ax_alpha)
        cb.set_label(r"$\alpha$", fontsize=fs)
        cb.ax.tick_params(labelsize=fs - 1)

    cmap_snr = mcolors.LinearSegmentedColormap.from_list(
        "snr", [lighten(TEAL, 0.7), TEAL, INDIGO], N=256,
    )
    norm_snr = mcolors.Normalize(vmin=min(snrs), vmax=max(snrs))

    for i, (label, col) in enumerate(METHODS):
        _draw_panel(axes_bot[i], df, col, df["snr"].values, cmap_snr, norm_snr,
                    "", show_ylabel=(i == 0), jitter_seed=99)

    if cbar_ax_snr is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap_snr, norm=norm_snr)
        cb = plt.colorbar(sm, cax=cbar_ax_snr)
        cb.set_label("SNR", fontsize=fs)
        cb.ax.tick_params(labelsize=fs - 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_style()
    fs = 5
    plt.rcParams.update(_nature_rc(fs))

    imputation = _load_imputation()
    recovery = _load_imputation_recovery()
    srf_perf = _load_srf_performance()
    rank_det = _load_rank_detection()

    ROW_H = 40 / 25.4  # 40mm per row
    fig = plt.figure(figsize=(FIG_WIDTH_IN, 2 * ROW_H + 0.5))

    # Outer grid: 2 rows x 2 cols. Left ~28%, right ~72%.
    # Equal height rows so b and d match.
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[0.28, 0.72],
        height_ratios=[1, 1],
        hspace=0.45, wspace=0.20,
    )

    # --- Row 1 left: a) Generative model placeholder ---
    ax_gen = fig.add_subplot(gs[0, 0])

    # --- Row 1 right: b) Two theory panels side by side ---
    gs_b = gs[0, 1].subgridspec(1, 2, wspace=0.55)
    ax_b_left = fig.add_subplot(gs_b[0, 0])
    ax_b_right = fig.add_subplot(gs_b[0, 1])

    # --- Row 2 left: c) Imputation (top) + Factor recovery (bottom) ---
    gs_c = gs[1, 0].subgridspec(2, 1, hspace=0.55)
    ax_imp = fig.add_subplot(gs_c[0])
    ax_fac = fig.add_subplot(gs_c[1])

    # --- Row 2 right: d) Rank detection 2x3 + colorbars ---
    # 3 plot cols + 1 narrow colorbar col
    gs_d = gs[1, 1].subgridspec(2, 4, hspace=0.40, wspace=0.08,
                                 width_ratios=[1, 1, 1, 0.06])
    axes_rank_top = [fig.add_subplot(gs_d[0, c]) for c in range(3)]
    axes_rank_bot = [fig.add_subplot(gs_d[1, c]) for c in range(3)]
    cbar_ax_alpha = fig.add_subplot(gs_d[0, 3])
    cbar_ax_snr = fig.add_subplot(gs_d[1, 3])

    # Panel labels
    for label, ax in [("a", ax_gen), ("b", ax_b_left),
                       ("c", ax_imp), ("d", axes_rank_top[0])]:
        x_off = -0.05 if ax is ax_gen else -0.25
        ax.text(x_off, 1.15, label, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="right")

    # Fill panels
    _panel_generative(ax_gen)
    _panel_theory_left(ax_b_left, fs)
    _panel_factor_alignment(ax_b_right, srf_perf, fs)
    _panel_imputation(ax_imp, imputation, fs)
    _panel_factor_recovery_missing(ax_fac, recovery, fs)
    _panel_rank_detection(axes_rank_top, axes_rank_bot, rank_det, fs,
                          cbar_ax_alpha=cbar_ax_alpha, cbar_ax_snr=cbar_ax_snr)

    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.06, top=0.96)
    _save(fig, "simulation")


if __name__ == "__main__":
    main()
