"""
Create all simulation experiment plots.

Usage:
    poetry run python experiments/simulation/plot.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.utils.figure_theme import (
    CMAP,
    GRAY,
    apply_theme,
    create_figure,
    despine,
    save_figure,
)

PROJECT_ROOT = Path(__file__).parents[2]
SIMULATION_DIR = PROJECT_ROOT / "outputs/experiments/simulation"
DATA_DIR = SIMULATION_DIR / "data"
PLOT_DIR = SIMULATION_DIR / "plots"

# Simulation parameters (consistent across all plots)
N = 30
K = 3
ALPHAS = np.logspace(-1, 2, 25)  # 0.1 to 100


def _plot_metric_by_alpha(
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    output_dir: Path,
    filename: str,
) -> None:
    """Scatter plot: x=alpha (log), y=metric, color/size=SNR."""
    fig, ax = create_figure("single")

    snr_values = sorted(df["snr"].unique())
    theme_blue = CMAP[1]
    light_blue = "#d6eaf8"
    cmap = LinearSegmentedColormap.from_list("blues", [light_blue, theme_blue])

    for snr in snr_values:
        subset = df[df["snr"] == snr].sort_values("alpha")
        snr_norm = (snr - min(snr_values)) / (max(snr_values) - min(snr_values))
        ax.scatter(
            subset["alpha"],
            subset[y_col],
            c=[cmap(snr_norm)],
            s=30 + snr_norm * 70,
            edgecolors="white",
            linewidths=0.5,
            label=f"{snr}",
            zorder=5 + int(snr * 10),
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(ylabel)

    alpha_vals = sorted(df["alpha"].unique())
    ax.set_xticks(alpha_vals)
    ax.set_xticklabels([str(a) for a in alpha_vals])

    despine(ax)
    ax.legend(title="SNR", frameon=False, fontsize=8)

    save_figure(fig, output_dir / filename)


def _plot_rank_scatter(
    df: pd.DataFrame,
    color_col: str,
    legend_title: str,
    output_dir: Path,
    filename: str,
) -> None:
    """Scatter plot: true rank vs selected rank."""
    rng = np.random.default_rng(42)
    jitter = 0.5
    df = df.copy()
    df["true_jit"] = df["true_rank"] + rng.uniform(-jitter, jitter, len(df))
    df["sel_jit"] = df["selected_rank"] + rng.uniform(-jitter, jitter, len(df))

    fig, ax = create_figure("single")

    color_vals = sorted(df[color_col].unique())
    markers = ["o", "s", "^", "D"]
    min_val = min(df["true_rank"].min(), df["selected_rank"].min())
    max_val = max(df["true_rank"].max(), df["selected_rank"].max())
    lims = [min_val - 1, max_val + 1]

    for i, val in enumerate(color_vals):
        subset = df[df[color_col] == val]
        ax.scatter(
            subset["true_jit"],
            subset["sel_jit"],
            c=CMAP[i % len(CMAP)],
            s=30,
            alpha=0.6,
            marker=markers[i % len(markers)],
            edgecolors="white",
            linewidths=0.3,
            label=f"{val}",
        )

    ax.plot(lims, lims, "--", color=GRAY["light"], lw=1, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True rank")
    ax.set_ylabel("Selected rank")
    despine(ax)
    ax.legend(title=legend_title, frameon=False, fontsize=7, loc="lower right")

    save_figure(fig, output_dir / filename)


def _plot_imputation(df: pd.DataFrame, output_dir: Path) -> None:
    """Line plot: imputation R² by obs/dof with underdetermined region."""
    df_plot = df[df["method"] != "Mean"].copy()
    if "obs_per_dof" not in df_plot.columns:
        return

    agg = (
        df_plot.groupby(["obs_per_dof", "method"])["r2"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    agg = agg.dropna()

    # Colors from CMAP: red, blue, green
    colors = {"SRF": CMAP[0], "KNN": CMAP[1], "Median": CMAP[2]}

    fig, ax = create_figure("single")

    # Underdetermined region
    ax.axvspan(0, 1.0, color=GRAY["faint"], zorder=0)
    ax.axvline(x=1.0, color=GRAY["light"], linestyle="--", linewidth=0.8, zorder=1)
    ax.text(
        0.9,
        50,
        "Underdetermined",
        rotation=90,
        va="center",
        ha="center",
        fontsize=7,
        color=GRAY["medium"],
    )

    for method in ["SRF", "KNN", "Median"]:
        data = agg[agg["method"] == method].sort_values("obs_per_dof")
        if data.empty:
            continue
        c = colors[method]
        lw = 1.8 if method == "SRF" else 1.2

        ax.plot(
            data["obs_per_dof"],
            data["mean"] * 100,
            label=method,
            color=c,
            lw=lw,
            zorder=3,
        )
        ax.fill_between(
            data["obs_per_dof"],
            (data["mean"] - data["sem"]) * 100,
            (data["mean"] + data["sem"]) * 100,
            color=c,
            alpha=0.2,
            linewidth=0,
            zorder=2,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Samples per degree of freedom")
    ax.set_ylabel("Held-out variance explained (%)")
    ax.set_xlim(0.7, 25)
    ax.set_ylim(0, 105)

    # Set major ticks only at desired positions, no minor ticks
    from matplotlib.ticker import FixedLocator, NullLocator

    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1", "2", "5", "10", "20"])

    despine(ax)
    ax.legend(loc="lower right", frameon=False)

    save_figure(fig, output_dir / "imputation_r2.pdf")


def _plot_alpha_factors(output_dir: Path) -> None:
    """Plot factor matrices W for different alpha values (vector PDF)."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from utils.simulation import simulation_dirichlet

    alphas = [0.1, 1.0, 10.0, 100.0]
    rng = np.random.default_rng(42)

    for alpha in alphas:
        factors = simulation_dirichlet(N, K, alpha, rng)
        fig, ax = create_figure("single")

        x = np.arange(K + 1)
        y = np.arange(N + 1)
        im = ax.pcolormesh(
            x, y, factors, cmap="RdYlBu_r", vmin=0, vmax=1, rasterized=False
        )
        ax.set_xlim(0, K)
        ax.set_ylim(N, 0)
        ax.set_aspect("auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=7)

        alpha_str = str(alpha).replace(".", "_")
        save_figure(fig, output_dir / f"alpha_factors_{alpha_str}.pdf")


def _plot_alpha_rsm(output_dir: Path) -> None:
    """Plot RSMs for different alpha values as individual vector PDFs."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from utils.simulation import simulation_dirichlet

    alphas = [0.1, 1.0, 10.0, 100.0]
    rng = np.random.default_rng(42)

    for alpha in alphas:
        factors = simulation_dirichlet(N, K, alpha, rng)
        rsm = factors @ factors.T

        fig, ax = create_figure("single")
        vmin, vmax = rsm.min(), rsm.max()

        x = np.arange(N + 1)
        y = np.arange(N + 1)
        im = ax.pcolormesh(
            x, y, rsm, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, rasterized=False
        )
        ax.set_xlim(0, N)
        ax.set_ylim(N, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(3))

        alpha_str = str(alpha).replace(".", "_")
        save_figure(fig, output_dir / f"alpha_rsm_{alpha_str}.pdf")


def _hoyer_sparsity(x: np.ndarray) -> float:
    """Hoyer's sparsity measure. 0 = uniform, 1 = maximally sparse."""
    n = len(x)
    l1 = np.sum(np.abs(x))
    l2 = np.sqrt(np.sum(x**2))
    if l2 == 0:
        return 0.0
    return (np.sqrt(n) - l1 / l2) / (np.sqrt(n) - 1)


def _plot_dirichlet_properties(output_dir: Path) -> None:
    """Plot entropy and Hoyer sparsity vs alpha (dual y-axis)."""
    from matplotlib.ticker import MultipleLocator

    from scipy.stats import entropy

    from utils.simulation import simulation_dirichlet

    n_samples = 50
    rng = np.random.default_rng(42)
    max_entropy = np.log(K)

    entropy_values = np.zeros((len(ALPHAS), n_samples))
    sparsity_values = np.zeros((len(ALPHAS), n_samples))

    for i, alpha in enumerate(ALPHAS):
        for j in range(n_samples):
            factors = simulation_dirichlet(N, K, alpha, rng)
            row_entropies = [entropy(row + 1e-10) for row in factors]
            entropy_values[i, j] = np.mean(row_entropies) / max_entropy
            row_sparsities = [_hoyer_sparsity(row) for row in factors]
            sparsity_values[i, j] = np.mean(row_sparsities)

    fig, ax1 = create_figure("single", pad_left=0.6, pad_right=0.6)

    # Shared axis settings
    tick_positions = [0, 0.25, 0.5, 0.75, 1.0]

    # Reference line for uniform distribution (entropy = 1)
    ax1.axhline(1.0, color=GRAY["light"], linestyle="--", linewidth=0.8, zorder=0)
    ax1.text(0.12, 0.96, "uniform", fontsize=7, color=GRAY["medium"], ha="left")

    # Left axis: Normalized entropy (increases with alpha)
    entropy_mean = entropy_values.mean(axis=1)
    entropy_sem = entropy_values.std(axis=1) / np.sqrt(n_samples)

    ax1.plot(
        ALPHAS, entropy_mean, "o-", color=CMAP[1], markersize=4, linewidth=1.5
    )
    ax1.fill_between(
        ALPHAS, entropy_mean - entropy_sem, entropy_mean + entropy_sem,
        color=CMAP[1], alpha=0.2, linewidth=0
    )
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel("Normalized entropy", color=CMAP[1])
    ax1.set_xlim(0.08, 120)
    ax1.set_ylim(0, 1.08)
    ax1.set_yticks(tick_positions)
    ax1.yaxis.set_minor_locator(MultipleLocator(0.125))
    despine(ax1)

    # Right axis: Hoyer sparsity (decreases with alpha)
    ax2 = ax1.twinx()
    sparsity_mean = sparsity_values.mean(axis=1)
    sparsity_sem = sparsity_values.std(axis=1) / np.sqrt(n_samples)

    ax2.plot(
        ALPHAS, sparsity_mean, "o-", color=CMAP[0], markersize=4, linewidth=1.5
    )
    ax2.fill_between(
        ALPHAS, sparsity_mean - sparsity_sem, sparsity_mean + sparsity_sem,
        color=CMAP[0], alpha=0.2, linewidth=0
    )
    ax2.set_ylabel("Hoyer sparsity", color=CMAP[0])
    ax2.set_ylim(0, 1.08)
    ax2.set_yticks(tick_positions)
    ax2.yaxis.set_minor_locator(MultipleLocator(0.125))
    ax2.spines["top"].set_visible(False)

    save_figure(fig, output_dir / "dirichlet_properties.pdf")


def _plot_srf_performance(output_dir: Path) -> None:
    """Plot reconstruction R² and factor recovery vs alpha (dual y-axis)."""
    from joblib import Parallel, delayed
    from matplotlib.ticker import MultipleLocator
    from pysrf import SRF
    from scipy.optimize import linear_sum_assignment

    from utils.simulation import simulation_dirichlet

    n_reps = 50

    def normalize(w):
        return w / (w.sum(axis=1, keepdims=True) + 1e-10)

    def align_and_correlate(true_w, learned_w):
        sim = normalize(true_w).T @ normalize(learned_w)
        _, col_ind = linear_sum_assignment(-sim)
        aligned = learned_w[:, col_ind]
        corrs = []
        for i in range(true_w.shape[1]):
            c = np.corrcoef(true_w[:, i], aligned[:, i])[0, 1]
            if not np.isnan(c):
                corrs.append(abs(c))
        return np.mean(corrs) if corrs else 0.0

    def run_single(alpha, seed):
        rng = np.random.default_rng(seed)
        factors = simulation_dirichlet(N, K, alpha, rng)
        similarity = factors @ factors.T
        srf = SRF(rank=K, random_state=seed)
        srf.fit(similarity)
        recon = srf.reconstruct()
        r2 = float(np.corrcoef(similarity.flatten(), recon.flatten())[0, 1] ** 2)
        factor_corr = align_and_correlate(factors, srf.w_)
        return {"alpha": alpha, "recon_r2": r2, "factor_corr": factor_corr}

    results = Parallel(n_jobs=-1)(
        delayed(run_single)(alpha, rep) for alpha in ALPHAS for rep in range(n_reps)
    )

    df = pd.DataFrame(results)
    df_recon = (
        df.groupby("alpha")
        .agg(mean=("recon_r2", "mean"), sem=("recon_r2", "sem"))
        .reset_index()
    )
    df_factor = (
        df.groupby("alpha")
        .agg(mean=("factor_corr", "mean"), sem=("factor_corr", "sem"))
        .reset_index()
    )

    fig, ax1 = create_figure("single", pad_left=0.6, pad_right=0.6)

    # Shared axis settings
    tick_positions = [0, 0.25, 0.5, 0.75, 1.0]

    # Chance level reference line for factor recovery
    ax1.axhline(0.07, color=GRAY["light"], linestyle="--", linewidth=0.8, zorder=0)
    ax1.text(0.12, 0.10, "chance", fontsize=7, color=GRAY["medium"], ha="left")

    # Left axis: Reconstruction R²
    ax1.plot(
        df_recon["alpha"], df_recon["mean"], "o-",
        color=CMAP[1], markersize=4, linewidth=1.5
    )
    ax1.fill_between(
        df_recon["alpha"],
        df_recon["mean"] - df_recon["sem"],
        df_recon["mean"] + df_recon["sem"],
        color=CMAP[1], alpha=0.2, linewidth=0
    )
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel(r"Reconstruction $R^2$", color=CMAP[1])
    ax1.set_xlim(0.08, 120)
    ax1.set_ylim(0, 1.08)
    ax1.set_yticks(tick_positions)
    ax1.yaxis.set_minor_locator(MultipleLocator(0.125))
    despine(ax1)

    # Right axis: Factor recovery
    ax2 = ax1.twinx()
    ax2.plot(
        df_factor["alpha"], df_factor["mean"], "o-",
        color=CMAP[0], markersize=4, linewidth=1.5
    )
    ax2.fill_between(
        df_factor["alpha"],
        df_factor["mean"] - df_factor["sem"],
        df_factor["mean"] + df_factor["sem"],
        color=CMAP[0], alpha=0.2, linewidth=0
    )
    ax2.set_ylabel("Factor recovery", color=CMAP[0])
    ax2.set_ylim(0, 1.08)
    ax2.set_yticks(tick_positions)
    ax2.yaxis.set_minor_locator(MultipleLocator(0.125))
    ax2.spines["top"].set_visible(False)

    save_figure(fig, output_dir / "srf_performance.pdf")


def _plot_rank_detection_by_alpha(df: pd.DataFrame, output_dir: Path) -> None:
    """Median selected rank with IQR ribbon, by alpha."""
    df = df[df["alpha"] != 5.0].copy()  # Keep 0.1, 1.0, 10.0

    summary = df.groupby(["true_rank", "alpha"])["selected_rank"].agg(
        ["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    ).reset_index()
    summary.columns = ["true_rank", "alpha", "median", "q25", "q75"]

    fig, ax = create_figure("single")

    alpha_colors = {0.1: CMAP[0], 1.0: CMAP[1], 10.0: CMAP[2]}

    ax.plot([0, 32], [0, 32], "--", color=GRAY["light"], lw=1.5, zorder=0)
    ax.text(28, 26, "identity", fontsize=7, color=GRAY["medium"], ha="right")

    for alpha in [0.1, 1.0, 10.0]:
        subset = summary[summary["alpha"] == alpha].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["median"], "o-",
                color=alpha_colors[alpha], markersize=4, linewidth=1.5,
                label=f"{alpha}")
        ax.fill_between(
            subset["true_rank"],
            subset["q25"],
            subset["q75"],
            color=alpha_colors[alpha], alpha=0.2, linewidth=0,
        )

    ax.set_xlabel("True rank")
    ax.set_ylabel("Selected rank")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 38)
    ax.legend(frameon=False, loc="upper left", fontsize=7, title=r"$\alpha$",
              title_fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "rank_detection_by_alpha.pdf")


def _plot_rank_detection_by_snr(df: pd.DataFrame, output_dir: Path) -> None:
    """Median selected rank with IQR ribbon, by SNR."""
    df = df[df["alpha"] != 5.0].copy()  # Keep 0.1, 1.0, 10.0

    summary = df.groupby(["true_rank", "snr"])["selected_rank"].agg(
        ["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    ).reset_index()
    summary.columns = ["true_rank", "snr", "median", "q25", "q75"]

    fig, ax = create_figure("single")

    snrs = [0.4, 0.6, 0.8, 1.0]
    snr_colors = {0.4: GRAY["medium"], 0.6: CMAP[2], 0.8: CMAP[1], 1.0: CMAP[0]}

    ax.plot([0, 32], [0, 32], "--", color=GRAY["light"], lw=1.5, zorder=0)
    ax.text(28, 26, "identity", fontsize=7, color=GRAY["medium"], ha="right")

    for snr in snrs:
        subset = summary[summary["snr"] == snr].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["median"], "o-",
                color=snr_colors[snr], markersize=4, linewidth=1.5,
                label=f"{snr}")
        ax.fill_between(
            subset["true_rank"],
            subset["q25"],
            subset["q75"],
            color=snr_colors[snr], alpha=0.2, linewidth=0,
        )

    ax.set_xlabel("True rank")
    ax.set_ylabel("Selected rank")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 38)
    ax.legend(frameon=False, loc="upper left", fontsize=7, title="SNR",
              title_fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "rank_detection_by_snr.pdf")


def _plot_stability(output_dir: Path) -> None:
    """Plot stability (cophenetic correlation) vs alpha."""
    from joblib import Parallel, delayed
    from pysrf import SRF
    from scipy.cluster.hierarchy import cophenet, linkage
    from scipy.spatial.distance import squareform
    from utils.simulation import simulation_dirichlet

    n_stability_runs = 20

    def run_stability(alpha, seed):
        def normalize(w):
            return w / (w.sum(axis=1, keepdims=True) + 1e-10)

        rng = np.random.default_rng(seed)
        factors = simulation_dirichlet(N, K, alpha, rng)
        similarity = factors @ factors.T

        consensus_sum = np.zeros((N, N))
        for i in range(n_stability_runs):
            srf = SRF(rank=K, random_state=seed + i)
            w = srf.fit_transform(similarity)
            assignments = np.argmax(normalize(w), axis=1)
            connectivity = (assignments[:, None] == assignments[None, :]).astype(float)
            consensus_sum += connectivity

        consensus = consensus_sum / n_stability_runs
        np.fill_diagonal(consensus, 1.0)
        condensed = squareform(1.0 - consensus, checks=False)
        linkage_matrix = linkage(condensed, method="average")
        coph, _ = cophenet(linkage_matrix, condensed)
        return {"alpha": alpha, "cophenetic": float(coph)}

    results = Parallel(n_jobs=-1)(
        delayed(run_stability)(alpha, int(alpha * 1000) + rep)
        for alpha in ALPHAS
        for rep in range(20)
    )

    import pandas as pd

    df = (
        pd.DataFrame(results)
        .groupby("alpha")
        .agg(mean=("cophenetic", "mean"), sem=("cophenetic", "sem"))
        .reset_index()
    )

    fig, ax = create_figure("single")

    from matplotlib.ticker import MultipleLocator

    tick_positions = [0, 0.25, 0.5, 0.75, 1.0]

    ax.axhline(1.0, color=GRAY["light"], linestyle="--", linewidth=0.8, zorder=0)
    ax.plot(df["alpha"], df["mean"], "o-", color=CMAP[1], markersize=4, linewidth=1.5)
    ax.fill_between(
        df["alpha"],
        df["mean"] - df["sem"],
        df["mean"] + df["sem"],
        color=CMAP[1],
        alpha=0.2,
        linewidth=0,
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Cophenetic correlation")
    ax.set_xlim(0.08, 120)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(tick_positions)
    ax.yaxis.set_minor_locator(MultipleLocator(0.125))
    despine(ax)

    save_figure(fig, output_dir / "stability.pdf")


def main():
    parser = argparse.ArgumentParser(description="Create simulation plots")
    parser.add_argument("--output-dir", type=Path, default=PLOT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Alpha effect on ground truth (4 + 4 + 1 = 9 plots)
    _plot_alpha_factors(output_dir)  # 4 individual PDFs
    _plot_alpha_rsm(output_dir)  # 4 individual PDFs
    _plot_dirichlet_properties(output_dir)  # 1 dual-axis PDF
    print("alpha_effect: 9 plots")

    # SRF validation (2 plots)
    print("Computing SRF validation plots...")
    _plot_srf_performance(output_dir)  # 1 dual-axis PDF
    _plot_stability(output_dir)
    print("srf_validation: 2 plots")

    # Rank detection (2 plots)
    csv = DATA_DIR / "rank_detection.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        _plot_rank_detection_by_alpha(df, output_dir)
        _plot_rank_detection_by_snr(df, output_dir)
        print("rank_detection: 2 plots")

    # Imputation (1 plot)
    csv = DATA_DIR / "imputation.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        _plot_imputation(df, output_dir)
        print("imputation: 1 plot")

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
