"""Plot SRF vs NMF comparison results by kernel type.

Usage:
    poetry run python sandbox/nmf_comparison/plot.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.colors import ROSE, TEAL
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()

# Method styles
METHODS = {
    "srf": {"color": ROSE, "marker": "o", "lw": 2.0, "label": "SRF"},
    "nmf": {"color": TEAL, "marker": "s", "lw": 1.2, "label": "NMF"},
}


def plot_factor_recovery_by_alpha(df: pd.DataFrame) -> None:
    """Bar plot comparing factor recovery by alpha for each kernel."""
    fig, axes = create_figure("full_width", nrows=1, ncols=2)

    for idx, kernel in enumerate(["linear", "rbf"]):
        ax = axes[idx]
        df_k = df[df["kernel"] == kernel]

        alphas = sorted(df_k["alpha"].unique())
        x = np.arange(len(alphas))
        width = 0.35

        srf_means = [df_k[df_k["alpha"] == a]["srf_factor_corr"].mean() for a in alphas]
        nmf_means = [df_k[df_k["alpha"] == a]["nmf_factor_corr"].mean() for a in alphas]
        srf_sems = [df_k[df_k["alpha"] == a]["srf_factor_corr"].sem() for a in alphas]
        nmf_sems = [df_k[df_k["alpha"] == a]["nmf_factor_corr"].sem() for a in alphas]

        ax.bar(x - width/2, srf_means, width, yerr=srf_sems,
               label="SRF", color=ROSE, capsize=3)
        ax.bar(x + width/2, nmf_means, width, yerr=nmf_sems,
               label="NMF", color=TEAL, capsize=3)

        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Factor recovery (correlation)")
        ax.set_title(f"{kernel.upper()} kernel", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([str(a) for a in alphas])
        ax.set_ylim(0, 1.05)
        despine(ax)
        ax.legend(frameon=False, loc="lower left")

    save_figure(fig, OUTPUT_DIR / "factor_recovery_by_kernel.pdf")


def plot_rsm_recovery_by_alpha(df: pd.DataFrame) -> None:
    """Bar plot comparing RSM recovery by alpha for each kernel."""
    fig, axes = create_figure("full_width", nrows=1, ncols=2)

    for idx, kernel in enumerate(["linear", "rbf"]):
        ax = axes[idx]
        df_k = df[df["kernel"] == kernel]

        alphas = sorted(df_k["alpha"].unique())
        x = np.arange(len(alphas))
        width = 0.35

        srf_means = [df_k[df_k["alpha"] == a]["srf_rsm_corr"].mean() for a in alphas]
        nmf_means = [df_k[df_k["alpha"] == a]["nmf_rsm_corr"].mean() for a in alphas]
        srf_sems = [df_k[df_k["alpha"] == a]["srf_rsm_corr"].sem() for a in alphas]
        nmf_sems = [df_k[df_k["alpha"] == a]["nmf_rsm_corr"].sem() for a in alphas]

        ax.bar(x - width/2, srf_means, width, yerr=srf_sems,
               label="SRF", color=ROSE, capsize=3)
        ax.bar(x + width/2, nmf_means, width, yerr=nmf_sems,
               label="NMF", color=TEAL, capsize=3)

        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("RSM recovery (correlation)")
        ax.set_title(f"{kernel.upper()} kernel", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([str(a) for a in alphas])
        ax.set_ylim(0.9, 1.01)
        despine(ax)
        ax.legend(frameon=False, loc="lower left")

    save_figure(fig, OUTPUT_DIR / "rsm_recovery_by_kernel.pdf")


def plot_symmetry_error(df: pd.DataFrame) -> None:
    """Bar plot showing symmetry error (SRF = 0 by construction)."""
    fig, ax = create_figure("single")

    kernels = ["linear", "rbf"]
    x = np.arange(len(kernels))
    width = 0.35

    srf_means = [df[df["kernel"] == k]["srf_symmetry_error"].mean() * 100 for k in kernels]
    nmf_means = [df[df["kernel"] == k]["nmf_symmetry_error"].mean() * 100 for k in kernels]

    ax.bar(x - width/2, srf_means, width, label="SRF", color=ROSE)
    ax.bar(x + width/2, nmf_means, width, label="NMF", color=TEAL)

    ax.set_xlabel("Kernel")
    ax.set_ylabel("Symmetry error (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in kernels])
    despine(ax)
    ax.legend(frameon=False)

    save_figure(fig, OUTPUT_DIR / "symmetry_error.pdf")


def main():
    csv_path = OUTPUT_DIR / "comparison.csv"
    if not csv_path.exists():
        print(f"Data not found: {csv_path}")
        print("Run: poetry run python sandbox/nmf_comparison/run.py")
        return

    df = pd.read_csv(csv_path)

    plot_factor_recovery_by_alpha(df)
    plot_rsm_recovery_by_alpha(df)
    plot_symmetry_error(df)

    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
