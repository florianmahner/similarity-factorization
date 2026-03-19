"""Plot SRF accuracy at coherence ranks vs VICE.

Loads results from srf_coherence_ranks, srf_seeds (VICE-matched),
and VICE lowdata comparison. Produces comparison figures.

Usage:
    poetry run python experiments/things_behavior/lowdata/plot_coherence_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.things_behavior.lowdata import get_vice_dims
from src.colors import ROSE, TEAL, CYAN, SAND, GRAY, GRAY_LIGHT
from src.utils.figure_theme import create_figure, despine, save_figure

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "experiments" / "things_behavior"


def load_all_results() -> dict[str, pd.DataFrame]:
    """Load results from all three sources."""
    # 1. SRF at coherence ranks (new)
    coherence_path = OUTPUT_BASE / "srf_coherence_ranks" / "results.csv"
    coherence_df = pd.read_csv(coherence_path)

    # 2. SRF at VICE-matched ranks (existing)
    vice_srf_path = OUTPUT_BASE / "srf_lowdata" / "results.csv"
    vice_srf_df = pd.read_csv(vice_srf_path)
    vice_srf_df["method"] = "VICE-matched"

    # 3. VICE results
    comparison_path = OUTPUT_BASE / "lowdata_comparison" / "lowdata_comparison.csv"
    comparison_df = pd.read_csv(comparison_path)
    vice_df = comparison_df[comparison_df["model"] == "VICE"].copy()
    vice_df["method"] = "VICE"

    return {
        "coherence": coherence_df,
        "vice_srf": vice_srf_df,
        "vice": vice_df,
    }


def plot_accuracy_vs_data(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Main comparison: accuracy vs training data percentage."""
    fig, ax = create_figure("single")

    vice_dims = get_vice_dims()
    pct_ranks = {5: 4, 10: 10, 20: 14, 50: 20, 100: 25}
    kappa_ranks = {5: 6, 10: 11, 20: 11, 50: 21, 100: 26}

    # Aggregate each method
    methods = [
        ("PCT", data["coherence"][data["coherence"]["method"] == "PCT"], TEAL, "o", "-"),
        ("kappa", data["coherence"][data["coherence"]["method"] == "kappa"], CYAN, "^", "-"),
        ("SRF @ VICE rank", data["vice_srf"], SAND, "s", "--"),
        ("VICE", data["vice"], ROSE, "D", "--"),
    ]

    for label, df, color, marker, ls in methods:
        grouped = df.groupby("pct")["val_acc"].agg(["mean", "std"]).reset_index()
        grouped = grouped.sort_values("pct")
        pcts = grouped["pct"].values
        x = np.arange(len(pcts))
        ax.errorbar(
            x, grouped["mean"] * 100, yerr=grouped["std"] * 100,
            marker=marker, markersize=5, color=color,
            capsize=2, capthick=0.8, linewidth=1.5, linestyle=ls,
            label=label,
        )

    # Reference lines
    ax.axhline(33.33, color=GRAY_LIGHT, linestyle=":", linewidth=0.8)
    ax.axhline(66.67, color=GRAY, linestyle=":", linewidth=0.8)

    # Use PCT percentages for x-axis (has 100%)
    pct_vals = sorted(data["coherence"]["pct"].unique())
    ax.set_xticks(range(len(pct_vals)))
    ax.set_xticklabels([f"{p}" for p in pct_vals])
    ax.set_xlabel("Training data (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(55, 67)
    ax.legend(fontsize=6.5, loc="lower right")
    despine(ax)

    save_figure(fig, output_dir / "accuracy_vs_data.pdf")
    print(f"Saved accuracy_vs_data.pdf")


def plot_rank_vs_data(output_dir: Path) -> None:
    """Show what rank each method uses."""
    fig, ax = create_figure("single")

    vice_dims = get_vice_dims()
    pct_ranks = {5: 4, 10: 10, 20: 14, 50: 20, 100: 25}
    kappa_ranks = {5: 6, 10: 11, 20: 11, 50: 21, 100: 26}

    pcts_coherence = sorted(pct_ranks.keys())
    pcts_vice = sorted(vice_dims.keys())

    ax.plot(pcts_coherence, [pct_ranks[p] for p in pcts_coherence],
            marker="o", color=TEAL, linewidth=2, label="PCT")
    ax.plot(pcts_coherence, [kappa_ranks[p] for p in pcts_coherence],
            marker="^", color=CYAN, linewidth=2, label="Kappa")
    ax.plot(pcts_vice, [vice_dims[p] for p in pcts_vice],
            marker="D", color=ROSE, linewidth=1.5, linestyle="--", label="VICE")

    ax.set_xlabel("Training data (%)")
    ax.set_ylabel("Rank (k)")
    ax.legend(fontsize=7)
    despine(ax)

    save_figure(fig, output_dir / "rank_vs_data.pdf")
    print(f"Saved rank_vs_data.pdf")


def print_summary(data: dict[str, pd.DataFrame]) -> None:
    """Print comparison table."""
    vice_dims = get_vice_dims()
    pct_ranks = {5: 4, 10: 10, 20: 14, 50: 20, 100: 25}
    kappa_ranks = {5: 6, 10: 11, 20: 11, 50: 21, 100: 26}

    print("\n" + "=" * 80)
    print("ACCURACY COMPARISON: SRF at coherence ranks vs VICE")
    print("=" * 80)
    print(f"{'pct':>5s}  {'PCT':>12s}  {'kappa':>12s}  {'SRF@VICE':>12s}  {'VICE':>12s}  {'PCT rank':>9s}  {'VICE rank':>10s}")
    print("-" * 80)

    for pct in sorted(set(pct_ranks.keys())):
        pct_df = data["coherence"][(data["coherence"]["method"] == "PCT") & (data["coherence"]["pct"] == pct)]
        kap_df = data["coherence"][(data["coherence"]["method"] == "kappa") & (data["coherence"]["pct"] == pct)]
        vsrf_df = data["vice_srf"][data["vice_srf"]["pct"] == pct]
        vice_df = data["vice"][data["vice"]["pct"] == pct]

        pct_acc = f"{pct_df['val_acc'].mean()*100:.2f}" if len(pct_df) > 0 else "--"
        kap_acc = f"{kap_df['val_acc'].mean()*100:.2f}" if len(kap_df) > 0 else "--"
        vsrf_acc = f"{vsrf_df['val_acc'].mean()*100:.2f}" if len(vsrf_df) > 0 else "--"
        vice_acc = f"{vice_df['val_acc'].mean()*100:.2f}" if len(vice_df) > 0 else "--"
        pr = str(pct_ranks.get(pct, "--"))
        vr = str(vice_dims.get(pct, "--"))

        print(f"{pct:>4d}%  {pct_acc:>12s}  {kap_acc:>12s}  {vsrf_acc:>12s}  {vice_acc:>12s}  {pr:>9s}  {vr:>10s}")


def main():
    output_dir = OUTPUT_BASE / "srf_coherence_ranks"
    data = load_all_results()

    print_summary(data)
    plot_accuracy_vs_data(data, output_dir)
    plot_rank_vs_data(output_dir)

    # Check convergence
    coh_df = data["coherence"]
    if "n_iter" in coh_df.columns:
        print("\nConvergence check:")
        print(coh_df.groupby(["method", "pct"])["n_iter"].agg(["mean", "max"]).round(0))


if __name__ == "__main__":
    main()
