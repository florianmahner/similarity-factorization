"""Imputation performance plots.

Generates:
- imputation_r2.pdf           -- held-out R² by fraction observed
- downstream_recovery.pdf     -- factor recovery after imputation vs oracle

Usage:
    poetry run python experiments/analyses/simulation/imputation/plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator

from src.colors import ROSE, TEAL, INDIGO, GRAY, GRAY_LIGHT, GRAY_PALE, setup_style
from src.utils.figure_theme import create_figure, despine, save_figure

TASK_DIR = Path(__file__).parent
DATA_PATH = TASK_DIR / "outputs" / "imputation.csv"
TRIPLET_PATH = TASK_DIR / "outputs" / "triplet_prediction.csv"
OUTPUT_DIR = TASK_DIR / "outputs"


def plot_imputation(df: pd.DataFrame, output_dir: Path) -> None:
    """Line plot: imputation R² by samples per model parameter."""
    df_plot = df[df["method"] != "Mean"].copy()
    if "obs_per_dof" not in df_plot.columns:
        return

    # Aggregate per (obs_per_dof, method), then interpolate for smooth curves
    agg = (
        df_plot.groupby(["obs_per_dof", "method"])["r2"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    agg = agg.dropna()

    colors = {"SRF": ROSE, "KNN": TEAL, "Median": INDIGO}

    fig, ax = create_figure("single")

    ax.axvspan(0, 1.0, color=GRAY_PALE, zorder=0)
    ax.axvline(x=1.0, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=1)
    ax.text(0.9, 50, "Underdetermined", rotation=90, va="center", ha="center",
            fontsize=7, color=GRAY)

    for method in ["SRF", "KNN", "Median"]:
        data = agg[agg["method"] == method].sort_values("obs_per_dof")
        if data.empty:
            continue
        c = colors[method]

        x = data["obs_per_dof"].values
        y = data["mean"].values * 100
        y_lo = (data["mean"] - data["sem"]).values * 100
        y_hi = (data["mean"] + data["sem"]).values * 100

        # Smooth interpolation for publication-quality curves
        from scipy.interpolate import make_interp_spline

        x_log = np.log(x)
        x_smooth_log = np.linspace(x_log.min(), x_log.max(), 200)
        x_smooth = np.exp(x_smooth_log)

        spl = make_interp_spline(x_log, y, k=3)
        spl_lo = make_interp_spline(x_log, y_lo, k=3)
        spl_hi = make_interp_spline(x_log, y_hi, k=3)

        y_smooth = spl(x_smooth_log)
        y_lo_smooth = spl_lo(x_smooth_log)
        y_hi_smooth = spl_hi(x_smooth_log)

        ax.plot(x_smooth, y_smooth, label=method, color=c, zorder=3)
        ax.fill_between(x_smooth, y_lo_smooth, y_hi_smooth,
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

    save_figure(fig, output_dir / "imputation_r2.pdf")


def plot_triplet_prediction(df: pd.DataFrame, output_dir: Path) -> None:
    """Held-out triplet prediction accuracy by fraction of unobserved pairs."""
    colors = {"SRF": ROSE, "KNN": TEAL, "Median": INDIGO}

    # x-axis: mean pct_missing per triplet_multiple (inverted: more missing = harder)
    pct_by_mult = df.groupby("triplet_multiple")["pct_missing"].mean()

    agg = (
        df.groupby(["triplet_multiple", "method"])["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    agg["pct_missing"] = agg["triplet_multiple"].map(pct_by_mult)

    oracle = df.groupby("triplet_multiple").agg(
        oracle=("oracle_accuracy", "mean"),
        pct_missing=("pct_missing", "mean"),
    )

    fig, ax = create_figure("single")

    # Chance level
    ax.axhline(100 / 3, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)
    ax.text(42, 34.5, "chance", fontsize=7, color=GRAY, ha="left")

    # Oracle
    ax.plot(oracle["pct_missing"], oracle["oracle"] * 100, "o--", color=GRAY, ms=3,
            markeredgecolor="white", markeredgewidth=0.5, zorder=1, label="Oracle (complete)")

    for method in ["SRF", "KNN", "Median"]:
        data = agg[agg["method"] == method].sort_values("pct_missing", ascending=False)
        if data.empty:
            continue
        c = colors[method]

        x = data["pct_missing"].values
        y = data["mean"].values * 100
        y_lo = (data["mean"] - data["sem"]).values * 100
        y_hi = (data["mean"] + data["sem"]).values * 100

        ax.plot(x, y, "o-", label=method, color=c, ms=4,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        ax.fill_between(x, y_lo, y_hi, color=c, alpha=0.2, linewidth=0, zorder=2)

    ax.set_xlabel("Unobserved pairs (%)")
    ax.set_ylabel("Held-out triplet accuracy (%)")
    ax.invert_xaxis()

    despine(ax)
    ax.legend(loc="lower left", frameon=False, fontsize=7)

    save_figure(fig, output_dir / "triplet_prediction.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded {len(df)} rows from imputation.csv")
        plot_imputation(df, OUTPUT_DIR)

    if TRIPLET_PATH.exists():
        df_trip = pd.read_csv(TRIPLET_PATH)
        print(f"Loaded {len(df_trip)} rows from triplet_prediction.csv")
        plot_triplet_prediction(df_trip, OUTPUT_DIR)

    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
