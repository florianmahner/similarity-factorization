"""Analyze VICE low-data experiment results.

Plots:
1. Estimated dimensionality vs training percentage
2. Validation accuracy vs training percentage
"""
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import get_output_dir
from src.colors import TEAL, ROSE
from src.utils.figure_theme import create_figure, despine, save_figure

OUTPUT_DIR = get_output_dir()
VICE_DIR = Path("/LOCAL/fmahner/similarity-factorization/outputs/experiments/things_behavior/vice_lowdata/models")


def parse_training_log(log_path: Path) -> dict | None:
    """Parse training log to extract final epoch metrics."""
    if not log_path.exists():
        return None

    text = log_path.read_text()
    lines = text.strip().split("\n")

    final_dim = None
    final_val_acc = None
    final_train_acc = None

    for line in reversed(lines):
        if "Dim:" in line and final_dim is None:
            match = re.search(r"Dim:\s*(\d+)", line)
            if match:
                final_dim = int(match.group(1))
        if "Val acc:" in line and final_val_acc is None:
            match = re.search(r"Val acc:\s*([\d.]+)", line)
            if match:
                final_val_acc = float(match.group(1))
        if "Train acc:" in line and final_train_acc is None:
            match = re.search(r"Train acc:\s*([\d.]+)", line)
            if match:
                final_train_acc = float(match.group(1))
        if final_dim and final_val_acc and final_train_acc:
            break

    if final_dim is None or final_val_acc is None:
        return None

    return {
        "dim": final_dim,
        "val_acc": final_val_acc,
        "train_acc": final_train_acc,
    }


def collect_results() -> pd.DataFrame:
    """Collect results from all VICE low-data runs."""
    records = []

    for model_dir in VICE_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        name = model_dir.name
        match = re.match(r"vice_(\d+)pct_part(\d+)", name)
        if not match:
            continue

        pct = int(match.group(1))
        part = int(match.group(2))

        log_paths = list(model_dir.glob("**/training.log"))
        if not log_paths:
            continue
        log_path = log_paths[0]
        metrics = parse_training_log(log_path)

        if metrics:
            records.append({
                "pct": pct,
                "part": part,
                "dim": metrics["dim"],
                "val_acc": metrics["val_acc"],
                "train_acc": metrics["train_acc"],
            })

    return pd.DataFrame(records)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = collect_results()
    df.to_csv(OUTPUT_DIR / "vice_lowdata_results.csv", index=False)
    print(f"Collected {len(df)} runs")
    print(df.groupby("pct")[["dim", "val_acc"]].agg(["mean", "std"]))

    # Aggregate by percentage
    agg = df.groupby("pct").agg({
        "dim": ["mean", "std", "count"],
        "val_acc": ["mean", "std"],
        "train_acc": ["mean", "std"],
    }).reset_index()
    agg.columns = ["pct", "dim_mean", "dim_std", "n_runs", "val_acc_mean", "val_acc_std", "train_acc_mean", "train_acc_std"]

    # Plot 1: Dimensionality vs training percentage
    fig, ax = create_figure("single")
    ax.errorbar(
        agg["pct"], agg["dim_mean"], yerr=agg["dim_std"],
        fmt="o-", color=TEAL, capsize=4, markersize=8, linewidth=2
    )
    ax.set_xlabel("Training data (%)")
    ax.set_ylabel("Estimated dimensionality")
    ax.set_xticks(agg["pct"])
    despine(ax)
    save_figure(fig, OUTPUT_DIR / "dim_vs_training_pct.png")
    plt.close(fig)

    # Plot 2: Validation accuracy vs training percentage
    fig, ax = create_figure("single")
    ax.errorbar(
        agg["pct"], agg["val_acc_mean"], yerr=agg["val_acc_std"],
        fmt="o-", color=ROSE, capsize=4, markersize=8, linewidth=2
    )
    ax.set_xlabel("Training data (%)")
    ax.set_ylabel("Validation accuracy (odd-one-out)")
    ax.set_xticks(agg["pct"])
    despine(ax)
    save_figure(fig, OUTPUT_DIR / "val_acc_vs_training_pct.png")
    plt.close(fig)

    # Plot 3: Combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    ax1.errorbar(
        agg["pct"], agg["dim_mean"], yerr=agg["dim_std"],
        fmt="o-", color=TEAL, capsize=4, markersize=8, linewidth=2
    )
    ax1.set_xlabel("Training data (%)")
    ax1.set_ylabel("Estimated dimensionality")
    ax1.set_xticks(agg["pct"])
    despine(ax1)

    ax2.errorbar(
        agg["pct"], agg["val_acc_mean"], yerr=agg["val_acc_std"],
        fmt="o-", color=ROSE, capsize=4, markersize=8, linewidth=2
    )
    ax2.set_xlabel("Training data (%)")
    ax2.set_ylabel("Validation accuracy")
    ax2.set_xticks(agg["pct"])
    despine(ax2)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "vice_lowdata_combined.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nSaved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
