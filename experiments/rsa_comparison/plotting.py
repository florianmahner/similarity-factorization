"""Plotting utilities for RSA comparison experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.figure_theme import (
    CMAP,
    HEATMAPS,
    clean_axis,
    create_figure,
    despine,
    save_figure,
)


def create_power_plot(
    df: pd.DataFrame,
    output_path: Path,
    n_objects: int | None = None,
    title: str | None = None,
) -> None:
    """Create statistical power vs SNR plot.

    Args:
        df: DataFrame with columns: snr, method, significant (and optionally n_objects)
        output_path: Path to save PDF
        n_objects: Filter to specific n_objects (if column exists)
        title: Optional title for the plot
    """
    # Filter by n_objects if specified and column exists
    if n_objects is not None and "n_objects" in df.columns:
        df = df[df["n_objects"] == n_objects].copy()

    # Compute power as percentage
    power_df = (
        df.groupby(["snr", "method"])["significant"]
        .mean()
        .reset_index()
    )
    power_df["power"] = power_df["significant"] * 100

    # Colors: RSA = red, SRF = blue
    colors = {
        "RSA": CMAP[0],  # red
        "SRF": CMAP[1],  # blue
    }

    fig, ax = create_figure("single")

    for method in ["RSA", "SRF"]:
        method_df = power_df[power_df["method"] == method].sort_values("snr")
        if method_df.empty:
            continue

        c = colors.get(method, CMAP[0])

        ax.plot(
            method_df["snr"],
            method_df["power"],
            label=method,
            color=c,
            linewidth=1.5,
            marker="o",
            markersize=5,
        )

    ax.set_xlabel("Signal-to-noise ratio")
    ax.set_ylabel("Statistical power (%)")

    if title:
        ax.set_title(title)

    # Add breathing room on both ends
    snr_min = power_df["snr"].min()
    snr_max = power_df["snr"].max()
    snr_range = snr_max - snr_min
    ax.set_xlim(snr_min - snr_range * 0.05, snr_max + snr_range * 0.05)
    ax.set_ylim(-5, 105)

    despine(ax)
    ax.legend(loc="lower right", frameon=False)

    save_figure(fig, output_path)


def plot_spose_comparison(output_dir: Path | None = None) -> None:
    """Plot SPoSE-based hypothesis testing comparison."""
    if output_dir is None:
        output_dir = Path("outputs/experiments/rsa_comparison")

    csv_path = output_dir / "spose.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Results not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Plot for each n_objects value
    for n_objects in sorted(df["n_objects"].unique()):
        create_power_plot(
            df,
            output_dir / f"spose_power_{n_objects}.pdf",
            n_objects=n_objects,
        )


def plot_factorial_comparison(output_dir: Path | None = None) -> None:
    """Plot factorial design hypothesis testing comparison."""
    if output_dir is None:
        output_dir = Path("outputs/experiments/rsa_comparison")

    csv_path = output_dir / "factorial.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Results not found: {csv_path}")

    df = pd.read_csv(csv_path)

    create_power_plot(
        df,
        output_dir / "factorial_power.pdf",
    )


def plot_rsm(
    rsm: np.ndarray,
    output_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
) -> None:
    """Plot a single RSM (similarity matrix) as vector PDF.

    Args:
        rsm: Square similarity matrix
        output_path: Path to save PDF
        vmin: Minimum value for colormap (default: data min)
        vmax: Maximum value for colormap (default: data max)
        colorbar: Whether to add colorbar
    """
    fig, ax = create_figure("square")

    if vmin is None:
        vmin = rsm.min()
    if vmax is None:
        vmax = rsm.max()

    im = ax.pcolormesh(
        rsm,
        cmap=HEATMAPS["diverging"],
        vmin=vmin,
        vmax=vmax,
        rasterized=False,
    )

    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    clean_axis(ax)
    ax.set_aspect("equal")

    save_figure(fig, output_path)


def plot_factors(
    factors: np.ndarray,
    output_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
) -> None:
    """Plot factor matrix as vector PDF.

    Args:
        factors: Factor matrix (n_items x n_factors)
        output_path: Path to save PDF
        vmin: Minimum value for colormap (default: 0)
        vmax: Maximum value for colormap (default: data max)
        colorbar: Whether to add colorbar
    """
    fig, ax = create_figure("single")

    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = factors.max()

    im = ax.pcolormesh(
        factors,
        cmap=HEATMAPS["diverging"],
        vmin=vmin,
        vmax=vmax,
        rasterized=False,
    )

    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    clean_axis(ax)

    save_figure(fig, output_path)


def plot_rsm_grid(
    rsms: list[np.ndarray],
    output_path: Path,
    labels: list[str] | None = None,
) -> None:
    """Plot multiple RSMs in a row.

    Args:
        rsms: List of square similarity matrices
        output_path: Path to save PDF
        labels: Optional labels for each RSM
    """
    n = len(rsms)
    fig, axes = create_figure("square", ncols=n)
    if n == 1:
        axes = [axes]

    # Global vmin/vmax for consistency
    all_values = np.concatenate([r.flatten() for r in rsms])
    vmin, vmax = all_values.min(), all_values.max()

    for i, (ax, rsm) in enumerate(zip(axes, rsms)):
        im = ax.pcolormesh(
            rsm,
            cmap=HEATMAPS["diverging"],
            vmin=vmin,
            vmax=vmax,
            rasterized=False,
        )
        clean_axis(ax)
        ax.set_aspect("equal")

    # Single colorbar on the right
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)

    save_figure(fig, output_path)
