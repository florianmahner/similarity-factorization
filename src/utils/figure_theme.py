"""Figure utilities for publication-quality plots."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from src.colors import GRAY_LIGHT, setup_style

# =============================================================================
# Figure Sizes
# =============================================================================

SIZES = {
    "single": (2.7, 2.2),
    "square": (2.2, 2.2),
    "wide": (3.4, 2.2),
    "full_width": (5.6, 2.2),
}

DEFAULT_PAD = {
    "left": 0.5,
    "right": 0.2,
    "bottom": 0.5,
    "top": 0.1,
}


# =============================================================================
# Figure Helpers
# =============================================================================


def create_figure(
    size: str = "single",
    nrows: int = 1,
    ncols: int = 1,
    pad_left: float | None = None,
    pad_right: float | None = None,
    pad_bottom: float | None = None,
    pad_top: float | None = None,
    **subplot_kw,
) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    """Create figure with fixed size and consistent plotting area.

    Figure size = plotting area + padding. All figures with the same size
    preset and padding will have identical dimensions, ensuring consistent
    axes size when scaled proportionally.

    Args:
        size: Plot area size preset - "single" (2.7x2.2"), "square" (2.2x2.2"),
              "wide" (3.4x2.2"), "full_width" (5.6x2.2")
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        pad_left: Left padding in inches (default 0.5)
        pad_right: Right padding in inches (default 0.2)
        pad_bottom: Bottom padding in inches (default 0.5)
        pad_top: Top padding in inches (default 0.1)
        **subplot_kw: Additional arguments passed to plt.subplots
    """
    setup_style()

    pl = pad_left if pad_left is not None else DEFAULT_PAD["left"]
    pr = pad_right if pad_right is not None else DEFAULT_PAD["right"]
    pb = pad_bottom if pad_bottom is not None else DEFAULT_PAD["bottom"]
    pt = pad_top if pad_top is not None else DEFAULT_PAD["top"]

    plot_w, plot_h = SIZES[size]
    total_plot_w = plot_w * ncols
    total_plot_h = plot_h * nrows

    fig_w = pl + total_plot_w + pr
    fig_h = pb + total_plot_h + pt

    fig, ax = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), **subplot_kw)

    left_frac = pl / fig_w
    right_frac = (pl + total_plot_w) / fig_w
    bottom_frac = pb / fig_h
    top_frac = (pb + total_plot_h) / fig_h

    fig.subplots_adjust(
        left=left_frac,
        right=right_frac,
        bottom=bottom_frac,
        top=top_frac,
    )

    return fig, ax


def save_figure(fig: plt.Figure, path: Path | str, close: bool = True, tight: bool = True) -> None:
    """Save figure as PDF.

    Args:
        fig: Figure to save
        path: Output path (will add .pdf suffix if missing)
        close: Whether to close figure after saving
        tight: If True (default), use bbox_inches='tight' to prevent label cutoff.
               Set to False for fixed dimensions (requires careful padding).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix != ".pdf":
        path = path.with_suffix(".pdf")
    if tight:
        fig.savefig(path, format="pdf", bbox_inches="tight")
    else:
        fig.savefig(path, format="pdf")
    if close:
        plt.close(fig)


# =============================================================================
# Axis Helpers
# =============================================================================

def despine(ax: plt.Axes, left: bool = False, bottom: bool = False) -> None:
    """Remove top/right spines, optionally left/bottom."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(False)


def clean_axis(ax: plt.Axes) -> None:
    """Clean axis for heatmaps."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def add_reference_line(
    ax: plt.Axes,
    value: float,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    color: str = GRAY_LIGHT,
    linestyle: str = "--",
    linewidth: float = 0.8,
) -> None:
    """Add a reference line."""
    if orientation == "horizontal":
        ax.axhline(value, color=color, linestyle=linestyle, linewidth=linewidth, zorder=0)
    else:
        ax.axvline(value, color=color, linestyle=linestyle, linewidth=linewidth, zorder=0)


def pad_limits(
    data_min: float,
    data_max: float,
    pad_frac: float = 0.05,
) -> tuple[float, float]:
    """Calculate axis limits with breathing room.

    Args:
        data_min: Minimum data value
        data_max: Maximum data value
        pad_frac: Fraction of range to add as padding (default 5%)

    Returns:
        Tuple of (min_limit, max_limit) with padding applied
    """
    data_range = data_max - data_min
    if data_range == 0:
        data_range = abs(data_max) * 0.1 if data_max != 0 else 1.0
    pad = data_range * pad_frac
    return data_min - pad, data_max + pad
