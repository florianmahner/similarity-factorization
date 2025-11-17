from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _prepare_data(data: pd.DataFrame, axis_col: str, value_col: str) -> pd.DataFrame:
    subset = data[[axis_col, value_col]].dropna().copy()
    subset = subset.sort_values(value_col, ascending=False)
    subset[value_col] = subset[value_col].astype(float)
    return subset


def plot_bars(
    data: pd.DataFrame,
    axis_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plot_data = _prepare_data(data, axis_col, value_col)
    if plot_data.empty:
        return

    sns.set_theme(style="ticks")
    plt.figure(figsize=(9, 4))
    colors = sns.color_palette("viridis", len(plot_data))
    ax = sns.barplot(
        data=plot_data,
        x=value_col,
        y=axis_col,
        hue=axis_col,
        palette=colors,
        dodge=False,
        legend=False,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(ylabel, fontsize=11)
    ax.set_ylabel("")
    ax.set_xlim(0, max(plot_data[value_col].max() * 1.1, 0.1))
    sns.despine(left=True, bottom=True)

    for idx, val in enumerate(plot_data[value_col]):
        ax.text(
            val + 0.01,
            idx,
            f"{val:.2f}",
            va="center",
            fontsize=9,
            color="black",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_grouped_bars(
    data: pd.DataFrame,
    category_col: str,
    value_col: str,
    group_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot grouped bar chart comparing multiple groups."""
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = sorted(data[category_col].unique())
    groups = sorted(data[group_col].unique())
    n_categories = len(categories)
    n_groups = len(groups)

    bar_width = 0.8 / n_groups
    x = np.arange(n_categories)
    colors = sns.color_palette("Greens", n_groups)
    for i, group in enumerate(groups):
        group_data = data[data[group_col] == group].set_index(category_col)
        values = [
            group_data.loc[cat, value_col] if cat in group_data.index else 0
            for cat in categories
        ]

        offset = (i - n_groups / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, values, bar_width, label=group, color=colors[i], alpha=0.85
        )

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_xlabel("Semantic Axis", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend(loc="upper right", frameon=True, shadow=True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
