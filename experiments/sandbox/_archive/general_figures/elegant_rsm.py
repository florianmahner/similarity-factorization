"""Elegant RSM visualization with soft cluster structure.

Minimal design, sophisticated colormap, subtle factor annotations.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

# Sophisticated muted palette
CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#c8c8c8'
MEDIUM_GRAY = '#888888'

# Muted factor colors
FACTOR_COLORS = ['#c75b5b', '#5b8fc7', '#5bc77a']


def create_elegant_data(n_per_cluster=4, seed=42):
    """Create synthetic soft membership data."""
    rng = np.random.default_rng(seed)

    W_list = []

    # Pure cluster 1 items
    for _ in range(n_per_cluster):
        w = [rng.uniform(0.8, 0.95), rng.uniform(0.02, 0.1), rng.uniform(0.02, 0.1)]
        W_list.append(w)

    # Pure cluster 2 items
    for _ in range(n_per_cluster):
        w = [rng.uniform(0.02, 0.1), rng.uniform(0.8, 0.95), rng.uniform(0.02, 0.1)]
        W_list.append(w)

    # Pure cluster 3 items
    for _ in range(n_per_cluster):
        w = [rng.uniform(0.02, 0.1), rng.uniform(0.02, 0.1), rng.uniform(0.8, 0.95)]
        W_list.append(w)

    # Overlap items
    W_list.append([0.45, 0.45, 0.10])  # F1-F2
    W_list.append([0.48, 0.08, 0.44])  # F1-F3
    W_list.append([0.08, 0.46, 0.46])  # F2-F3

    W = np.array(W_list)
    W = W / W.sum(axis=1, keepdims=True)

    S = W @ W.T

    return W, S


def create_elegant_colormap():
    """Create a sophisticated single-hue colormap."""
    # White to deep navy
    colors = ['#ffffff', '#f0f4f8', '#d4e1ed', '#a8c5db',
              '#6a9fc0', '#3d7ea6', '#1a5a7a', '#0d3d54']
    return mcolors.LinearSegmentedColormap.from_list('elegant', colors)


def main():
    W, S = create_elegant_data()
    n = len(W)

    # Reorder by dominant factor for clean block structure
    dominant = np.argmax(W, axis=1)
    order = np.lexsort((-W.max(axis=1), dominant))
    W_ordered = W[order]
    S_ordered = S[np.ix_(order, order)]

    # Create figure with specific proportions
    fig = plt.figure(figsize=(3.0, 3.2))

    # Main RSM axes
    ax_rsm = fig.add_axes([0.15, 0.08, 0.72, 0.72])

    # Custom colormap
    cmap = create_elegant_colormap()

    # Plot RSM
    im = ax_rsm.imshow(S_ordered, cmap=cmap, vmin=0, vmax=1, aspect='equal')

    # Remove all spines and ticks
    ax_rsm.set_xticks([])
    ax_rsm.set_yticks([])
    for spine in ax_rsm.spines.values():
        spine.set_visible(False)

    # Factor membership bars on top
    bar_height = 0.06
    ax_top = fig.add_axes([0.15, 0.82, 0.72, bar_height])

    for i in range(n):
        for k in range(3):
            alpha = W_ordered[i, k]
            if alpha > 0.05:
                rect = mpatches.Rectangle(
                    (i / n, 0), 1 / n * 0.95, 1,
                    facecolor=FACTOR_COLORS[k],
                    alpha=alpha,
                    edgecolor='none'
                )
                ax_top.add_patch(rect)

    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)
    ax_top.axis('off')

    # Factor membership bars on left
    ax_left = fig.add_axes([0.06, 0.08, bar_height, 0.72])

    for i in range(n):
        for k in range(3):
            alpha = W_ordered[i, k]
            if alpha > 0.05:
                rect = mpatches.Rectangle(
                    (0, 1 - (i + 1) / n), 1, 1 / n * 0.95,
                    facecolor=FACTOR_COLORS[k],
                    alpha=alpha,
                    edgecolor='none'
                )
                ax_left.add_patch(rect)

    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)
    ax_left.axis('off')

    # Minimal colorbar
    ax_cbar = fig.add_axes([0.89, 0.08, 0.025, 0.72])
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7, length=2, color=LIGHT_GRAY)
    cbar.set_ticks([0, 0.5, 1])

    # Factor legend (minimal)
    ax_leg = fig.add_axes([0.15, 0.91, 0.72, 0.06])
    ax_leg.axis('off')
    for k in range(3):
        x_pos = 0.1 + k * 0.35
        rect = mpatches.Rectangle(
            (x_pos, 0.2), 0.06, 0.6,
            facecolor=FACTOR_COLORS[k],
            edgecolor='none'
        )
        ax_leg.add_patch(rect)
        ax_leg.text(x_pos + 0.09, 0.5, f'F{k+1}', fontsize=7,
                    va='center', color=MEDIUM_GRAY)
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)

    # Save
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    fig.savefig(output_dir / "elegant_rsm.svg", format='svg',
                bbox_inches='tight', pad_inches=0.02)
    fig.savefig(output_dir / "elegant_rsm.png", dpi=200,
                bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    print("Saved: elegant_rsm.svg")


if __name__ == "__main__":
    main()
