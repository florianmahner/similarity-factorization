"""Elegant combined visualization: Simplex + RSM side by side.

Publication-quality figure showing soft clustering structure.
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
FAINT_GRAY = '#e8e8e8'

# Muted factor colors
FACTOR_COLORS = ['#c75b5b', '#5b8fc7', '#5bc77a']


def barycentric_to_cartesian(weights):
    """Convert barycentric coordinates to 2D cartesian."""
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    weights = np.array(weights)
    weights = weights / weights.sum()
    return weights @ corners


def create_data(seed=42):
    """Create synthetic soft membership data."""
    rng = np.random.default_rng(seed)
    W_list = []

    # Pure cluster items (4 each)
    for _ in range(4):
        W_list.append([rng.uniform(0.82, 0.92), rng.uniform(0.03, 0.09), rng.uniform(0.03, 0.09)])
    for _ in range(4):
        W_list.append([rng.uniform(0.03, 0.09), rng.uniform(0.82, 0.92), rng.uniform(0.03, 0.09)])
    for _ in range(4):
        W_list.append([rng.uniform(0.03, 0.09), rng.uniform(0.03, 0.09), rng.uniform(0.82, 0.92)])

    # Overlap items
    W_list.append([0.46, 0.46, 0.08])  # F1-F2
    W_list.append([0.47, 0.07, 0.46])  # F1-F3
    W_list.append([0.07, 0.47, 0.46])  # F2-F3

    W = np.array(W_list)
    W = W / W.sum(axis=1, keepdims=True)
    S = W @ W.T
    return W, S


def create_colormap():
    """Sophisticated single-hue colormap."""
    colors = ['#ffffff', '#f0f4f8', '#d4e1ed', '#a8c5db',
              '#6a9fc0', '#3d7ea6', '#1a5a7a', '#0d3d54']
    return mcolors.LinearSegmentedColormap.from_list('elegant', colors)


def main():
    W, S = create_data()
    n = len(W)

    # Create figure
    fig = plt.figure(figsize=(6.0, 2.8))

    # =========== LEFT: Simplex ===========
    ax_simplex = fig.add_axes([0.02, 0.08, 0.42, 0.88])

    # Triangle frame
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    ax_simplex.plot(corners[:, 0], corners[:, 1],
                    color=LIGHT_GRAY, linewidth=0.8, zorder=1)

    # Positions
    positions = np.array([barycentric_to_cartesian(W[i]) for i in range(n)])

    # Similarity edges
    threshold = 0.25
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > threshold:
                alpha = ((S[i, j] - threshold) / (1 - threshold)) ** 0.7
                ax_simplex.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    color=LIGHT_GRAY, linewidth=0.4,
                    alpha=alpha * 0.7, zorder=2
                )

    # Nodes
    for i in range(n):
        x, y = positions[i]
        dominant = np.argmax(W[i])
        is_mixed = W[i].max() < 0.6

        if is_mixed:
            circle = mpatches.Circle(
                (x, y), 0.032, facecolor='white',
                edgecolor=CHARCOAL, linewidth=0.8, zorder=4
            )
        else:
            circle = mpatches.Circle(
                (x, y), 0.032, facecolor=FACTOR_COLORS[dominant],
                edgecolor='white', linewidth=0.6, zorder=4
            )
        ax_simplex.add_patch(circle)

    # Factor labels
    for (x, y, label, dy) in [(0, 0, 'F₁', -0.08), (1, 0, 'F₂', -0.08), (0.5, np.sqrt(3)/2, 'F₃', 0.06)]:
        ax_simplex.text(x, y + dy, label, fontsize=8, ha='center', va='center',
                        color=MEDIUM_GRAY, fontweight='medium')

    ax_simplex.set_xlim(-0.1, 1.1)
    ax_simplex.set_ylim(-0.15, 1.0)
    ax_simplex.set_aspect('equal')
    ax_simplex.axis('off')

    # =========== RIGHT: RSM ===========
    # Reorder
    dominant = np.argmax(W, axis=1)
    order = np.lexsort((-W.max(axis=1), dominant))
    W_ordered = W[order]
    S_ordered = S[np.ix_(order, order)]

    ax_rsm = fig.add_axes([0.52, 0.12, 0.38, 0.76])
    cmap = create_colormap()
    im = ax_rsm.imshow(S_ordered, cmap=cmap, vmin=0, vmax=1, aspect='equal')

    ax_rsm.set_xticks([])
    ax_rsm.set_yticks([])
    for spine in ax_rsm.spines.values():
        spine.set_visible(False)

    # Top membership bar
    ax_top = fig.add_axes([0.52, 0.90, 0.38, 0.05])
    for i in range(n):
        for k in range(3):
            if W_ordered[i, k] > 0.05:
                rect = mpatches.Rectangle(
                    (i / n, 0), 1 / n * 0.92, 1,
                    facecolor=FACTOR_COLORS[k], alpha=W_ordered[i, k], edgecolor='none'
                )
                ax_top.add_patch(rect)
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)
    ax_top.axis('off')

    # Left membership bar
    ax_left = fig.add_axes([0.46, 0.12, 0.05, 0.76])
    for i in range(n):
        for k in range(3):
            if W_ordered[i, k] > 0.05:
                rect = mpatches.Rectangle(
                    (0, 1 - (i + 1) / n), 1, 1 / n * 0.92,
                    facecolor=FACTOR_COLORS[k], alpha=W_ordered[i, k], edgecolor='none'
                )
                ax_left.add_patch(rect)
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)
    ax_left.axis('off')

    # Colorbar
    ax_cbar = fig.add_axes([0.92, 0.12, 0.015, 0.76])
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=6, length=2, color=LIGHT_GRAY)
    cbar.set_ticks([0, 0.5, 1])

    # Save
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    fig.savefig(output_dir / "elegant_combined.svg", format='svg',
                bbox_inches='tight', pad_inches=0.03)
    fig.savefig(output_dir / "elegant_combined.png", dpi=200,
                bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)

    print("Saved: elegant_combined.svg")


if __name__ == "__main__":
    main()
