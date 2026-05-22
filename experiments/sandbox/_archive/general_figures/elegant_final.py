"""Final elegant combined: RSM + Arc diagram.

Publication-quality figure showing soft clustering structure.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#c8c8c8'
MEDIUM_GRAY = '#888888'
FACTOR_COLORS = ['#c75b5b', '#5b8fc7', '#5bc77a']


def create_data(seed=42):
    rng = np.random.default_rng(seed)
    W_list = []

    for _ in range(4):
        W_list.append([rng.uniform(0.82, 0.92), rng.uniform(0.03, 0.09), rng.uniform(0.03, 0.09)])
    for _ in range(4):
        W_list.append([rng.uniform(0.03, 0.09), rng.uniform(0.82, 0.92), rng.uniform(0.03, 0.09)])
    for _ in range(4):
        W_list.append([rng.uniform(0.03, 0.09), rng.uniform(0.03, 0.09), rng.uniform(0.82, 0.92)])

    W_list.append([0.46, 0.46, 0.08])
    W_list.append([0.47, 0.07, 0.46])
    W_list.append([0.07, 0.47, 0.46])

    W = np.array(W_list)
    W = W / W.sum(axis=1, keepdims=True)
    S = W @ W.T
    return W, S


def create_colormap():
    colors = ['#ffffff', '#f0f4f8', '#d4e1ed', '#a8c5db',
              '#6a9fc0', '#3d7ea6', '#1a5a7a', '#0d3d54']
    return mcolors.LinearSegmentedColormap.from_list('elegant', colors)


def draw_arc(ax, x1, x2, y_base=0, height_scale=0.5, **kwargs):
    theta = np.linspace(0, np.pi, 50)
    center = (x1 + x2) / 2
    width = abs(x2 - x1)
    height = width * height_scale
    x = center + (width / 2) * np.cos(theta)
    y = y_base + height * np.sin(theta) / 2
    ax.plot(x, y, **kwargs)


def main():
    W, S = create_data()
    n = len(W)

    # Order by dominant factor
    dominant = np.argmax(W, axis=1)
    order = np.lexsort((-W.max(axis=1), dominant))
    W_ordered = W[order]
    S_ordered = S[np.ix_(order, order)]

    fig = plt.figure(figsize=(5.5, 3.5))

    # =========== TOP: Arc diagram ===========
    ax_arc = fig.add_axes([0.08, 0.58, 0.88, 0.38])

    # X positions with gaps
    positions = {}
    x = 0
    prev_dom = -1
    for idx, node in enumerate(order):
        dom = dominant[node]
        if dom != prev_dom and prev_dom >= 0:
            x += 0.4
        positions[node] = x
        x += 1
        prev_dom = dom

    max_x = max(positions.values())
    positions = {k: v / max_x for k, v in positions.items()}

    # Draw arcs
    threshold = 0.18
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > threshold:
                x1, x2 = positions[i], positions[j]
                if x1 > x2:
                    x1, x2 = x2, x1
                alpha = 0.2 + 0.55 * (S[i, j] - threshold) / (1 - threshold)
                lw = 0.4 + 0.6 * (S[i, j] - threshold) / (1 - threshold)
                draw_arc(ax_arc, x1, x2, height_scale=0.9,
                        color=LIGHT_GRAY, linewidth=lw, alpha=alpha, zorder=1)

    # Draw nodes
    node_radius = 0.022
    for node in range(n):
        x = positions[node]
        dom = dominant[node]
        max_w = W[node].max()

        if max_w < 0.55:
            circle = mpatches.Circle((x, 0), node_radius, facecolor='white',
                                      edgecolor=CHARCOAL, linewidth=1.0, zorder=3)
        else:
            circle = mpatches.Circle((x, 0), node_radius, facecolor=FACTOR_COLORS[dom],
                                      edgecolor='white', linewidth=0.5, zorder=3)
        ax_arc.add_patch(circle)

    # Factor labels
    for k in range(3):
        group_nodes = [node for node in range(n) if dominant[node] == k]
        if group_nodes:
            center_x = np.mean([positions[node] for node in group_nodes])
            ax_arc.text(center_x, -0.1, f'F{k+1}', fontsize=8, ha='center', va='top',
                       color=FACTOR_COLORS[k], fontweight='medium')

    ax_arc.set_xlim(-0.03, 1.03)
    ax_arc.set_ylim(-0.15, 0.55)
    ax_arc.set_aspect('equal')
    ax_arc.axis('off')

    # =========== BOTTOM: RSM ===========
    ax_rsm = fig.add_axes([0.25, 0.08, 0.45, 0.45])
    cmap = create_colormap()
    im = ax_rsm.imshow(S_ordered, cmap=cmap, vmin=0, vmax=1, aspect='equal')

    ax_rsm.set_xticks([])
    ax_rsm.set_yticks([])
    for spine in ax_rsm.spines.values():
        spine.set_visible(False)

    # Top bar
    ax_top = fig.add_axes([0.25, 0.54, 0.45, 0.025])
    for i in range(n):
        for k in range(3):
            if W_ordered[i, k] > 0.05:
                rect = mpatches.Rectangle((i/n, 0), 1/n*0.92, 1,
                                           facecolor=FACTOR_COLORS[k], alpha=W_ordered[i, k], edgecolor='none')
                ax_top.add_patch(rect)
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)
    ax_top.axis('off')

    # Left bar
    ax_left = fig.add_axes([0.215, 0.08, 0.025, 0.45])
    for i in range(n):
        for k in range(3):
            if W_ordered[i, k] > 0.05:
                rect = mpatches.Rectangle((0, 1-(i+1)/n), 1, 1/n*0.92,
                                           facecolor=FACTOR_COLORS[k], alpha=W_ordered[i, k], edgecolor='none')
                ax_left.add_patch(rect)
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)
    ax_left.axis('off')

    # Colorbar
    ax_cbar = fig.add_axes([0.72, 0.08, 0.015, 0.45])
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=6, length=2, color=LIGHT_GRAY)
    cbar.set_ticks([0, 0.5, 1])

    # Save
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    fig.savefig(output_dir / "elegant_final.svg", format='svg', bbox_inches='tight', pad_inches=0.03)
    fig.savefig(output_dir / "elegant_final.png", dpi=200, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)

    print("Saved: elegant_final.svg")


if __name__ == "__main__":
    main()
