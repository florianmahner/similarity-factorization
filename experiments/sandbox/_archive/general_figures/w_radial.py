"""Radial visualization for W matrix - factors as spokes, items orbit by loading."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#c8c8c8'
MEDIUM_GRAY = '#888888'

FACTOR_COLORS = np.array([
    [0.88, 0.44, 0.25],
    [0.31, 0.44, 0.75],
    [0.44, 0.69, 0.25],
])
FACTOR_HEX = ['#E07040', '#5070C0', '#70B040']


def blend_color(weights):
    weights = np.array(weights) / np.sum(weights)
    return weights @ FACTOR_COLORS


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
    dominant = np.argmax(W, axis=1)
    order = np.lexsort((-W.max(axis=1), dominant))
    return W, order


def draw_radial(ax, W, item_labels=None):
    """Radial layout - factors as spokes, items positioned by their loadings."""
    n, k = W.shape
    center = np.array([0.5, 0.5])

    # Factor angles (120° apart)
    angles = np.array([np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3])

    # Draw factor spokes
    spoke_length = 0.38
    for j in range(k):
        end = center + spoke_length * np.array([np.cos(angles[j]), np.sin(angles[j])])
        # Gradient line
        ax.plot([center[0], end[0]], [center[1], end[1]],
                color=FACTOR_HEX[j], lw=3, alpha=0.3, zorder=1)
        # Arrow at end
        ax.annotate('', xy=end, xytext=center,
                   arrowprops=dict(arrowstyle='->', color=FACTOR_HEX[j], lw=1.5))

    # Factor labels
    label_dist = 0.44
    factor_labels = ['animate', 'round', 'natural']
    for j in range(k):
        lpos = center + label_dist * np.array([np.cos(angles[j]), np.sin(angles[j])])
        ax.text(lpos[0], lpos[1], factor_labels[j], fontsize=8, ha='center', va='center',
                color=FACTOR_HEX[j], fontweight='medium', style='italic')

    # Position items using barycentric → polar
    # Each item's angle is determined by its factor weights
    # Distance from center = how "pure" the assignment is
    for i in range(n):
        # Weighted average of angles
        item_angle = np.sum(W[i] * angles)

        # Distance: items with pure assignments are further out
        # Items with mixed assignments are closer to center
        purity = np.max(W[i])  # how dominant is the strongest factor
        item_dist = 0.15 + purity * 0.22

        pos = center + item_dist * np.array([np.cos(item_angle), np.sin(item_angle)])
        color = blend_color(W[i])

        circle = mpatches.Circle(pos, 0.022, facecolor=color,
                                  edgecolor='white', lw=0.8, zorder=3)
        ax.add_patch(circle)

        if item_labels and i < len(item_labels):
            # Label offset
            label_offset = 0.04 * np.array([np.cos(item_angle), np.sin(item_angle)])
            ax.text(pos[0] + label_offset[0], pos[1] + label_offset[1],
                    item_labels[i], fontsize=4.5, ha='center', va='center',
                    color=CHARCOAL, alpha=0.8)

    # Central hub
    hub = mpatches.Circle(center, 0.03, facecolor='white',
                          edgecolor=MEDIUM_GRAY, lw=1, zorder=2)
    ax.add_patch(hub)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_radar_grid(ax, W, item_labels=None, highlight_items=None):
    """Radar/spider plot showing a few representative items."""
    if highlight_items is None:
        # Pick representative items: one pure per factor, one mixed
        highlight_items = [0, 4, 8, 12]  # one from each cluster + one mixed

    n_show = len(highlight_items)
    k = W.shape[1]

    # Radar setup
    angles = np.linspace(0, 2*np.pi, k, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    factor_labels = ['animate', 'round', 'natural']

    # Draw axis grid
    center = (0.5, 0.5)
    for r in [0.25, 0.5, 0.75, 1.0]:
        points = []
        for a in angles[:-1]:
            x = 0.5 + r * 0.35 * np.cos(a - np.pi/2)
            y = 0.5 + r * 0.35 * np.sin(a - np.pi/2)
            points.append((x, y))
        points.append(points[0])
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=LIGHT_GRAY, lw=0.5, alpha=0.5)

    # Factor labels
    for j, a in enumerate(angles[:-1]):
        x = 0.5 + 1.15 * 0.35 * np.cos(a - np.pi/2)
        y = 0.5 + 1.15 * 0.35 * np.sin(a - np.pi/2)
        ax.text(x, y, factor_labels[j], fontsize=7, ha='center', va='center',
                color=FACTOR_HEX[j], fontweight='medium', style='italic')

    # Draw spokes
    for a in angles[:-1]:
        x = 0.5 + 0.35 * np.cos(a - np.pi/2)
        y = 0.5 + 0.35 * np.sin(a - np.pi/2)
        ax.plot([0.5, x], [0.5, y], color=LIGHT_GRAY, lw=0.5)

    # Plot highlighted items
    for idx in highlight_items:
        values = W[idx].tolist()
        values += values[:1]

        points = []
        for v, a in zip(values, angles):
            x = 0.5 + v * 0.35 * np.cos(a - np.pi/2)
            y = 0.5 + v * 0.35 * np.sin(a - np.pi/2)
            points.append((x, y))
        xs, ys = zip(*points)

        color = blend_color(W[idx])
        ax.fill(xs, ys, color=color, alpha=0.15)
        ax.plot(xs, ys, color=color, lw=1.5, alpha=0.8)

        # Mark vertices
        for x, y in points[:-1]:
            ax.scatter(x, y, c=[color], s=20, edgecolor='white', linewidth=0.4, zorder=3)

    # Legend
    if item_labels:
        for i, idx in enumerate(highlight_items):
            color = blend_color(W[idx])
            y_leg = 0.12 - i * 0.035
            circle = mpatches.Circle((0.15, y_leg), 0.012, facecolor=color, edgecolor='white', lw=0.5)
            ax.add_patch(circle)
            ax.text(0.18, y_leg, item_labels[idx], fontsize=5, va='center', color=CHARCOAL)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_force_layout(ax, W, item_labels=None, seed=42):
    """Force-directed layout where items are attracted to factors."""
    rng = np.random.default_rng(seed)
    n, k = W.shape

    # Factor positions (fixed anchors in triangle)
    factor_pos = np.array([[0.5, 0.85], [0.15, 0.2], [0.85, 0.2]])

    # Initialize item positions
    item_pos = rng.uniform(0.2, 0.8, (n, 2))

    # Simple force-directed simulation
    for _ in range(100):
        forces = np.zeros((n, 2))

        # Attraction to factors (proportional to loading)
        for i in range(n):
            for j in range(k):
                direction = factor_pos[j] - item_pos[i]
                dist = np.linalg.norm(direction) + 0.01
                forces[i] += W[i, j] * direction / dist * 0.1

        # Repulsion between items
        for i in range(n):
            for i2 in range(i + 1, n):
                diff = item_pos[i] - item_pos[i2]
                dist = np.linalg.norm(diff) + 0.01
                if dist < 0.15:
                    repel = diff / dist * 0.01 / (dist**2)
                    forces[i] += repel
                    forces[i2] -= repel

        item_pos += forces
        item_pos = np.clip(item_pos, 0.1, 0.9)

    # Draw factor anchors
    factor_labels = ['animate', 'round', 'natural']
    for j in range(k):
        circle = mpatches.Circle(factor_pos[j], 0.045, facecolor=FACTOR_HEX[j],
                                  edgecolor='white', lw=1.5, zorder=4)
        ax.add_patch(circle)
        ax.text(factor_pos[j][0], factor_pos[j][1] + 0.08, factor_labels[j],
                fontsize=7, ha='center', color=FACTOR_HEX[j], fontweight='medium', style='italic')

    # Draw attraction lines (faint)
    for i in range(n):
        for j in range(k):
            if W[i, j] > 0.15:
                alpha = W[i, j] * 0.3
                ax.plot([item_pos[i, 0], factor_pos[j, 0]],
                       [item_pos[i, 1], factor_pos[j, 1]],
                       color=FACTOR_HEX[j], alpha=alpha, lw=0.5, zorder=1)

    # Draw items
    for i in range(n):
        color = blend_color(W[i])
        circle = mpatches.Circle(item_pos[i], 0.025, facecolor=color,
                                  edgecolor='white', lw=0.8, zorder=3)
        ax.add_patch(circle)

        if item_labels and i < len(item_labels):
            ax.text(item_pos[i, 0] + 0.04, item_pos[i, 1], item_labels[i],
                    fontsize=4.5, ha='left', va='center', color=CHARCOAL, alpha=0.8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    W, order = create_data()
    W_ord = W[order]

    item_labels = ['dog', 'cat', 'bird', 'fish',
                   'ball', 'orange', 'apple', 'wheel',
                   'tree', 'rock', 'leaf', 'cloud',
                   'turtle', 'egg', 'snail']
    item_labels_ord = [item_labels[i] for i in order]

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # 1. Radial layout
    fig1, ax1 = plt.subplots(figsize=(4.5, 4.5))
    draw_radial(ax1, W_ord, item_labels_ord)
    fig1.text(0.5, 0.96, 'Radial Factor Layout', fontsize=10, ha='center',
              color=CHARCOAL, fontweight='medium')
    fig1.savefig(output_dir / "w_radial.svg", format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig1)
    print("Saved: w_radial.svg")

    # 2. Radar grid (representative items)
    fig2, ax2 = plt.subplots(figsize=(4, 4.5))
    draw_radar_grid(ax2, W_ord, item_labels_ord, highlight_items=[0, 4, 8, 12])
    fig2.text(0.5, 0.96, 'Factor Profiles (Radar)', fontsize=10, ha='center',
              color=CHARCOAL, fontweight='medium')
    fig2.savefig(output_dir / "w_radar.svg", format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig2)
    print("Saved: w_radar.svg")

    # 3. Force-directed layout
    fig3, ax3 = plt.subplots(figsize=(4.5, 4.5))
    draw_force_layout(ax3, W_ord, item_labels_ord)
    fig3.text(0.5, 0.96, 'Force-Directed Layout', fontsize=10, ha='center',
              color=CHARCOAL, fontweight='medium')
    fig3.savefig(output_dir / "w_force.svg", format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig3)
    print("Saved: w_force.svg")

    # Combined comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    draw_radial(axes[0], W_ord, item_labels_ord)
    axes[0].set_title('Radial', fontsize=9, color=CHARCOAL, pad=5)

    draw_radar_grid(axes[1], W_ord, item_labels_ord, highlight_items=[0, 4, 8, 12])
    axes[1].set_title('Radar Profiles', fontsize=9, color=CHARCOAL, pad=5)

    draw_force_layout(axes[2], W_ord, item_labels_ord)
    axes[2].set_title('Force-Directed', fontsize=9, color=CHARCOAL, pad=5)

    fig.suptitle('More Alternatives for Factor Loadings',
                 fontsize=11, color=CHARCOAL, fontweight='medium', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / "w_radial_comparison.svg", format='svg',
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("Saved: w_radial_comparison.svg")


if __name__ == "__main__":
    main()
