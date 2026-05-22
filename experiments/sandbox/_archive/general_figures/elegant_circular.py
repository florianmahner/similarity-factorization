"""Elegant circular graph - nodes on circle, grouped by factor.

Clean, modern look with edges inside the circle.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Sophisticated palette
CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#d0d0d0'
MEDIUM_GRAY = '#888888'

# Muted factor colors
FACTOR_COLORS = ['#c75b5b', '#5b8fc7', '#5bc77a']


def create_data(seed=42):
    """Create synthetic soft membership data."""
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


def circular_layout(W):
    """Position nodes on circle, grouped by dominant factor."""
    n = len(W)
    dominant = np.argmax(W, axis=1)

    # Sort by dominant factor, then by membership strength
    order = np.lexsort((-W.max(axis=1), dominant))

    # Assign angles - leave gaps between factor groups
    positions = {}
    angles = []

    # Calculate angles with gaps between groups
    group_gap = 0.15  # gap between factor groups (in radians proportion)
    usable = 2 * np.pi * (1 - 3 * group_gap)

    # Count per group
    counts = [np.sum(dominant == k) for k in range(3)]

    current_angle = np.pi / 2  # start at top

    for k in range(3):
        group_items = [i for i in order if dominant[i] == k]
        n_group = len(group_items)

        if n_group > 0:
            arc_length = usable * (n_group / n)
            angles_group = np.linspace(
                current_angle,
                current_angle - arc_length,
                n_group,
                endpoint=False
            )

            for idx, node in enumerate(group_items):
                positions[node] = angles_group[idx]

            current_angle -= arc_length + 2 * np.pi * group_gap

    return positions


def main():
    W, S = create_data()
    n = len(W)

    angles = circular_layout(W)
    radius = 0.8

    # Convert to cartesian
    pos = {i: (radius * np.cos(angles[i]), radius * np.sin(angles[i])) for i in range(n)}

    fig, ax = plt.subplots(figsize=(3.2, 3.2))

    # Draw edges (curved bezier through center)
    threshold = 0.20
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > threshold:
                x1, y1 = pos[i]
                x2, y2 = pos[j]

                # Alpha based on similarity
                alpha = 0.15 + 0.5 * (S[i, j] - threshold) / (1 - threshold)

                # Draw curved edge (quadratic bezier toward center)
                # Control point toward center
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                dist = np.sqrt(mid_x**2 + mid_y**2)
                # Pull toward center proportional to chord length
                chord = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                pull = 0.3 * chord
                ctrl_x = mid_x * (1 - pull / (dist + 0.01))
                ctrl_y = mid_y * (1 - pull / (dist + 0.01))

                # Draw as simple line (bezier is complex in matplotlib)
                ax.plot([x1, ctrl_x, x2], [y1, ctrl_y, y2],
                        color=LIGHT_GRAY, linewidth=0.5, alpha=alpha, zorder=1)

    # Draw nodes
    for i in range(n):
        x, y = pos[i]
        dominant = np.argmax(W[i])
        max_weight = W[i].max()

        node_size = 0.07

        if max_weight < 0.55:
            circle = mpatches.Circle(
                (x, y), node_size,
                facecolor='white',
                edgecolor=CHARCOAL,
                linewidth=1.2,
                zorder=3
            )
        else:
            circle = mpatches.Circle(
                (x, y), node_size,
                facecolor=FACTOR_COLORS[dominant],
                edgecolor='white',
                linewidth=0.8,
                zorder=3
            )
        ax.add_patch(circle)

    # Factor labels outside circle
    label_radius = 1.05
    label_angles = [np.pi/2 + 0.15, np.pi/2 - 2*np.pi/3 + 0.1, np.pi/2 + 2*np.pi/3 - 0.1]
    for k, angle in enumerate(label_angles):
        x = label_radius * np.cos(angle)
        y = label_radius * np.sin(angle)
        ax.text(x, y, f'F{k+1}', fontsize=9, ha='center', va='center',
                color=FACTOR_COLORS[k], fontweight='medium')

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect('equal')
    ax.axis('off')

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    fig.savefig(output_dir / "elegant_circular.svg", format='svg',
                bbox_inches='tight', pad_inches=0.05)
    fig.savefig(output_dir / "elegant_circular.png", dpi=200,
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    print("Saved: elegant_circular.svg")


if __name__ == "__main__":
    main()
