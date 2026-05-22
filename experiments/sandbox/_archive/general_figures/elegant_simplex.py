"""Elegant ternary simplex visualization for soft clustering.

Node position encodes factor membership via barycentric coordinates.
No garish colors - position tells the story.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Sophisticated muted palette
CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#c8c8c8'
MEDIUM_GRAY = '#888888'
FAINT_GRAY = '#e8e8e8'

# Muted factor colors (for subtle accents only)
FACTOR_COLORS = ['#c75b5b', '#5b8fc7', '#5bc77a']  # muted red, blue, green


def barycentric_to_cartesian(weights):
    """Convert barycentric coordinates to 2D cartesian on equilateral triangle."""
    # Triangle corners
    corners = np.array([
        [0, 0],           # Factor 1 (bottom left)
        [1, 0],           # Factor 2 (bottom right)
        [0.5, np.sqrt(3)/2]  # Factor 3 (top)
    ])
    weights = np.array(weights)
    weights = weights / weights.sum()  # normalize
    return weights @ corners


def create_elegant_data(n_per_cluster=4, n_overlap=3, seed=42):
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

    # Overlap items (between pairs)
    # F1-F2 overlap
    W_list.append([0.45, 0.45, 0.10])
    # F1-F3 overlap
    W_list.append([0.48, 0.08, 0.44])
    # F2-F3 overlap
    W_list.append([0.08, 0.46, 0.46])

    W = np.array(W_list)
    W = W / W.sum(axis=1, keepdims=True)  # normalize rows

    # Compute similarity matrix
    S = W @ W.T

    return W, S


def draw_triangle_frame(ax):
    """Draw subtle triangle frame."""
    corners = np.array([
        [0, 0],
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [0, 0]  # close
    ])
    ax.plot(corners[:, 0], corners[:, 1],
            color=LIGHT_GRAY, linewidth=0.8, zorder=1)


def draw_factor_labels(ax):
    """Add subtle factor labels at corners."""
    labels = [
        (0, 0, 'F₁', 'right', 'top'),
        (1, 0, 'F₂', 'left', 'top'),
        (0.5, np.sqrt(3)/2, 'F₃', 'center', 'bottom'),
    ]
    for x, y, label, ha, va in labels:
        # Offset slightly outside triangle
        if va == 'top':
            y -= 0.06
        else:
            y += 0.06
        if ha == 'right':
            x -= 0.04
        elif ha == 'left':
            x += 0.04

        ax.text(x, y, label, fontsize=9, ha='center', va='center',
                color=MEDIUM_GRAY, fontweight='medium')


def main():
    W, S = create_elegant_data()
    n = len(W)

    # Convert to positions
    positions = np.array([barycentric_to_cartesian(W[i]) for i in range(n)])

    # Create figure
    fig, ax = plt.subplots(figsize=(3.2, 3.0))

    # Draw triangle frame
    draw_triangle_frame(ax)

    # Draw similarity edges (only above threshold, with alpha)
    threshold = 0.25
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > threshold:
                alpha = (S[i, j] - threshold) / (1 - threshold)
                alpha = alpha ** 0.7  # compress range slightly
                ax.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    color=LIGHT_GRAY,
                    linewidth=0.5,
                    alpha=alpha * 0.8,
                    zorder=2
                )

    # Draw nodes
    for i in range(n):
        x, y = positions[i]

        # Determine dominant factor for subtle color accent
        dominant = np.argmax(W[i])
        is_mixed = W[i].max() < 0.6  # mixed membership

        if is_mixed:
            # Mixed nodes: hollow with gray stroke
            circle = mpatches.Circle(
                (x, y), 0.028,
                facecolor='white',
                edgecolor=CHARCOAL,
                linewidth=1.0,
                zorder=4
            )
        else:
            # Pure nodes: filled with muted factor color
            circle = mpatches.Circle(
                (x, y), 0.028,
                facecolor=FACTOR_COLORS[dominant],
                edgecolor='white',
                linewidth=0.8,
                zorder=4
            )
        ax.add_patch(circle)

    # Add factor labels
    draw_factor_labels(ax)

    # Clean up axes
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.15, 1.02)
    ax.set_aspect('equal')
    ax.axis('off')

    # Save
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    fig.savefig(output_dir / "elegant_simplex.svg", format='svg',
                bbox_inches='tight', pad_inches=0.05)
    fig.savefig(output_dir / "elegant_simplex.png", dpi=200,
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    print("Saved: elegant_simplex.svg")


if __name__ == "__main__":
    main()
