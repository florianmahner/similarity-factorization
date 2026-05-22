"""Arc diagram - standalone figure matching RSM ordering."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#c8c8c8'
MEDIUM_GRAY = '#888888'

# Colors chosen for distinct blends:
# F1+F2 = warm purple, F2+F3 = teal, F1+F3 = olive
FACTOR_COLORS = np.array([
    [0.88, 0.44, 0.25],  # orange-coral (#E07040)
    [0.31, 0.44, 0.75],  # blue-indigo (#5070C0)
    [0.44, 0.69, 0.25],  # yellow-green (#70B040)
])
FACTOR_HEX = ['#E07040', '#5070C0', '#70B040']


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
    # Normalize so diagonal is 1
    d = np.sqrt(np.diag(S))
    S = S / np.outer(d, d)

    dominant = np.argmax(W, axis=1)
    order = np.lexsort((-W.max(axis=1), dominant))

    return W, S, order


def blend_color(weights):
    """Blend factor colors based on membership weights."""
    weights = np.array(weights)
    weights = weights / weights.sum()
    rgb = weights @ FACTOR_COLORS
    return rgb


def draw_arc(ax, x1, x2, y_base=0, height_scale=0.5, clip_on=False, **kwargs):
    theta = np.linspace(0, np.pi, 50)
    center = (x1 + x2) / 2
    width = abs(x2 - x1)
    height = width * height_scale
    x = center + (width / 2) * np.cos(theta)
    y = y_base + height * np.sin(theta) / 2
    line, = ax.plot(x, y, **kwargs)
    line.set_clip_on(clip_on)


def main():
    W, S, order = create_data()
    n = len(W)
    dominant = np.argmax(W, axis=1)

    # Scrambled order - random permutation to break visible structure
    rng = np.random.default_rng(123)
    scramble = rng.permutation(n)

    # X positions evenly spaced (no gaps since order is random)
    positions = {node: i / (n - 1) for i, node in enumerate(scramble)}

    fig, ax = plt.subplots(figsize=(5.5, 2.0))

    # Draw arcs - thickness/alpha based on similarity
    threshold = 0.15
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > threshold:
                x1, x2 = positions[i], positions[j]
                if x1 > x2:
                    x1, x2 = x2, x1

                # Scale alpha and linewidth by similarity
                strength = (S[i, j] - threshold) / (1 - threshold)
                alpha = 0.15 + 0.6 * strength
                lw = 0.3 + 0.8 * strength

                draw_arc(ax, x1, x2, height_scale=0.85,
                        color=LIGHT_GRAY, linewidth=lw, alpha=alpha, zorder=1, clip_on=False)

    # Draw nodes with blended colors (no clipping)
    node_radius = 0.022
    for node in range(n):
        x = positions[node]
        color = blend_color(W[node])

        circle = mpatches.Circle(
            (x, 0), node_radius,
            facecolor=color,
            edgecolor='white',
            linewidth=0.8,
            zorder=3,
            clip_on=False
        )
        ax.add_patch(circle)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.16, 0.58)
    ax.set_aspect('equal')
    ax.axis('off')

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "figure_arc.svg", format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("Saved: figure_arc.svg")


if __name__ == "__main__":
    main()
