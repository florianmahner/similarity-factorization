"""Standalone W embedding matrix figure - scrambled order."""

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
    return W


def blend_color(weights):
    weights = np.array(weights)
    weights = weights / weights.sum()
    return weights @ FACTOR_COLORS


def draw_w_matrix(ax, W):
    n, k = W.shape
    cell_h = 0.9 / n
    cell_w = 0.20

    for i in range(n):
        y = 0.95 - (i + 1) * cell_h
        for j in range(k):
            val = W[i, j]
            rect = mpatches.Rectangle(
                (0.40 + j * cell_w, y), cell_w * 0.9, cell_h * 0.9,
                facecolor=FACTOR_HEX[j], alpha=val, edgecolor='none')
            ax.add_patch(rect)

        color = blend_color(W[i])
        circle = mpatches.Circle((0.18, y + cell_h * 0.45), 0.035,
                                  facecolor=color, edgecolor='white', lw=0.6)
        ax.add_patch(circle)

        ax.plot([0.22, 0.38], [y + cell_h * 0.45, y + cell_h * 0.45],
                color=LIGHT_GRAY, lw=0.4, alpha=0.5)

    ax.text(0.18, 0.02, 'nodes', fontsize=7, ha='center', color=MEDIUM_GRAY)
    for j in range(k):
        ax.text(0.40 + (j + 0.5) * cell_w, 0.02, f'F{j+1}',
                fontsize=7, ha='center', color=FACTOR_HEX[j], fontweight='medium')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def main():
    W = create_data()
    n = len(W)

    # Scrambled order
    rng = np.random.default_rng(123)
    scramble = rng.permutation(n)
    W_scrambled = W[scramble]

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(2.5, 4.0))
    draw_w_matrix(ax, W_scrambled)
    fig.savefig(output_dir / "figure_w_scrambled.svg", format='svg',
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("Saved: figure_w_scrambled.svg")


if __name__ == "__main__":
    main()
