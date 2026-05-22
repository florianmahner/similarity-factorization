"""Complexity figure - showing W matrices and RSMs at different complexity levels."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
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


def create_rsm_colormap():
    """Elegant blue colormap for RSM."""
    colors = ['#ffffff', '#f0f4f8', '#d4e1ed', '#a8c5db',
              '#6a9fc0', '#3d7ea6', '#1a5a7a', '#0d3d54']
    return mcolors.LinearSegmentedColormap.from_list('elegant', colors)


def blend_color(weights):
    weights = np.array(weights) / (np.sum(weights) + 1e-10)
    return weights @ FACTOR_COLORS


def create_w_simple(n=20, k=3, seed=42):
    """Simple/sparse W - clear block structure, low complexity."""
    rng = np.random.default_rng(seed)
    W = np.zeros((n, k))
    items_per_factor = n // k

    for j in range(k):
        start = j * items_per_factor
        end = start + items_per_factor if j < k - 1 else n
        for i in range(start, end):
            W[i, j] = rng.uniform(0.85, 0.95)
            for other in range(k):
                if other != j:
                    W[i, other] = rng.uniform(0.02, 0.08)

    W = W / W.sum(axis=1, keepdims=True)
    return W


def create_w_medium(n=20, k=3, seed=42):
    """Medium complexity W - some overlap between factors."""
    rng = np.random.default_rng(seed)
    W = np.zeros((n, k))
    items_per_factor = n // k

    for j in range(k):
        start = j * items_per_factor
        end = start + items_per_factor if j < k - 1 else n
        for i in range(start, end):
            W[i, j] = rng.uniform(0.5, 0.7)
            for other in range(k):
                if other != j:
                    W[i, other] = rng.uniform(0.1, 0.25)

    # Add some mixed items
    for i in range(0, n, 4):
        W[i] = rng.uniform(0.2, 0.5, k)

    W = W / W.sum(axis=1, keepdims=True)
    return W


def create_w_complex(n=20, k=3, seed=42):
    """High complexity W - distributed, no clear clusters."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(0.2, 0.6, (n, k))
    # Add some variation
    W += rng.normal(0, 0.1, (n, k))
    W = np.clip(W, 0.05, 1)
    W = W / W.sum(axis=1, keepdims=True)
    return W


def draw_w_matrix(ax, W):
    """Draw W matrix with factor colors."""
    n, k = W.shape
    cell_h = 1.0 / n
    cell_w = 1.0 / k

    for i in range(n):
        for j in range(k):
            val = W[i, j]
            rect = mpatches.Rectangle(
                (j * cell_w, 1 - (i + 1) * cell_h),
                cell_w * 0.95, cell_h * 0.95,
                facecolor=FACTOR_HEX[j], alpha=val, edgecolor='none')
            ax.add_patch(rect)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_rsm(ax, S, cmap):
    """Draw RSM as vector graphic."""
    n = len(S)
    cell = 1.0 / n

    for i in range(n):
        for j in range(n):
            rect = mpatches.Rectangle(
                (j * cell, 1 - (i + 1) * cell), cell, cell,
                facecolor=cmap(S[i, j]), edgecolor='none')
            ax.add_patch(rect)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_w_with_nodes(ax, W):
    """W matrix with colored node indicators on the left."""
    n, k = W.shape
    cell_h = 0.9 / n
    cell_w = 0.22
    x_offset = 0.25

    for i in range(n):
        y = 0.95 - (i + 1) * cell_h
        for j in range(k):
            val = W[i, j]
            rect = mpatches.Rectangle(
                (x_offset + j * cell_w, y),
                cell_w * 0.92, cell_h * 0.88,
                facecolor=FACTOR_HEX[j], alpha=val, edgecolor='none')
            ax.add_patch(rect)

        # Node indicator
        color = blend_color(W[i])
        circle = mpatches.Circle((0.1, y + cell_h * 0.44), 0.028,
                                  facecolor=color, edgecolor='white', lw=0.6)
        ax.add_patch(circle)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def main():
    n = 20
    k = 3
    cmap = create_rsm_colormap()

    # Create W matrices at different complexity levels
    W_simple = create_w_simple(n, k, seed=42)
    W_medium = create_w_medium(n, k, seed=43)
    W_complex = create_w_complex(n, k, seed=44)

    # Compute RSMs
    S_simple = W_simple @ W_simple.T
    S_medium = W_medium @ W_medium.T
    S_complex = W_complex @ W_complex.T

    # Normalize for visualization
    for S in [S_simple, S_medium, S_complex]:
        S /= S.max()

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # === Option A: Clean grid layout ===
    fig_a = plt.figure(figsize=(10, 5.5))

    # Complexity arrow at top
    ax_arrow = fig_a.add_axes([0.15, 0.88, 0.7, 0.08])
    ax_arrow.annotate('', xy=(0.95, 0.5), xytext=(0.05, 0.5),
                      arrowprops=dict(arrowstyle='->', color=CHARCOAL, lw=1.5))
    ax_arrow.text(0.5, 0.5, 'Complexity', fontsize=10, ha='center', va='bottom',
                  color=CHARCOAL, fontweight='medium')
    ax_arrow.set_xlim(0, 1)
    ax_arrow.set_ylim(0, 1)
    ax_arrow.axis('off')

    # Column labels
    col_labels = ['Sparse', 'Mixed', 'Distributed']
    col_x = [0.18, 0.48, 0.78]
    for x, label in zip(col_x, col_labels):
        fig_a.text(x, 0.82, label, fontsize=9, ha='center', color=MEDIUM_GRAY, style='italic')

    # Row labels
    fig_a.text(0.04, 0.62, 'W', fontsize=11, ha='center', va='center', color=CHARCOAL, fontweight='medium')
    fig_a.text(0.04, 0.28, 'S', fontsize=11, ha='center', va='center', color=CHARCOAL, fontweight='medium')

    # W matrices (top row)
    w_height = 0.32
    w_width = 0.22
    y_w = 0.48

    ax_w1 = fig_a.add_axes([0.08, y_w, w_width, w_height])
    draw_w_matrix(ax_w1, W_simple)

    ax_w2 = fig_a.add_axes([0.38, y_w, w_width, w_height])
    draw_w_matrix(ax_w2, W_medium)

    ax_w3 = fig_a.add_axes([0.68, y_w, w_width, w_height])
    draw_w_matrix(ax_w3, W_complex)

    # RSMs (bottom row)
    s_size = 0.28
    y_s = 0.10

    ax_s1 = fig_a.add_axes([0.10, y_s, s_size, s_size])
    draw_rsm(ax_s1, S_simple, cmap)

    ax_s2 = fig_a.add_axes([0.40, y_s, s_size, s_size])
    draw_rsm(ax_s2, S_medium, cmap)

    ax_s3 = fig_a.add_axes([0.70, y_s, s_size, s_size])
    draw_rsm(ax_s3, S_complex, cmap)

    # Arrows connecting W to S
    for x in [0.19, 0.49, 0.79]:
        fig_a.text(x, 0.44, '↓', fontsize=12, ha='center', color=LIGHT_GRAY)

    fig_a.savefig(output_dir / "complexity_optionA.svg", format='svg',
                  bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_a)
    print("Saved: complexity_optionA.svg")

    # === Option B: With node indicators ===
    fig_b = plt.figure(figsize=(11, 5.5))

    # Complexity arrow
    ax_arrow = fig_b.add_axes([0.12, 0.88, 0.76, 0.08])
    ax_arrow.annotate('', xy=(0.95, 0.5), xytext=(0.05, 0.5),
                      arrowprops=dict(arrowstyle='->', color=CHARCOAL, lw=1.5))
    ax_arrow.text(0.5, 0.5, 'Complexity', fontsize=10, ha='center', va='bottom',
                  color=CHARCOAL, fontweight='medium')
    ax_arrow.set_xlim(0, 1)
    ax_arrow.set_ylim(0, 1)
    ax_arrow.axis('off')

    # Column labels
    for x, label in zip([0.17, 0.48, 0.79], col_labels):
        fig_b.text(x, 0.82, label, fontsize=9, ha='center', color=MEDIUM_GRAY, style='italic')

    # Row labels
    fig_b.text(0.02, 0.62, 'Dimensions\n(W)', fontsize=9, ha='center', va='center', color=CHARCOAL)
    fig_b.text(0.02, 0.25, 'Similarity\n(S = WWᵀ)', fontsize=9, ha='center', va='center', color=CHARCOAL)

    # W matrices with nodes
    w_width = 0.26
    w_height = 0.32
    y_w = 0.48

    ax_w1 = fig_b.add_axes([0.06, y_w, w_width, w_height])
    draw_w_with_nodes(ax_w1, W_simple)

    ax_w2 = fig_b.add_axes([0.37, y_w, w_width, w_height])
    draw_w_with_nodes(ax_w2, W_medium)

    ax_w3 = fig_b.add_axes([0.68, y_w, w_width, w_height])
    draw_w_with_nodes(ax_w3, W_complex)

    # RSMs
    s_size = 0.28
    y_s = 0.08

    ax_s1 = fig_b.add_axes([0.08, y_s, s_size, s_size])
    draw_rsm(ax_s1, S_simple, cmap)

    ax_s2 = fig_b.add_axes([0.39, y_s, s_size, s_size])
    draw_rsm(ax_s2, S_medium, cmap)

    ax_s3 = fig_b.add_axes([0.70, y_s, s_size, s_size])
    draw_rsm(ax_s3, S_complex, cmap)

    # Connecting arrows
    for x in [0.19, 0.50, 0.81]:
        fig_b.text(x, 0.42, '↓', fontsize=12, ha='center', color=LIGHT_GRAY)

    fig_b.savefig(output_dir / "complexity_optionB.svg", format='svg',
                  bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_b)
    print("Saved: complexity_optionB.svg")

    # === Option C: Horizontal flow with equation ===
    fig_c = plt.figure(figsize=(12, 4))

    # Title/arrow
    fig_c.text(0.5, 0.95, 'Effect of Dimension Structure on Similarity',
               fontsize=11, ha='center', color=CHARCOAL, fontweight='medium')

    y_row = 0.18
    h_mat = 0.65

    # Three columns
    for col, (W, S, label) in enumerate([(W_simple, S_simple, 'Sparse\n(clear clusters)'),
                                          (W_medium, S_medium, 'Mixed\n(soft boundaries)'),
                                          (W_complex, S_complex, 'Distributed\n(uniform)')]):
        x_base = 0.02 + col * 0.33

        # W matrix
        ax_w = fig_c.add_axes([x_base, y_row, 0.12, h_mat])
        draw_w_matrix(ax_w, W)
        fig_c.text(x_base + 0.06, 0.08, 'W', fontsize=9, ha='center', color=CHARCOAL)

        # Arrow
        fig_c.text(x_base + 0.145, 0.50, '→', fontsize=14, ha='center', color=MEDIUM_GRAY)

        # RSM
        ax_s = fig_c.add_axes([x_base + 0.17, y_row + 0.08, h_mat * 0.7, h_mat * 0.7])
        draw_rsm(ax_s, S, cmap)
        fig_c.text(x_base + 0.17 + h_mat * 0.35, 0.08, 'S = WWᵀ', fontsize=9, ha='center', color=CHARCOAL)

        # Column label
        fig_c.text(x_base + 0.15, 0.88, label, fontsize=8, ha='center',
                   color=MEDIUM_GRAY, style='italic', linespacing=1.3)

    # Complexity arrow at bottom
    ax_arr = fig_c.add_axes([0.1, 0.01, 0.8, 0.05])
    ax_arr.annotate('', xy=(0.95, 0.5), xytext=(0.05, 0.5),
                    arrowprops=dict(arrowstyle='->', color=CHARCOAL, lw=1.2))
    ax_arr.text(0.5, 0.8, 'Complexity', fontsize=9, ha='center', color=CHARCOAL)
    ax_arr.axis('off')

    fig_c.savefig(output_dir / "complexity_optionC.svg", format='svg',
                  bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_c)
    print("Saved: complexity_optionC.svg")


if __name__ == "__main__":
    main()
