"""Consensus visualization - three options for showing candidate embeddings."""

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
    return W[order]


def draw_w_matrix_mini(ax, W, perm=None, add_noise=0, show_labels=True):
    """Small W matrix with optional permutation and noise."""
    n, k = W.shape

    if perm is not None:
        W = W[:, perm]
        colors = [FACTOR_HEX[p] for p in perm]
    else:
        colors = FACTOR_HEX

    if add_noise > 0:
        rng = np.random.default_rng(int(add_noise * 1000))
        W = W + rng.normal(0, add_noise, W.shape)
        W = np.clip(W, 0.02, 1)
        W = W / W.sum(axis=1, keepdims=True)

    cell_h = 0.88 / n
    cell_w = 0.25

    for i in range(n):
        y = 0.94 - (i + 1) * cell_h
        for j in range(k):
            val = W[i, j]
            rect = mpatches.Rectangle(
                (0.12 + j * cell_w, y), cell_w * 0.9, cell_h * 0.9,
                facecolor=colors[j], alpha=val, edgecolor='none')
            ax.add_patch(rect)

    if show_labels:
        for j in range(k):
            if perm is not None:
                label = f'F{perm[j]+1}'
            else:
                label = f'F{j+1}'
            ax.text(0.12 + (j + 0.5) * cell_w, 0.02, label,
                    fontsize=6, ha='center', color=colors[j], fontweight='medium')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_w_matrix_full(ax, W):
    """Full W matrix with nodes linked to rows."""
    n, k = W.shape
    cell_h = 0.85 / n
    cell_w = 0.22

    for i in range(n):
        y = 0.92 - (i + 1) * cell_h
        for j in range(k):
            val = W[i, j]
            rect = mpatches.Rectangle(
                (0.38 + j * cell_w, y), cell_w * 0.88, cell_h * 0.88,
                facecolor=FACTOR_HEX[j], alpha=val, edgecolor='none')
            ax.add_patch(rect)

        color = blend_color(W[i])
        circle = mpatches.Circle((0.15, y + cell_h * 0.44), 0.028,
                                  facecolor=color, edgecolor='white', lw=0.6)
        ax.add_patch(circle)
        ax.plot([0.18, 0.36], [y + cell_h * 0.44, y + cell_h * 0.44],
                color=LIGHT_GRAY, lw=0.4, alpha=0.5)

    for j in range(k):
        ax.text(0.38 + (j + 0.5) * cell_w, 0.02, f'F{j+1}',
                fontsize=7, ha='center', color=FACTOR_HEX[j], fontweight='medium')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_nodes_colored(ax, W, perm=None, add_noise=0, seed=42):
    """Nodes colored by membership."""
    n = len(W)
    rng = np.random.default_rng(seed)

    if perm is not None:
        W = W[:, perm]

    if add_noise > 0:
        W = W + rng.normal(0, add_noise, W.shape)
        W = np.clip(W, 0.02, 1)
        W = W / W.sum(axis=1, keepdims=True)

    # Arrange in a grid
    cols = 5
    rows = (n + cols - 1) // cols

    for i in range(n):
        row = i // cols
        col = i % cols
        x = 0.12 + col * 0.19
        y = 0.85 - row * 0.25

        color = blend_color(W[i])
        circle = mpatches.Circle((x, y), 0.07, facecolor=color,
                                  edgecolor='white', lw=0.8)
        ax.add_patch(circle)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_dimension_clustering(ax, rng):
    """Dimensions from multiple runs cluster together."""
    cluster_centers = [(0.22, 0.72), (0.72, 0.68), (0.47, 0.22)]
    n_points = 8

    for k, ((cx, cy), col) in enumerate(zip(cluster_centers, FACTOR_HEX)):
        ellipse = mpatches.Ellipse((cx, cy), 0.32, 0.28, angle=rng.uniform(-15, 15),
                                    facecolor=col, alpha=0.12, edgecolor=col, lw=1.2)
        ax.add_patch(ellipse)
        for _ in range(n_points):
            px = cx + rng.normal(0, 0.07)
            py = cy + rng.normal(0, 0.06)
            ax.scatter(px, py, c=[col], s=25, alpha=0.75, edgecolor='white', linewidth=0.4)
        ax.text(cx, cy - 0.22, f'Dim {k+1}', fontsize=7, ha='center', color=col, fontweight='medium')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_rank_selection(ax):
    """Bar chart for rank selection."""
    ranks = [2, 3, 4, 5]
    values = [0.5, 0.92, 0.75, 0.45]
    selected = 1

    bar_width = 0.15
    for i, (r, v) in enumerate(zip(ranks, values)):
        x = 0.2 + i * 0.2
        color = FACTOR_HEX[1] if i == selected else LIGHT_GRAY
        lw = 2 if i == selected else 0
        rect = mpatches.Rectangle((x, 0.25), bar_width, v * 0.55,
                                   facecolor=color, edgecolor=CHARCOAL if i == selected else 'none', lw=lw)
        ax.add_patch(rect)
        ax.text(x + bar_width/2, 0.18, str(r), fontsize=7, ha='center', color=MEDIUM_GRAY)

    ax.text(0.5, 0.08, 'k (rank)', fontsize=7, ha='center', color=MEDIUM_GRAY, style='italic')
    ax.annotate('', xy=(0.2 + selected * 0.2 + bar_width/2, 0.88),
               xytext=(0.2 + selected * 0.2 + bar_width/2, 0.95),
               arrowprops=dict(arrowstyle='->', color=FACTOR_HEX[1], lw=1.5))
    ax.text(0.2 + selected * 0.2 + bar_width/2, 0.97, 'k*', fontsize=8,
            ha='center', color=FACTOR_HEX[1], fontweight='medium')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def create_different_w(seed, n=15):
    """Create a W matrix that's structurally different (not just permuted)."""
    rng = np.random.default_rng(seed)
    W_list = []

    # Different cluster assignments and overlaps per run
    if seed == 1:
        # Run 1: clean 3-factor structure
        for _ in range(5):
            W_list.append([rng.uniform(0.75, 0.90), rng.uniform(0.03, 0.12), rng.uniform(0.03, 0.12)])
        for _ in range(5):
            W_list.append([rng.uniform(0.03, 0.12), rng.uniform(0.75, 0.90), rng.uniform(0.03, 0.12)])
        for _ in range(5):
            W_list.append([rng.uniform(0.03, 0.12), rng.uniform(0.03, 0.12), rng.uniform(0.75, 0.90)])
    elif seed == 2:
        # Run 2: factor 1 and 2 more blended, factor 3 split
        for _ in range(4):
            W_list.append([rng.uniform(0.55, 0.70), rng.uniform(0.20, 0.35), rng.uniform(0.03, 0.12)])
        for _ in range(5):
            W_list.append([rng.uniform(0.15, 0.30), rng.uniform(0.60, 0.75), rng.uniform(0.05, 0.15)])
        for _ in range(3):
            W_list.append([rng.uniform(0.03, 0.12), rng.uniform(0.03, 0.12), rng.uniform(0.80, 0.92)])
        for _ in range(3):
            W_list.append([rng.uniform(0.03, 0.15), rng.uniform(0.25, 0.40), rng.uniform(0.50, 0.65)])
    else:
        # Run 3: different structure again
        for _ in range(6):
            W_list.append([rng.uniform(0.70, 0.88), rng.uniform(0.05, 0.15), rng.uniform(0.03, 0.12)])
        for _ in range(4):
            W_list.append([rng.uniform(0.08, 0.20), rng.uniform(0.65, 0.82), rng.uniform(0.08, 0.18)])
        for _ in range(5):
            W_list.append([rng.uniform(0.10, 0.25), rng.uniform(0.05, 0.15), rng.uniform(0.65, 0.80)])

    W = np.array(W_list)
    W = W / W.sum(axis=1, keepdims=True)
    return W


def option_a():
    """Option A: 3 small W matrices with genuinely different factor structures."""
    W_consensus = create_data()
    rng = np.random.default_rng(42)

    fig = plt.figure(figsize=(12, 3.5))
    fig.text(0.5, 0.95, 'Option A: Different runs find different factors',
             fontsize=10, ha='center', color=CHARCOAL)

    y_main = 0.12
    h_main = 0.72

    # 3 candidate W matrices with genuinely different structures
    for i in range(3):
        ax = fig.add_axes([0.02 + i * 0.105, y_main, 0.09, h_main])
        W_run = create_different_w(seed=i+1)
        draw_w_matrix_mini(ax, W_run, perm=None, add_noise=0)
        ax.set_title(f'Run {i+1}', fontsize=7, color=MEDIUM_GRAY, pad=2)

    fig.text(0.35, 0.50, '→', fontsize=16, color=MEDIUM_GRAY, ha='center')

    # Dimension clustering
    ax_clust = fig.add_axes([0.38, y_main, 0.16, h_main])
    draw_dimension_clustering(ax_clust, rng)
    fig.text(0.46, 0.04, 'Cluster dims', fontsize=8, ha='center', color=MEDIUM_GRAY)

    fig.text(0.555, 0.50, '→', fontsize=16, color=MEDIUM_GRAY, ha='center')

    # Rank selection
    ax_rank = fig.add_axes([0.58, y_main, 0.12, h_main])
    draw_rank_selection(ax_rank)
    fig.text(0.64, 0.04, 'Select rank', fontsize=8, ha='center', color=MEDIUM_GRAY)

    fig.text(0.715, 0.50, '→', fontsize=16, color=MEDIUM_GRAY, ha='center')

    # Consensus
    ax_cons = fig.add_axes([0.74, y_main, 0.24, h_main])
    draw_w_matrix_full(ax_cons, W_consensus)
    fig.text(0.86, 0.04, 'Consensus W*', fontsize=8, ha='center', color=MEDIUM_GRAY)

    box = mpatches.FancyBboxPatch((0.73, y_main - 0.02), 0.26, h_main + 0.06,
                                   boxstyle="round,pad=0.01", facecolor='none',
                                   edgecolor=FACTOR_HEX[1], lw=1.5, transform=fig.transFigure)
    fig.add_artist(box)

    return fig, 'consensus_optionA.svg'


def option_b():
    """Option B: Same nodes with different colorings across runs."""
    W = create_data()
    rng = np.random.default_rng(42)

    fig = plt.figure(figsize=(12, 3.5))
    fig.text(0.5, 0.95, 'Option B: Same nodes, different colorings per run',
             fontsize=10, ha='center', color=CHARCOAL)

    y_main = 0.12
    h_main = 0.72

    # 3 node grids with different colorings
    perms = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    noises = [0.08, 0.12, 0.10]

    for i, (perm, noise) in enumerate(zip(perms, noises)):
        ax = fig.add_axes([0.02 + i * 0.105, y_main, 0.09, h_main])
        draw_nodes_colored(ax, W, perm=perm, add_noise=noise, seed=42+i)
        ax.set_title(f'Run {i+1}', fontsize=7, color=MEDIUM_GRAY, pad=2)

    fig.text(0.35, 0.50, '→', fontsize=16, color=MEDIUM_GRAY, ha='center')

    # Dimension clustering
    ax_clust = fig.add_axes([0.38, y_main, 0.16, h_main])
    draw_dimension_clustering(ax_clust, rng)
    fig.text(0.46, 0.04, 'Cluster dims', fontsize=8, ha='center', color=MEDIUM_GRAY)

    fig.text(0.555, 0.50, '→', fontsize=16, color=MEDIUM_GRAY, ha='center')

    # Rank selection
    ax_rank = fig.add_axes([0.58, y_main, 0.12, h_main])
    draw_rank_selection(ax_rank)
    fig.text(0.64, 0.04, 'Select rank', fontsize=8, ha='center', color=MEDIUM_GRAY)

    fig.text(0.715, 0.50, '→', fontsize=16, color=MEDIUM_GRAY, ha='center')

    # Consensus
    ax_cons = fig.add_axes([0.74, y_main, 0.24, h_main])
    draw_w_matrix_full(ax_cons, W)
    fig.text(0.86, 0.04, 'Consensus W*', fontsize=8, ha='center', color=MEDIUM_GRAY)

    box = mpatches.FancyBboxPatch((0.73, y_main - 0.02), 0.26, h_main + 0.06,
                                   boxstyle="round,pad=0.01", facecolor='none',
                                   edgecolor=FACTOR_HEX[1], lw=1.5, transform=fig.transFigure)
    fig.add_artist(box)

    return fig, 'consensus_optionB.svg'


def option_c():
    """Option C: Single clean candidate, instability implied by 'multiple runs'."""
    W = create_data()
    rng = np.random.default_rng(42)

    fig = plt.figure(figsize=(12, 3.5))
    fig.text(0.5, 0.95, 'Option C: Clean candidate → multiple runs → consensus',
             fontsize=10, ha='center', color=CHARCOAL)

    y_main = 0.12
    h_main = 0.72

    # Single candidate W
    ax1 = fig.add_axes([0.02, y_main, 0.18, h_main])
    draw_w_matrix_full(ax1, W)
    fig.text(0.11, 0.04, 'Candidate W', fontsize=8, ha='center', color=MEDIUM_GRAY)

    fig.text(0.215, 0.50, '→', fontsize=16, color=MEDIUM_GRAY, ha='center')
    fig.text(0.215, 0.40, '×n runs', fontsize=7, ha='center', color=MEDIUM_GRAY, style='italic')

    # Dimension clustering
    ax_clust = fig.add_axes([0.26, y_main, 0.18, h_main])
    draw_dimension_clustering(ax_clust, rng)
    fig.text(0.35, 0.04, 'Cluster dims', fontsize=8, ha='center', color=MEDIUM_GRAY)

    fig.text(0.455, 0.50, '→', fontsize=16, color=MEDIUM_GRAY, ha='center')

    # Rank selection
    ax_rank = fig.add_axes([0.49, y_main, 0.14, h_main])
    draw_rank_selection(ax_rank)
    fig.text(0.56, 0.04, 'Select rank', fontsize=8, ha='center', color=MEDIUM_GRAY)

    fig.text(0.645, 0.50, '→', fontsize=16, color=MEDIUM_GRAY, ha='center')

    # Consensus
    ax_cons = fig.add_axes([0.68, y_main, 0.30, h_main])
    draw_w_matrix_full(ax_cons, W)
    fig.text(0.83, 0.04, 'Consensus W*', fontsize=8, ha='center', color=MEDIUM_GRAY)

    box = mpatches.FancyBboxPatch((0.67, y_main - 0.02), 0.32, h_main + 0.06,
                                   boxstyle="round,pad=0.01", facecolor='none',
                                   edgecolor=FACTOR_HEX[1], lw=1.5, transform=fig.transFigure)
    fig.add_artist(box)

    return fig, 'consensus_optionC.svg'


def main():
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    for option_fn in [option_a, option_b, option_c]:
        fig, filename = option_fn()
        fig.savefig(output_dir / filename, format='svg', bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f"Saved: {filename}")


if __name__ == "__main__":
    main()
