"""Overview v6 - Matrix decomposition view: S = W × W^T."""

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
    dominant = np.argmax(W, axis=1)
    order = np.lexsort((-W.max(axis=1), dominant))
    return W, S, order


def blend_color(weights):
    weights = np.array(weights) / np.sum(weights)
    return weights @ FACTOR_COLORS


def create_colormap():
    colors = ['#ffffff', '#f0f4f8', '#d4e1ed', '#a8c5db',
              '#6a9fc0', '#3d7ea6', '#1a5a7a', '#0d3d54']
    return mcolors.LinearSegmentedColormap.from_list('elegant', colors)


def draw_s_matrix(ax, S, cmap):
    """Draw S matrix (n×n)."""
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


def draw_w_matrix(ax, W):
    """Draw W matrix (n×k) with colored columns."""
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


def draw_wt_matrix(ax, W):
    """Draw W^T matrix (k×n) with colored rows."""
    n, k = W.shape
    cell_h = 1.0 / k
    cell_w = 1.0 / n

    for i in range(k):
        for j in range(n):
            val = W[j, i]  # transposed
            rect = mpatches.Rectangle(
                (j * cell_w, 1 - (i + 1) * cell_h),
                cell_w * 0.95, cell_h * 0.95,
                facecolor=FACTOR_HEX[i], alpha=val, edgecolor='none')
            ax.add_patch(rect)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    W, S, order = create_data()
    W_ord = W[order]
    S_ord = S[np.ix_(order, order)]
    cmap = create_colormap()
    n, k = W_ord.shape

    fig = plt.figure(figsize=(10, 4))

    # Equation at top
    fig.text(0.5, 0.92, 'Similarity-based Representation Factorization',
             fontsize=12, ha='center', color=CHARCOAL, fontweight='medium')

    # Matrix sizes for layout
    s_size = 0.28  # S is square n×n
    w_width = 0.08  # W is n×k (tall)
    w_height = s_size
    wt_width = s_size  # W^T is k×n (wide)
    wt_height = 0.08

    y_center = 0.38

    # S matrix
    ax_s = fig.add_axes([0.05, y_center - s_size/2, s_size, s_size])
    draw_s_matrix(ax_s, S_ord, cmap)
    fig.text(0.05 + s_size/2, y_center - s_size/2 - 0.06, 'S', fontsize=14, ha='center', color=CHARCOAL)
    fig.text(0.05 + s_size/2, y_center - s_size/2 - 0.12, f'{n} × {n}', fontsize=9, ha='center', color=MEDIUM_GRAY)

    # ≈
    fig.text(0.37, y_center, '≈', fontsize=18, ha='center', va='center', color=CHARCOAL)

    # W matrix
    w_left = 0.42
    ax_w = fig.add_axes([w_left, y_center - w_height/2, w_width, w_height])
    draw_w_matrix(ax_w, W_ord)
    fig.text(w_left + w_width/2, y_center - w_height/2 - 0.06, 'W', fontsize=14, ha='center', color=CHARCOAL)
    fig.text(w_left + w_width/2, y_center - w_height/2 - 0.12, f'{n} × {k}', fontsize=9, ha='center', color=MEDIUM_GRAY)

    # ×
    fig.text(0.535, y_center, '×', fontsize=16, ha='center', va='center', color=CHARCOAL)

    # W^T matrix
    wt_left = 0.57
    ax_wt = fig.add_axes([wt_left, y_center - wt_height/2, wt_width, wt_height])
    draw_wt_matrix(ax_wt, W_ord)
    fig.text(wt_left + wt_width/2, y_center - wt_height/2 - 0.06, 'Wᵀ', fontsize=14, ha='center', color=CHARCOAL)
    fig.text(wt_left + wt_width/2, y_center - wt_height/2 - 0.12, f'{k} × {n}', fontsize=9, ha='center', color=MEDIUM_GRAY)

    # Constraint box on right
    ax_box = fig.add_axes([0.88, y_center - 0.12, 0.10, 0.24])
    box = mpatches.FancyBboxPatch((0.05, 0.1), 0.9, 0.8, boxstyle="round,pad=0.05",
                                   facecolor='#f8fbfd', edgecolor=FACTOR_HEX[1], lw=1.2)
    ax_box.add_patch(box)
    ax_box.text(0.5, 0.5, 'W ≥ 0', fontsize=10, ha='center', va='center', color=CHARCOAL)
    ax_box.set_xlim(0, 1)
    ax_box.set_ylim(0, 1)
    ax_box.axis('off')

    # Factor legend at bottom
    for j in range(k):
        x = 0.38 + j * 0.12
        rect = mpatches.Rectangle((x, 0.04), 0.025, 0.06, transform=fig.transFigure,
                                   facecolor=FACTOR_HEX[j], edgecolor='none', clip_on=False)
        fig.add_artist(rect)
        fig.text(x + 0.035, 0.07, f'Factor {j+1}', fontsize=8, va='center', color=MEDIUM_GRAY)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "overview_v6.svg", format='svg', bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print("Saved: overview_v6.svg")


if __name__ == "__main__":
    main()
