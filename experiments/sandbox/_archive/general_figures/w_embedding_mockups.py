"""W embedding visualizations - different ways to show n×k factor loadings."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#c8c8c8'
MEDIUM_GRAY = '#888888'

FACTOR_COLORS = ['#E07040', '#5070C0', '#70B040', '#9b7eb5']  # 4 factors


def create_data(seed=42, k=4, n=16, sparse=False):
    """Create W matrix with distributed profiles (not block diagonal)."""
    rng = np.random.default_rng(seed)

    W_list = []
    for i in range(n):
        if sparse:
            # Sparse: most items load on 1-2 factors only
            row = rng.dirichlet(np.ones(k) * 0.3)  # Sparse Dirichlet
            # Threshold small values to zero
            row[row < 0.1] = 0
            if row.sum() == 0:
                row[rng.integers(k)] = 1.0
            row = row / row.sum()
        else:
            # Each item has a mix of factor loadings
            row = rng.dirichlet(np.ones(k) * 1.5)
            row = row ** 0.7
            row = row / row.sum()
        W_list.append(row)

    W = np.array(W_list)

    # Sort by dominant factor for visual clarity
    dominant = np.argmax(W, axis=1)
    order = np.lexsort((-W.max(axis=1), dominant))
    W = W[order]

    return W


def draw_w_bars_horizontal(ax, W):
    """V1: Horizontal stacked bars per item."""
    n, k = W.shape
    cell_h = 0.9 / n

    for i in range(n):
        y = 0.95 - (i + 1) * cell_h
        x_offset = 0.1
        for j in range(k):
            width = W[i, j] * 0.75
            rect = mpatches.Rectangle((x_offset, y), width, cell_h * 0.85,
                facecolor=FACTOR_COLORS[j], edgecolor='none', alpha=0.9)
            ax.add_patch(rect)
            x_offset += width

    # Factor labels
    for j in range(k):
        ax.add_patch(mpatches.Rectangle((0.1 + j*0.18, 0.01), 0.06, 0.03,
            facecolor=FACTOR_COLORS[j], edgecolor='none'))
        ax.text(0.18 + j*0.18, 0.025, f'F{j+1}', fontsize=6, va='center', color=CHARCOAL)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_w_bars_vertical(ax, W, wide=False):
    """V2: Vertical bars - columns are factors, rows are items."""
    n, k = W.shape

    if wide:
        # Wider layout with more space between columns
        margin_left = 0.05
        margin_right = 0.05
        col_gap = 0.08
        total_w = 1 - margin_left - margin_right - (k - 1) * col_gap
        col_w = total_w / k
    else:
        col_w = 0.8 / k
        margin_left = 0.1
        col_gap = 0

    cell_h = 0.85 / n

    for i in range(n):
        y = 0.88 - (i + 1) * cell_h
        for j in range(k):
            x = margin_left + j * (col_w + col_gap)
            # Only draw if value > threshold
            if W[i, j] > 0.02:
                width = W[i, j] * col_w * 0.9
                rect = mpatches.Rectangle((x, y + cell_h*0.1), width, cell_h * 0.8,
                    facecolor=FACTOR_COLORS[j], edgecolor='none', alpha=0.9)
                ax.add_patch(rect)

    # Column headers
    for j in range(k):
        x = margin_left + j * (col_w + col_gap) + col_w * 0.4
        ax.text(x, 0.94, f'F{j+1}', fontsize=9, ha='center', va='center',
               color=FACTOR_COLORS[j], fontweight='medium')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_w_heatmap(ax, W):
    """V3: Heatmap with factor-colored columns."""
    n, k = W.shape
    cell_w = 0.7 / k
    cell_h = 0.85 / n

    for i in range(n):
        y = 0.90 - (i + 1) * cell_h
        for j in range(k):
            x = 0.15 + j * cell_w
            # Color intensity based on value
            base_color = mcolors.to_rgb(FACTOR_COLORS[j])
            alpha = 0.15 + 0.85 * W[i, j]
            color = [c * alpha + 1 * (1 - alpha) for c in base_color]
            rect = mpatches.Rectangle((x, y), cell_w * 0.95, cell_h * 0.95,
                facecolor=color, edgecolor='none')
            ax.add_patch(rect)

    # Column headers
    for j in range(k):
        x = 0.15 + j * cell_w + cell_w * 0.45
        ax.text(x, 0.95, f'F{j+1}', fontsize=8, ha='center', va='center',
               color=FACTOR_COLORS[j], fontweight='medium')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_w_dots(ax, W):
    """V4: Dot plot - dot size proportional to loading."""
    n, k = W.shape
    col_w = 0.7 / k
    cell_h = 0.85 / n

    for i in range(n):
        y = 0.90 - (i + 0.5) * cell_h
        for j in range(k):
            x = 0.18 + j * col_w + col_w * 0.3
            radius = W[i, j] * 0.025
            circle = mpatches.Circle((x, y), radius,
                facecolor=FACTOR_COLORS[j], edgecolor='white', lw=0.3)
            ax.add_patch(circle)

    # Column headers
    for j in range(k):
        x = 0.18 + j * col_w + col_w * 0.3
        ax.text(x, 0.95, f'F{j+1}', fontsize=8, ha='center', va='center',
               color=FACTOR_COLORS[j], fontweight='medium')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_w_sparklines(ax, W):
    """V5: Mini sparkline bars for each item."""
    n, k = W.shape
    row_h = 0.85 / n
    bar_w = 0.12

    for i in range(n):
        y_base = 0.90 - (i + 1) * row_h
        for j in range(k):
            x = 0.12 + j * (bar_w + 0.04)
            height = W[i, j] * row_h * 0.9
            rect = mpatches.Rectangle((x, y_base + row_h*0.05), bar_w * 0.8, height,
                facecolor=FACTOR_COLORS[j], edgecolor='none', alpha=0.85)
            ax.add_patch(rect)

    # Column headers
    for j in range(k):
        x = 0.12 + j * (bar_w + 0.04) + bar_w * 0.4
        ax.text(x, 0.95, f'F{j+1}', fontsize=7, ha='center', va='center',
               color=FACTOR_COLORS[j], fontweight='medium')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_w_blocks(ax, W):
    """V6: Block matrix - clean grid with intensity."""
    n, k = W.shape
    cell_w = 0.65 / k
    cell_h = 0.82 / n
    gap = 0.005

    for i in range(n):
        y = 0.88 - (i + 1) * cell_h
        for j in range(k):
            x = 0.18 + j * cell_w
            # Intensity mapping
            val = W[i, j]
            base = mcolors.to_rgb(FACTOR_COLORS[j])
            # White to color gradient
            color = [1 - val * (1 - c) for c in base]
            rect = mpatches.Rectangle((x + gap, y + gap),
                cell_w - 2*gap, cell_h - 2*gap,
                facecolor=color, edgecolor='none')
            ax.add_patch(rect)

    # Headers with colored squares
    for j in range(k):
        x = 0.18 + j * cell_w + cell_w * 0.5
        rect = mpatches.Rectangle((x - 0.02, 0.92), 0.04, 0.04,
            facecolor=FACTOR_COLORS[j], edgecolor='none')
        ax.add_patch(rect)
        ax.text(x, 0.89, f'{j+1}', fontsize=7, ha='center', va='top', color=CHARCOAL)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def main():
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Sparse W for better visualization
    W_sparse = create_data(seed=42, k=4, n=16, sparse=True)

    # Vertical sparse version - wider figure
    fig, ax = plt.subplots(figsize=(4, 3.5))
    draw_w_bars_vertical(ax, W_sparse, wide=True)
    fig.savefig(output_dir / "w_vertical_sparse.svg", format='svg',
               bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close(fig)
    print("Saved: w_vertical_sparse.svg")

    # Even wider version
    fig, ax = plt.subplots(figsize=(5, 3))
    draw_w_bars_vertical(ax, W_sparse, wide=True)
    fig.savefig(output_dir / "w_vertical_sparse_wide.svg", format='svg',
               bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close(fig)
    print("Saved: w_vertical_sparse_wide.svg")

    # Compact version
    fig, ax = plt.subplots(figsize=(3, 4))
    draw_w_bars_vertical(ax, W_sparse, wide=True)
    fig.savefig(output_dir / "w_vertical_sparse_compact.svg", format='svg',
               bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close(fig)
    print("Saved: w_vertical_sparse_compact.svg")


if __name__ == "__main__":
    main()
