"""Imputation and CV mockup - unified view with graph visualization."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx

# Colors from figure_graph.py
CHARCOAL = '#3d3d3d'
MEDIUM_GRAY = '#888888'
LIGHT_GRAY = '#d0d0d0'
ARROW_GRAY = '#999999'

FACTOR_COLORS = np.array([
    [0.88, 0.44, 0.25],  # orange
    [0.31, 0.44, 0.75],  # blue
    [0.44, 0.69, 0.25],  # green
])
FACTOR_HEX = ['#E07040', '#5070C0', '#70B040']

TRAIN_BLACK = '#333333'
VALIDATION_RED = '#cc4444'
MISSING_COLOR = '#f5f5f5'


def blend_color(weights):
    weights = np.array(weights)
    weights = weights / weights.sum()
    return weights @ FACTOR_COLORS


def create_similarity_cmap():
    # Saturated blue gradient (white to deep blue)
    colors = ['#ffffff', '#c6dbef', '#6baed6', '#2171b5', '#08519c', '#08306b']
    return mcolors.LinearSegmentedColormap.from_list('sim', colors)


def create_graph_data(seed=42):
    """Create graph with cluster structure - matching figure_graph style."""
    rng = np.random.default_rng(seed)
    W_list = []

    # Strong cluster members (4 per cluster)
    for _ in range(4):
        W_list.append([rng.uniform(0.82, 0.92), rng.uniform(0.03, 0.09), rng.uniform(0.03, 0.09)])
    for _ in range(4):
        W_list.append([rng.uniform(0.03, 0.09), rng.uniform(0.82, 0.92), rng.uniform(0.03, 0.09)])
    for _ in range(4):
        W_list.append([rng.uniform(0.03, 0.09), rng.uniform(0.03, 0.09), rng.uniform(0.82, 0.92)])

    # Mixed membership nodes
    W_list.append([0.46, 0.46, 0.08])
    W_list.append([0.47, 0.07, 0.46])
    W_list.append([0.07, 0.47, 0.46])

    W = np.array(W_list)
    W = W / W.sum(axis=1, keepdims=True)

    # Sort by dominant cluster for block structure in RSM
    dominant = np.argmax(W, axis=1)
    order = np.lexsort((-W.max(axis=1), dominant))
    W = W[order]

    S = W @ W.T
    return W, S


def draw_mini_graph(ax, W, pos=None, seed=42):
    """Draw a small graph matching figure_graph style."""
    n = len(W)
    S = W @ W.T

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > 0.15:
                G.add_edge(i, j, weight=S[i, j])

    if pos is None:
        factor_centers = {
            0: np.array([-0.35, -0.2]),
            1: np.array([0.35, -0.2]),
            2: np.array([0.0, 0.35]),
        }
        pos = {}
        rng = np.random.default_rng(seed)
        for i in range(n):
            center = sum(W[i, k] * factor_centers[k] for k in range(3))
            offset = rng.uniform(-0.08, 0.08, 2)
            pos[i] = center + offset
        pos = nx.spring_layout(G, pos=pos, k=0.3, iterations=50, seed=seed)

    # Draw edges
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        alpha = 0.2 + 0.5 * weight
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
               color=LIGHT_GRAY, lw=0.8, alpha=alpha, zorder=1)

    # Draw nodes
    for i in range(n):
        color = blend_color(W[i])
        circle = mpatches.Circle(pos[i], 0.055, facecolor=color,
                                edgecolor='white', lw=0.6, zorder=2)
        ax.add_patch(circle)

    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.45, 0.55)
    ax.set_aspect('equal')
    ax.axis('off')
    return pos


def draw_matrix(ax, S, mask=None, mask_type='missing', title=None):
    """Draw similarity matrix with optional mask."""
    n = S.shape[0]
    cmap = create_similarity_cmap()
    cell = 1.0 / n

    # First pass: draw all cells
    for i in range(n):
        for j in range(n):
            x, y = j * cell, 1 - (i + 1) * cell

            if mask is not None and mask[i, j]:
                if mask_type == 'missing':
                    # Missing: white background
                    rect = mpatches.Rectangle((x, y), cell, cell,
                        facecolor=MISSING_COLOR, edgecolor='none')
                    ax.add_patch(rect)
                elif mask_type == 'holdout':
                    # Held-out: show the actual value
                    rect = mpatches.Rectangle((x, y), cell, cell,
                        facecolor=cmap(S[i, j]), edgecolor='none')
                    ax.add_patch(rect)
            else:
                rect = mpatches.Rectangle((x, y), cell, cell,
                    facecolor=cmap(S[i, j]), edgecolor='none')
                ax.add_patch(rect)

    # Second pass: draw borders and X marks on top
    for i in range(n):
        for j in range(n):
            x, y = j * cell, 1 - (i + 1) * cell

            if mask is not None and mask[i, j]:
                if mask_type == 'missing':
                    # X mark
                    ax.plot([x + cell*0.2, x + cell*0.8], [y + cell*0.2, y + cell*0.8],
                           color=LIGHT_GRAY, lw=0.5, zorder=2)
                    ax.plot([x + cell*0.2, x + cell*0.8], [y + cell*0.8, y + cell*0.2],
                           color=LIGHT_GRAY, lw=0.5, zorder=2)
                elif mask_type == 'holdout':
                    # Red border as separate rectangle with no fill
                    border = mpatches.Rectangle((x, y), cell, cell,
                        facecolor='none', edgecolor=VALIDATION_RED, lw=1.2, zorder=2)
                    ax.add_patch(border)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_cv_curve(ax):
    """Draw train/validation curves."""
    ranks = np.linspace(1, 10, 50)
    train = 0.55 * np.exp(-0.4 * ranks) + 0.02
    validation = 0.22 * np.exp(-0.5 * ranks) + 0.012 * (ranks - 4)**2 + 0.055

    optimal_k = 4

    ax.plot(ranks, train, color=TRAIN_BLACK, lw=1.5, label='Train')
    ax.plot(ranks, validation, color=VALIDATION_RED, lw=1.5, label='Validation')

    ax.axvline(optimal_k, color=LIGHT_GRAY, linestyle='--', lw=0.8)
    opt_val = 0.22 * np.exp(-0.5 * optimal_k) + 0.012 * (optimal_k - 4)**2 + 0.055
    ax.scatter([optimal_k], [opt_val], color=VALIDATION_RED, s=35, zorder=5)
    ax.text(optimal_k + 0.3, opt_val - 0.02, 'k*', fontsize=9, color=CHARCOAL)

    ax.set_xlabel('Rank', fontsize=8, color=CHARCOAL)
    ax.set_ylabel('Error', fontsize=8, color=CHARCOAL)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 0.55)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(MEDIUM_GRAY)
    ax.spines['bottom'].set_color(MEDIUM_GRAY)
    ax.tick_params(colors=MEDIUM_GRAY, labelsize=6)
    ax.legend(frameon=False, fontsize=7, loc='upper right')


def draw_w_bars(ax, W):
    """Draw W matrix as horizontal bars."""
    n, k = W.shape
    bar_h = 0.85 / n
    gap = bar_h * 0.1

    for col in range(k):
        x_offset = 0.12 + col * 0.30
        for row in range(n):
            y = 0.92 - (row + 1) * (bar_h + gap)
            width = W[row, col] * 0.26
            rect = mpatches.Rectangle((x_offset, y), width, bar_h,
                facecolor=FACTOR_HEX[col], edgecolor='none', alpha=0.85)
            ax.add_patch(rect)

        ax.text(x_offset + 0.13, 0.02, f'F{col+1}', fontsize=7, ha='center',
               color=FACTOR_HEX[col], fontweight='medium')

    # Colored dots on left
    for row in range(n):
        y = 0.92 - (row + 1) * (bar_h + gap) + bar_h/2
        cluster = row // (n // k)
        if cluster >= k:
            cluster = k - 1
        circle = mpatches.Circle((0.05, y), 0.015, facecolor=FACTOR_HEX[cluster],
                                 edgecolor='white', lw=0.4)
        ax.add_patch(circle)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def main():
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    W, S = create_graph_data(seed=42)
    n = len(W)

    # Create masks - same pattern for visual alignment
    rng = np.random.default_rng(123)
    base_mask = rng.random((n, n)) < 0.15
    base_mask = base_mask | base_mask.T
    np.fill_diagonal(base_mask, False)

    # === Unified figure with two aligned RSM columns ===
    fig = plt.figure(figsize=(8, 5))

    # Column positions
    rsm_width = 0.18
    rsm_height = 0.35

    # Row 1: Cross-validation (held-out)
    row1_y = 0.55

    # RSM with held-out entries
    ax1 = fig.add_axes([0.08, row1_y, rsm_width, rsm_height])
    draw_matrix(ax1, S, mask=base_mask, mask_type='holdout')
    ax1.set_title('Held-out entries', fontsize=9, color=CHARCOAL, pad=5)

    # Arrow
    fig.text(0.29, row1_y + rsm_height/2, '→', fontsize=14, color=ARROW_GRAY, ha='center', va='center')

    # CV curves
    ax2 = fig.add_axes([0.34, row1_y + 0.02, 0.22, rsm_height - 0.04])
    draw_cv_curve(ax2)
    ax2.set_title('Find optimal k', fontsize=9, color=CHARCOAL, pad=5)

    # Arrow
    fig.text(0.59, row1_y + rsm_height/2, '→', fontsize=14, color=ARROW_GRAY, ha='center', va='center')

    # W embedding
    ax3 = fig.add_axes([0.64, row1_y, 0.14, rsm_height])
    draw_w_bars(ax3, W)
    ax3.set_title('W', fontsize=9, color=CHARCOAL, pad=5)

    # Arrow
    fig.text(0.80, row1_y + rsm_height/2, '→', fontsize=14, color=ARROW_GRAY, ha='center', va='center')

    # Reconstructed RSM
    ax4 = fig.add_axes([0.84, row1_y, rsm_width, rsm_height])
    draw_matrix(ax4, S)
    ax4.set_title('Ŝ = WWᵀ', fontsize=9, color=CHARCOAL, pad=5)

    # Row label
    fig.text(0.02, row1_y + rsm_height/2, 'CV', fontsize=10, ha='center', va='center',
            color=CHARCOAL, fontweight='medium', rotation=90)

    # Row 2: Imputation (missing)
    row2_y = 0.10

    # RSM with missing entries
    ax5 = fig.add_axes([0.08, row2_y, rsm_width, rsm_height])
    draw_matrix(ax5, S, mask=base_mask, mask_type='missing')
    ax5.set_title('Missing entries', fontsize=9, color=CHARCOAL, pad=5)

    # Arrow
    fig.text(0.29, row2_y + rsm_height/2, '→', fontsize=14, color=ARROW_GRAY, ha='center', va='center')

    # SRF box
    ax6 = fig.add_axes([0.34, row2_y, 0.22, rsm_height])
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')

    # Trapezoid
    trap = mpatches.Polygon([(0.12, 0.08), (0.88, 0.08), (0.72, 0.92), (0.28, 0.92)],
                           facecolor='#a8d4e6', edgecolor='white', lw=1.5, alpha=0.7)
    ax6.add_patch(trap)
    ax6.text(0.5, 0.58, 'SRF', fontsize=11, ha='center', va='center', color=CHARCOAL)
    ax6.text(0.5, 0.38, 'S ≈ WWᵀ', fontsize=8, ha='center', va='center',
            color=CHARCOAL, style='italic')
    ax6.set_title('Low-rank factorization', fontsize=9, color=CHARCOAL, pad=5)

    # Arrow
    fig.text(0.59, row2_y + rsm_height/2, '→', fontsize=14, color=ARROW_GRAY, ha='center', va='center')

    # W embedding (same as above)
    ax7 = fig.add_axes([0.64, row2_y, 0.14, rsm_height])
    draw_w_bars(ax7, W)
    ax7.set_title('W', fontsize=9, color=CHARCOAL, pad=5)

    # Arrow
    fig.text(0.80, row2_y + rsm_height/2, '→', fontsize=14, color=ARROW_GRAY, ha='center', va='center')

    # Imputed RSM
    ax8 = fig.add_axes([0.84, row2_y, rsm_width, rsm_height])
    draw_matrix(ax8, S)
    ax8.set_title('Imputed Ŝ', fontsize=9, color=CHARCOAL, pad=5)

    # Row label
    fig.text(0.02, row2_y + rsm_height/2, 'Impute', fontsize=10, ha='center', va='center',
            color=CHARCOAL, fontweight='medium', rotation=90)

    # Title
    fig.text(0.50, 0.97, 'Cross-validation and imputation: same mechanism', fontsize=11,
            ha='center', va='center', color=CHARCOAL, fontweight='medium')

    # Bracket connecting both rows on left
    ax_bracket = fig.add_axes([0.04, 0.15, 0.025, 0.70])
    ax_bracket.set_xlim(0, 1)
    ax_bracket.set_ylim(0, 1)
    ax_bracket.axis('off')
    ax_bracket.plot([0.9, 0.3, 0.3, 0.9], [0.92, 0.80, 0.20, 0.08],
                   color=MEDIUM_GRAY, lw=1.0)

    # Legend at bottom right
    leg_y = 0.02
    # Held-out legend
    ax_leg1 = fig.add_axes([0.75, leg_y, 0.015, 0.03])
    ax_leg1.add_patch(mpatches.Rectangle((0, 0), 1, 1, facecolor='#98cce4',
                                         edgecolor=VALIDATION_RED, lw=2))
    ax_leg1.axis('off')
    fig.text(0.77, leg_y + 0.015, 'Held-out', fontsize=7, color=CHARCOAL, va='center')

    # Missing legend
    ax_leg2 = fig.add_axes([0.86, leg_y, 0.015, 0.03])
    ax_leg2.add_patch(mpatches.Rectangle((0, 0), 1, 1, facecolor=MISSING_COLOR,
                                         edgecolor=LIGHT_GRAY, lw=1))
    ax_leg2.plot([0.2, 0.8], [0.2, 0.8], color=LIGHT_GRAY, lw=1)
    ax_leg2.plot([0.2, 0.8], [0.8, 0.2], color=LIGHT_GRAY, lw=1)
    ax_leg2.axis('off')
    ax_leg2.set_xlim(0, 1)
    ax_leg2.set_ylim(0, 1)
    fig.text(0.88, leg_y + 0.015, 'Missing', fontsize=7, color=CHARCOAL, va='center')

    fig.savefig(output_dir / "imputation_cv.svg", format='svg',
               bbox_inches='tight', pad_inches=0.08, transparent=True)
    plt.close(fig)

    print("Saved: imputation_cv.svg")

    # Plain RSM for manual editing
    fig_rsm, ax_rsm = plt.subplots(figsize=(3, 3))
    draw_matrix(ax_rsm, S)
    fig_rsm.savefig(output_dir / "figure_rsm_plain.svg", format='svg',
                   bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close(fig_rsm)
    print("Saved: figure_rsm_plain.svg")


if __name__ == "__main__":
    main()
