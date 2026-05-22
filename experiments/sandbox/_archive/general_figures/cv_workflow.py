"""CV workflow visualization - showing rank-k factorization and error measurement."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#c8c8c8'
MEDIUM_GRAY = '#888888'
ARROW_GRAY = '#999999'

FACTOR_COLORS = ['#E07040', '#5070C0', '#70B040', '#9b7eb5']  # 4 factors
HOLDOUT_RED = '#cc4444'
MISSING_COLOR = '#f5f5f5'


def create_data(seed=42, k=4):
    """Create data with k factors."""
    rng = np.random.default_rng(seed)
    n = 16
    items_per_cluster = n // k

    W_list = []
    for cluster in range(k):
        for _ in range(items_per_cluster):
            row = [rng.uniform(0.03, 0.08) for _ in range(k)]
            row[cluster] = rng.uniform(0.75, 0.90)
            W_list.append(row)

    W = np.array(W_list)
    W = W / W.sum(axis=1, keepdims=True)

    # Sort by dominant factor
    dominant = np.argmax(W, axis=1)
    order = np.lexsort((-W.max(axis=1), dominant))
    W = W[order]

    S = W @ W.T
    return W, S


def create_colormap():
    colors = ['#ffffff', '#f0f4f8', '#d4e1ed', '#a8c5db',
              '#6a9fc0', '#3d7ea6', '#1a5a7a', '#0d3d54']
    return mcolors.LinearSegmentedColormap.from_list('elegant', colors)


def draw_rsm(ax, S, cmap, holdout_mask=None, show_error=False):
    """Draw RSM with optional held-out highlighting."""
    n = S.shape[0]
    cell = 1.0 / n

    for i in range(n):
        for j in range(n):
            x, y = j * cell, 1 - (i + 1) * cell
            color = cmap(S[i, j])
            rect = mpatches.Rectangle((x, y), cell, cell,
                facecolor=color, edgecolor='none')
            ax.add_patch(rect)

    # Draw held-out borders on top
    if holdout_mask is not None:
        for i in range(n):
            for j in range(n):
                if holdout_mask[i, j]:
                    x, y = j * cell, 1 - (i + 1) * cell
                    if show_error:
                        # Show as error indicator
                        border = mpatches.Rectangle((x, y), cell, cell,
                            facecolor='none', edgecolor=HOLDOUT_RED, lw=1.2, zorder=2)
                    else:
                        border = mpatches.Rectangle((x, y), cell, cell,
                            facecolor='none', edgecolor=HOLDOUT_RED, lw=1.2, zorder=2)
                    ax.add_patch(border)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_w_matrix(ax, W, factor_colors):
    """Draw W as colored bars showing factor loadings."""
    n, k = W.shape
    cell_h = 1.0 / n
    col_w = 0.8 / k

    for i in range(n):
        y = 1 - (i + 1) * cell_h
        for j in range(k):
            x = 0.1 + j * col_w
            width = W[i, j] * col_w * 0.9
            rect = mpatches.Rectangle((x, y + cell_h * 0.1), width, cell_h * 0.8,
                facecolor=factor_colors[j], edgecolor='none', alpha=0.85)
            ax.add_patch(rect)

    # Factor labels at bottom
    for j in range(k):
        x = 0.1 + j * col_w + col_w * 0.4
        ax.text(x, -0.05, f'F{j+1}', fontsize=7, ha='center', va='top',
               color=factor_colors[j], fontweight='medium')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_cv_curve(ax, optimal_k=4):
    """Draw CV error curve."""
    ranks = np.arange(1, 9)
    # U-shaped validation curve with minimum at optimal_k
    train_err = 0.5 * np.exp(-0.4 * ranks) + 0.02
    val_err = 0.15 * np.exp(-0.3 * ranks) + 0.008 * (ranks - optimal_k)**2 + 0.04

    ax.plot(ranks, train_err, 'o-', color=CHARCOAL, lw=1.5, ms=4, label='Train')
    ax.plot(ranks, val_err, 's-', color=HOLDOUT_RED, lw=1.5, ms=4, label='Validation')

    # Mark optimal
    ax.axvline(optimal_k, color=LIGHT_GRAY, ls='--', lw=0.8)
    ax.scatter([optimal_k], [val_err[optimal_k-1]], color=HOLDOUT_RED, s=60, zorder=5,
              edgecolor='white', lw=1.5)
    ax.text(optimal_k + 0.3, val_err[optimal_k-1], f'k*={optimal_k}', fontsize=8,
           color=CHARCOAL, va='center')

    ax.set_xlabel('Rank k', fontsize=9, color=CHARCOAL)
    ax.set_ylabel('Error', fontsize=9, color=CHARCOAL)
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0, 0.5)
    ax.set_xticks(ranks)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(MEDIUM_GRAY)
    ax.spines['bottom'].set_color(MEDIUM_GRAY)
    ax.tick_params(colors=MEDIUM_GRAY, labelsize=7)
    ax.legend(frameon=False, fontsize=7, loc='upper right')


def main():
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    k = 4
    W, S = create_data(seed=42, k=k)
    n = len(W)
    cmap = create_colormap()

    # Create holdout mask
    rng = np.random.default_rng(123)
    holdout_mask = np.zeros((n, n), dtype=bool)
    for _ in range(15):
        i, j = rng.integers(0, n, 2)
        if i != j:
            holdout_mask[i, j] = True
            holdout_mask[j, i] = True

    # === Main workflow figure ===
    fig = plt.figure(figsize=(10, 3.5))

    # 1. Original RSM with held-out
    ax1 = fig.add_axes([0.02, 0.15, 0.18, 0.70])
    draw_rsm(ax1, S, cmap, holdout_mask=holdout_mask)
    ax1.set_title('S with held-out', fontsize=9, color=CHARCOAL, pad=5)

    # Arrow
    fig.text(0.215, 0.50, '→', fontsize=18, color=ARROW_GRAY, ha='center', va='center')

    # 2. Factorize box
    ax2 = fig.add_axes([0.24, 0.20, 0.12, 0.55])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    trap = mpatches.Polygon([(0.1, 0.05), (0.9, 0.05), (0.75, 0.95), (0.25, 0.95)],
                           facecolor='#a8d4e6', edgecolor='white', lw=1.5, alpha=0.7)
    ax2.add_patch(trap)
    ax2.text(0.5, 0.60, 'SRF', fontsize=10, ha='center', va='center', color=CHARCOAL)
    ax2.text(0.5, 0.40, f'rank={k}', fontsize=8, ha='center', va='center',
            color=CHARCOAL, style='italic')

    # Arrow
    fig.text(0.375, 0.50, '→', fontsize=18, color=ARROW_GRAY, ha='center', va='center')

    # 3. W embedding
    ax3 = fig.add_axes([0.40, 0.15, 0.12, 0.70])
    draw_w_matrix(ax3, W, FACTOR_COLORS)
    ax3.set_title(f'W (n×{k})', fontsize=9, color=CHARCOAL, pad=5)

    # Arrow
    fig.text(0.535, 0.50, '→', fontsize=18, color=ARROW_GRAY, ha='center', va='center')

    # 4. Predicted RSM
    ax4 = fig.add_axes([0.56, 0.15, 0.18, 0.70])
    S_pred = W @ W.T
    draw_rsm(ax4, S_pred, cmap, holdout_mask=holdout_mask, show_error=True)
    ax4.set_title('Ŝ = WWᵀ', fontsize=9, color=CHARCOAL, pad=5)

    # Arrow
    fig.text(0.755, 0.50, '→', fontsize=18, color=ARROW_GRAY, ha='center', va='center')

    # 5. CV curve
    ax5 = fig.add_axes([0.80, 0.22, 0.18, 0.55])
    draw_cv_curve(ax5, optimal_k=k)
    ax5.set_title('Error on held-out', fontsize=9, color=CHARCOAL, pad=5)

    # Annotation
    fig.text(0.50, 0.02, 'Repeat for k = 1, 2, 3, ... → find optimal k*',
            fontsize=9, ha='center', va='bottom', color=MEDIUM_GRAY, style='italic')

    fig.savefig(output_dir / "cv_workflow.svg", format='svg',
               bbox_inches='tight', pad_inches=0.05, transparent=True)
    plt.close(fig)
    print("Saved: cv_workflow.svg")

    # === Also save individual components ===
    # W matrix alone
    fig_w, ax_w = plt.subplots(figsize=(2, 3))
    draw_w_matrix(ax_w, W, FACTOR_COLORS)
    fig_w.savefig(output_dir / "w_embedding_k4.pdf", format='pdf',
                 bbox_inches='tight', pad_inches=0.02)
    plt.close(fig_w)
    print("Saved: w_embedding_k4.pdf")

    # CV curve alone
    fig_cv, ax_cv = plt.subplots(figsize=(3, 2.5))
    draw_cv_curve(ax_cv, optimal_k=k)
    fig_cv.savefig(output_dir / "cv_curve.pdf", format='pdf',
                  bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_cv)
    print("Saved: cv_curve.pdf")


if __name__ == "__main__":
    main()
