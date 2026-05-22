"""RSM with soft membership bars - standalone figure."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#c8c8c8'
MEDIUM_GRAY = '#888888'

# Colors chosen for distinct blends
FACTOR_COLORS = ['#E07040', '#5070C0', '#70B040']


def create_data(seed=42):
    """Shared data generation - same across all figures."""
    rng = np.random.default_rng(seed)
    W_list = []

    for _ in range(4):
        W_list.append([rng.uniform(0.82, 0.92), rng.uniform(0.03, 0.09), rng.uniform(0.03, 0.09)])
    for _ in range(4):
        W_list.append([rng.uniform(0.03, 0.09), rng.uniform(0.82, 0.92), rng.uniform(0.03, 0.09)])
    for _ in range(4):
        W_list.append([rng.uniform(0.03, 0.09), rng.uniform(0.03, 0.09), rng.uniform(0.82, 0.92)])

    # Overlap items
    W_list.append([0.46, 0.46, 0.08])  # F1-F2
    W_list.append([0.47, 0.07, 0.46])  # F1-F3
    W_list.append([0.07, 0.47, 0.46])  # F2-F3

    W = np.array(W_list)
    W = W / W.sum(axis=1, keepdims=True)
    S = W @ W.T
    # Normalize so diagonal is 1 (proper similarity matrix)
    d = np.sqrt(np.diag(S))
    S = S / np.outer(d, d)

    # Order by dominant factor
    dominant = np.argmax(W, axis=1)
    order = np.lexsort((-W.max(axis=1), dominant))

    return W, S, order


def create_colormap():
    colors = ['#ffffff', '#f0f4f8', '#d4e1ed', '#a8c5db',
              '#6a9fc0', '#3d7ea6', '#1a5a7a', '#0d3d54']
    return mcolors.LinearSegmentedColormap.from_list('elegant', colors)


def main():
    W, S, order = create_data()
    n = len(W)

    W_ordered = W[order]
    S_ordered = S[np.ix_(order, order)]

    # Scrambled version - random permutation to break visible block structure
    # This simulates "alphabetical" or arbitrary ordering where structure isn't visible
    rng = np.random.default_rng(123)
    scramble = rng.permutation(n)
    W_scrambled = W[scramble]
    S_scrambled = S[np.ix_(scramble, scramble)]

    fig = plt.figure(figsize=(3.4, 3.4))
    cmap = create_colormap()

    # Main RSM - draw as rectangles for true vector output
    # Use square axes in a square figure for proper alignment
    rsm_left = 0.18
    rsm_bottom = 0.10
    rsm_size = 0.65
    ax_rsm = fig.add_axes([rsm_left, rsm_bottom, rsm_size, rsm_size])
    cell_size = 1.0 / n
    for i in range(n):
        for j in range(n):
            color = cmap(S_ordered[i, j])
            rect = mpatches.Rectangle(
                (j * cell_size, 1 - (i + 1) * cell_size),
                cell_size, cell_size,
                facecolor=color, edgecolor='none'
            )
            ax_rsm.add_patch(rect)

    ax_rsm.set_xlim(0, 1)
    ax_rsm.set_ylim(0, 1)
    ax_rsm.set_xticks([])
    ax_rsm.set_yticks([])
    for spine in ax_rsm.spines.values():
        spine.set_visible(False)

    # Top membership bar - positioned exactly above RSM
    bar_height = 0.035
    ax_top = fig.add_axes([rsm_left, rsm_bottom + rsm_size, rsm_size, bar_height])
    for i in range(n):
        for k in range(3):
            if W_ordered[i, k] > 0.05:
                rect = mpatches.Rectangle(
                    (i * cell_size, 0), cell_size, 1,
                    facecolor=FACTOR_COLORS[k], alpha=W_ordered[i, k], edgecolor='none'
                )
                ax_top.add_patch(rect)
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)
    ax_top.axis('off')

    # Left membership bar - aligned with RSM rows
    bar_width = 0.035
    ax_left = fig.add_axes([rsm_left - bar_width - 0.01, rsm_bottom, bar_width, rsm_size])
    for i in range(n):
        for k in range(3):
            if W_ordered[i, k] > 0.05:
                rect = mpatches.Rectangle(
                    (0, 1 - (i + 1) * cell_size), 1, cell_size,
                    facecolor=FACTOR_COLORS[k], alpha=W_ordered[i, k], edgecolor='none'
                )
                ax_left.add_patch(rect)
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)
    ax_left.axis('off')

    # Colorbar - draw as vector rectangles
    cbar_width = 0.02
    ax_cbar = fig.add_axes([rsm_left + rsm_size + 0.02, rsm_bottom, cbar_width, rsm_size])
    n_cbar = 100  # number of color steps
    for i in range(n_cbar):
        val = i / (n_cbar - 1)
        color = cmap(val)
        rect = mpatches.Rectangle(
            (0, i / n_cbar), 1, 1 / n_cbar,
            facecolor=color, edgecolor='none'
        )
        ax_cbar.add_patch(rect)
    ax_cbar.set_xlim(0, 1)
    ax_cbar.set_ylim(0, 1)
    ax_cbar.set_xticks([])
    ax_cbar.set_yticks([0, 0.5, 1])
    ax_cbar.yaxis.tick_right()
    ax_cbar.tick_params(axis='y', labelsize=7, length=2, color=LIGHT_GRAY)
    for spine in ax_cbar.spines.values():
        spine.set_visible(False)

    # Factor legend
    ax_leg = fig.add_axes([rsm_left, rsm_bottom + rsm_size + bar_height + 0.02, rsm_size, 0.06])
    ax_leg.axis('off')
    for k in range(3):
        x_pos = 0.12 + k * 0.32
        rect = mpatches.Rectangle(
            (x_pos, 0.2), 0.06, 0.6,
            facecolor=FACTOR_COLORS[k], edgecolor='none'
        )
        ax_leg.add_patch(rect)
        ax_leg.text(x_pos + 0.10, 0.5, f'F{k+1}', fontsize=8, va='center', color=MEDIUM_GRAY)
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)

    # Save
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "figure_rsm.svg", format='svg', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print("Saved: figure_rsm.svg")

    # Plain RSM without bars/legend/colorbar - PDF for clean vector editing
    fig_plain, ax_plain = plt.subplots(figsize=(3, 3))
    for i in range(n):
        for j in range(n):
            color = cmap(S_ordered[i, j])
            rect = mpatches.Rectangle(
                (j * cell_size, 1 - (i + 1) * cell_size),
                cell_size, cell_size,
                facecolor=color, edgecolor='none'
            )
            ax_plain.add_patch(rect)
    ax_plain.set_xlim(0, 1)
    ax_plain.set_ylim(0, 1)
    ax_plain.axis('off')
    fig_plain.savefig(output_dir / "figure_rsm_plain.pdf", format='pdf',
                      bbox_inches='tight', pad_inches=0.02)
    plt.close(fig_plain)
    print("Saved: figure_rsm_plain.pdf")

    # RSM with missing and held-out entries
    MISSING_COLOR = '#f5f5f5'
    HOLDOUT_RED = '#cc4444'

    rng = np.random.default_rng(42)
    # Create separate masks for missing and held-out
    missing_mask = np.zeros((n, n), dtype=bool)
    holdout_mask = np.zeros((n, n), dtype=bool)

    # Scatter some missing entries (symmetric)
    for _ in range(12):
        i, j = rng.integers(0, n, 2)
        if i != j:
            missing_mask[i, j] = True
            missing_mask[j, i] = True

    # Scatter some held-out entries (symmetric, non-overlapping)
    for _ in range(12):
        i, j = rng.integers(0, n, 2)
        if i != j and not missing_mask[i, j]:
            holdout_mask[i, j] = True
            holdout_mask[j, i] = True

    fig_masked, ax_masked = plt.subplots(figsize=(3, 3))

    # First pass: draw all cells
    for i in range(n):
        for j in range(n):
            x = j * cell_size
            y = 1 - (i + 1) * cell_size

            if missing_mask[i, j]:
                # Missing: white/light gray background
                rect = mpatches.Rectangle((x, y), cell_size, cell_size,
                    facecolor=MISSING_COLOR, edgecolor='none')
                ax_masked.add_patch(rect)
            else:
                # Normal or held-out: show value
                color = cmap(S_ordered[i, j])
                rect = mpatches.Rectangle((x, y), cell_size, cell_size,
                    facecolor=color, edgecolor='none')
                ax_masked.add_patch(rect)

    # Second pass: draw X marks for missing and borders for held-out
    for i in range(n):
        for j in range(n):
            x = j * cell_size
            y = 1 - (i + 1) * cell_size

            if missing_mask[i, j]:
                # X mark
                ax_masked.plot([x + cell_size*0.2, x + cell_size*0.8],
                              [y + cell_size*0.2, y + cell_size*0.8],
                              color=LIGHT_GRAY, lw=0.8, zorder=2)
                ax_masked.plot([x + cell_size*0.2, x + cell_size*0.8],
                              [y + cell_size*0.8, y + cell_size*0.2],
                              color=LIGHT_GRAY, lw=0.8, zorder=2)
            elif holdout_mask[i, j]:
                # Red border
                border = mpatches.Rectangle((x, y), cell_size, cell_size,
                    facecolor='none', edgecolor=HOLDOUT_RED, lw=1.5, zorder=2)
                ax_masked.add_patch(border)

    ax_masked.set_xlim(0, 1)
    ax_masked.set_ylim(0, 1)
    ax_masked.axis('off')
    fig_masked.savefig(output_dir / "figure_rsm_masked.pdf", format='pdf',
                       bbox_inches='tight', pad_inches=0.02)
    plt.close(fig_masked)
    print("Saved: figure_rsm_masked.pdf")

    # Unsorted RSM - shows that structure isn't visible without clustering
    fig_unsorted = plt.figure(figsize=(3.4, 3.4))
    ax_unsorted = fig_unsorted.add_axes([rsm_left, rsm_bottom, rsm_size, rsm_size])
    for i in range(n):
        for j in range(n):
            color = cmap(S_scrambled[i, j])
            rect = mpatches.Rectangle(
                (j * cell_size, 1 - (i + 1) * cell_size),
                cell_size, cell_size,
                facecolor=color, edgecolor='none'
            )
            ax_unsorted.add_patch(rect)
    ax_unsorted.set_xlim(0, 1)
    ax_unsorted.set_ylim(0, 1)
    ax_unsorted.set_xticks([])
    ax_unsorted.set_yticks([])
    for spine in ax_unsorted.spines.values():
        spine.set_visible(False)

    # Top membership bar (unsorted)
    ax_top_unsorted = fig_unsorted.add_axes([rsm_left, rsm_bottom + rsm_size, rsm_size, bar_height])
    for i in range(n):
        for k in range(3):
            if W_scrambled[i, k] > 0.05:
                rect = mpatches.Rectangle(
                    (i * cell_size, 0), cell_size, 1,
                    facecolor=FACTOR_COLORS[k], alpha=W_scrambled[i, k], edgecolor='none'
                )
                ax_top_unsorted.add_patch(rect)
    ax_top_unsorted.set_xlim(0, 1)
    ax_top_unsorted.set_ylim(0, 1)
    ax_top_unsorted.axis('off')

    # Left membership bar (unsorted)
    ax_left_unsorted = fig_unsorted.add_axes([rsm_left - bar_width - 0.01, rsm_bottom, bar_width, rsm_size])
    for i in range(n):
        for k in range(3):
            if W_scrambled[i, k] > 0.05:
                rect = mpatches.Rectangle(
                    (0, 1 - (i + 1) * cell_size), 1, cell_size,
                    facecolor=FACTOR_COLORS[k], alpha=W_scrambled[i, k], edgecolor='none'
                )
                ax_left_unsorted.add_patch(rect)
    ax_left_unsorted.set_xlim(0, 1)
    ax_left_unsorted.set_ylim(0, 1)
    ax_left_unsorted.axis('off')

    # Colorbar
    ax_cbar_unsorted = fig_unsorted.add_axes([rsm_left + rsm_size + 0.02, rsm_bottom, cbar_width, rsm_size])
    for i in range(n_cbar):
        val = i / (n_cbar - 1)
        color = cmap(val)
        rect = mpatches.Rectangle(
            (0, i / n_cbar), 1, 1 / n_cbar,
            facecolor=color, edgecolor='none'
        )
        ax_cbar_unsorted.add_patch(rect)
    ax_cbar_unsorted.set_xlim(0, 1)
    ax_cbar_unsorted.set_ylim(0, 1)
    ax_cbar_unsorted.set_xticks([])
    ax_cbar_unsorted.set_yticks([0, 0.5, 1])
    ax_cbar_unsorted.yaxis.tick_right()
    ax_cbar_unsorted.tick_params(axis='y', labelsize=7, length=2, color=LIGHT_GRAY)
    for spine in ax_cbar_unsorted.spines.values():
        spine.set_visible(False)

    # Factor legend
    ax_leg_unsorted = fig_unsorted.add_axes([rsm_left, rsm_bottom + rsm_size + bar_height + 0.02, rsm_size, 0.06])
    ax_leg_unsorted.axis('off')
    for k in range(3):
        x_pos = 0.12 + k * 0.32
        rect = mpatches.Rectangle(
            (x_pos, 0.2), 0.06, 0.6,
            facecolor=FACTOR_COLORS[k], edgecolor='none'
        )
        ax_leg_unsorted.add_patch(rect)
        ax_leg_unsorted.text(x_pos + 0.10, 0.5, f'F{k+1}', fontsize=8, va='center', color=MEDIUM_GRAY)
    ax_leg_unsorted.set_xlim(0, 1)
    ax_leg_unsorted.set_ylim(0, 1)

    fig_unsorted.savefig(output_dir / "figure_rsm_scrambled.svg", format='svg',
                         bbox_inches='tight', pad_inches=0.02)
    plt.close(fig_unsorted)
    print("Saved: figure_rsm_scrambled.svg")


if __name__ == "__main__":
    main()
