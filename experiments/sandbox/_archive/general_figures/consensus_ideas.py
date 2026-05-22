"""Consensus visualization ideas."""

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


def idea_1_multiple_to_one():
    """Multiple noisy solutions → one stable consensus."""
    fig = plt.figure(figsize=(10, 3))

    rng = np.random.default_rng(42)
    n_nodes = 8
    n_runs = 4

    # True W (what we want to recover)
    W_true = np.array([
        [0.9, 0.05, 0.05],
        [0.85, 0.1, 0.05],
        [0.1, 0.85, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.1, 0.85],
        [0.1, 0.05, 0.85],
        [0.45, 0.45, 0.1],  # overlap
        [0.1, 0.45, 0.45],  # overlap
    ])
    W_true = W_true / W_true.sum(axis=1, keepdims=True)

    # Multiple runs with noise and permutations
    y_positions = np.linspace(0.75, 0.25, n_nodes)

    for run in range(n_runs):
        ax = fig.add_axes([0.02 + run * 0.18, 0.15, 0.15, 0.75])

        # Add noise to W
        W_noisy = W_true + rng.normal(0, 0.08, W_true.shape)
        W_noisy = np.clip(W_noisy, 0.01, 1)
        W_noisy = W_noisy / W_noisy.sum(axis=1, keepdims=True)

        # Random permutation of factors (the alignment problem)
        perm = rng.permutation(3)
        W_perm = W_noisy[:, perm]
        colors_perm = [FACTOR_HEX[p] for p in perm]

        # Draw nodes with this run's coloring
        for i in range(n_nodes):
            color = blend_color(W_perm[i])
            circle = mpatches.Circle((0.5, y_positions[i]), 0.08,
                                      facecolor=color, edgecolor='white', lw=0.8)
            ax.add_patch(circle)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f'Run {run+1}', fontsize=8, color=MEDIUM_GRAY)

    # Arrow
    fig.text(0.75, 0.52, '→', fontsize=24, color=MEDIUM_GRAY, ha='center')
    fig.text(0.75, 0.42, 'align &\naverage', fontsize=7, ha='center', color=MEDIUM_GRAY)

    # Consensus result
    ax_cons = fig.add_axes([0.82, 0.15, 0.15, 0.75])
    for i in range(n_nodes):
        color = blend_color(W_true[i])
        circle = mpatches.Circle((0.5, y_positions[i]), 0.08,
                                  facecolor=color, edgecolor='white', lw=1.2)
        ax_cons.add_patch(circle)

    ax_cons.set_xlim(0, 1)
    ax_cons.set_ylim(0, 1)
    ax_cons.axis('off')
    ax_cons.set_title('Consensus', fontsize=9, color=CHARCOAL, fontweight='medium')

    # Box around consensus
    box = mpatches.FancyBboxPatch((0.1, 0.05), 0.8, 0.9, boxstyle="round,pad=0.02",
                                   facecolor='none', edgecolor=FACTOR_HEX[1], lw=1.5,
                                   transform=ax_cons.transAxes)
    ax_cons.add_patch(box)

    return fig, 'consensus_idea1.svg'


def idea_2_dimension_clustering():
    """Dimensions from multiple runs cluster together."""
    fig = plt.figure(figsize=(8, 3.5))

    rng = np.random.default_rng(42)

    # Left: scattered dimensions from multiple runs
    ax1 = fig.add_axes([0.05, 0.12, 0.38, 0.78])

    # 3 true clusters, each with points from multiple runs
    cluster_centers = [(0.2, 0.7), (0.75, 0.65), (0.5, 0.2)]
    n_runs = 5

    for k, ((cx, cy), col) in enumerate(zip(cluster_centers, FACTOR_HEX)):
        for run in range(n_runs):
            x = cx + rng.normal(0, 0.08)
            y = cy + rng.normal(0, 0.06)
            # Different marker for each run
            marker = ['o', 's', '^', 'D', 'v'][run]
            ax1.scatter(x, y, c=[col], s=60, marker=marker, alpha=0.7,
                       edgecolor='white', linewidth=0.5)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Dimensions from multiple runs', fontsize=9, color=CHARCOAL)

    # Arrow
    fig.text(0.47, 0.50, '→', fontsize=20, color=MEDIUM_GRAY, ha='center')
    fig.text(0.47, 0.40, 'cluster', fontsize=8, ha='center', color=MEDIUM_GRAY)

    # Right: clustered and labeled
    ax2 = fig.add_axes([0.55, 0.12, 0.40, 0.78])

    for k, ((cx, cy), col) in enumerate(zip(cluster_centers, FACTOR_HEX)):
        # Draw cluster ellipse
        ellipse = mpatches.Ellipse((cx, cy), 0.28, 0.22, angle=0,
                                    facecolor=col, alpha=0.15, edgecolor=col, lw=1.5)
        ax2.add_patch(ellipse)

        # Points inside
        for run in range(n_runs):
            x = cx + rng.normal(0, 0.06)
            y = cy + rng.normal(0, 0.04)
            ax2.scatter(x, y, c=[col], s=50, alpha=0.8, edgecolor='white', linewidth=0.5)

        # Cluster label
        ax2.text(cx, cy + 0.18, f'F{k+1}', fontsize=10, ha='center',
                color=col, fontweight='bold')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Consensus factors', fontsize=9, color=CHARCOAL)

    return fig, 'consensus_idea2.svg'


def idea_3_stability():
    """Show stability: uncertain → certain."""
    fig = plt.figure(figsize=(9, 3.5))

    rng = np.random.default_rng(42)
    n_nodes = 6

    # Left: unstable (multiple overlapping colors per node = uncertainty)
    ax1 = fig.add_axes([0.05, 0.15, 0.35, 0.75])

    y_pos = np.linspace(0.85, 0.15, n_nodes)

    # True memberships
    W_true = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.85, 0.05],
        [0.05, 0.1, 0.85],
        [0.5, 0.4, 0.1],   # uncertain
        [0.35, 0.35, 0.3], # very uncertain
        [0.1, 0.5, 0.4],   # uncertain
    ])

    for i in range(n_nodes):
        # Show multiple "ghost" circles to indicate uncertainty
        n_ghosts = 6
        for g in range(n_ghosts):
            # Perturbed W
            W_pert = W_true[i] + rng.normal(0, 0.15, 3)
            W_pert = np.clip(W_pert, 0.01, 1)
            W_pert = W_pert / W_pert.sum()

            color = blend_color(W_pert)
            offset_x = rng.uniform(-0.08, 0.08)
            offset_y = rng.uniform(-0.03, 0.03)

            circle = mpatches.Circle((0.5 + offset_x, y_pos[i] + offset_y), 0.065,
                                      facecolor=color, alpha=0.25, edgecolor='none')
            ax1.add_patch(circle)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Single run: unstable', fontsize=9, color=CHARCOAL)

    # Arrow
    fig.text(0.44, 0.52, '→', fontsize=22, color=MEDIUM_GRAY, ha='center')
    fig.text(0.44, 0.42, 'consensus', fontsize=8, ha='center', color=MEDIUM_GRAY)

    # Right: stable (solid colors)
    ax2 = fig.add_axes([0.55, 0.15, 0.35, 0.75])

    for i in range(n_nodes):
        color = blend_color(W_true[i])
        circle = mpatches.Circle((0.5, y_pos[i]), 0.07,
                                  facecolor=color, edgecolor='white', lw=1.2)
        ax2.add_patch(circle)

        # Small bar showing membership
        bar_x = 0.7
        bar_w = 0.25
        bar_h = 0.06
        x_start = bar_x
        for k in range(3):
            w = W_true[i, k] * bar_w
            rect = mpatches.Rectangle((x_start, y_pos[i] - bar_h/2), w, bar_h,
                                       facecolor=FACTOR_HEX[k], edgecolor='none')
            ax2.add_patch(rect)
            x_start += w

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Consensus: stable', fontsize=9, color=CHARCOAL)

    return fig, 'consensus_idea3.svg'


def idea_4_funnel():
    """Multiple embeddings funnel into one."""
    fig = plt.figure(figsize=(8, 4))

    ax = fig.add_axes([0.05, 0.1, 0.9, 0.8])

    rng = np.random.default_rng(42)
    n_runs = 5

    # W matrices on the left
    for run in range(n_runs):
        y_center = 0.15 + run * 0.175
        x_left = 0.05

        # Small W matrix
        rect = mpatches.Rectangle((x_left, y_center - 0.06), 0.08, 0.12,
                                   facecolor='#f5f5f5', edgecolor=LIGHT_GRAY, lw=1)
        ax.add_patch(rect)
        ax.text(x_left + 0.04, y_center, f'W{run+1}', fontsize=7, ha='center', va='center', color=MEDIUM_GRAY)

        # Arrow curving to center
        ax.annotate('', xy=(0.45, 0.5), xytext=(x_left + 0.1, y_center),
                   arrowprops=dict(arrowstyle='->', color=LIGHT_GRAY, lw=1,
                                  connectionstyle=f'arc3,rad={-0.2 + run*0.1}'))

    # Center: consensus process
    box = mpatches.FancyBboxPatch((0.38, 0.35), 0.24, 0.30, boxstyle="round,pad=0.03",
                                   facecolor='#f8fbfd', edgecolor=FACTOR_HEX[1], lw=2)
    ax.add_patch(box)
    ax.text(0.50, 0.55, 'Cluster &', fontsize=9, ha='center', color=CHARCOAL)
    ax.text(0.50, 0.45, 'Align', fontsize=9, ha='center', color=CHARCOAL)

    # Arrow to right
    ax.annotate('', xy=(0.78, 0.5), xytext=(0.64, 0.5),
               arrowprops=dict(arrowstyle='->', color=CHARCOAL, lw=2))

    # Consensus W* on right
    rect_final = mpatches.FancyBboxPatch((0.78, 0.35), 0.12, 0.30, boxstyle="round,pad=0.02",
                                          facecolor='#e8f4fc', edgecolor=FACTOR_HEX[1], lw=2)
    ax.add_patch(rect_final)
    ax.text(0.84, 0.5, 'W*', fontsize=14, ha='center', va='center', color=CHARCOAL, fontweight='medium')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    fig.text(0.5, 0.92, 'Consensus: Multiple runs → Stable embedding', fontsize=10,
             ha='center', color=CHARCOAL)

    return fig, 'consensus_idea4.svg'


def main():
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    ideas = [idea_1_multiple_to_one, idea_2_dimension_clustering,
             idea_3_stability, idea_4_funnel]

    for idea_fn in ideas:
        fig, filename = idea_fn()
        fig.savefig(output_dir / filename, format='svg', bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f"Saved: {filename}")


if __name__ == "__main__":
    main()
