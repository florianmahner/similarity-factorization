"""Overview figure mockups - multiple variants for the SRF pipeline."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
import matplotlib.lines as mlines
import numpy as np

CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#c8c8c8'
MEDIUM_GRAY = '#888888'

# Factor colors (matching RSM figure)
COLORS = {
    'red': '#E07040',
    'blue': '#5070C0',
    'green': '#70B040',
    'purple': '#9060A0',
}


def draw_arrow(ax, start, end, color=MEDIUM_GRAY):
    """Draw a simple arrow."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))


def variant_1_minimal():
    """Minimal, clean variant with simple shapes."""
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle('Variant 1: Minimal', fontsize=12, color=CHARCOAL)

    for ax in axes.flat:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    # Row 1: SRF Embedding
    # 1a: Graph as adjacency/similarity matrix
    ax = axes[0, 0]
    ax.set_title('Input', fontsize=9, color=MEDIUM_GRAY)
    n = 12
    rng = np.random.default_rng(42)
    for i in range(n):
        for j in range(n):
            block = (i // 4, j // 4)
            if block[0] == block[1]:
                val = rng.uniform(0.6, 0.9)
            else:
                val = rng.uniform(0.1, 0.3)
            color = plt.cm.Blues(val)
            rect = mpatches.Rectangle((j/n + 0.1, 1 - (i+1)/n), 0.8/n, 0.8/n,
                                       facecolor=color, edgecolor='none')
            ax.add_patch(rect)

    # 1b: Cross-validation curves
    ax = axes[0, 1]
    ax.set_title('Rank Selection', fontsize=9, color=MEDIUM_GRAY)
    x = np.linspace(0.1, 0.9, 50)
    train = 0.7 * np.exp(-3*x) + 0.1
    val = 0.5 * np.exp(-2*x) + 0.15 + 0.3 * (x - 0.4)**2
    ax.plot(x, train, color=CHARCOAL, lw=1.5, label='Train')
    ax.plot(x, val, color=COLORS['red'], lw=1.5, label='Val')
    ax.axvline(0.45, color=LIGHT_GRAY, ls='--', lw=1)
    ax.text(0.47, 0.15, 'k*', fontsize=8, color=MEDIUM_GRAY, style='italic')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 1c: SRF box
    ax = axes[0, 2]
    ax.set_title('SRF', fontsize=9, color=MEDIUM_GRAY)
    box = FancyBboxPatch((0.2, 0.25), 0.6, 0.5, boxstyle="round,pad=0.05",
                         facecolor='#e8f4fc', edgecolor=COLORS['blue'], lw=1.5)
    ax.add_patch(box)
    ax.text(0.5, 0.5, 'W ≥ 0\nS ≈ WWᵀ', ha='center', va='center',
            fontsize=8, color=CHARCOAL)

    # 1d: Embedding matrix
    ax = axes[0, 3]
    ax.set_title('Embedding', fontsize=9, color=MEDIUM_GRAY)
    rect = mpatches.Rectangle((0.25, 0.15), 0.2, 0.7,
                               facecolor=LIGHT_GRAY, edgecolor=CHARCOAL, lw=1)
    ax.add_patch(rect)
    ax.text(0.35, 0.5, 'W', ha='center', va='center', fontsize=10, color=CHARCOAL)
    ax.text(0.35, 0.08, 'n × k', ha='center', fontsize=7, color=MEDIUM_GRAY)

    # Row 2: Consensus
    # 2a: Multiple embeddings
    ax = axes[1, 0]
    ax.set_title('Repeats', fontsize=9, color=MEDIUM_GRAY)
    for i in range(4):
        offset = i * 0.08
        rect = mpatches.Rectangle((0.2 + offset, 0.15 + offset), 0.15, 0.5,
                                   facecolor='white', edgecolor=LIGHT_GRAY, lw=1)
        ax.add_patch(rect)
    rect = mpatches.Rectangle((0.2 + 0.32, 0.15 + 0.32), 0.15, 0.5,
                               facecolor=LIGHT_GRAY, edgecolor=CHARCOAL, lw=1)
    ax.add_patch(rect)

    # 2b: Clustering in dimension space
    ax = axes[1, 1]
    ax.set_title('Clustering', fontsize=9, color=MEDIUM_GRAY)
    centers = [(0.25, 0.7), (0.7, 0.7), (0.5, 0.25)]
    for (cx, cy), col in zip(centers, [COLORS['red'], COLORS['blue'], COLORS['green']]):
        for _ in range(8):
            x, y = cx + rng.normal(0, 0.08), cy + rng.normal(0, 0.08)
            circle = Circle((x, y), 0.025, facecolor=col, alpha=0.7, edgecolor='none')
            ax.add_patch(circle)

    # 2c: Select rank
    ax = axes[1, 2]
    ax.set_title('Select Rank', fontsize=9, color=MEDIUM_GRAY)
    bars = [0.6, 0.85, 0.5, 0.3]
    for i, h in enumerate(bars):
        rect = mpatches.Rectangle((0.15 + i*0.2, 0.2), 0.12, h * 0.6,
                                   facecolor=COLORS['blue'] if i == 1 else LIGHT_GRAY,
                                   edgecolor='none')
        ax.add_patch(rect)
    ax.text(0.5, 0.1, 'k', ha='center', fontsize=8, color=MEDIUM_GRAY, style='italic')

    # 2d: Consensus embedding
    ax = axes[1, 3]
    ax.set_title('Consensus', fontsize=9, color=MEDIUM_GRAY)
    rect = mpatches.Rectangle((0.25, 0.15), 0.2, 0.7,
                               facecolor='#e8f4fc', edgecolor=COLORS['blue'], lw=1.5)
    ax.add_patch(rect)
    ax.text(0.35, 0.5, 'W*', ha='center', va='center', fontsize=10, color=CHARCOAL)

    plt.tight_layout()
    return fig, 'variant_1_minimal.svg'


def variant_2_graph_focused():
    """Variant emphasizing the graph structure and soft clusters."""
    fig = plt.figure(figsize=(11, 4))

    # Create custom grid
    ax1 = fig.add_axes([0.02, 0.15, 0.18, 0.7])  # Graph
    ax2 = fig.add_axes([0.24, 0.15, 0.14, 0.7])  # RSM
    ax3 = fig.add_axes([0.42, 0.15, 0.14, 0.7])  # Curves
    ax4 = fig.add_axes([0.60, 0.15, 0.08, 0.7])  # Embedding
    ax5 = fig.add_axes([0.74, 0.15, 0.22, 0.7])  # Soft clusters

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    fig.text(0.5, 0.95, 'Variant 2: Graph-Focused', ha='center', fontsize=12, color=CHARCOAL)

    # 1: Graph with community structure
    rng = np.random.default_rng(42)
    positions = []
    node_colors = []
    # 3 clusters
    for cluster, (cx, cy), col in [(0, (0.25, 0.75), COLORS['red']),
                                     (1, (0.75, 0.75), COLORS['blue']),
                                     (2, (0.5, 0.25), COLORS['green'])]:
        for _ in range(5):
            x, y = cx + rng.normal(0, 0.12), cy + rng.normal(0, 0.1)
            positions.append((x, y))
            node_colors.append(col)

    # Draw edges (within and between clusters)
    positions = np.array(positions)
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            ci, cj = i // 5, j // 5
            if ci == cj:
                prob = 0.7
                alpha = 0.4
            else:
                prob = 0.15
                alpha = 0.15
            if rng.random() < prob:
                ax1.plot([positions[i, 0], positions[j, 0]],
                        [positions[i, 1], positions[j, 1]],
                        color=LIGHT_GRAY, lw=0.8, alpha=alpha, zorder=1)

    # Draw nodes
    for (x, y), col in zip(positions, node_colors):
        circle = Circle((x, y), 0.045, facecolor=col, edgecolor='white', lw=1, zorder=2)
        ax1.add_patch(circle)
    ax1.set_title('Graph', fontsize=9, color=MEDIUM_GRAY, pad=5)

    # Arrow
    fig.text(0.205, 0.5, '→', fontsize=16, color=MEDIUM_GRAY, ha='center', va='center')

    # 2: RSM with block structure
    n = 15
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            bi, bj = i // 5, j // 5
            if bi == bj:
                S[i, j] = rng.uniform(0.6, 0.95)
            else:
                S[i, j] = rng.uniform(0.05, 0.25)
    np.fill_diagonal(S, 1)

    cell = 1.0 / n
    for i in range(n):
        for j in range(n):
            color = plt.cm.Blues(S[i, j])
            rect = mpatches.Rectangle((j*cell, 1-(i+1)*cell), cell, cell,
                                       facecolor=color, edgecolor='none')
            ax2.add_patch(rect)
    ax2.set_title('Similarity', fontsize=9, color=MEDIUM_GRAY, pad=5)

    # Arrow
    fig.text(0.395, 0.5, '→', fontsize=16, color=MEDIUM_GRAY, ha='center', va='center')

    # 3: Train/val curves
    x = np.linspace(0.05, 0.95, 50)
    train = 0.8 * np.exp(-4*x) + 0.08
    val = 0.6 * np.exp(-2.5*x) + 0.12 + 0.25 * (x - 0.35)**2
    ax3.plot(x, train, color=CHARCOAL, lw=1.5)
    ax3.plot(x, val, color=COLORS['red'], lw=1.5)
    k_star = 0.35
    ax3.axvline(k_star, color=LIGHT_GRAY, ls='--', lw=1)
    ax3.scatter([k_star], [val[17]], color=COLORS['red'], s=30, zorder=3)
    ax3.text(0.5, 0.05, 'Rank', fontsize=8, color=MEDIUM_GRAY, ha='center')
    ax3.set_title('Cross-val', fontsize=9, color=MEDIUM_GRAY, pad=5)

    # Arrow
    fig.text(0.575, 0.5, '→', fontsize=16, color=MEDIUM_GRAY, ha='center', va='center')

    # 4: Embedding matrix W
    rect = mpatches.Rectangle((0.2, 0.1), 0.6, 0.8,
                               facecolor='#f5f5f5', edgecolor=CHARCOAL, lw=1)
    ax4.add_patch(rect)
    # Show some structure
    for i in range(15):
        row_y = 0.85 - i * 0.05
        cluster = i // 5
        for k in range(3):
            if k == cluster:
                w = rng.uniform(0.6, 0.9)
            else:
                w = rng.uniform(0.05, 0.2)
            col = [COLORS['red'], COLORS['blue'], COLORS['green']][k]
            rect = mpatches.Rectangle((0.25 + k*0.18, row_y), 0.14, 0.04,
                                       facecolor=col, alpha=w, edgecolor='none')
            ax4.add_patch(rect)
    ax4.set_title('W', fontsize=9, color=MEDIUM_GRAY, pad=5)

    # Arrow
    fig.text(0.70, 0.5, '→', fontsize=16, color=MEDIUM_GRAY, ha='center', va='center')

    # 5: Soft cluster visualization (like our RSM margins)
    ax5.set_title('Soft Clusters', fontsize=9, color=MEDIUM_GRAY, pad=5)

    # Simplex-like representation
    corners = np.array([[0.5, 0.9], [0.15, 0.2], [0.85, 0.2]])
    triangle = Polygon(corners, fill=False, edgecolor=LIGHT_GRAY, lw=1)
    ax5.add_patch(triangle)

    # Label corners
    ax5.text(0.5, 0.95, 'F1', fontsize=8, ha='center', color=COLORS['red'])
    ax5.text(0.08, 0.18, 'F2', fontsize=8, ha='center', color=COLORS['blue'])
    ax5.text(0.92, 0.18, 'F3', fontsize=8, ha='center', color=COLORS['green'])

    # Place points based on W weights
    for i in range(15):
        cluster = i // 5
        if cluster == 0:
            w = np.array([0.8, 0.1, 0.1]) + rng.normal(0, 0.05, 3)
        elif cluster == 1:
            w = np.array([0.1, 0.8, 0.1]) + rng.normal(0, 0.05, 3)
        else:
            w = np.array([0.1, 0.1, 0.8]) + rng.normal(0, 0.05, 3)
        w = np.clip(w, 0, 1)
        w = w / w.sum()
        pos = w @ corners

        # Blend color
        rgb = (w[0] * np.array([0.88, 0.44, 0.25]) +
               w[1] * np.array([0.31, 0.44, 0.75]) +
               w[2] * np.array([0.44, 0.69, 0.25]))
        circle = Circle(pos, 0.035, facecolor=rgb, edgecolor='white', lw=0.5)
        ax5.add_patch(circle)

    return fig, 'variant_2_graph.svg'


def variant_3_horizontal_flow():
    """Clean horizontal flow with icons."""
    fig = plt.figure(figsize=(12, 2.5))

    # Single row of panels
    n_panels = 6
    panel_width = 0.13
    gap = 0.02
    start_x = 0.04

    titles = ['Data', 'Similarity', 'Rank Selection', 'Factorization', 'Embedding', 'Interpretation']

    for i in range(n_panels):
        ax = fig.add_axes([start_x + i * (panel_width + gap), 0.2, panel_width, 0.65])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(titles[i], fontsize=8, color=MEDIUM_GRAY, pad=3)

        rng = np.random.default_rng(42 + i)

        if i == 0:  # Data - simple graph icon
            for _ in range(6):
                x, y = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
                circle = Circle((x, y), 0.06, facecolor=COLORS['blue'], alpha=0.7, edgecolor='none')
                ax.add_patch(circle)

        elif i == 1:  # Similarity matrix
            n = 10
            for r in range(n):
                for c in range(n):
                    val = 0.8 if abs(r-c) < 3 else 0.2
                    val += rng.uniform(-0.1, 0.1)
                    rect = mpatches.Rectangle((c/n + 0.05, 0.9 - (r+1)/n), 0.9/n, 0.9/n,
                                               facecolor=plt.cm.Blues(val), edgecolor='none')
                    ax.add_patch(rect)

        elif i == 2:  # Rank selection curves
            x = np.linspace(0.1, 0.9, 30)
            train = 0.8 * np.exp(-4*x) + 0.1
            val = 0.6 * np.exp(-2*x) + 0.15 + 0.2 * (x - 0.4)**2
            ax.plot(x, train, color=CHARCOAL, lw=1.2)
            ax.plot(x, val, color=COLORS['red'], lw=1.2)
            ax.axvline(0.4, color=LIGHT_GRAY, ls='--', lw=0.8)

        elif i == 3:  # Factorization
            ax.text(0.5, 0.55, 'S ≈ WWᵀ', ha='center', va='center', fontsize=9, color=CHARCOAL)
            ax.text(0.5, 0.35, 'W ≥ 0', ha='center', va='center', fontsize=8, color=MEDIUM_GRAY)
            box = FancyBboxPatch((0.15, 0.2), 0.7, 0.6, boxstyle="round,pad=0.03",
                                 facecolor='#f0f7fc', edgecolor=COLORS['blue'], lw=1)
            ax.add_patch(box)

        elif i == 4:  # Embedding matrix
            rect = mpatches.Rectangle((0.3, 0.1), 0.4, 0.8,
                                       facecolor='#f5f5f5', edgecolor=CHARCOAL, lw=1)
            ax.add_patch(rect)
            ax.text(0.5, 0.5, 'W', ha='center', va='center', fontsize=11, color=CHARCOAL)
            ax.text(0.5, 0.02, 'n × k', ha='center', fontsize=7, color=MEDIUM_GRAY)

        elif i == 5:  # Interpretation - bar chart of loadings
            cols = [COLORS['red'], COLORS['blue'], COLORS['green']]
            for j, (h, c) in enumerate(zip([0.7, 0.5, 0.8], cols)):
                rect = mpatches.Rectangle((0.15 + j*0.25, 0.15), 0.18, h * 0.7,
                                           facecolor=c, alpha=0.8, edgecolor='none')
                ax.add_patch(rect)

        # Arrows between panels
        if i < n_panels - 1:
            fig.text(start_x + (i + 1) * (panel_width + gap) - gap/2, 0.52,
                    '→', fontsize=14, color=LIGHT_GRAY, ha='center', va='center')

    fig.text(0.5, 0.92, 'Variant 3: Horizontal Flow', ha='center', fontsize=11, color=CHARCOAL)

    return fig, 'variant_3_horizontal.svg'


def variant_4_two_stage():
    """Two-stage pipeline with clear separation."""
    fig = plt.figure(figsize=(10, 5))

    # Stage 1 box
    stage1 = fig.add_axes([0.05, 0.52, 0.9, 0.42])
    stage1.set_xlim(0, 1)
    stage1.set_ylim(0, 1)
    stage1.axis('off')

    # Stage 1 background
    box1 = FancyBboxPatch((0.01, 0.05), 0.98, 0.9, boxstyle="round,pad=0.02",
                          facecolor='#fafafa', edgecolor=LIGHT_GRAY, lw=1)
    stage1.add_patch(box1)
    stage1.text(0.02, 0.92, 'Stage 1: Embedding', fontsize=10, color=CHARCOAL,
                fontweight='bold', va='top')

    # Stage 1 content
    rng = np.random.default_rng(42)

    # Graph
    for _ in range(8):
        x, y = rng.uniform(0.08, 0.22), rng.uniform(0.25, 0.75)
        col = [COLORS['red'], COLORS['blue'], COLORS['green']][rng.integers(3)]
        circle = Circle((x, y), 0.025, facecolor=col, edgecolor='white', lw=0.5)
        stage1.add_patch(circle)

    stage1.text(0.26, 0.5, '→', fontsize=14, color=MEDIUM_GRAY, ha='center', va='center')

    # RSM
    n = 8
    for i in range(n):
        for j in range(n):
            bi, bj = i // 3, j // 3
            val = 0.8 if bi == bj else 0.2
            rect = mpatches.Rectangle((0.30 + j*0.022, 0.68 - (i+1)*0.05), 0.02, 0.045,
                                       facecolor=plt.cm.Blues(val), edgecolor='none')
            stage1.add_patch(rect)

    stage1.text(0.50, 0.5, '→', fontsize=14, color=MEDIUM_GRAY, ha='center', va='center')

    # Cross-val
    x = np.linspace(0.54, 0.70, 30)
    y_scale = np.linspace(0.25, 0.75, 30)
    train = 0.75 - 0.4 * (x - 0.54) / 0.16
    val_base = 0.7 - 0.25 * (x - 0.54) / 0.16
    val = val_base + 0.15 * ((x - 0.60) / 0.08)**2
    stage1.plot(x, train, color=CHARCOAL, lw=1)
    stage1.plot(x, val, color=COLORS['red'], lw=1)
    stage1.axvline(0.62, ymin=0.2, ymax=0.8, color=LIGHT_GRAY, ls='--', lw=0.8)
    stage1.text(0.62, 0.18, 'k*', fontsize=7, color=MEDIUM_GRAY, ha='center', style='italic')

    stage1.text(0.74, 0.5, '→', fontsize=14, color=MEDIUM_GRAY, ha='center', va='center')

    # SRF box
    srf_box = FancyBboxPatch((0.77, 0.3), 0.08, 0.4, boxstyle="round,pad=0.01",
                             facecolor='#e8f4fc', edgecolor=COLORS['blue'], lw=1)
    stage1.add_patch(srf_box)
    stage1.text(0.81, 0.5, 'SRF', fontsize=8, color=CHARCOAL, ha='center', va='center')

    stage1.text(0.88, 0.5, '→', fontsize=14, color=MEDIUM_GRAY, ha='center', va='center')

    # W matrix
    w_rect = mpatches.Rectangle((0.91, 0.3), 0.06, 0.4,
                                 facecolor=LIGHT_GRAY, edgecolor=CHARCOAL, lw=1)
    stage1.add_patch(w_rect)
    stage1.text(0.94, 0.5, 'W', fontsize=9, color=CHARCOAL, ha='center', va='center')

    # Stage 2 box
    stage2 = fig.add_axes([0.05, 0.05, 0.9, 0.42])
    stage2.set_xlim(0, 1)
    stage2.set_ylim(0, 1)
    stage2.axis('off')

    box2 = FancyBboxPatch((0.01, 0.05), 0.98, 0.9, boxstyle="round,pad=0.02",
                          facecolor='#fafafa', edgecolor=LIGHT_GRAY, lw=1)
    stage2.add_patch(box2)
    stage2.text(0.02, 0.92, 'Stage 2: Consensus', fontsize=10, color=CHARCOAL,
                fontweight='bold', va='top')

    # Multiple W matrices
    for i in range(4):
        rect = mpatches.Rectangle((0.08 + i*0.04, 0.28 + i*0.08), 0.05, 0.35,
                                   facecolor='white', edgecolor=LIGHT_GRAY, lw=0.8)
        stage2.add_patch(rect)
    rect = mpatches.Rectangle((0.08 + 4*0.04, 0.28 + 4*0.08), 0.05, 0.35,
                               facecolor=LIGHT_GRAY, edgecolor=CHARCOAL, lw=1)
    stage2.add_patch(rect)
    stage2.text(0.20, 0.18, 'repeats', fontsize=7, color=MEDIUM_GRAY, ha='center', style='italic')

    stage2.text(0.36, 0.5, '→', fontsize=14, color=MEDIUM_GRAY, ha='center', va='center')

    # Clustering
    centers = [(0.45, 0.65), (0.55, 0.65), (0.50, 0.35)]
    for (cx, cy), col in zip(centers, [COLORS['red'], COLORS['blue'], COLORS['green']]):
        for _ in range(6):
            x, y = cx + rng.normal(0, 0.03), cy + rng.normal(0, 0.05)
            circle = Circle((x, y), 0.015, facecolor=col, alpha=0.7, edgecolor='none')
            stage2.add_patch(circle)
    stage2.text(0.50, 0.18, 'cluster', fontsize=7, color=MEDIUM_GRAY, ha='center', style='italic')

    stage2.text(0.64, 0.5, '→', fontsize=14, color=MEDIUM_GRAY, ha='center', va='center')

    # Rank histogram
    bars = [0.3, 0.7, 0.5, 0.2]
    for i, h in enumerate(bars):
        rect = mpatches.Rectangle((0.68 + i*0.04, 0.3), 0.03, h * 0.4,
                                   facecolor=COLORS['blue'] if i == 1 else LIGHT_GRAY,
                                   edgecolor='none')
        stage2.add_patch(rect)
    stage2.text(0.76, 0.18, 'select k', fontsize=7, color=MEDIUM_GRAY, ha='center', style='italic')

    stage2.text(0.84, 0.5, '→', fontsize=14, color=MEDIUM_GRAY, ha='center', va='center')

    # Consensus W
    w_final = FancyBboxPatch((0.88, 0.3), 0.08, 0.4, boxstyle="round,pad=0.01",
                             facecolor='#e8f4fc', edgecolor=COLORS['blue'], lw=1.5)
    stage2.add_patch(w_final)
    stage2.text(0.92, 0.5, 'W*', fontsize=9, color=CHARCOAL, ha='center', va='center')

    fig.text(0.5, 0.97, 'Variant 4: Two-Stage Pipeline', ha='center', fontsize=11, color=CHARCOAL)

    return fig, 'variant_4_twostage.svg'


def variant_5_equation_focused():
    """Variant emphasizing the mathematical formulation."""
    fig = plt.figure(figsize=(10, 3.5))

    ax = fig.add_axes([0.05, 0.1, 0.9, 0.75])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    fig.text(0.5, 0.92, 'Variant 5: Equation-Focused', ha='center', fontsize=11, color=CHARCOAL)

    # Main equation
    ax.text(0.5, 0.75, r'$\mathbf{S} \approx \mathbf{W}\mathbf{W}^\top$',
            fontsize=16, ha='center', va='center', color=CHARCOAL)
    ax.text(0.5, 0.60, r'$\mathbf{W} \geq 0$, $\quad \mathbf{W} \in \mathbb{R}^{n \times k}$',
            fontsize=11, ha='center', va='center', color=MEDIUM_GRAY)

    # Visual representation below
    rng = np.random.default_rng(42)

    # S matrix (left)
    n = 12
    for i in range(n):
        for j in range(n):
            bi, bj = i // 4, j // 4
            val = 0.8 if bi == bj else 0.15
            val += rng.uniform(-0.1, 0.1)
            rect = mpatches.Rectangle((0.08 + j*0.018, 0.42 - (i+1)*0.028),
                                       0.016, 0.026,
                                       facecolor=plt.cm.Blues(val), edgecolor='none')
            ax.add_patch(rect)
    ax.text(0.19, 0.08, 'S', fontsize=11, ha='center', color=CHARCOAL)

    # Equals
    ax.text(0.35, 0.25, '≈', fontsize=14, ha='center', va='center', color=CHARCOAL)

    # W matrix (middle)
    k = 3
    for i in range(n):
        for j in range(k):
            cluster = i // 4
            if j == cluster:
                val = rng.uniform(0.7, 0.95)
            else:
                val = rng.uniform(0.05, 0.2)
            col = [COLORS['red'], COLORS['blue'], COLORS['green']][j]
            rect = mpatches.Rectangle((0.42 + j*0.03, 0.42 - (i+1)*0.028),
                                       0.025, 0.026,
                                       facecolor=col, alpha=val, edgecolor='none')
            ax.add_patch(rect)
    ax.text(0.48, 0.08, 'W', fontsize=11, ha='center', color=CHARCOAL)

    # Times
    ax.text(0.56, 0.25, '×', fontsize=12, ha='center', va='center', color=CHARCOAL)

    # W^T matrix (right)
    for i in range(k):
        for j in range(n):
            cluster = j // 4
            if i == cluster:
                val = rng.uniform(0.7, 0.95)
            else:
                val = rng.uniform(0.05, 0.2)
            col = [COLORS['red'], COLORS['blue'], COLORS['green']][i]
            rect = mpatches.Rectangle((0.62 + j*0.018, 0.35 - (i+1)*0.08),
                                       0.016, 0.07,
                                       facecolor=col, alpha=val, edgecolor='none')
            ax.add_patch(rect)
    ax.text(0.73, 0.08, 'Wᵀ', fontsize=11, ha='center', color=CHARCOAL)

    return fig, 'variant_5_equation.svg'


def main():
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    variants = [
        variant_1_minimal,
        variant_2_graph_focused,
        variant_3_horizontal_flow,
        variant_4_two_stage,
        variant_5_equation_focused,
    ]

    for variant_fn in variants:
        fig, filename = variant_fn()
        fig.savefig(output_dir / filename, format='svg', bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f"Saved: {filename}")


if __name__ == "__main__":
    main()
