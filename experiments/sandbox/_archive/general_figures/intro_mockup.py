"""Intro figure mockup - from representations to dimensions."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#c8c8c8'
MEDIUM_GRAY = '#888888'
FAINT_GRAY = '#e8e8e8'

FACTOR_COLORS = np.array([
    [0.88, 0.44, 0.25],
    [0.31, 0.44, 0.75],
    [0.44, 0.69, 0.25],
])
FACTOR_HEX = ['#E07040', '#5070C0', '#70B040']

# Source colors
BRAIN_COLOR = '#7eb5d6'
DNN_COLOR = '#9b7eb5'
BEHAVIOR_COLOR = '#7eb58a'


def create_colormap():
    colors = ['#ffffff', '#f0f4f8', '#d4e1ed', '#a8c5db',
              '#6a9fc0', '#3d7ea6', '#1a5a7a', '#0d3d54']
    return mcolors.LinearSegmentedColormap.from_list('elegant', colors)


def blend_color(weights):
    weights = np.array(weights) / np.sum(weights)
    return weights @ FACTOR_COLORS


def draw_brain_icon(ax, x, y, size=0.08, color=BRAIN_COLOR):
    """Brain silhouette with two hemispheres and characteristic folds."""
    from matplotlib.path import Path as MPath
    import matplotlib.patches as mpatch

    s = size  # shorthand

    # Brain outline using bezier curves - side view with lobes
    verts = [
        # Start at bottom center (brain stem area)
        (x, y - s * 0.35),
        # Right side - up along back
        (x + s * 0.15, y - s * 0.4),
        (x + s * 0.45, y - s * 0.25),
        (x + s * 0.5, y + s * 0.1),
        # Top right lobe (occipital/parietal bulge)
        (x + s * 0.48, y + s * 0.35),
        (x + s * 0.35, y + s * 0.5),
        (x + s * 0.15, y + s * 0.45),
        # Top center dip
        (x, y + s * 0.35),
        # Top left lobe (frontal bulge)
        (x - s * 0.15, y + s * 0.45),
        (x - s * 0.35, y + s * 0.5),
        (x - s * 0.48, y + s * 0.35),
        # Left side - down along front
        (x - s * 0.5, y + s * 0.1),
        (x - s * 0.45, y - s * 0.25),
        (x - s * 0.15, y - s * 0.4),
        # Back to start
        (x, y - s * 0.35),
    ]

    codes = [MPath.MOVETO] + [MPath.CURVE3] * (len(verts) - 1)
    path = MPath(verts, codes)
    patch = mpatch.PathPatch(path, facecolor=color, edgecolor='white', lw=0.8)
    ax.add_patch(patch)

    # Central sulcus (vertical groove dividing hemispheres)
    sulcus_verts = [
        (x, y + s * 0.35),
        (x - s * 0.02, y + s * 0.15),
        (x + s * 0.02, y - s * 0.05),
        (x, y - s * 0.2),
    ]
    sulcus_codes = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3, MPath.CURVE3]
    sulcus_path = MPath(sulcus_verts, sulcus_codes)
    sulcus_patch = mpatch.PathPatch(
        sulcus_path, facecolor='none', edgecolor='white', lw=0.6, alpha=0.7
    )
    ax.add_patch(sulcus_patch)

    # Add horizontal fold lines for texture (simplified gyri)
    for offset in [-0.15, 0.1]:
        fold_y = y + s * offset
        fold_verts = [
            (x - s * 0.3, fold_y),
            (x - s * 0.1, fold_y + s * 0.08),
            (x + s * 0.1, fold_y - s * 0.05),
            (x + s * 0.3, fold_y + s * 0.03),
        ]
        fold_codes = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3, MPath.CURVE3]
        fold_path = MPath(fold_verts, fold_codes)
        fold_patch = mpatch.PathPatch(
            fold_path, facecolor='none', edgecolor='white', lw=0.4, alpha=0.5
        )
        ax.add_patch(fold_patch)


def draw_dnn_icon(ax, x, y, size=0.06, color=DNN_COLOR):
    """Simple neural network icon."""
    layers = [3, 4, 3]
    layer_x = [x - size, x, x + size]

    for li, (lx, n_nodes) in enumerate(zip(layer_x, layers)):
        for ni in range(n_nodes):
            ny = y + (ni - (n_nodes-1)/2) * size * 0.5
            circle = mpatches.Circle((lx, ny), size*0.15, facecolor=color, edgecolor='white', lw=0.3)
            ax.add_patch(circle)

            # Connections to next layer
            if li < len(layers) - 1:
                next_n = layers[li+1]
                for nj in range(next_n):
                    next_y = y + (nj - (next_n-1)/2) * size * 0.5
                    ax.plot([lx, layer_x[li+1]], [ny, next_y], color=color, alpha=0.3, lw=0.3)


def draw_behavior_icon(ax, x, y, size=0.08, color=BEHAVIOR_COLOR):
    """Simple person/group icon."""
    # Head
    head = mpatches.Circle((x, y + size*0.4), size*0.25, facecolor=color, edgecolor='white', lw=0.5)
    ax.add_patch(head)
    # Body
    body = mpatches.Ellipse((x, y - size*0.1), size*0.5, size*0.6, facecolor=color, edgecolor='white', lw=0.5)
    ax.add_patch(body)


def draw_rsm(ax, n=10, seed=42):
    """Draw a similarity matrix."""
    rng = np.random.default_rng(seed)
    cmap = create_colormap()

    # Create block structure
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            bi, bj = i // 4, j // 4
            if bi == bj:
                S[i, j] = rng.uniform(0.6, 0.95)
            else:
                S[i, j] = rng.uniform(0.1, 0.35)
    np.fill_diagonal(S, 1)

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


def draw_items_with_dimensions(ax, seed=42):
    """Show items positioned in 2D dimension space."""
    rng = np.random.default_rng(seed)

    # Items positioned in 2D "dimension space" - (x=roundness, y=animacy)
    items = [
        (0.2, 0.80, 'dog', FACTOR_HEX[0]),
        (0.25, 0.75, 'cat', FACTOR_HEX[0]),
        (0.15, 0.65, 'bird', FACTOR_HEX[0]),
        (0.80, 0.20, 'ball', FACTOR_HEX[1]),
        (0.75, 0.25, 'orange', FACTOR_HEX[1]),
        (0.70, 0.15, 'apple', FACTOR_HEX[1]),
        (0.35, 0.40, 'turtle', FACTOR_HEX[2]),
        (0.55, 0.35, 'egg', FACTOR_HEX[2]),
    ]

    # Draw dimension axes
    ax.annotate('', xy=(0.95, 0.08), xytext=(0.08, 0.08),
               arrowprops=dict(arrowstyle='->', color=MEDIUM_GRAY, lw=1.2))
    ax.text(0.55, 0.01, 'roundness', fontsize=8, ha='center', color=MEDIUM_GRAY, style='italic')

    ax.annotate('', xy=(0.08, 0.95), xytext=(0.08, 0.12),
               arrowprops=dict(arrowstyle='->', color=MEDIUM_GRAY, lw=1.2))
    ax.text(0.02, 0.55, 'animacy', fontsize=8, ha='center', color=MEDIUM_GRAY, style='italic', rotation=90)

    # Draw items as colored circles with labels
    for x, y, name, color in items:
        circle = mpatches.Circle((x, y), 0.045, facecolor=color, edgecolor='white', lw=0.8)
        ax.add_patch(circle)
        ax.text(x + 0.07, y, name, fontsize=6, ha='left', va='center', color=CHARCOAL)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_w_matrix_compact(ax, n=12, k=3):
    """Compact W matrix showing factor loadings."""
    rng = np.random.default_rng(42)

    W = np.zeros((n, k))
    for i in range(n):
        cluster = i // 4
        for j in range(k):
            if j == cluster:
                W[i, j] = rng.uniform(0.7, 0.95)
            else:
                W[i, j] = rng.uniform(0.03, 0.15)
    W = W / W.sum(axis=1, keepdims=True)

    cell_h = 0.85 / n
    cell_w = 0.25

    for i in range(n):
        y = 0.92 - (i + 1) * cell_h
        for j in range(k):
            val = W[i, j]
            rect = mpatches.Rectangle(
                (0.15 + j * cell_w, y), cell_w * 0.9, cell_h * 0.9,
                facecolor=FACTOR_HEX[j], alpha=val, edgecolor='none')
            ax.add_patch(rect)

    # Labels
    labels = ['animacy', 'size', 'color']
    for j in range(k):
        ax.text(0.15 + (j + 0.5) * cell_w, 0.02, labels[j],
                fontsize=6, ha='center', color=FACTOR_HEX[j], style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def main():
    fig = plt.figure(figsize=(12, 7))

    # === TOP ROW: Different sources → RSA ===

    # Brain
    ax1 = fig.add_axes([0.02, 0.68, 0.12, 0.25])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    draw_brain_icon(ax1, 0.5, 0.55, size=0.25, color=BRAIN_COLOR)
    ax1.text(0.5, 0.15, 'Brain', fontsize=9, ha='center', color=CHARCOAL)

    # DNN
    ax2 = fig.add_axes([0.14, 0.68, 0.12, 0.25])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    draw_dnn_icon(ax2, 0.5, 0.55, size=0.18, color=DNN_COLOR)
    ax2.text(0.5, 0.15, 'DNN', fontsize=9, ha='center', color=CHARCOAL)

    # Behavior
    ax3 = fig.add_axes([0.26, 0.68, 0.12, 0.25])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    draw_behavior_icon(ax3, 0.5, 0.55, size=0.2, color=BEHAVIOR_COLOR)
    ax3.text(0.5, 0.15, 'Behavior', fontsize=9, ha='center', color=CHARCOAL)

    # Arrow → RSA
    fig.text(0.40, 0.80, '→', fontsize=20, color=MEDIUM_GRAY, ha='center')
    fig.text(0.40, 0.73, 'abstract to\nsimilarity', fontsize=7, ha='center', color=MEDIUM_GRAY)

    # RSM (RSA)
    ax4 = fig.add_axes([0.45, 0.65, 0.18, 0.28])
    draw_rsm(ax4)
    fig.text(0.54, 0.62, 'Similarity Matrix', fontsize=9, ha='center', color=CHARCOAL)
    fig.text(0.54, 0.58, '(RSA)', fontsize=8, ha='center', color=MEDIUM_GRAY)

    # Question mark / gap
    ax_q = fig.add_axes([0.66, 0.68, 0.10, 0.22])
    ax_q.set_xlim(0, 1)
    ax_q.set_ylim(0, 1)
    ax_q.axis('off')
    ax_q.text(0.5, 0.6, '?', fontsize=36, ha='center', va='center', color=LIGHT_GRAY)
    ax_q.text(0.5, 0.2, 'What\ndimensions?', fontsize=8, ha='center', color=MEDIUM_GRAY)

    # === MIDDLE: The insight ===
    fig.text(0.5, 0.50, 'RSA tells us IF representations are similar, but not WHY',
             fontsize=10, ha='center', color=CHARCOAL, style='italic')

    # === BOTTOM ROW: SRF → Dimensions ===

    # RSM again (input)
    ax5 = fig.add_axes([0.05, 0.12, 0.15, 0.28])
    draw_rsm(ax5, seed=42)
    fig.text(0.125, 0.08, 'Similarity S', fontsize=9, ha='center', color=CHARCOAL)

    # Arrow → SRF
    fig.text(0.23, 0.26, '→', fontsize=18, color=MEDIUM_GRAY, ha='center')

    # SRF box
    ax6 = fig.add_axes([0.27, 0.15, 0.14, 0.22])
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    box = mpatches.FancyBboxPatch((0.05, 0.1), 0.9, 0.8, boxstyle="round,pad=0.05",
                                   facecolor='#f8fbfd', edgecolor=FACTOR_HEX[1], lw=1.5)
    ax6.add_patch(box)
    ax6.text(0.5, 0.55, 'S ≈ WWᵀ', fontsize=10, ha='center', va='center', color=CHARCOAL)
    ax6.text(0.5, 0.35, 'W ≥ 0', fontsize=8, ha='center', va='center', color=MEDIUM_GRAY)
    fig.text(0.34, 0.08, 'SRF', fontsize=9, ha='center', color=CHARCOAL)

    # Arrow → W
    fig.text(0.44, 0.26, '→', fontsize=18, color=MEDIUM_GRAY, ha='center')

    # W matrix (dimensions)
    ax7 = fig.add_axes([0.48, 0.10, 0.15, 0.32])
    draw_w_matrix_compact(ax7)
    fig.text(0.555, 0.08, 'Dimensions W', fontsize=9, ha='center', color=CHARCOAL)

    # Arrow → interpretation
    fig.text(0.66, 0.26, '→', fontsize=18, color=MEDIUM_GRAY, ha='center')

    # Dimension space visualization
    ax8 = fig.add_axes([0.70, 0.10, 0.28, 0.35])
    draw_items_with_dimensions(ax8)
    fig.text(0.84, 0.08, 'Interpretable Space', fontsize=9, ha='center', color=CHARCOAL)

    # Title
    fig.text(0.5, 0.97, 'From Similarity to Dimensions', fontsize=12, ha='center',
             color=CHARCOAL, fontweight='medium')

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "intro_mockup.svg", format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("Saved: intro_mockup.svg")


if __name__ == "__main__":
    main()
