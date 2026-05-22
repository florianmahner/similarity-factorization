"""Elegant figure: From measurements to representational similarity.

Beautiful, Nature-quality abstract figure.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.path import Path as MPath
import numpy as np

CHARCOAL = '#2d2d2d'
LIGHT_GRAY = '#d0d0d0'
MEDIUM_GRAY = '#808080'
FAINT_GRAY = '#f5f5f5'

# Elegant, harmonious palette
PALETTE = {
    'coral': '#e07860',
    'ocean': '#5a8fba',
    'sage': '#7ba876',
    'lavender': '#9a7bb5',
    'gold': '#c9a857',
    'rose': '#c27888',
}


def create_colormap():
    """Elegant blue-gray colormap for similarity."""
    colors = ['#fafbfc', '#e8eef4', '#c5d5e4', '#94b5cf',
              '#6394b5', '#3d7292', '#1f4f6b', '#0d3348']
    return mcolors.LinearSegmentedColormap.from_list('elegant', colors)


def draw_blob(ax, x, y, size, color, seed=0, n_points=8):
    """Draw organic blob shape."""
    rng = np.random.default_rng(seed)

    # Generate smooth blob using polar coordinates
    theta = np.linspace(0, 2 * np.pi, 100)

    # Random radius variations
    r_base = size
    r_var = size * 0.25

    # Create smooth variations using sin waves
    r = r_base + r_var * (
        0.3 * np.sin(2 * theta + rng.uniform(0, 2*np.pi)) +
        0.2 * np.sin(3 * theta + rng.uniform(0, 2*np.pi)) +
        0.15 * np.sin(4 * theta + rng.uniform(0, 2*np.pi))
    )

    xs = x + r * np.cos(theta)
    ys = y + r * np.sin(theta)

    ax.fill(xs, ys, facecolor=color, edgecolor='white', lw=1.5, zorder=3)


def draw_abstract_stimulus(ax, x, y, size, style, color):
    """Draw various abstract stimulus types."""

    if style == 'blob_a':
        draw_blob(ax, x, y, size, color, seed=42)
    elif style == 'blob_b':
        draw_blob(ax, x, y, size, color, seed=123)
    elif style == 'blob_c':
        draw_blob(ax, x, y, size, color, seed=789)
    elif style == 'star':
        # Soft star shape
        n_points = 5
        theta_outer = np.linspace(0, 2*np.pi, n_points, endpoint=False) - np.pi/2
        theta_inner = theta_outer + np.pi/n_points
        r_outer, r_inner = size, size * 0.5

        verts = []
        for i in range(n_points):
            verts.append((x + r_outer * np.cos(theta_outer[i]),
                         y + r_outer * np.sin(theta_outer[i])))
            verts.append((x + r_inner * np.cos(theta_inner[i]),
                         y + r_inner * np.sin(theta_inner[i])))

        patch = mpatches.Polygon(verts, facecolor=color, edgecolor='white', lw=1.5, zorder=3)
        ax.add_patch(patch)
    elif style == 'rounded_rect':
        patch = mpatches.FancyBboxPatch(
            (x - size, y - size * 0.6), size * 2, size * 1.2,
            boxstyle="round,pad=0,rounding_size=0.15",
            facecolor=color, edgecolor='white', lw=1.5, zorder=3
        )
        ax.add_patch(patch)
    elif style == 'hexagon':
        n = 6
        theta = np.linspace(0, 2*np.pi, n, endpoint=False) + np.pi/6
        verts = [(x + size * np.cos(t), y + size * np.sin(t)) for t in theta]
        patch = mpatches.Polygon(verts, facecolor=color, edgecolor='white', lw=1.5, zorder=3)
        ax.add_patch(patch)
    else:
        # Default circle
        patch = mpatches.Circle((x, y), size, facecolor=color, edgecolor='white', lw=1.5, zorder=3)
        ax.add_patch(patch)


def draw_response_bars(ax, x, y, values, width=0.08, height=0.35, color_base=MEDIUM_GRAY):
    """Draw elegant horizontal response bars."""
    n = len(values)
    bar_h = height / n * 0.85
    gap = height / n * 0.15

    for i, v in enumerate(values):
        yi = y + height/2 - (i + 1) * (bar_h + gap) + gap/2
        bar_w = width * v

        # Gradient-like effect with alpha
        alpha = 0.3 + 0.6 * v
        rect = mpatches.FancyBboxPatch(
            (x, yi), bar_w, bar_h,
            boxstyle="round,pad=0,rounding_size=0.01",
            facecolor=color_base, alpha=alpha, edgecolor='none'
        )
        ax.add_patch(rect)


def draw_curved_arrow(ax, x1, y1, x2, y2, color=LIGHT_GRAY, lw=1.5):
    """Draw elegant curved arrow."""
    # Simple straight arrow with nice styling
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle='-|>',
            color=color,
            lw=lw,
            shrinkA=0, shrinkB=0,
            mutation_scale=12
        )
    )


def draw_rsm(ax, S, cmap):
    """Draw similarity matrix with soft styling."""
    n = len(S)
    cell = 1.0 / n

    for i in range(n):
        for j in range(n):
            # Slight rounding on cells
            rect = mpatches.FancyBboxPatch(
                (j * cell + cell*0.02, 1 - (i + 1) * cell + cell*0.02),
                cell * 0.96, cell * 0.96,
                boxstyle="round,pad=0,rounding_size=0.005",
                facecolor=cmap(S[i, j]), edgecolor='none'
            )
            ax.add_patch(rect)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    fig = plt.figure(figsize=(11, 4.5))
    cmap = create_colormap()

    # Define stimuli - 5 items with distinct styles
    stimuli = [
        ('blob_a', PALETTE['coral']),
        ('blob_b', PALETTE['coral']),
        ('hexagon', PALETTE['ocean']),
        ('rounded_rect', PALETTE['ocean']),
        ('star', PALETTE['sage']),
    ]
    n_stimuli = len(stimuli)

    # Create response patterns with clear structure
    responses = np.array([
        [0.95, 0.85, 0.15, 0.10, 0.12, 0.08],  # coral group
        [0.88, 0.80, 0.18, 0.14, 0.10, 0.11],  # coral group
        [0.12, 0.15, 0.90, 0.85, 0.20, 0.15],  # ocean group
        [0.15, 0.12, 0.82, 0.88, 0.22, 0.18],  # ocean group
        [0.35, 0.40, 0.38, 0.42, 0.92, 0.88],  # sage (mixed)
    ])

    # Compute similarity
    S = responses @ responses.T
    S = S / S.max()

    # === PANEL 1: Stimuli column ===
    ax1 = fig.add_axes([0.03, 0.12, 0.12, 0.78])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    for i, (style, color) in enumerate(stimuli):
        y = 0.88 - i * 0.19
        draw_abstract_stimulus(ax1, 0.5, y, 0.11, style, color)

    fig.text(0.09, 0.04, 'stimuli', fontsize=11, ha='center', color=CHARCOAL,
             fontweight='light', style='italic')

    # Arrow 1
    ax_arrow1 = fig.add_axes([0.15, 0.35, 0.08, 0.3])
    ax_arrow1.set_xlim(0, 1)
    ax_arrow1.set_ylim(0, 1)
    ax_arrow1.axis('off')
    draw_curved_arrow(ax_arrow1, 0.1, 0.5, 0.9, 0.5, color=LIGHT_GRAY, lw=1.8)
    fig.text(0.19, 0.42, 'measure', fontsize=9, ha='center', color=MEDIUM_GRAY, style='italic')

    # === PANEL 2: Response patterns ===
    ax2 = fig.add_axes([0.24, 0.12, 0.22, 0.78])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    for i, (style, color) in enumerate(stimuli):
        y = 0.88 - i * 0.19

        # Small stimulus
        draw_abstract_stimulus(ax2, 0.08, y, 0.055, style, color)

        # Thin connecting line
        ax2.plot([0.16, 0.28], [y, y], color=LIGHT_GRAY, lw=1, zorder=1)

        # Response pattern
        draw_response_bars(ax2, 0.30, y, responses[i], width=0.60, height=0.14,
                          color_base=color)

    # Dimension label
    ax2.text(0.95, 0.50, 'features', fontsize=8, ha='center', va='center',
             color=MEDIUM_GRAY, rotation=90, style='italic')

    fig.text(0.35, 0.04, 'response patterns', fontsize=11, ha='center',
             color=CHARCOAL, fontweight='light', style='italic')

    # Arrow 2
    ax_arrow2 = fig.add_axes([0.46, 0.35, 0.08, 0.3])
    ax_arrow2.set_xlim(0, 1)
    ax_arrow2.set_ylim(0, 1)
    ax_arrow2.axis('off')
    draw_curved_arrow(ax_arrow2, 0.1, 0.5, 0.9, 0.5, color=LIGHT_GRAY, lw=1.8)
    fig.text(0.50, 0.42, 'compare', fontsize=9, ha='center', color=MEDIUM_GRAY, style='italic')

    # === PANEL 3: Pairwise comparisons ===
    ax3 = fig.add_axes([0.54, 0.12, 0.18, 0.78])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    # Show select pairwise comparisons
    pairs = [(0, 1, 0.82), (2, 3, 0.76), (0, 2, 0.22), (1, 4, 0.42)]

    for idx, (i, j, _) in enumerate(pairs):
        y = 0.85 - idx * 0.22
        sim = S[i, j]

        style_i, color_i = stimuli[i]
        style_j, color_j = stimuli[j]

        # Left stimulus
        draw_abstract_stimulus(ax3, 0.12, y, 0.055, style_i, color_i)

        # Connection line colored by similarity
        line_color = cmap(sim)
        ax3.plot([0.22, 0.52], [y, y], color=line_color, lw=4,
                solid_capstyle='round', zorder=2)

        # Right stimulus
        draw_abstract_stimulus(ax3, 0.62, y, 0.055, style_j, color_j)

        # Similarity value
        ax3.text(0.82, y, f'{sim:.2f}', fontsize=9, ha='left', va='center',
                color=CHARCOAL, fontweight='light')

    # Ellipsis
    ax3.text(0.37, 0.06, '⋮', fontsize=16, ha='center', color=LIGHT_GRAY)

    fig.text(0.63, 0.04, 'pairwise similarity', fontsize=11, ha='center',
             color=CHARCOAL, fontweight='light', style='italic')

    # Arrow 3
    ax_arrow3 = fig.add_axes([0.72, 0.35, 0.06, 0.3])
    ax_arrow3.set_xlim(0, 1)
    ax_arrow3.set_ylim(0, 1)
    ax_arrow3.axis('off')
    draw_curved_arrow(ax_arrow3, 0.1, 0.5, 0.9, 0.5, color=LIGHT_GRAY, lw=1.8)

    # === PANEL 4: Similarity matrix ===
    rsm_left = 0.78
    rsm_bottom = 0.18
    rsm_size = 0.18

    ax4 = fig.add_axes([rsm_left, rsm_bottom, rsm_size, rsm_size * 11/4.5])
    draw_rsm(ax4, S, cmap)

    # Stimulus icons on margins
    icon_size = 0.022
    for i, (style, color) in enumerate(stimuli):
        # Top margin
        x = rsm_left + (i + 0.5) * rsm_size / n_stimuli
        y = rsm_bottom + rsm_size * 11/4.5 + 0.02

        ax_icon = fig.add_axes([x - icon_size, y, icon_size * 2, icon_size * 2 * 4.5/11])
        ax_icon.set_xlim(0, 1)
        ax_icon.set_ylim(0, 1)
        ax_icon.axis('off')
        draw_abstract_stimulus(ax_icon, 0.5, 0.5, 0.38, style, color)

        # Left margin
        x = rsm_left - 0.035
        y = rsm_bottom + (n_stimuli - i - 0.5) * rsm_size * 11/4.5 / n_stimuli

        ax_icon = fig.add_axes([x - icon_size, y - icon_size * 4.5/11,
                               icon_size * 2, icon_size * 2 * 4.5/11])
        ax_icon.set_xlim(0, 1)
        ax_icon.set_ylim(0, 1)
        ax_icon.axis('off')
        draw_abstract_stimulus(ax_icon, 0.5, 0.5, 0.38, style, color)

    fig.text(0.87, 0.04, 'similarity matrix', fontsize=11, ha='center',
             color=CHARCOAL, fontweight='light', style='italic')

    # Subtle colorbar
    cbar_width = 0.012
    cbar_height = rsm_size * 11/4.5 * 0.6
    ax_cbar = fig.add_axes([rsm_left + rsm_size + 0.015,
                            rsm_bottom + (rsm_size * 11/4.5 - cbar_height)/2,
                            cbar_width, cbar_height])

    n_cbar = 50
    for i in range(n_cbar):
        val = i / (n_cbar - 1)
        rect = mpatches.Rectangle(
            (0, i / n_cbar), 1, 1 / n_cbar,
            facecolor=cmap(val), edgecolor='none'
        )
        ax_cbar.add_patch(rect)

    ax_cbar.set_xlim(0, 1)
    ax_cbar.set_ylim(0, 1)
    ax_cbar.axis('off')
    ax_cbar.text(1.8, 0, '0', fontsize=7, ha='left', va='center', color=MEDIUM_GRAY)
    ax_cbar.text(1.8, 1, '1', fontsize=7, ha='left', va='center', color=MEDIUM_GRAY)

    # Save
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    fig.savefig(output_dir / "representation_similarity_elegant.svg", format='svg',
                bbox_inches='tight', pad_inches=0.08)
    fig.savefig(output_dir / "representation_similarity_elegant.pdf", format='pdf',
                bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print("Saved: representation_similarity_elegant.svg/pdf")


if __name__ == "__main__":
    main()
