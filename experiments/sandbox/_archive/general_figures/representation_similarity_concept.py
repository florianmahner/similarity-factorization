"""Abstract figure: From measurements to representational similarity."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#c8c8c8'
MEDIUM_GRAY = '#888888'
FAINT_GRAY = '#f0f0f0'

# Soft, elegant colors
STIM_COLORS = ['#7eb5d6', '#9b7eb5', '#7eb58a', '#d6a87e', '#b57e7e']


def create_colormap():
    colors = ['#ffffff', '#f0f4f8', '#d4e1ed', '#a8c5db',
              '#6a9fc0', '#3d7ea6', '#1a5a7a', '#0d3d54']
    return mcolors.LinearSegmentedColormap.from_list('elegant', colors)


def draw_stimulus_icon(ax, x, y, size, color, shape='circle'):
    """Draw abstract stimulus as simple geometric shape."""
    if shape == 'circle':
        patch = mpatches.Circle((x, y), size, facecolor=color, edgecolor='white', lw=1.2)
    elif shape == 'square':
        patch = mpatches.Rectangle((x - size, y - size), size * 2, size * 2,
                                    facecolor=color, edgecolor='white', lw=1.2)
    elif shape == 'triangle':
        verts = [(x, y + size), (x - size, y - size * 0.7), (x + size, y - size * 0.7)]
        patch = mpatches.Polygon(verts, facecolor=color, edgecolor='white', lw=1.2)
    elif shape == 'diamond':
        verts = [(x, y + size), (x + size, y), (x, y - size), (x - size, y)]
        patch = mpatches.Polygon(verts, facecolor=color, edgecolor='white', lw=1.2)
    else:
        patch = mpatches.Circle((x, y), size, facecolor=color, edgecolor='white', lw=1.2)
    ax.add_patch(patch)


def draw_response_pattern(ax, x, y, values, width=0.12, height=0.4):
    """Draw a vertical bar pattern representing a response vector."""
    n = len(values)
    bar_h = height / n

    for i, v in enumerate(values):
        yi = y - height / 2 + i * bar_h
        gray = 1 - v * 0.85  # 0 = white, 1 = dark
        color = (gray, gray, gray)
        rect = mpatches.Rectangle((x - width / 2, yi), width, bar_h * 0.9,
                                   facecolor=color, edgecolor='none')
        ax.add_patch(rect)

    # Outline
    outline = mpatches.Rectangle((x - width / 2, y - height / 2), width, height,
                                  facecolor='none', edgecolor=LIGHT_GRAY, lw=0.8)
    ax.add_patch(outline)


def draw_similarity_computation(ax, x, y, size=0.08):
    """Draw abstract comparison symbol."""
    # Two small bars being compared
    ax.plot([x - size, x - size * 0.3], [y + size * 0.5, y + size * 0.5],
            color=MEDIUM_GRAY, lw=2, solid_capstyle='round')
    ax.plot([x - size, x - size * 0.3], [y, y],
            color=MEDIUM_GRAY, lw=2, solid_capstyle='round')
    ax.plot([x - size, x - size * 0.3], [y - size * 0.5, y - size * 0.5],
            color=MEDIUM_GRAY, lw=2, solid_capstyle='round')

    # Comparison arrows
    ax.annotate('', xy=(x + size * 0.3, y), xytext=(x - size * 0.2, y),
                arrowprops=dict(arrowstyle='->', color=MEDIUM_GRAY, lw=1.2))


def draw_rsm(ax, S, cmap):
    """Draw similarity matrix."""
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


def main():
    fig = plt.figure(figsize=(10, 4))
    cmap = create_colormap()
    rng = np.random.default_rng(42)

    # Generate data
    n_stimuli = 5
    n_dims = 6

    # Create response patterns with structure
    responses = np.zeros((n_stimuli, n_dims))
    responses[0] = [0.9, 0.8, 0.2, 0.1, 0.15, 0.1]
    responses[1] = [0.85, 0.75, 0.25, 0.15, 0.1, 0.12]
    responses[2] = [0.1, 0.15, 0.85, 0.9, 0.2, 0.15]
    responses[3] = [0.15, 0.1, 0.8, 0.85, 0.25, 0.2]
    responses[4] = [0.4, 0.35, 0.45, 0.5, 0.8, 0.85]

    # Compute similarity
    S = responses @ responses.T
    S = S / S.max()

    shapes = ['circle', 'circle', 'square', 'square', 'triangle']
    colors = [STIM_COLORS[0], STIM_COLORS[0], STIM_COLORS[1], STIM_COLORS[1], STIM_COLORS[2]]

    # === PANEL 1: Stimuli ===
    ax1 = fig.add_axes([0.02, 0.15, 0.18, 0.7])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # Draw stimuli in a column
    for i in range(n_stimuli):
        y = 0.85 - i * 0.18
        draw_stimulus_icon(ax1, 0.5, y, 0.08, colors[i], shapes[i])

    fig.text(0.11, 0.08, 'Stimuli', fontsize=10, ha='center', color=CHARCOAL)

    # Arrow
    fig.text(0.21, 0.5, '→', fontsize=20, color=LIGHT_GRAY, ha='center', va='center')
    fig.text(0.21, 0.40, 'measure', fontsize=8, ha='center', color=MEDIUM_GRAY, style='italic')

    # === PANEL 2: Response patterns ===
    ax2 = fig.add_axes([0.26, 0.15, 0.22, 0.7])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    # Draw response patterns
    for i in range(n_stimuli):
        y = 0.85 - i * 0.18
        # Small stimulus icon
        draw_stimulus_icon(ax2, 0.12, y, 0.045, colors[i], shapes[i])
        # Arrow to pattern
        ax2.annotate('', xy=(0.32, y), xytext=(0.20, y),
                     arrowprops=dict(arrowstyle='->', color=LIGHT_GRAY, lw=1))
        # Response pattern
        draw_response_pattern(ax2, 0.55, y, responses[i], width=0.35, height=0.12)

    fig.text(0.37, 0.08, 'Response patterns', fontsize=10, ha='center', color=CHARCOAL)

    # Arrow
    fig.text(0.52, 0.5, '→', fontsize=20, color=LIGHT_GRAY, ha='center', va='center')
    fig.text(0.52, 0.40, 'compare', fontsize=8, ha='center', color=MEDIUM_GRAY, style='italic')

    # === PANEL 3: Pairwise comparison (abstract) ===
    ax3 = fig.add_axes([0.56, 0.15, 0.16, 0.7])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    # Show a few pairwise comparisons abstractly
    comparison_pairs = [(0, 1), (2, 3), (0, 2)]
    pair_y = [0.78, 0.50, 0.22]

    for (i, j), y in zip(comparison_pairs, pair_y):
        # Left stimulus
        draw_stimulus_icon(ax3, 0.15, y, 0.05, colors[i], shapes[i])
        # Right stimulus
        draw_stimulus_icon(ax3, 0.45, y, 0.05, colors[j], shapes[j])
        # Comparison line with similarity value
        sim = S[i, j]
        line_color = cmap(sim)
        ax3.plot([0.22, 0.38], [y, y], color=line_color, lw=3, solid_capstyle='round')
        # Similarity value
        ax3.text(0.70, y, f'{sim:.2f}', fontsize=8, ha='left', va='center', color=CHARCOAL)

    # Dots to indicate more
    ax3.text(0.30, 0.08, '⋮', fontsize=14, ha='center', va='center', color=LIGHT_GRAY)

    fig.text(0.64, 0.02, 'Pairwise\nsimilarity', fontsize=10, ha='center', color=CHARCOAL)

    # Arrow
    fig.text(0.74, 0.5, '→', fontsize=20, color=LIGHT_GRAY, ha='center', va='center')

    # === PANEL 4: Similarity matrix ===
    ax4 = fig.add_axes([0.78, 0.20, 0.20, 0.60])
    draw_rsm(ax4, S, cmap)

    # Add stimulus icons on margins
    margin_size = 0.035
    for i in range(n_stimuli):
        # Top margin
        x = 0.78 + (i + 0.5) * 0.20 / n_stimuli
        y = 0.82
        # Draw tiny icon
        ax_icon = fig.add_axes([x - margin_size/2, y, margin_size, margin_size])
        ax_icon.set_xlim(0, 1)
        ax_icon.set_ylim(0, 1)
        ax_icon.axis('off')
        draw_stimulus_icon(ax_icon, 0.5, 0.5, 0.4, colors[i], shapes[i])

        # Left margin
        x = 0.75
        y = 0.20 + (n_stimuli - i - 0.5) * 0.60 / n_stimuli
        ax_icon = fig.add_axes([x - margin_size/2, y - margin_size/2, margin_size, margin_size])
        ax_icon.set_xlim(0, 1)
        ax_icon.set_ylim(0, 1)
        ax_icon.axis('off')
        draw_stimulus_icon(ax_icon, 0.5, 0.5, 0.4, colors[i], shapes[i])

    fig.text(0.88, 0.08, 'Similarity matrix', fontsize=10, ha='center', color=CHARCOAL)

    # Title
    fig.text(0.5, 0.94, 'From Measurements to Representational Similarity',
             fontsize=12, ha='center', color=CHARCOAL, fontweight='medium')

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "representation_similarity_concept.svg", format='svg',
                bbox_inches='tight', pad_inches=0.05)
    fig.savefig(output_dir / "representation_similarity_concept.pdf", format='pdf',
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("Saved: representation_similarity_concept.svg/pdf")


if __name__ == "__main__":
    main()
