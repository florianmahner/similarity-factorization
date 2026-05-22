"""DNN icon - abstract deep neural network visualization."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DNN_COLOR = '#9b7eb5'


def draw_dnn_icon(ax, x, y, size, color):
    """Abstract deep neural network with multiple layers."""
    s = size * 0.5

    # Layer configuration: nodes per layer
    layers = [3, 5, 6, 5, 3]
    n_layers = len(layers)

    # Horizontal spacing between layers
    layer_xs = np.linspace(x - s * 0.7, x + s * 0.7, n_layers)

    # Store node positions
    all_positions = []

    for li, (lx, n_nodes) in enumerate(zip(layer_xs, layers)):
        layer_positions = []
        # Vertical spacing for nodes in this layer
        if n_nodes == 1:
            ys = [y]
        else:
            ys = np.linspace(y - s * 0.6, y + s * 0.6, n_nodes)

        for ny in ys:
            layer_positions.append((lx, ny))
        all_positions.append(layer_positions)

    # Draw connections (edges) first
    for li in range(n_layers - 1):
        for (x1, y1) in all_positions[li]:
            for (x2, y2) in all_positions[li + 1]:
                # Vary alpha based on position for visual interest
                alpha = 0.15 + 0.15 * np.random.random()
                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha,
                       lw=0.4, zorder=1)

    # Draw nodes
    node_sizes = [0.07, 0.055, 0.05, 0.055, 0.07]  # Larger at input/output
    for li, layer_positions in enumerate(all_positions):
        node_size = s * node_sizes[li]
        for (px, py) in layer_positions:
            circle = mpatches.Circle((px, py), node_size,
                                    facecolor=color, edgecolor='white',
                                    lw=0.5, zorder=2)
            ax.add_patch(circle)


def draw_dnn_icon_v2(ax, x, y, size, color):
    """Abstract hourglass - fewer, larger nodes."""
    s = size * 0.5

    layers = [2, 3, 2, 3, 2]
    n_layers = len(layers)
    layer_xs = np.linspace(x - s * 0.55, x + s * 0.55, n_layers)

    all_positions = []
    for li, (lx, n_nodes) in enumerate(zip(layer_xs, layers)):
        layer_positions = []
        height = s * 0.4 * (n_nodes / max(layers))
        if n_nodes == 1:
            ys = [y]
        else:
            ys = np.linspace(y - height, y + height, n_nodes)
        for ny in ys:
            layer_positions.append((lx, ny))
        all_positions.append(layer_positions)

    # Draw connections - thin lines
    for li in range(n_layers - 1):
        for (x1, y1) in all_positions[li]:
            for (x2, y2) in all_positions[li + 1]:
                ax.plot([x1, x2], [y1, y2], color=color, alpha=0.3,
                       lw=0.3, zorder=1)

    # Draw nodes - large
    for layer_positions in all_positions:
        for (px, py) in layer_positions:
            circle = mpatches.Circle((px, py), s * 0.11,
                                    facecolor=color, edgecolor='white',
                                    lw=0.6, zorder=2)
            ax.add_patch(circle)


def draw_dnn_icon_v3(ax, x, y, size, color):
    """Clean minimal - fewer nodes, cleaner connections."""
    s = size * 0.5

    layers = [2, 4, 4, 2]
    n_layers = len(layers)
    layer_xs = np.linspace(x - s * 0.55, x + s * 0.55, n_layers)

    all_positions = []
    for li, (lx, n_nodes) in enumerate(zip(layer_xs, layers)):
        layer_positions = []
        ys = np.linspace(y - s * 0.5, y + s * 0.5, n_nodes)
        for ny in ys:
            layer_positions.append((lx, ny))
        all_positions.append(layer_positions)

    # Draw connections - all same alpha for clean look
    for li in range(n_layers - 1):
        for (x1, y1) in all_positions[li]:
            for (x2, y2) in all_positions[li + 1]:
                ax.plot([x1, x2], [y1, y2], color=color, alpha=0.25,
                       lw=0.5, zorder=1)

    # Draw nodes
    for layer_positions in all_positions:
        for (px, py) in layer_positions:
            circle = mpatches.Circle((px, py), s * 0.065,
                                    facecolor=color, edgecolor='white',
                                    lw=0.5, zorder=2)
            ax.add_patch(circle)


def draw_dnn_icon_v4(ax, x, y, size, color):
    """Deep and narrow - emphasizes depth."""
    s = size * 0.5

    layers = [2, 3, 4, 4, 3, 2]
    n_layers = len(layers)
    layer_xs = np.linspace(x - s * 0.7, x + s * 0.7, n_layers)

    all_positions = []
    for li, (lx, n_nodes) in enumerate(zip(layer_xs, layers)):
        layer_positions = []
        height = s * 0.4 * (n_nodes / max(layers))
        if n_nodes == 1:
            ys = [y]
        else:
            ys = np.linspace(y - height, y + height, n_nodes)
        for ny in ys:
            layer_positions.append((lx, ny))
        all_positions.append(layer_positions)

    # Draw connections
    np.random.seed(99)
    for li in range(n_layers - 1):
        for (x1, y1) in all_positions[li]:
            for (x2, y2) in all_positions[li + 1]:
                alpha = 0.18 + 0.1 * np.random.random()
                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha,
                       lw=0.4, zorder=1)

    # Draw nodes - gradient size
    for li, layer_positions in enumerate(all_positions):
        # Smaller in middle, larger at edges
        dist_from_edge = min(li, n_layers - 1 - li)
        node_size = s * (0.055 + 0.02 * dist_from_edge / (n_layers // 2))
        for (px, py) in layer_positions:
            circle = mpatches.Circle((px, py), node_size,
                                    facecolor=color, edgecolor='white',
                                    lw=0.45, zorder=2)
            ax.add_patch(circle)


def main():
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    versions = [
        (draw_dnn_icon, "V1: Classic layers"),
        (draw_dnn_icon_v2, "V2: Hourglass"),
        (draw_dnn_icon_v3, "V3: Minimal clean"),
        (draw_dnn_icon_v4, "V4: Deep narrow"),
    ]

    for ax, (func, title) in zip(axes.flat, versions):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor('#fafafa')
        func(ax, 0.5, 0.5, size=0.85, color=DNN_COLOR)
        ax.set_title(title, fontsize=11, color='#3d3d3d', pad=8)

    plt.tight_layout()

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "dnn_icons.pdf", format='pdf', bbox_inches='tight')
    plt.close(fig)

    # Save hourglass version as standalone SVG
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.axis('off')
    draw_dnn_icon_v2(ax2, 0.5, 0.5, size=0.9, color=DNN_COLOR)
    fig2.savefig(output_dir / "dnn_icon.svg", format='svg', bbox_inches='tight',
                 pad_inches=0.05, transparent=True)
    plt.close(fig2)

    print("Saved: dnn_icons.pdf, dnn_icon.svg")


if __name__ == "__main__":
    main()
