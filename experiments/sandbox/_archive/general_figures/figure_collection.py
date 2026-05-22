"""Complete figure collection for paper - all scrambled order."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx

OUTPUT_DIR = Path(__file__).parent / "paper_figures"

CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#c8c8c8'
MEDIUM_GRAY = '#888888'

FACTOR_COLORS_3 = np.array([
    [0.88, 0.44, 0.25],
    [0.31, 0.44, 0.75],
    [0.44, 0.69, 0.25],
])
FACTOR_HEX_3 = ['#E07040', '#5070C0', '#70B040']
FACTOR_HEX_4 = ['#E07040', '#5070C0', '#70B040', '#9b7eb5']


def create_colormap():
    colors = ['#ffffff', '#f0f4f8', '#d4e1ed', '#a8c5db',
              '#6a9fc0', '#3d7ea6', '#1a5a7a', '#0d3d54']
    return mcolors.LinearSegmentedColormap.from_list('elegant', colors)


def create_data_3factors(seed=42):
    """Create W with 3 factors, normalized S with diag=1."""
    rng = np.random.default_rng(seed)
    W_list = []
    # F1 dominant (animate): bird, cat, dog, fish
    for _ in range(4):
        W_list.append([rng.uniform(0.82, 0.92), rng.uniform(0.03, 0.09), rng.uniform(0.03, 0.09)])
    # F2 dominant (round): ball, coin, globe, wheel
    for _ in range(4):
        W_list.append([rng.uniform(0.03, 0.09), rng.uniform(0.82, 0.92), rng.uniform(0.03, 0.09)])
    # F3 dominant (natural): bark, fern, leaf, moss
    for _ in range(4):
        W_list.append([rng.uniform(0.03, 0.09), rng.uniform(0.03, 0.09), rng.uniform(0.82, 0.92)])
    # Overlap items
    W_list.append([0.46, 0.46, 0.08])  # F1+F2: snail (animate + round shell)
    W_list.append([0.47, 0.07, 0.46])  # F1+F3: frog (animate + natural)
    W_list.append([0.07, 0.47, 0.46])  # F2+F3: berry (round + natural)

    W = np.array(W_list)
    W = W / W.sum(axis=1, keepdims=True)
    S = W @ W.T
    d = np.sqrt(np.diag(S))
    S = S / np.outer(d, d)

    # Labels matching factor loadings - when sorted alphabetically they align with colors
    # Original order: 0-3 animate, 4-7 round, 8-11 natural, 12 anim+round, 13 anim+nat, 14 round+nat
    labels = ['bat', 'bird', 'fish', 'lion',     # F1: animate (orange)
              'ball', 'coin', 'globe', 'hoop',   # F2: round (blue)
              'clay', 'fern', 'leaf', 'moss',    # F3: natural (green)
              'clam', 'owl', 'berry']            # overlaps: anim+round, anim+nat, round+nat

    return W, S, labels


def create_data_4factors(seed=42, n=16):
    """Create W with 4 factors, sparse profiles."""
    rng = np.random.default_rng(seed)
    W_list = []
    for _ in range(n):
        row = rng.dirichlet(np.ones(4) * 0.3)
        row[row < 0.1] = 0
        if row.sum() == 0:
            row[rng.integers(4)] = 1.0
        row = row / row.sum()
        W_list.append(row)
    W = np.array(W_list)
    S = W @ W.T
    d = np.sqrt(np.diag(S))
    S = S / np.outer(d, d)
    return W, S


def blend_color(weights):
    weights = np.array(weights)
    weights = weights / weights.sum()
    return weights @ FACTOR_COLORS_3


def draw_rsm_with_bars(fig, S, W, cmap):
    """Draw RSM with membership color bars on top and left."""
    n = len(S)
    cell = 1.0 / n
    vmin, vmax = S.min(), S.max()

    rsm_left = 0.18
    rsm_bottom = 0.10
    rsm_size = 0.65
    bar_height = 0.035
    bar_width = 0.035

    # Main RSM
    ax_rsm = fig.add_axes([rsm_left, rsm_bottom, rsm_size, rsm_size])
    for i in range(n):
        for j in range(n):
            val = (S[i, j] - vmin) / (vmax - vmin)
            rect = mpatches.Rectangle(
                (j * cell, 1 - (i + 1) * cell), cell, cell,
                facecolor=cmap(val), edgecolor='none')
            ax_rsm.add_patch(rect)
    ax_rsm.set_xlim(0, 1)
    ax_rsm.set_ylim(0, 1)
    ax_rsm.set_xticks([])
    ax_rsm.set_yticks([])
    for spine in ax_rsm.spines.values():
        spine.set_visible(False)

    # Top membership bar
    ax_top = fig.add_axes([rsm_left, rsm_bottom + rsm_size, rsm_size, bar_height])
    for i in range(n):
        for k in range(3):
            if W[i, k] > 0.05:
                rect = mpatches.Rectangle(
                    (i * cell, 0), cell, 1,
                    facecolor=FACTOR_HEX_3[k], alpha=W[i, k], edgecolor='none')
                ax_top.add_patch(rect)
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)
    ax_top.axis('off')

    # Left membership bar
    ax_left = fig.add_axes([rsm_left - bar_width - 0.01, rsm_bottom, bar_width, rsm_size])
    for i in range(n):
        for k in range(3):
            if W[i, k] > 0.05:
                rect = mpatches.Rectangle(
                    (0, 1 - (i + 1) * cell), 1, cell,
                    facecolor=FACTOR_HEX_3[k], alpha=W[i, k], edgecolor='none')
                ax_left.add_patch(rect)
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)
    ax_left.axis('off')

    return ax_rsm


def draw_rsm_with_missing(ax, S, cmap, missing_mask, holdout_mask):
    """Draw RSM with missing (X) and held-out (red border) entries."""
    n = len(S)
    cell = 1.0 / n
    vmin, vmax = S[~missing_mask].min(), S[~missing_mask].max()

    MISSING_COLOR = '#f5f5f5'
    HOLDOUT_RED = '#cc4444'

    for i in range(n):
        for j in range(n):
            x = j * cell
            y = 1 - (i + 1) * cell
            if missing_mask[i, j]:
                rect = mpatches.Rectangle((x, y), cell, cell,
                    facecolor=MISSING_COLOR, edgecolor='none')
                ax.add_patch(rect)
                ax.plot([x + cell*0.2, x + cell*0.8], [y + cell*0.2, y + cell*0.8],
                        color=LIGHT_GRAY, lw=0.8, zorder=2)
                ax.plot([x + cell*0.2, x + cell*0.8], [y + cell*0.8, y + cell*0.2],
                        color=LIGHT_GRAY, lw=0.8, zorder=2)
            else:
                val = (S[i, j] - vmin) / (vmax - vmin)
                rect = mpatches.Rectangle((x, y), cell, cell,
                    facecolor=cmap(val), edgecolor='none')
                ax.add_patch(rect)
                if holdout_mask[i, j]:
                    border = mpatches.Rectangle((x, y), cell, cell,
                        facecolor='none', edgecolor=HOLDOUT_RED, lw=1.5, zorder=2)
                    ax.add_patch(border)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_w_vertical(ax, W, factor_colors, factor_names=None):
    """Draw W matrix with vertical bars per factor."""
    n, k = W.shape
    margin_left = 0.05
    margin_right = 0.05
    col_gap = 0.08
    total_w = 1 - margin_left - margin_right - (k - 1) * col_gap
    col_w = total_w / k
    cell_h = 0.85 / n

    for i in range(n):
        y = 0.88 - (i + 1) * cell_h
        for j in range(k):
            x = margin_left + j * (col_w + col_gap)
            if W[i, j] > 0.02:
                width = W[i, j] * col_w * 0.9
                rect = mpatches.Rectangle((x, y + cell_h*0.1), width, cell_h * 0.8,
                    facecolor=factor_colors[j], edgecolor='none', alpha=0.9)
                ax.add_patch(rect)

    if factor_names is None:
        factor_names = [f'F{j+1}' for j in range(k)]
    for j in range(k):
        x = margin_left + j * (col_w + col_gap) + col_w * 0.4
        ax.text(x, 0.94, factor_names[j], fontsize=9, ha='center', va='center',
               color=factor_colors[j], fontweight='medium')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_w_with_labels(ax, W, factor_colors, labels=None, factor_names=None):
    """Draw W matrix with nodes on left, factors as columns - overview_v5 style."""
    n, k = W.shape

    if labels is None:
        labels = [f'item {i+1}' for i in range(n)]
    if factor_names is None:
        factor_names = [f'F{j+1}' for j in range(k)]

    cell_h = 0.9 / n
    cell_w = 0.20

    # W matrix cells
    for i in range(n):
        y = 0.95 - (i + 1) * cell_h
        for j in range(k):
            val = W[i, j]
            rect = mpatches.Rectangle(
                (0.40 + j * cell_w, y), cell_w * 0.9, cell_h * 0.9,
                facecolor=factor_colors[j], alpha=val, edgecolor='none')
            ax.add_patch(rect)

        # Node circle on left
        if k == 3:
            color = blend_color(W[i])
        else:
            color = factor_colors[np.argmax(W[i])]
        circle = mpatches.Circle((0.18, y + cell_h * 0.45), 0.035,
                                  facecolor=color, edgecolor='white', lw=0.6)
        ax.add_patch(circle)

        # Connection line
        ax.plot([0.22, 0.38], [y + cell_h * 0.45, y + cell_h * 0.45],
                color=LIGHT_GRAY, lw=0.4, alpha=0.5)

    # Labels
    ax.text(0.18, 0.02, 'nodes', fontsize=7, ha='center', color=MEDIUM_GRAY)
    for j in range(k):
        ax.text(0.40 + (j + 0.5) * cell_w, 0.02, factor_names[j],
                fontsize=7, ha='center', color=factor_colors[j], fontweight='medium')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_graph(ax, W, S):
    """Draw force-directed graph with nodes colored by membership."""
    n = len(W)
    rng = np.random.default_rng(42)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    threshold = 0.12
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > threshold:
                G.add_edge(i, j, weight=S[i, j])

    factor_centers = {0: np.array([-0.5, -0.3]), 1: np.array([0.5, -0.3]), 2: np.array([0.0, 0.5])}
    pos = {}
    for i in range(n):
        center = sum(W[i, k] * factor_centers[k] for k in range(3))
        pos[i] = center + rng.uniform(-0.1, 0.1, 2)
    pos = nx.spring_layout(G, pos=pos, k=0.5, iterations=60, seed=42)

    for u, v, data in G.edges(data=True):
        weight = data['weight']
        strength = (weight - threshold) / (1 - threshold)
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=LIGHT_GRAY, lw=0.4 + 0.6 * strength,
                alpha=0.15 + 0.5 * strength, zorder=1, clip_on=False)

    for i in range(n):
        x, y = pos[i]
        color = blend_color(W[i])
        circle = mpatches.Circle((x, y), 0.08, facecolor=color,
                                  edgecolor='white', lw=1, zorder=3, clip_on=False)
        ax.add_patch(circle)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.8, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_arc(ax, W, S, positions):
    """Draw arc diagram with nodes at given positions."""
    n = len(W)
    threshold = 0.15

    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > threshold:
                x1, x2 = positions[i], positions[j]
                if x1 > x2:
                    x1, x2 = x2, x1
                strength = (S[i, j] - threshold) / (1 - threshold)
                alpha = 0.15 + 0.6 * strength
                lw = 0.3 + 0.8 * strength

                theta = np.linspace(0, np.pi, 50)
                center = (x1 + x2) / 2
                width = abs(x2 - x1)
                height = width * 0.85
                arc_x = center + (width / 2) * np.cos(theta)
                arc_y = height * np.sin(theta) / 2
                ax.plot(arc_x, arc_y, color=LIGHT_GRAY, linewidth=lw, alpha=alpha,
                       zorder=1, clip_on=False)

    node_radius = 0.022
    for node in range(n):
        x = positions[node]
        color = blend_color(W[node])
        circle = mpatches.Circle((x, 0), node_radius, facecolor=color,
                                  edgecolor='white', linewidth=0.8, zorder=3, clip_on=False)
        ax.add_patch(circle)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    cmap = create_colormap()

    # Create data with labels
    W3, S3, labels3 = create_data_3factors()
    W4, S4 = create_data_4factors(n=16)
    n3, n4 = len(W3), len(W4)

    # Alphabetical order (simulates arbitrary/unsorted real data)
    alpha_order = np.argsort(labels3)
    labels3_alpha = [labels3[i] for i in alpha_order]
    W3_alpha = W3[alpha_order]
    S3_alpha = S3[np.ix_(alpha_order, alpha_order)]

    # Scrambled for 4-factor version
    rng = np.random.default_rng(123)
    scramble4 = rng.permutation(n4)
    W4_scr = W4[scramble4]
    S4_scr = S4[np.ix_(scramble4, scramble4)]

    # Positions for arc (alphabetical order)
    positions3 = {i: idx / (n3 - 1) for idx, i in enumerate(alpha_order)}

    # 1. RSM input (alphabetical order - no visible block structure)
    fig = plt.figure(figsize=(3.4, 3.4))
    draw_rsm_with_bars(fig, S3_alpha, W3_alpha, cmap)
    fig.savefig(OUTPUT_DIR / "rsm_input.svg", format='svg', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print("Saved: rsm_input.svg")

    # 2. RSM with missing and held-out (input - incomplete data)
    rng_mask = np.random.default_rng(42)
    missing_mask = np.zeros((n3, n3), dtype=bool)
    holdout_mask = np.zeros((n3, n3), dtype=bool)
    for _ in range(12):
        i, j = rng_mask.integers(0, n3, 2)
        if i != j:
            missing_mask[i, j] = missing_mask[j, i] = True
    for _ in range(12):
        i, j = rng_mask.integers(0, n3, 2)
        if i != j and not missing_mask[i, j]:
            holdout_mask[i, j] = holdout_mask[j, i] = True

    # Apply alpha ordering to masks
    missing_alpha = missing_mask[np.ix_(alpha_order, alpha_order)]
    holdout_alpha = holdout_mask[np.ix_(alpha_order, alpha_order)]

    fig, ax = plt.subplots(figsize=(3, 3))
    draw_rsm_with_missing(ax, S3_alpha, cmap, missing_alpha, holdout_alpha)
    fig.savefig(OUTPUT_DIR / "rsm_missing_holdout.svg", format='svg', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print("Saved: rsm_missing_holdout.svg")

    # 3. RSM predicted (reconstruction - noise on missing and held-out entries)
    rng_noise = np.random.default_rng(99)
    S3_pred = S3_alpha.copy()
    # Add noise to missing and held-out entries (imputed values differ from true)
    noise = rng_noise.normal(0, 0.15, S3_alpha.shape)
    imputed_mask = missing_alpha | holdout_alpha
    S3_pred[imputed_mask] += noise[imputed_mask]
    S3_pred = np.clip(S3_pred, 0, 1)
    S3_pred = (S3_pred + S3_pred.T) / 2  # Keep symmetric
    np.fill_diagonal(S3_pred, 1.0)

    fig, ax = plt.subplots(figsize=(3, 3))
    n = len(S3_pred)
    cell = 1.0 / n
    vmin, vmax = S3_pred.min(), S3_pred.max()
    for i in range(n):
        for j in range(n):
            val = (S3_pred[i, j] - vmin) / (vmax - vmin)
            rect = mpatches.Rectangle(
                (j * cell, 1 - (i + 1) * cell), cell, cell,
                facecolor=cmap(val), edgecolor='none')
            ax.add_patch(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig(OUTPUT_DIR / "rsm_predicted.svg", format='svg', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print("Saved: rsm_predicted.svg")

    # 4. W embedding (3 factors, alphabetical order) - overview_v5 style
    fig, ax = plt.subplots(figsize=(2.5, 4.0))
    draw_w_with_labels(ax, W3_alpha, FACTOR_HEX_3)
    fig.savefig(OUTPUT_DIR / "w_embedding.svg", format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("Saved: w_embedding.svg")

    # 5. W vertical wide (4 factors, scrambled)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    draw_w_vertical(ax, W4_scr, FACTOR_HEX_4)
    fig.savefig(OUTPUT_DIR / "w_vertical_wide_scrambled.svg", format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("Saved: w_vertical_wide_scrambled.svg")

    # 6. Graph
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    draw_graph(ax, W3, S3)
    fig.savefig(OUTPUT_DIR / "graph.svg", format='svg', bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.close(fig)
    print("Saved: graph.svg")

    # 7. Arc diagram (alphabetical order)
    positions_alpha = {node: idx / (n3 - 1) for idx, node in enumerate(alpha_order)}
    fig, ax = plt.subplots(figsize=(5.5, 2.0))
    draw_arc(ax, W3, S3, positions_alpha)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 0.58)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig(OUTPUT_DIR / "arc.svg", format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("Saved: arc.svg")

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
