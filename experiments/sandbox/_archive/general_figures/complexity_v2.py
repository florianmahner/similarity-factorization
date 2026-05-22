"""Complexity figure v2 - meaningful complexity gradient with proper S = WW^T."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
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


def create_rsm_colormap():
    colors = ['#ffffff', '#f0f4f8', '#d4e1ed', '#a8c5db',
              '#6a9fc0', '#3d7ea6', '#1a5a7a', '#0d3d54']
    return mcolors.LinearSegmentedColormap.from_list('elegant', colors)


def blend_color(weights):
    weights = np.array(weights) / (np.sum(weights) + 1e-10)
    return weights @ FACTOR_COLORS


def create_discrete_clusters(n=18, k=3, seed=42):
    """Discrete clusters - each item belongs to exactly one factor.

    Real-world example: Objects grouped by clear categories
    (animals vs vehicles vs furniture) with no overlap.
    """
    rng = np.random.default_rng(seed)
    W = np.zeros((n, k))
    items_per = n // k

    for j in range(k):
        start = j * items_per
        end = start + items_per if j < k - 1 else n
        for i in range(start, end):
            W[i, j] = 1.0  # Pure assignment

    # Add tiny noise to avoid perfect zeros (numerical stability)
    W = W + rng.uniform(0.001, 0.01, W.shape)
    W = W / W.sum(axis=1, keepdims=True)
    return W


def create_soft_clusters(n=18, k=3, seed=42):
    """Soft clusters - items primarily belong to one factor but have overlap.

    Real-world example: Objects that share properties across categories
    (a turtle is animate but also has shell-like properties).
    """
    rng = np.random.default_rng(seed)
    W = np.zeros((n, k))
    items_per = n // k

    for j in range(k):
        start = j * items_per
        end = start + items_per if j < k - 1 else n
        for i in range(start, end):
            W[i, j] = rng.uniform(0.6, 0.8)  # Dominant factor
            for other in range(k):
                if other != j:
                    W[i, other] = rng.uniform(0.08, 0.2)  # Some loading on others

    # Add a few bridge items that load on two factors equally
    bridge_indices = [items_per - 1, 2 * items_per - 1]
    for idx in bridge_indices:
        if idx < n:
            j1, j2 = idx // items_per, (idx // items_per + 1) % k
            W[idx, j1] = 0.45
            W[idx, j2] = 0.45
            W[idx, (j1 + 2) % k] = 0.1

    W = W / W.sum(axis=1, keepdims=True)
    return W


def create_continuum(n=18, k=3, seed=42):
    """Continuum structure - smooth gradients, no discrete clusters.

    Real-world example: Items varying smoothly along dimensions
    (size gradient from small to large, not discrete size categories).
    """
    rng = np.random.default_rng(seed)
    W = np.zeros((n, k))

    # Items are positioned along a smooth trajectory in factor space
    # Like going around the simplex
    for i in range(n):
        t = i / (n - 1)  # 0 to 1
        # Smooth transition through factors
        # Start at factor 0, move to factor 1, then to factor 2
        if t < 0.33:
            s = t / 0.33
            W[i, 0] = 1 - s * 0.7
            W[i, 1] = s * 0.7
            W[i, 2] = 0.1 + s * 0.1
        elif t < 0.67:
            s = (t - 0.33) / 0.34
            W[i, 0] = 0.3 - s * 0.2
            W[i, 1] = 0.7 - s * 0.5
            W[i, 2] = 0.2 + s * 0.5
        else:
            s = (t - 0.67) / 0.33
            W[i, 0] = 0.1 + s * 0.3
            W[i, 1] = 0.2 + s * 0.1
            W[i, 2] = 0.7 - s * 0.2

        # Add noise
        W[i] += rng.uniform(0, 0.08, k)

    W = np.clip(W, 0.01, None)
    W = W / W.sum(axis=1, keepdims=True)
    return W


def draw_w_matrix(ax, W):
    """Draw W matrix with factor colors, alpha = loading."""
    n, k = W.shape
    cell_h = 1.0 / n
    cell_w = 1.0 / k

    for i in range(n):
        for j in range(k):
            val = W[i, j]
            rect = mpatches.Rectangle(
                (j * cell_w, 1 - (i + 1) * cell_h),
                cell_w * 0.94, cell_h * 0.94,
                facecolor=FACTOR_HEX[j], alpha=val, edgecolor='none')
            ax.add_patch(rect)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_rsm(ax, S, cmap, vmin=0, vmax=1):
    """Draw RSM as vector graphic."""
    n = len(S)
    cell = 1.0 / n

    for i in range(n):
        for j in range(n):
            val = (S[i, j] - vmin) / (vmax - vmin)
            val = np.clip(val, 0, 1)
            rect = mpatches.Rectangle(
                (j * cell, 1 - (i + 1) * cell), cell, cell,
                facecolor=cmap(val), edgecolor='none')
            ax.add_patch(rect)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    n = 18
    k = 3
    cmap = create_rsm_colormap()

    # Create W matrices
    W_discrete = create_discrete_clusters(n, k, seed=42)
    W_soft = create_soft_clusters(n, k, seed=43)
    W_continuum = create_continuum(n, k, seed=44)

    # Compute RSMs: S = WW^T (this is the key - genuine computation)
    S_discrete = W_discrete @ W_discrete.T
    S_soft = W_soft @ W_soft.T
    S_continuum = W_continuum @ W_continuum.T

    # For visualization, normalize to [0, 1] range
    # (diagonal will be highest since S[i,i] = ||W[i]||^2)
    all_S = [S_discrete, S_soft, S_continuum]
    global_max = max(S.max() for S in all_S)
    global_min = min(S.min() for S in all_S)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # === Main figure ===
    fig = plt.figure(figsize=(11, 5.5))

    # Title
    fig.text(0.5, 0.94, 'Dimension Structure Complexity',
             fontsize=11, ha='center', color=CHARCOAL, fontweight='medium')

    # Complexity arrow
    ax_arrow = fig.add_axes([0.12, 0.85, 0.76, 0.06])
    ax_arrow.annotate('', xy=(0.95, 0.5), xytext=(0.05, 0.5),
                      arrowprops=dict(arrowstyle='->', color=MEDIUM_GRAY, lw=1.2))
    ax_arrow.set_xlim(0, 1)
    ax_arrow.set_ylim(0, 1)
    ax_arrow.axis('off')

    # Column setup
    col_x = [0.06, 0.37, 0.68]
    col_labels = [
        'Discrete Clusters',
        'Soft Clusters',
        'Continuum'
    ]
    col_desc = [
        '(categorical)',
        '(overlapping)',
        '(gradient)'
    ]

    # Row positions
    y_w = 0.50
    y_s = 0.08
    h_w = 0.32
    w_w = 0.18
    s_size = 0.30

    # Draw columns
    for col, (W, S, label, desc) in enumerate(zip(
        [W_discrete, W_soft, W_continuum],
        [S_discrete, S_soft, S_continuum],
        col_labels, col_desc
    )):
        x = col_x[col]

        # Column label
        fig.text(x + 0.14, 0.82, label, fontsize=9, ha='center',
                 color=CHARCOAL, fontweight='medium')
        fig.text(x + 0.14, 0.78, desc, fontsize=7, ha='center',
                 color=MEDIUM_GRAY, style='italic')

        # W matrix
        ax_w = fig.add_axes([x + 0.05, y_w, w_w, h_w])
        draw_w_matrix(ax_w, W)

        # Arrow
        fig.text(x + 0.14, y_w - 0.04, '↓', fontsize=12, ha='center', color=LIGHT_GRAY)
        fig.text(x + 0.14, y_w - 0.08, 'S = WWᵀ', fontsize=7, ha='center', color=MEDIUM_GRAY)

        # RSM
        ax_s = fig.add_axes([x + 0.02, y_s, s_size, s_size])
        draw_rsm(ax_s, S, cmap, vmin=global_min, vmax=global_max)

    # Row labels
    fig.text(0.02, y_w + h_w/2, 'W', fontsize=10, ha='center', va='center',
             color=CHARCOAL, fontweight='medium')
    fig.text(0.02, y_s + s_size/2, 'S', fontsize=10, ha='center', va='center',
             color=CHARCOAL, fontweight='medium')

    fig.savefig(output_dir / "complexity_v2.svg", format='svg',
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("Saved: complexity_v2.svg")

    # === Debug: Print statistics to verify ===
    print("\nVerification:")
    for name, W, S in [("Discrete", W_discrete, S_discrete),
                        ("Soft", W_soft, S_soft),
                        ("Continuum", W_continuum, S_continuum)]:
        print(f"\n{name}:")
        print(f"  W range: [{W.min():.3f}, {W.max():.3f}], rows sum to 1: {np.allclose(W.sum(1), 1)}")
        print(f"  S range: [{S.min():.3f}, {S.max():.3f}]")
        print(f"  S diagonal (self-similarity): [{S.diagonal().min():.3f}, {S.diagonal().max():.3f}]")
        # Check that S = WW^T
        S_check = W @ W.T
        print(f"  S == WW^T: {np.allclose(S, S_check)}")


if __name__ == "__main__":
    main()
