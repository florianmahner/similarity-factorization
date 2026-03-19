"""SRF on Samuel mathematical concepts with coherence-based rank selection."""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from wordcloud import WordCloud

from pysrf import SRF
from src.tools.coherence import impute_similarity_matrix
from sandbox.coherence.projected_iproj.run import (
    compute_projected_coherence,
    estimate_rank_from_activations,
    plot_iproj_curves,
    plot_activation_summary,
)
from src.utils import get_output_dir
from src.colors import CYCLE
from src.utils.figure_theme import create_figure, despine, save_figure

OUTPUT_DIR = get_output_dir()
DATA_PATH = Path("data/samuel/similarityMatrix.csv")


def load_and_preprocess() -> tuple[np.ndarray, list[str]]:
    """Load, symmetrize, and set diagonal."""
    df = pd.read_csv(DATA_PATH, index_col=0)
    labels = df.index.tolist()
    s = df.values.astype(float)
    n = s.shape[0]

    st = s.T
    has_s, has_st = ~np.isnan(s), ~np.isnan(st)
    sym = np.full((n, n), np.nan)
    sym[has_s & has_st] = (s[has_s & has_st] + st[has_s & has_st]) / 2
    sym[has_s & ~has_st] = s[has_s & ~has_st]
    sym[~has_s & has_st] = st[~has_s & has_st]

    max_val = np.nanmax(sym)
    np.fill_diagonal(sym, max_val)

    n_obs = np.sum(~np.isnan(sym))
    print(f"Loaded {n} concepts, {n_obs} observed ({100*n_obs/(n*n):.1f}%), diagonal={max_val:.2f}")
    return sym, labels


def plot_wordclouds(embedding: np.ndarray, labels: list[str]) -> plt.Figure:
    """One wordcloud per dimension, arranged in a grid."""
    k = embedding.shape[1]
    n_cols = min(k, 4)
    n_rows = (k + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
    if k == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    colors_per_dim = ["Blues", "Oranges", "Greens", "Purples", "Reds",
                      "YlGn", "BuPu", "OrRd", "GnBu", "YlOrRd"]

    for d in range(k):
        row, col = d // n_cols, d % n_cols
        ax = axes[row, col]

        weights = embedding[:, d]
        top_idx = np.argsort(weights)[::-1][:40]
        freq = {labels[i]: float(weights[i]) for i in top_idx if weights[i] > 1e-6}

        if freq:
            cmap = colors_per_dim[d % len(colors_per_dim)]
            wc = WordCloud(
                width=600, height=400, background_color="white",
                max_words=40, colormap=cmap, prefer_horizontal=0.7,
                relative_scaling=0.5, min_font_size=8,
            )
            wc.generate_from_frequencies(freq)
            ax.imshow(wc, interpolation="bilinear")

        norm = np.linalg.norm(weights)
        ax.set_title(f"Dimension {d+1}  (norm = {norm:.1f})", fontsize=12, fontweight="bold")
        ax.axis("off")

    for d in range(k, n_rows * n_cols):
        row, col = d // n_cols, d % n_cols
        axes[row, col].axis("off")

    plt.suptitle("SRF Dimensions — Samuel Mathematical Concepts",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_top_words(embedding: np.ndarray, labels: list[str], top_k: int = 15) -> plt.Figure:
    """Horizontal bar chart of top words per dimension."""
    k = embedding.shape[1]
    n_cols = min(k, 4)
    n_rows = (k + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 0.4 * top_k * n_rows))
    if k == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    dim_colors = [CYCLE[i % len(CYCLE)] for i in range(k)]

    for d in range(k):
        row, col = d // n_cols, d % n_cols
        ax = axes[row, col]

        weights = embedding[:, d]
        top_idx = np.argsort(weights)[::-1][:top_k]
        top_labs = [labels[i] for i in top_idx]
        top_w = weights[top_idx]

        ax.barh(range(top_k), top_w[::-1], color=dim_colors[d], alpha=0.85)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_labs[::-1], fontsize=10)
        ax.set_xlabel("Weight")
        ax.set_title(f"Dimension {d+1}", fontsize=11, fontweight="bold")
        despine(ax)

    for d in range(k, n_rows * n_cols):
        row, col = d // n_cols, d % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    return fig


def plot_predicted_matrix(embedding: np.ndarray, labels: list[str]) -> plt.Figure:
    """Plot reconstructed similarity matrix."""
    recon = embedding @ embedding.T
    n = len(labels)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(recon, cmap="magma", aspect="auto", vmin=0)
    plt.colorbar(im, ax=ax, label="Predicted similarity", shrink=0.8)

    step = max(1, n // 20)
    tick_pos = list(range(0, n, step))
    tick_labs = [labels[i] for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labs, rotation=90, fontsize=7)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labs, fontsize=7)

    ax.set_title("Predicted Similarity Matrix (WW$^T$)", fontsize=13, fontweight="bold")
    return fig


def main():
    print("=" * 50)
    print("Samuel Mathematical Concepts — SRF")
    print("=" * 50)

    # Load
    sym, labels = load_and_preprocess()

    # Step 1: Impute for coherence analysis
    print("\n--- Imputing for coherence analysis ---")
    imputation_rank = 20
    s_imp = impute_similarity_matrix(sym, rank=imputation_rank, max_outer=50, tol=1e-4)
    print(f"Imputed with rank={imputation_rank}")

    # Step 2: Projected coherence on imputed matrix
    print("\n--- Projected Coherence ---")
    k_list = list(range(1, 31))
    p_list = np.linspace(0.05, 0.95, 25)

    result = compute_projected_coherence(
        s_imp,
        k_list,
        p_list,
        b=50,
        random_state=42,
        use_baseline_correction=True,
        clip_baseline=False,
        n_jobs=-1,
    )

    # Estimate rank
    estimated_rank = estimate_rank_from_activations(result, p_threshold=0.5)
    print(f"Estimated rank (p_threshold=0.5): {estimated_rank}")

    print("\nActivation points:")
    for i, k in enumerate(result["k_list"][:30]):
        p_act = result["activation_p"][i]
        status = f"p={p_act:.3f}" if np.isfinite(p_act) else "never"
        print(f"  k={k:2d}: {status}")

    # Plot coherence curves
    plot_iproj_curves(result, OUTPUT_DIR / "iproj_curves.pdf", show_ci=True)
    print("Saved iproj_curves.pdf")

    plot_activation_summary(result, OUTPUT_DIR / "activation_summary.pdf")
    print("Saved activation_summary.pdf")

    # Use estimated rank (fallback to 10 if 0)
    best_rank = estimated_rank if estimated_rank > 0 else 10
    print(f"\n--- Fitting SRF (rank={best_rank}) ---")
    model = SRF(rank=best_rank, missing_values=np.nan, rho=0.2, verbose=1)
    embedding = model.fit_transform(sym)

    recon = embedding @ embedding.T
    mask = ~np.isnan(sym)
    mse = np.mean((sym[mask] - recon[mask])**2)
    ss_res = np.sum((sym[mask] - recon[mask])**2)
    ss_tot = np.sum((sym[mask] - np.mean(sym[mask]))**2)
    r2 = 1 - ss_res / ss_tot
    print(f"MSE={mse:.4f}, R²={r2:.4f}")

    # Plots
    print("\n--- Generating plots ---")
    fig = plot_wordclouds(embedding, labels)
    fig.savefig(OUTPUT_DIR / "wordclouds.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved wordclouds.pdf")

    fig = plot_top_words(embedding, labels)
    fig.savefig(OUTPUT_DIR / "top_words.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved top_words.pdf")

    fig = plot_predicted_matrix(embedding, labels)
    fig.savefig(OUTPUT_DIR / "predicted_similarity.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved predicted_similarity.pdf")

    # Save CSVs
    recon_df = pd.DataFrame(recon, index=labels, columns=labels)
    recon_df.to_csv(OUTPUT_DIR / "predicted_similarity.csv")

    emb_df = pd.DataFrame(embedding, index=labels, columns=[f"dim_{d+1}" for d in range(best_rank)])
    emb_df.index.name = "concept"
    emb_df.to_csv(OUTPUT_DIR / "embedding.csv")

    np.save(OUTPUT_DIR / "embedding.npy", embedding)

    print(f"\nDone! All results in:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
