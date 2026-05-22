"""
Label SRF embedding dimensions by their semantic correlates.

For each of the 50 SWOW embedding dimensions, compute correlations with
all available semantic ratings to understand what each dimension encodes.

Usage:
    poetry run python sandbox/semantic_prediction/label_dimensions.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.utils import get_output_dir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns

from src.colors import ROSE, TEAL, GRAY_DARK, setup_style
from src.utils.figure_theme import despine, save_figure

PROJECT_ROOT = Path(__file__).parents[2]
EMBEDDING_DIR = PROJECT_ROOT / "outputs/experiments/word_association/generate_embedding"
NORMS_DIR = PROJECT_ROOT / "data/semantic_norms"
OUTPUT_DIR = get_output_dir()


def load_embedding(rank: int = 50) -> tuple[np.ndarray, dict[str, int]]:
    """Load SWOW embedding."""
    emb_path = EMBEDDING_DIR / str(rank)
    embeddings = np.load(emb_path / "word_embedding.npy")
    with open(emb_path / "metadata.json") as f:
        metadata = json.load(f)
    vocabulary = metadata["vocabulary"]
    word_to_idx = {w.lower(): i for i, w in enumerate(vocabulary)}
    return embeddings, word_to_idx


def load_all_norms() -> pd.DataFrame:
    """Load and merge all semantic norms into one dataframe."""
    dfs = []

    # Warriner VAD
    df = pd.read_csv(NORMS_DIR / "warriner_vad.csv")
    df = df.rename(columns={
        "Word": "word", "V.Mean.Sum": "valence_w",
        "A.Mean.Sum": "arousal_w", "D.Mean.Sum": "dominance_w"
    })
    df["word"] = df["word"].str.lower()
    dfs.append(df[["word", "valence_w", "arousal_w", "dominance_w"]])

    # Lancaster sensorimotor
    df = pd.read_csv(NORMS_DIR / "lancaster_sensorimotor.csv")
    df = df.rename(columns={"Word": "word"})
    df["word"] = df["word"].str.lower()
    cols = {
        "Auditory.mean": "auditory", "Gustatory.mean": "gustatory",
        "Haptic.mean": "haptic", "Interoceptive.mean": "interoceptive",
        "Olfactory.mean": "olfactory", "Visual.mean": "visual",
        "Foot_leg.mean": "foot_leg", "Hand_arm.mean": "hand_arm",
        "Head.mean": "head", "Mouth.mean": "mouth", "Torso.mean": "torso",
    }
    df = df.rename(columns=cols)
    dfs.append(df[["word"] + list(cols.values())])

    # Glasgow
    df = pd.read_csv(NORMS_DIR / "glasgow_norms.csv")
    df["word"] = df["word"].str.lower()
    df = df.rename(columns={
        "arousal": "arousal_g", "valence": "valence_g", "dominance": "dominance_g",
        "concreteness": "concrete_g", "imageability": "imagery",
        "familiarity": "familiar", "aoa": "aoa", "semsize": "size_g", "gender": "gender"
    })
    dfs.append(df[["word", "arousal_g", "valence_g", "dominance_g", "concrete_g",
                   "imagery", "familiar", "aoa", "size_g", "gender"]])

    # Brysbaert concreteness
    df = pd.read_excel(PROJECT_ROOT / "data/concreteness_ratings_brysbaert.xlsx")
    df = df.rename(columns={"Word": "word", "Conc.M": "concrete_b"})
    df["word"] = df["word"].str.lower()
    dfs.append(df[["word", "concrete_b"]])

    # THINGS properties
    props = pd.read_csv(PROJECT_ROOT / "data/things/things_property_ratings.csv")
    size = pd.read_csv(PROJECT_ROOT / "data/things/size_ratings.csv")
    props = props.rename(columns={
        "Word": "word", "manmade_mean": "manmade", "precious_mean": "precious",
        "lives_mean": "animacy", "heavy_mean": "heavy", "natural_mean": "natural",
        "moves_mean": "moves", "grasp_mean": "graspable", "hold_mean": "holdable",
        "be.moved_mean": "moveable", "pleasant_mean": "pleasant",
    })
    props["word"] = props["word"].str.lower()
    size = size.rename(columns={"Word": "word", "Size_mean": "size_t"})
    size["word"] = size["word"].str.lower()
    df = props.merge(size[["word", "size_t"]], on="word", how="outer")
    dfs.append(df[["word", "manmade", "precious", "animacy", "heavy", "natural",
                   "moves", "graspable", "holdable", "moveable", "pleasant", "size_t"]])

    # Merge all
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="word", how="outer")

    return merged


def compute_dimension_correlations(
    embeddings: np.ndarray,
    word_to_idx: dict[str, int],
    norms: pd.DataFrame,
) -> pd.DataFrame:
    """Compute correlation of each embedding dimension with each semantic property."""
    semantic_cols = [c for c in norms.columns if c != "word"]
    n_dims = embeddings.shape[1]

    results = []
    for sem_col in semantic_cols:
        valid = norms[["word", sem_col]].dropna()
        valid = valid[valid["word"].isin(word_to_idx)]

        if len(valid) < 100:
            continue

        word_indices = np.array([word_to_idx[w] for w in valid["word"]])
        X = embeddings[word_indices]
        y = valid[sem_col].to_numpy()

        for dim in range(n_dims):
            r, p = spearmanr(X[:, dim], y)
            results.append({
                "dimension": dim,
                "property": sem_col,
                "correlation": r,
                "pvalue": p,
                "n_words": len(y),
            })

    return pd.DataFrame(results)


def label_dimensions(corr_df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """For each dimension, find top correlated properties."""
    labels = []
    for dim in corr_df["dimension"].unique():
        dim_corrs = corr_df[corr_df["dimension"] == dim]
        # Get top positive and negative correlates
        top_pos = dim_corrs.nlargest(top_k, "correlation")
        top_neg = dim_corrs.nsmallest(top_k, "correlation")

        best = dim_corrs.loc[dim_corrs["correlation"].abs().idxmax()]

        labels.append({
            "dimension": dim,
            "best_property": best["property"],
            "best_correlation": best["correlation"],
            "top_positive": ", ".join(f"{r['property']}({r['correlation']:.2f})"
                                      for _, r in top_pos.iterrows()),
            "top_negative": ", ".join(f"{r['property']}({r['correlation']:.2f})"
                                      for _, r in top_neg.iterrows()),
        })

    return pd.DataFrame(labels).sort_values("dimension")


def plot_correlation_heatmap(corr_df: pd.DataFrame) -> plt.Figure:
    """Create heatmap of dimension × property correlations."""
    setup_style()
    plt.rcParams["figure.constrained_layout.use"] = False

    # Pivot to matrix
    matrix = corr_df.pivot(index="property", columns="dimension", values="correlation")

    # Cluster properties by similarity
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    # Cluster rows (properties)
    row_linkage = linkage(pdist(matrix.fillna(0)), method="ward")
    row_order = leaves_list(row_linkage)
    matrix = matrix.iloc[row_order]

    # Cluster columns (dimensions)
    col_linkage = linkage(pdist(matrix.fillna(0).T), method="ward")
    col_order = leaves_list(col_linkage)
    matrix = matrix.iloc[:, col_order]

    fig, ax = plt.subplots(figsize=(12, 8), layout=None)

    sns.heatmap(
        matrix, ax=ax, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
        cbar_kws={"label": "Spearman ρ", "shrink": 0.6},
        xticklabels=True, yticklabels=True,
    )

    ax.set_xlabel("SRF Dimension (clustered)")
    ax.set_ylabel("Semantic Property")
    ax.set_title("Semantic correlates of SWOW embedding dimensions", fontsize=12, pad=10)

    fig.tight_layout()
    return fig


def plot_top_dimensions(corr_df: pd.DataFrame, labels_df: pd.DataFrame) -> plt.Figure:
    """Plot dimensions with strongest semantic correlates."""
    setup_style()

    # Get dimensions with strongest correlations (by absolute value)
    strongest = labels_df.copy()
    strongest["abs_corr"] = strongest["best_correlation"].abs()
    strongest = strongest.nlargest(20, "abs_corr").sort_values("best_correlation")

    fig, ax = plt.subplots(figsize=(5, 6))

    colors = [ROSE if c < 0 else TEAL for c in strongest["best_correlation"]]
    bars = ax.barh(range(len(strongest)), strongest["best_correlation"],
                   color=colors, edgecolor="white", linewidth=0.5)

    labels = [f"Dim {int(r['dimension'])}: {r['best_property']}"
              for _, r in strongest.iterrows()]
    ax.set_yticks(range(len(strongest)))
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel("Spearman correlation (ρ)")
    ax.axvline(0, color=GRAY_DARK, linewidth=0.8)

    despine(ax)
    ax.set_title("Dimensions with strongest semantic correlates", fontsize=11, pad=10)

    plt.tight_layout()
    return fig


def plot_property_profiles(corr_df: pd.DataFrame) -> plt.Figure:
    """Show which dimensions encode key properties."""
    setup_style()

    key_props = ["valence_w", "arousal_w", "concrete_b", "animacy",
                 "size_t", "auditory", "visual", "haptic"]
    key_props = [p for p in key_props if p in corr_df["property"].values]

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    axes = axes.flatten()

    for ax, prop in zip(axes, key_props):
        prop_data = corr_df[corr_df["property"] == prop].sort_values("dimension")
        colors = [ROSE if c < 0 else TEAL for c in prop_data["correlation"]]

        ax.bar(prop_data["dimension"], prop_data["correlation"],
               color=colors, width=1, edgecolor="none")
        ax.axhline(0, color=GRAY_DARK, linewidth=0.5)
        ax.set_xlabel("Dimension", fontsize=8)
        ax.set_ylabel("ρ", fontsize=8)
        ax.set_title(prop.replace("_", " ").title(), fontsize=9)
        ax.set_ylim(-0.5, 0.5)
        despine(ax)

    for ax in axes[len(key_props):]:
        ax.set_visible(False)

    plt.suptitle("Dimension profiles for key semantic properties", fontsize=11, y=1.02)
    plt.tight_layout()
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading embedding...")
    embeddings, word_to_idx = load_embedding(rank=50)
    print(f"Embedding: {embeddings.shape[0]} words × {embeddings.shape[1]} dims")

    print("Loading semantic norms...")
    norms = load_all_norms()
    n_overlap = norms["word"].isin(word_to_idx).sum()
    print(f"Merged norms: {len(norms)} words, {n_overlap} in vocabulary")
    print(f"Properties: {[c for c in norms.columns if c != 'word']}")

    print("\nComputing dimension correlations...")
    corr_df = compute_dimension_correlations(embeddings, word_to_idx, norms)
    corr_df.to_csv(OUTPUT_DIR / "dimension_correlations.csv", index=False)

    print("\nLabeling dimensions...")
    labels_df = label_dimensions(corr_df)
    labels_df.to_csv(OUTPUT_DIR / "dimension_labels.csv", index=False)

    # Print summary
    print("\n" + "="*70)
    print("DIMENSION LABELS (by strongest correlate)")
    print("="*70)
    for _, row in labels_df.iterrows():
        sign = "+" if row["best_correlation"] > 0 else ""
        print(f"Dim {int(row['dimension']):2d}: {row['best_property']:15s} "
              f"(ρ={sign}{row['best_correlation']:.3f})")

    # Plots
    print("\nGenerating plots...")

    fig1 = plot_correlation_heatmap(corr_df)
    save_figure(fig1, OUTPUT_DIR / "dimension_heatmap.pdf")
    print("Saved: dimension_heatmap.pdf")

    fig2 = plot_top_dimensions(corr_df, labels_df)
    save_figure(fig2, OUTPUT_DIR / "top_semantic_dimensions.pdf")
    print("Saved: top_semantic_dimensions.pdf")

    fig3 = plot_property_profiles(corr_df)
    save_figure(fig3, OUTPUT_DIR / "property_profiles.pdf")
    print("Saved: property_profiles.pdf")

    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
