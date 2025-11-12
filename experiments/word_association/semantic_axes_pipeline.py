import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import argparse

from analysis.words.validation import load_behavioral_ratings


def load_glove_for_words(glove_path: Path, words: list[str]) -> dict[str, np.ndarray]:
    """Load GloVe vectors for specific words."""
    word_set = set(w.lower() for w in words)
    vectors = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if parts[0].lower() in word_set:
                try:
                    vectors[parts[0].lower()] = np.array(parts[1:], dtype=np.float32)
                except (ValueError, IndexError):
                    continue
            if len(vectors) == len(word_set):
                break
    return vectors


def project_glove_words(
    words: list[str],
    glove_path: Path,
    transform_path: Path,
    metadata_path: Path,
    axes: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Project arbitrary words via GloVe transformation."""
    with open(metadata_path) as f:
        meta = json.load(f)

    transform = np.load(transform_path)
    glove_mean = np.array(meta["preprocessing"]["glove_mean"])
    glove_std = np.array(meta["preprocessing"]["glove_std"])
    srf_mean = np.array(meta["preprocessing"]["srf_mean"])

    glove_vecs = load_glove_for_words(glove_path, words)

    results = []
    for word, vec in glove_vecs.items():
        glove_z = (vec - glove_mean) / glove_std
        srf_pred = glove_z @ transform + srf_mean
        srf_pred = np.maximum(srf_pred, 0)
        srf_norm = srf_pred / np.linalg.norm(srf_pred)

        proj = {name: float(srf_norm @ axis) for name, axis in axes.items()}
        results.append({"word": word, **proj})

    return pd.DataFrame(results)


def load_embeddings(
    embedding_dir: Path,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Load SRF embeddings and vocabulary."""
    w = np.load(embedding_dir / "srf_factors.npy")
    vocabulary = (embedding_dir / "vocabulary.txt").read_text().strip().split("\n")
    word_to_idx = {word.lower(): idx for idx, word in enumerate(vocabulary)}
    return w, vocabulary, word_to_idx


def load_semantic_axes(axes_file: Path) -> dict:
    """Load semantic axes from JSON file."""
    with open(axes_file, "r") as f:
        return json.load(f)


def compute_axis_vectors(
    axes_config: dict, w: np.ndarray, word_to_idx: dict
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Compute semantic axis vectors from pole words."""
    axes, metadata = {}, {}

    for axis_name, config in axes_config.items():
        pos_idx = [
            word_to_idx[word.lower()]
            for word in config["positive"]
            if word.lower() in word_to_idx
        ]
        neg_idx = [
            word_to_idx[word.lower()]
            for word in config["negative"]
            if word.lower() in word_to_idx
        ]

        if len(pos_idx) < 2 or len(neg_idx) < 2:
            continue

        pos_vecs, neg_vecs = w[pos_idx], w[neg_idx]
        axis = np.median(pos_vecs, axis=0) - np.median(neg_vecs, axis=0)
        axis /= np.linalg.norm(axis)

        # Ensure correct orientation
        pos_proj = pos_vecs @ axis
        neg_proj = neg_vecs @ axis
        if np.median(pos_proj) <= np.median(neg_proj):
            axis = -axis

        axes[axis_name] = axis
        metadata[axis_name] = {
            "n_positive": len(pos_idx),
            "n_negative": len(neg_idx),
            "positive_words": [
                w for w in config["positive"] if w.lower() in word_to_idx
            ][:10],
            "negative_words": [
                w for w in config["negative"] if w.lower() in word_to_idx
            ][:10],
        }

    return axes, metadata


def project_vocabulary(
    w: np.ndarray, vocabulary: list[str], axes: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Project all vocabulary words onto semantic axes."""
    w_normalized = w / np.linalg.norm(w, axis=1, keepdims=True)
    axis_matrix = np.array(list(axes.values())).T
    projections = w_normalized @ axis_matrix

    return pd.DataFrame(
        {
            "word": [w.lower() for w in vocabulary],
            **{name: projections[:, i] for i, name in enumerate(axes.keys())},
        }
    )


def compute_topk_precision(
    projections_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    axis_name: str,
    rating_col: str,
    k: int = 50,
) -> dict:
    """Compute precision of top-k predictions - only from words with ratings."""
    # First, filter to only words that have ratings
    valid_ratings = ratings_df[["word", rating_col]].dropna()
    rated_projections = projections_df[
        projections_df["word"].isin(valid_ratings["word"])
    ]

    if len(rated_projections) < k * 2:
        return {
            "n_rated_words": len(rated_projections),
            "top_k_mean_projection": np.nan,
            "bottom_k_mean_projection": np.nan,
            "top_k_mean_rating": np.nan,
            "bottom_k_mean_rating": np.nan,
            "rating_separation": np.nan,
        }

    # Get top and bottom k from ONLY rated words
    top_k = rated_projections.nlargest(k, axis_name)
    bottom_k = rated_projections.nsmallest(k, axis_name)

    # Merge with ratings
    top_k_with_ratings = top_k.merge(valid_ratings, on="word")
    bottom_k_with_ratings = bottom_k.merge(valid_ratings, on="word")

    return {
        "n_rated_words": len(rated_projections),
        "top_k_mean_projection": top_k[axis_name].mean(),
        "bottom_k_mean_projection": bottom_k[axis_name].mean(),
        "top_k_mean_rating": top_k_with_ratings[rating_col].mean(),
        "bottom_k_mean_rating": bottom_k_with_ratings[rating_col].mean(),
        "rating_separation": top_k_with_ratings[rating_col].mean()
        - bottom_k_with_ratings[rating_col].mean(),
    }


def validate_pole_words(
    axes_config: dict,
    ratings_df: pd.DataFrame,
    axis_rating_pairs: dict[str, str],
) -> list[dict]:
    """Validate that pole words used to define axes have appropriate ratings."""
    pole_results = []

    for axis_name, rating_col in axis_rating_pairs.items():
        if axis_name not in axes_config or rating_col not in ratings_df.columns:
            continue

        config = axes_config[axis_name]
        valid_ratings = ratings_df[["word", rating_col]].dropna()

        # Check positive pole words
        pos_words_lower = [w.lower() for w in config["positive"]]
        pos_ratings = valid_ratings[valid_ratings["word"].isin(pos_words_lower)]

        # Check negative pole words
        neg_words_lower = [w.lower() for w in config["negative"]]
        neg_ratings = valid_ratings[valid_ratings["word"].isin(neg_words_lower)]

        pole_results.append(
            {
                "axis": axis_name,
                "rating": rating_col,
                "n_pos_pole": len(config["positive"]),
                "n_pos_rated": len(pos_ratings),
                "pos_mean_rating": (
                    pos_ratings[rating_col].mean() if len(pos_ratings) > 0 else np.nan
                ),
                "n_neg_pole": len(config["negative"]),
                "n_neg_rated": len(neg_ratings),
                "neg_mean_rating": (
                    neg_ratings[rating_col].mean() if len(neg_ratings) > 0 else np.nan
                ),
                "pole_rating_separation": (
                    (pos_ratings[rating_col].mean() - neg_ratings[rating_col].mean())
                    if len(pos_ratings) > 0 and len(neg_ratings) > 0
                    else np.nan
                ),
            }
        )

    return pole_results


def validate_axes(
    projections_df: pd.DataFrame,
    data_dir: Path,
    axes_config: dict,
    axis_rating_pairs: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate axes using correlation, top-k precision, and pole words."""
    ratings_df = load_behavioral_ratings(data_dir)
    merged = projections_df.merge(ratings_df, on="word", how="left")

    validation_results = []
    for axis_col, rating_col in axis_rating_pairs.items():
        if (
            axis_col not in projections_df.columns
            or rating_col not in ratings_df.columns
        ):
            continue

        # Correlation validation
        valid_data = merged[[axis_col, rating_col]].dropna()
        if len(valid_data) >= 10:
            corr, pval = spearmanr(valid_data[axis_col], valid_data[rating_col])
        else:
            corr, pval = np.nan, np.nan

        # Top-k precision validation
        topk_metrics = compute_topk_precision(
            projections_df, ratings_df, axis_col, rating_col, k=50
        )

        validation_results.append(
            {
                "axis": axis_col,
                "rating": rating_col,
                "n_words": len(valid_data),
                "correlation": corr,
                "pvalue": pval,
                **topk_metrics,
            }
        )

    # Pole words validation
    pole_results = validate_pole_words(axes_config, ratings_df, axis_rating_pairs)

    return merged, pd.DataFrame(validation_results), pd.DataFrame(pole_results)


def main(
    embedding_dir: Path,
    axes_file: Path,
    data_dir: Path,
    output_dir: Path,
    glove_path: Path | None = None,
    glove_words: list[str] | None = None,
) -> None:
    """Run complete semantic axes pipeline."""

    # Load embeddings
    w, vocabulary, word_to_idx = load_embeddings(embedding_dir)

    # Load and compute axes
    axes_config = load_semantic_axes(axes_file)
    axes, axes_metadata = compute_axis_vectors(axes_config, w, word_to_idx)

    if not axes:
        raise ValueError("No valid axes created")

    # Project vocabulary onto axes
    projections_df = project_vocabulary(w, vocabulary, axes)

    # Optionally project GloVe words
    if glove_path and glove_words:
        transform_path = embedding_dir / "glove_to_srf_transform.npy"
        metadata_path = embedding_dir / "transform_metadata.json"
        if transform_path.exists() and metadata_path.exists():
            glove_proj = project_glove_words(
                glove_words, glove_path, transform_path, metadata_path, axes
            )
            projections_df = pd.concat([projections_df, glove_proj], ignore_index=True)

    # Define axis-rating pairs for validation
    axis_rating_pairs = {
        "concreteness": "concreteness_rating",
        "size": "size_rating",
        "valence": "valence_rating",
        "heaviness": "heaviness_rating",
        "animacy": "animacy_rating",
    }
    # Filter to available axes
    axis_rating_pairs = {k: v for k, v in axis_rating_pairs.items() if k in axes}

    # Validate axes
    merged_df, validation_summary, pole_summary = validate_axes(
        projections_df, data_dir, axes_config, axis_rating_pairs
    )

    # Add top-k indicators
    for axis_name in axes.keys():
        top_50 = merged_df.nlargest(50, axis_name)["word"].values
        bottom_50 = merged_df.nsmallest(50, axis_name)["word"].values
        merged_df[f"is_top_{axis_name}"] = merged_df["word"].isin(top_50)
        merged_df[f"is_bottom_{axis_name}"] = merged_df["word"].isin(bottom_50)

    # Save outputs to final folder
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    merged_df.to_csv(final_dir / "semantic_validation.csv", index=False)
    validation_summary.to_csv(final_dir / "validation_summary.csv", index=False)
    pole_summary.to_csv(final_dir / "pole_validation.csv", index=False)

    # Save axes
    np.save(final_dir / "semantic_axes.npy", np.array(list(axes.values())))
    (final_dir / "axis_names.txt").write_text("\n".join(axes.keys()))

    # Print summaries
    print("\n=== CORRELATION & TOP-K VALIDATION ===")
    for _, row in validation_summary.iterrows():
        print(
            f"{row['axis']:15s} | r={row['correlation']:.3f} (n={row['n_words']:4.0f}) | "
            f"top-k (from {row['n_rated_words']:4.0f} rated): {row['top_k_mean_rating']:.2f} vs "
            f"{row['bottom_k_mean_rating']:.2f} (Δ={row['rating_separation']:.2f})"
        )

    print("\n=== POLE WORDS VALIDATION ===")
    for _, row in pole_summary.iterrows():
        print(
            f"{row['axis']:15s} | pos: {row['n_pos_rated']}/{row['n_pos_pole']} rated (μ={row['pos_mean_rating']:.2f}) | "
            f"neg: {row['n_neg_rated']}/{row['n_neg_pole']} rated (μ={row['neg_mean_rating']:.2f}) | "
            f"Δ={row['pole_rating_separation']:.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic axes pipeline")
    parser.add_argument("--embedding-dir", type=str, required=True)
    parser.add_argument("--axes-file", type=str, default="semantic_axes.json")
    parser.add_argument(
        "--data-dir", type=str, default="/LOCAL/fmahner/similarity-factorization/data"
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--glove-path", type=str, default=None)
    parser.add_argument("--glove-words", type=str, nargs="*", default=None)

    args = parser.parse_args()

    embedding_dir = Path(args.embedding_dir)
    output_dir = Path(args.output_dir) if args.output_dir else embedding_dir

    main(
        embedding_dir=embedding_dir,
        axes_file=Path(args.axes_file),
        data_dir=Path(args.data_dir),
        output_dir=output_dir,
        glove_path=Path(args.glove_path) if args.glove_path else None,
        glove_words=args.glove_words,
    )
