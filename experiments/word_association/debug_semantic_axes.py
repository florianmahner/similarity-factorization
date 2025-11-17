from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from experiments.word_association.lib.semantic_axes import (
    load_semantic_axes,
    compute_axis_vectors,
    project_vocabulary,
)
from experiments.word_association.lib.validation import (
    load_behavioral_ratings,
    match_words_to_ratings,
    compute_axis_correlations,
    get_extreme_words,
)


def load_embeddings(embedding_dir: Path) -> tuple[np.ndarray, list[str], dict[str, int]]:
    w = np.load(embedding_dir / "srf_factors.npy")
    vocabulary = (embedding_dir / "vocabulary.txt").read_text().strip().split("\n")
    word_to_idx = {word.lower(): idx for idx, word in enumerate(vocabulary)}
    return w, vocabulary, word_to_idx


def debug_axes(
    embedding_dir: Path,
    axes_file: Path,
    ratings_dir: Path,
    probe_words: list[str],
) -> None:
    w, vocabulary, word_to_idx = load_embeddings(embedding_dir)

    axes_config = load_semantic_axes(axes_file)
    axes, metadata = compute_axis_vectors(axes_config, w, word_to_idx)
    if not axes:
        raise ValueError("No axes could be constructed from the provided config")

    projections = project_vocabulary(w, vocabulary, axes)
    ratings = load_behavioral_ratings(ratings_dir)
    merged = match_words_to_ratings(projections, ratings)
    print(f"Merged rows: {len(merged)} of {len(projections)} projections")
    print(f"Merged columns: {list(merged.columns)}")

    axis_rating_pairs = {
        "concreteness": "concreteness_rating",
        "size": "size_rating",
        "valence": "valence_rating",
        "heaviness": "heaviness_rating",
        "animacy": "animacy_rating",
    }
    axis_rating_pairs = {
        axis: rating
        for axis, rating in axis_rating_pairs.items()
        if axis in merged.columns and rating in merged.columns
    }

    print("Axis coverage (n_low, n_high):")
    for axis_name, info in metadata.items():
        print(f"  {axis_name:12s}: low={info['n_low']:3d}, high={info['n_high']:3d}")

    print("\nAxis–rating correlations:")
    corr_df = compute_axis_correlations(merged, axis_rating_pairs)
    if not corr_df.empty and "correlation" in corr_df.columns:
        print(corr_df.sort_values("correlation", ascending=False))
    else:
        print(corr_df)

    for axis_name, rating_col in axis_rating_pairs.items():
        print(f"\n=== {axis_name} (rating: {rating_col}) ===")
        sub = merged[["word", axis_name, rating_col]].dropna()
        if sub.empty:
            print("  No overlapping words with ratings.")
            continue
        top, bottom = get_extreme_words(sub, axis_name, n_top=15)
        print("  Top projected words:")
        print(top.head(15))
        print("  Bottom projected words:")
        print(bottom.head(15))

    if probe_words:
        probe = [w.lower() for w in probe_words]
        probe_df = merged[merged["word"].isin(probe)]
        print("\nProbe words:")
        if probe_df.empty:
            print("  None of the probe words found in merged data.")
        else:
            cols = list(axis_rating_pairs.keys()) + list(axis_rating_pairs.values())
            cols = [c for c in cols if c in probe_df.columns]
            print(probe_df.set_index("word")[cols])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug semantic axes projections and correlations."
    )
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--axes-file", type=Path, required=True)
    parser.add_argument("--ratings-dir", type=Path, required=True)
    parser.add_argument(
        "--probe-words",
        nargs="*",
        default=["africa", "america", "elephant", "needle", "truck", "banana"],
    )
    args = parser.parse_args()

    debug_axes(
        embedding_dir=args.embedding_dir,
        axes_file=args.axes_file,
        ratings_dir=args.ratings_dir,
        probe_words=args.probe_words,
    )


if __name__ == "__main__":
    main()
