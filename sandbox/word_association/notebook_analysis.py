from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.word_association.plotting import (
    plot_analogy,
    plot_reconstruction_quality,
    plot_top_words_facet_grid,
    plot_word_clouds_grid,
)


def main():
    output_dir = Path(
        "experiments/word_association/outputs/generate_embedding"
    )

    run_dir = Path("experiments/word_association/development") / datetime.now().strftime("%y%m%d") / datetime.now().strftime("%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Word Association Network Analysis")
    print("=" * 80)

    print(f"\nLoading data from: {output_dir}")
    print(f"Saving outputs to: {run_dir}\n")

    W = np.load(output_dir / "word_embedding.npy")
    similarity = np.load(output_dir / "similarity.npy")

    with open(output_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    vocabulary = metadata["vocabulary"]
    word_to_idx = metadata["words_to_idx"]
    top_words = metadata["top_words"]

    top_words_df = pd.DataFrame(top_words)

    n_words, rank = W.shape

    print("Data loaded successfully!")
    print("=" * 80)
    print(f"Factor matrix (W): {W.shape}")
    print(f"Number of words: {n_words}")
    print(f"Number of dimensions: {rank}")
    print("=" * 80)

    print("\n1. Reconstruction Quality")
    print("-" * 80)
    plot_reconstruction_quality(
        similarity=similarity,
        embedding=W,
        output_path=run_dir / "reconstruction_quality.png",
    )
    print(f"✓ Saved: {run_dir / 'reconstruction_quality.png'}")

    print("\n2. Top Words per Dimension (FacetGrid)")
    print("-" * 80)
    plot_top_words_facet_grid(
        top_words_df=top_words_df,
        output_path=run_dir / "top_words_facet_grid.png",
        n_words=10,
    )
    print(f"✓ Saved: {run_dir / 'top_words_facet_grid.png'}")

    print("\n3. Word Clouds for All Dimensions")
    print("-" * 80)
    plot_word_clouds_grid(
        top_words_df=top_words_df,
        output_path=run_dir / "word_clouds_grid.png",
        n_words=100,
    )
    print(f"✓ Saved: {run_dir / 'word_clouds_grid.png'}")

    print("\n4. Analogy Analysis")
    print("-" * 80)
    analogies = [
        ("king", "man", "woman"),
        ("actor", "man", "woman"),
        ("Berlin", "Germany", "France"),
        ("Paris", "France", "Italy"),
        ("good", "bad", "evil"),
    ]

    for a, b, c in analogies:
        if all(word in word_to_idx for word in [a, b, c]):
            plot_analogy(
                embedding=W,
                vocabulary=vocabulary,
                word_to_idx=word_to_idx,
                top_words_df=top_words_df,
                a=a,
                b=b,
                c=c,
                output_path=run_dir / f"analogy_{a}_{b}_{c}.png",
                top_k=10,
            )
            print(f"✓ Saved: analogy_{a}_{b}_{c}.png")
        else:
            missing = [w for w in [a, b, c] if w not in word_to_idx]
            print(f"✗ Skipped {a}-{b}+{c}: missing words {missing}")

    print("\n" + "=" * 80)
    print(f"Analysis complete! All outputs saved to:")
    print(f"  {run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
