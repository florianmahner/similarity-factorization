"""Test SRF on SWOW random walk similarity."""

from pathlib import Path
import numpy as np
import pandas as pd
from pysrf import SRF
from src.datasets import load_dataset
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()


def main():
    print("Loading SWOW RW similarity...")
    ds = load_dataset(
        "swow",
        root="data/small-world-of-words",
        similarity_method="rw",
        use_all_responses=True,
    )
    S = ds.rsm
    vocab = np.array(ds.metadata["vocabulary"])

    print(f"Shape: {S.shape}")
    print(f"Range: [{S.min():.4f}, {S.max():.4f}]")

    print("\nFitting SRF with rank=10...")
    model = SRF(rank=10, max_outer=100, verbose=1)
    model.fit(S)

    W = model.w_
    print(f"\nEmbedding shape: {W.shape}")

    # Top words per dimension
    print("\n" + "=" * 60)
    print("Top 15 words per dimension:")
    print("=" * 60)

    results = []
    for d in range(10):
        top_idx = np.argsort(W[:, d])[::-1][:15]
        top_words = [vocab[i] for i in top_idx]
        top_weights = [W[i, d] for i in top_idx]
        print(f"\nDim {d}: {', '.join(top_words[:10])}")
        for i, (w, wt) in enumerate(zip(top_words, top_weights)):
            results.append({"dim": d, "rank": i, "word": w, "weight": wt})

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "top_words.csv", index=False)
    np.save(OUTPUT_DIR / "embedding.npy", W)
    np.save(OUTPUT_DIR / "vocabulary.npy", vocab)
    np.save(OUTPUT_DIR / "similarity.npy", S)

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
