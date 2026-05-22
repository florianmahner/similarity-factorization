"""Recompute dimension reliability across random restarts."""

import json
from datetime import datetime
import os
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from pysrf import SRF

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.things_behavior.common import compute_similarity_matrix_from_triplets


OUTPUT_DIR = Path(os.environ.get("SANDBOX_OUTPUT_DIR", Path(__file__).parent / "outputs" / "dev"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_srf(similarity: np.ndarray, rank: int, seed: int) -> np.ndarray:
    model = SRF(rank=rank, random_state=seed, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
    return model.fit_transform(similarity)


def fisher_z(r: float) -> float:
    r = np.clip(r, -0.999, 0.999)
    return 0.5 * np.log((1 + r) / (1 - r))


def inverse_fisher_z(z: float) -> float:
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def main():
    print(f"Output directory: {OUTPUT_DIR}")

    # Load and compute similarity
    print("Loading triplets...")
    train = np.loadtxt("data/things/triplets_47/trainset.txt").astype(int)
    print(f"Computing similarity matrix from {len(train):,} triplets...")
    similarity = compute_similarity_matrix_from_triplets(1854, train)

    # Save similarity matrix
    np.save(OUTPUT_DIR / "similarity.npy", similarity)
    print(f"Saved similarity matrix to {OUTPUT_DIR / 'similarity.npy'}")

    # Fit with multiple seeds
    n_runs = 20
    n_jobs = 20
    rank = 66

    print(f"Fitting SRF {n_runs + 1} times with {n_jobs} parallel jobs...")
    embeddings = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(fit_srf)(similarity, rank=rank, seed=seed) for seed in range(n_runs + 1)
    )

    # Save all embeddings
    for i, emb in enumerate(embeddings):
        np.save(OUTPUT_DIR / f"embedding_seed{i}.npy", emb)
    print(f"Saved {len(embeddings)} embeddings")

    # Compute reliability
    original = embeddings[0]
    references = embeddings[1:]

    print("\nComputing dimension reliability...")
    reliabilities = []
    for dim_idx in range(rank):
        orig_dim = original[:, dim_idx]
        best_corrs = []
        for ref in references:
            corrs = [abs(np.corrcoef(orig_dim, ref[:, j])[0, 1]) for j in range(rank)]
            best_corrs.append(max(corrs))
        mean_z = np.mean([fisher_z(c) for c in best_corrs])
        reliabilities.append(inverse_fisher_z(mean_z))

    reliabilities = np.array(reliabilities)

    # Save results
    results = {
        "n_runs": n_runs,
        "rank": rank,
        "mean": float(reliabilities.mean()),
        "std": float(reliabilities.std()),
        "min": float(reliabilities.min()),
        "max": float(reliabilities.max()),
        "dims_above_0.9": int((reliabilities > 0.9).sum()),
        "dims_above_0.95": int((reliabilities > 0.95).sum()),
        "dims_above_0.99": int((reliabilities > 0.99).sum()),
        "per_dimension": reliabilities.tolist(),
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Dimension Reliability (n_runs={n_runs})")
    print(f"{'='*50}")
    print(f"Mean: {reliabilities.mean():.4f}")
    print(f"Std:  {reliabilities.std():.4f}")
    print(f"Min:  {reliabilities.min():.4f}")
    print(f"Max:  {reliabilities.max():.4f}")
    print(f"Dims > 0.9:  {(reliabilities > 0.9).sum()}")
    print(f"Dims > 0.95: {(reliabilities > 0.95).sum()}")
    print(f"Dims > 0.99: {(reliabilities > 0.99).sum()}")
    print(f"\nResults saved to: {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
