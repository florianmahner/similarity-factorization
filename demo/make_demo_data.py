"""Generate the small simulated dataset shipped with the demo.

The dataset matches what SRF assumes: a sparse, non-negative ground-truth
embedding ``X`` (100 objects x 6 dimensions) with similarity ``S = X @ X.T``.
Running SRF on ``S`` at rank 6 should recover ``X`` up to a permutation and
scaling of its columns.

``demo_similarity.npz`` is already committed, so this script is only needed to
change the simulation. It depends only on NumPy.

Usage
-----
    python demo/make_demo_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Small enough to run in seconds, large enough to make recovery non-trivial.
N_OBJECTS = 100
N_DIMS = 6
SPARSITY = 0.5  # fraction of zero entries in the embedding
SEED = 0

OUTPUT = Path(__file__).resolve().parent / "demo_similarity.npz"


def main() -> None:
    rng = np.random.default_rng(SEED)
    embedding = np.abs(rng.standard_normal((N_OBJECTS, N_DIMS)))
    embedding *= rng.random((N_OBJECTS, N_DIMS)) > SPARSITY
    similarity = embedding @ embedding.T

    np.savez_compressed(
        OUTPUT,
        similarity=similarity,
        embedding=embedding,
        n_dims=np.int64(N_DIMS),
        seed=np.int64(SEED),
    )
    print(f"wrote {OUTPUT}")
    print(f"  similarity: {similarity.shape}, embedding: {embedding.shape}")
    print(f"  true rank: {N_DIMS}")


if __name__ == "__main__":
    main()
