"""Test reliability vs RBF sigma for CLIP RN50 features.

Compares sigma multipliers [0.2, 0.4, 0.6, 0.8, 1.0] of median heuristic.
For each sigma, fits 10 SRF runs at rank=15, computes per-dimension reliability.
"""

from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from pysrf import SRF
from pysrf.consensus import AlignedConsensus, EnsembleEmbedding
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances, pairwise_kernels
from sklearn.pipeline import Pipeline

from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()

RANK = 15
N_RUNS = 10
SIGMA_MULTS = [0.2, 0.4, 0.6, 0.8, 1.0]


def _build_rbf(features, sigma_mult):
    dist = pairwise_distances(features, metric="euclidean")
    median_dist = np.median(dist[np.triu_indices(len(dist), k=1)])
    sigma = median_dist * sigma_mult
    gamma = 1 / (2 * sigma**2)
    s = pairwise_kernels(features, metric="rbf", gamma=gamma)
    return s, sigma


def _reliability_from_aligned(aligned):
    n_runs, n_samples, rank = aligned.shape
    rel = np.zeros(rank)
    for d in range(rank):
        pairs = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                r = np.corrcoef(aligned[i, :, d], aligned[j, :, d])[0, 1]
                if np.isfinite(r):
                    pairs.append(r)
        if pairs:
            z = np.arctanh(np.clip(pairs, -0.999, 0.999))
            rel[d] = float(np.tanh(np.mean(z)))
    return rel


def main():
    features = np.load("data/features/clip_rn50/clip_rn50_features.npy")
    print(f"Features: {features.shape}")

    # Also test linear kernel
    s_linear = features @ features.T
    print(f"Linear kernel: [{s_linear.min():.3f}, {s_linear.max():.3f}], mean={s_linear.mean():.3f}")

    rows = []

    # Linear kernel
    print(f"\n=== Linear kernel, rank={RANK} ===")
    pipe = Pipeline([
        ("ensemble", EnsembleEmbedding(SRF(rank=RANK), n_runs=N_RUNS, n_jobs=-1)),
        ("consensus", AlignedConsensus(rank=RANK, aggregation="select")),
    ])
    pipe.fit(s_linear)
    emb = pipe.transform(s_linear)
    rel = _reliability_from_aligned(pipe.named_steps["consensus"].aligned_embeddings_)
    print(f"  mean_rel={rel.mean():.3f}, per-dim={np.sort(rel)[::-1].round(3)}")
    rows.append({"kernel": "linear", "sigma_mult": None, "sigma": None,
                 "sim_mean": s_linear.mean(), "mean_rel": rel.mean(),
                 "min_rel": rel.min(), "max_rel": rel.max()})

    # RBF at different sigmas
    for mult in SIGMA_MULTS:
        s, sigma = _build_rbf(features, mult)
        print(f"\n=== RBF sigma={sigma:.1f} ({mult}x median), sim=[{s.min():.3f}, {s.max():.3f}], mean={s.mean():.3f} ===")

        pipe = Pipeline([
            ("ensemble", EnsembleEmbedding(SRF(rank=RANK), n_runs=N_RUNS, n_jobs=-1)),
            ("consensus", AlignedConsensus(rank=RANK, aggregation="select")),
        ])
        pipe.fit(s)
        emb = pipe.transform(s)
        rel = _reliability_from_aligned(pipe.named_steps["consensus"].aligned_embeddings_)
        print(f"  mean_rel={rel.mean():.3f}, per-dim={np.sort(rel)[::-1].round(3)}")

        rows.append({"kernel": "rbf", "sigma_mult": mult, "sigma": sigma,
                     "sim_mean": s.mean(), "mean_rel": rel.mean(),
                     "min_rel": rel.min(), "max_rel": rel.max()})

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    print(f"\n=== Summary ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
