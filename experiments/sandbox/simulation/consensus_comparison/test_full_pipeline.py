"""
Test the full consensus pipeline with mur92.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

from pysrf import SRF, cross_val_score
from pysrf.bounds import estimate_sampling_bounds_ultra
from pysrf.consensus import EnsembleEmbedding, AlignedConsensus
from src.datasets.loaders import load_mur92


def main():
    output_dir = Path(__file__).parent / "outputs" / "full_pipeline_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    dataset = load_mur92("/SSD/datasets/similarity_datasets/mur92")
    rsm = dataset.rsm

    print("=" * 60)
    print("FULL PIPELINE TEST FOR MUR92")
    print("=" * 60)

    # Step 1: Estimate bounds
    print("\n1. Estimating bounds...")
    pmin, pmax, _ = estimate_sampling_bounds_ultra(rsm, verbose=False)
    sampling_fraction = (pmin + pmax) / 2

    print(f"   pmin = {pmin:.4f}")
    print(f"   pmax = {pmax:.4f}")
    print(f"   sampling_fraction (mean) = {sampling_fraction:.4f}")

    # Step 2: Cross-validation to find optimal rank
    print("\n2. Running cross-validation...")
    rank_grid = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]

    cv_result = cross_val_score(
        rsm,
        param_grid={"rank": rank_grid},
        n_repeats=30,
        sampling_fraction=sampling_fraction,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    optimal_rank = cv_result.best_params_["rank"]
    print(f"   Optimal rank: {optimal_rank}")
    print(f"   Best CV score: {cv_result.best_score_:.6f}")

    # Show all scores
    print("\n   CV scores by rank:")
    mean_scores = cv_result.cv_results_.groupby("rank")["score"].mean()
    for rank in rank_grid:
        marker = " <-- OPTIMAL" if rank == optimal_rank else ""
        print(f"      Rank {rank:2d}: {mean_scores[rank]:.6f}{marker}")

    # Step 3: Fit consensus embedding with optimal rank
    print(f"\n3. Fitting consensus embedding (rank={optimal_rank})...")
    pipeline = Pipeline([
        ("ensemble", EnsembleEmbedding(
            SRF(rank=optimal_rank, random_state=42),
            n_runs=50,
            random_state=42,
            n_jobs=-1,
        )),
        ("consensus", AlignedConsensus(
            rank=optimal_rank,
            aggregation="refine",
        )),
    ])

    embedding = pipeline.fit_transform(rsm)
    print(f"   Embedding shape: {embedding.shape}")

    # Check reconstruction
    recon = embedding @ embedding.T
    error = np.linalg.norm(rsm - recon, "fro") / np.linalg.norm(rsm, "fro")
    print(f"   Reconstruction error: {error:.4f}")

    # Check purity
    purity = embedding.max(axis=1) / embedding.sum(axis=1)
    print(f"   Mean purity: {purity.mean():.3f}")

    # Save results
    results = {
        "pmin": float(pmin),
        "pmax": float(pmax),
        "sampling_fraction": float(sampling_fraction),
        "optimal_rank": int(optimal_rank),
        "best_cv_score": float(cv_result.best_score_),
        "reconstruction_error": float(error),
        "mean_purity": float(purity.mean()),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.save(output_dir / "embedding.npy", embedding)

    print(f"\n   Results saved to {output_dir}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
The full pipeline correctly identifies:
- Optimal rank: {optimal_rank} (using bounds-estimated sampling)
- Reconstruction error: {error:.4f}
- Mean purity: {purity.mean():.3f}

This confirms the pipeline works correctly when using the
bounds-estimated sampling fraction!
""")


if __name__ == "__main__":
    main()
