"""
THINGS Macaque Embedding: SRF factorization of IT neural responses.

Load THINGS-2k macaque data, compute RBF kernel similarity,
and extract SRF embedding to visualize dimensions.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import json
import logging
import argparse

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from datasets.loaders import load_things_monkey_2k, load_things_monkey
from tools.metrics import gaussian_kernel_similarity
from sklearn.metrics.pairwise import cosine_similarity
from pysrf import SRF

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def analyze_embedding(embedding: np.ndarray) -> dict:
    """Analyze structure of SRF embedding."""
    sparsity = (np.abs(embedding) < 0.01).mean()

    norms = np.linalg.norm(embedding, axis=0)
    norms = norms / norms.sum()
    entropy = -np.sum(norms * np.log(norms + 1e-10))
    eff_dim = np.exp(entropy)

    dim_variance = np.var(embedding, axis=0)
    dim_contribution = dim_variance / dim_variance.sum()

    return {
        "sparsity": float(sparsity),
        "effective_dim": float(eff_dim),
        "dim_contributions": dim_contribution.tolist(),
        "max_dim_contribution": float(dim_contribution.max()),
    }


def compute_rsm(data: np.ndarray, metric: str = "gaussian") -> np.ndarray:
    """Compute RSM with specified metric."""
    if metric == "gaussian":
        rsm = gaussian_kernel_similarity(data, data, sigma=None)
    elif metric == "shifted_cosine":
        rsm = (cosine_similarity(data) + 1) / 2
    else:
        raise ValueError(f"Unknown metric: {metric}")
    np.fill_diagonal(rsm, 1.0)
    return rsm


def main(
    rank: int = 5,
    monkey_type: str = "F",
    roi: str = "it",
    metric: str = "gaussian",
    dataset: str = "2k",
    max_outer: int = 200,
    max_inner: int = 30,
    output_dir: Path | None = None,
):
    if output_dir is None:
        output_dir = Path(__file__).parent / "outputs" / f"{dataset}_{metric}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(f"THINGS Macaque Embedding ({dataset}, rank={rank}, monkey={monkey_type}, roi={roi}, metric={metric})")
    log.info("=" * 60)

    # Load data
    log.info(f"\n[1] Loading THINGS-{dataset} macaque data...")
    if dataset == "2k":
        ds = load_things_monkey_2k(monkey_type=monkey_type, roi=roi)
    else:
        ds = load_things_monkey(monkey_type=monkey_type, roi=roi)

    log.info(f"  Neural data shape: {ds.data.shape}")
    log.info(f"  N concepts: {len(ds.metadata['filenames'])}")

    # Compute RSM
    log.info(f"\n[2] Computing RSM (metric={metric})...")
    rsm = compute_rsm(ds.data, metric=metric)
    log.info(f"  RSM shape: {rsm.shape}")
    log.info(f"  RSM range: [{rsm.min():.3f}, {rsm.max():.3f}]")

    # Run SRF
    log.info(f"\n[3] Running SRF (rank={rank}, max_outer={max_outer})...")
    model = SRF(
        rank=rank,
        rho=1.0,
        max_outer=max_outer,
        max_inner=max_inner,
        tol=1e-4,
        verbose=1,
        random_state=42,
    )
    embedding = model.fit_transform(rsm)
    reconstruction = embedding @ embedding.T

    recon_rmse = np.sqrt(np.mean((rsm - reconstruction) ** 2))
    log.info(f"  Reconstruction RMSE: {recon_rmse:.4f}")

    # Analyze embedding
    log.info("\n[4] Analyzing embedding structure...")
    structure = analyze_embedding(embedding)
    log.info(f"  Sparsity: {structure['sparsity']:.3f}")
    log.info(f"  Effective dimensionality: {structure['effective_dim']:.2f}")
    log.info(f"  Dimension contributions: {[f'{c:.3f}' for c in structure['dim_contributions']]}")

    # Save results
    log.info("\n[5] Saving results...")

    np.save(output_dir / "embedding.npy", embedding)
    np.save(output_dir / "rsm.npy", rsm)
    np.save(output_dir / "reconstruction.npy", reconstruction)

    filenames = ds.metadata["filenames"]
    np.savetxt(output_dir / "filenames.txt", filenames, fmt="%s")

    results = {
        "monkey_type": monkey_type,
        "roi": roi,
        "rank": rank,
        "metric": metric,
        "n_concepts": int(ds.data.shape[0]),
        "n_channels": int(ds.data.shape[1]),
        "recon_rmse": float(recon_rmse),
        **structure,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save per-dimension top concepts for quick inspection
    log.info("\n[6] Top concepts per dimension:")
    top_k = 10
    for dim in range(rank):
        loadings = embedding[:, dim]
        top_idx = np.argsort(loadings)[::-1][:top_k]
        top_concepts = filenames[top_idx]
        log.info(f"  Dim {dim+1}: {', '.join(top_concepts[:5])}...")

    dim_results = []
    for dim in range(rank):
        loadings = embedding[:, dim]
        top_idx = np.argsort(loadings)[::-1][:top_k]
        for i, idx in enumerate(top_idx):
            dim_results.append({
                "dimension": dim + 1,
                "rank": i + 1,
                "concept": filenames[idx],
                "loading": float(loadings[idx]),
            })

    pd.DataFrame(dim_results).to_csv(output_dir / "top_concepts.csv", index=False)

    log.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=5, help="Number of SRF dimensions")
    parser.add_argument("--monkey", type=str, default="F", choices=["F", "N"])
    parser.add_argument("--roi", type=str, default="it", choices=["v1", "v4", "it"])
    parser.add_argument("--metric", type=str, default="gaussian",
                        choices=["gaussian", "shifted_cosine"])
    parser.add_argument("--dataset", type=str, default="2k", choices=["2k", "22k"])
    parser.add_argument("--max-outer", type=int, default=200, help="Max outer iterations")
    parser.add_argument("--max-inner", type=int, default=30, help="Max inner iterations")
    args = parser.parse_args()
    main(rank=args.rank, monkey_type=args.monkey, roi=args.roi,
         metric=args.metric, dataset=args.dataset, max_outer=args.max_outer,
         max_inner=args.max_inner)
