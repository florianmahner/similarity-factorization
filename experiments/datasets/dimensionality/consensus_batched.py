"""Consensus embedding from cached similarity + CV-determined rank.

Bypasses the legacy `experiments/datasets/consensus/run.py` (which reads
ranks from `ranks/kappa/`). Uses the dimensionality CV's argmin_rank instead,
and the cached similarity from `outputs/cache/<dataset>.npy`.

For each dataset: runs EnsembleFit(n_runs=N, n_jobs=N) + AlignedConsensus,
saves embedding.npy, runs.npy, reliability.npy, summary.json into
outputs/consensus/<dataset>/.

Usage
-----
    python -m experiments.datasets.dimensionality.consensus_batched \
        --dataset swow --n-runs 30 --n-jobs 30

    # Override rank explicitly (skip CV lookup):
    python -m experiments.datasets.dimensionality.consensus_batched \
        --dataset swow --rank 268 --n-runs 30 --n-jobs 30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from pysrf import SRF
from pysrf.consensus import AlignedConsensus, EnsembleFit
from sklearn.pipeline import Pipeline

from tools.stats import dimension_reliability

log = logging.getLogger("consensus_batched")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

# Match the SRF kwargs used in the dimensionality CV so the consensus runs
# converge to comparable solutions.
SRF_KWARGS = {
    "rho": 3.0,
    "max_inner": 30,
    "tol": 0.0,
    "max_outer": 200,
    "check_input": False,
}


def main() -> None:
    args = _parse_args()

    dim_dir = Path(args.dim_output_dir).resolve()
    cache_path = dim_dir / "cache" / f"{args.dataset}.npy"
    cv_path = dim_dir / args.dataset / "cross_validation.json"
    consensus_out = Path(args.consensus_output_dir).resolve() / args.dataset
    consensus_out.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        raise FileNotFoundError(f"Cached similarity not found: {cache_path}")

    similarity = np.load(cache_path)
    n = similarity.shape[0]

    rank = args.rank if args.rank is not None else _read_argmin_rank(cv_path, args.variant, args.rank_kind)
    log.info(f"dataset={args.dataset}  n={n}  rank={rank}  n_runs={args.n_runs}  n_jobs={args.n_jobs}")

    pipeline = Pipeline([
        ("ensemble", EnsembleFit(
            SRF(rank=rank, random_state=args.random_state, **SRF_KWARGS),
            n_runs=args.n_runs,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )),
        ("consensus", AlignedConsensus(rank=rank)),
    ])

    t0 = time.time()
    pipeline.fit(similarity)
    embedding = pipeline.transform(similarity)
    elapsed = time.time() - t0
    log.info(f"consensus fit done in {elapsed:.1f}s")

    consensus = pipeline.named_steps["consensus"]
    aligned = consensus.aligned_embeddings_

    # Quality metrics
    recon = embedding @ embedding.T
    mask = np.isfinite(similarity)
    np.fill_diagonal(mask, False)
    recon_rmse = float(np.sqrt(np.mean((similarity[mask] - recon[mask])**2))) if mask.any() else float("nan")
    recon_r = float(np.corrcoef(similarity[mask].ravel(), recon[mask].ravel())[0, 1]) if mask.any() else float("nan")
    sparsity = float((embedding == 0).mean())
    cv_rel = dimension_reliability(aligned)

    log.info(f"recon RMSE={recon_rmse:.4f}  r={recon_r:.4f}  sparsity={sparsity:.3f}  "
             f"reliability mean={float(np.mean(cv_rel)):.3f}  min={float(cv_rel.min()):.3f}")

    np.save(consensus_out / "embedding.npy", embedding)
    np.save(consensus_out / "runs.npy", aligned)
    np.save(consensus_out / "cv_reliability.npy", cv_rel)

    summary = {
        "dataset": args.dataset,
        "n_samples": int(n),
        "rank": int(rank),
        "rank_kind": args.rank_kind,
        "n_runs": int(args.n_runs),
        "n_jobs": int(args.n_jobs),
        "elapsed_sec": elapsed,
        "selected_run_idx": int(consensus.selected_run_idx_),
        "reconstruction_rmse": recon_rmse,
        "reconstruction_r": recon_r,
        "sparsity": sparsity,
        "cv_reliability_mean": float(np.mean(cv_rel)),
        "cv_reliability_min": float(cv_rel.min()),
        "cv_reliability_max": float(cv_rel.max()),
        "srf_kwargs": SRF_KWARGS,
    }
    (consensus_out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"wrote {consensus_out}")


def _read_argmin_rank(cv_path: Path, variant: str, rank_kind: str) -> int:
    payload = json.loads(cv_path.read_text())
    block = payload["validations"][variant]
    if rank_kind == "argmin":
        return int(block["argmin_rank"])
    if rank_kind == "one_se":
        return int(block["one_se_rank"])
    raise ValueError(f"unknown rank_kind: {rank_kind}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rank", type=int, default=None, help="Override rank (skip CV lookup).")
    parser.add_argument("--rank-kind", choices=["argmin", "one_se"], default="argmin")
    parser.add_argument("--variant", default="5fold")
    parser.add_argument("--n-runs", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dim-output-dir", default="/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/datasets/dimensionality/outputs",
                        help="Where to read cached similarity + CV JSON from.")
    parser.add_argument("--consensus-output-dir", default="/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/datasets/consensus/outputs",
                        help="Where to write embedding.npy + runs.npy + summary.json.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
