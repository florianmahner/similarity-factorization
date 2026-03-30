"""Generate consensus embeddings using pre-computed optimal rank.

Uses AlignedConsensus with Hungarian alignment to produce a principled
consensus: aligns all runs via optimal permutation matching, then selects
the most central run (preserving valid factorization WW^T ~ S).

Requires ranks/kappa/ to have been run first for the dataset.

Usage:
    ./scripts/submit experiments/datasets/consensus/run.py dataset=mur92
    ./scripts/submit experiments/datasets/consensus/run.py dataset=nsd subject_id=1

Outputs:
    embedding.npy  - Final consensus embedding (n_samples, rank)
    runs.npy       - All aligned run embeddings (n_runs, n_samples, rank)
    summary.json   - Metadata and quality metrics
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from pysrf import SRF
from pysrf.consensus import AlignedConsensus, EnsembleEmbedding
from sklearn.pipeline import Pipeline

from similarity import build_similarity

log = logging.getLogger(__name__)

KAPPA_DIR = Path(__file__).resolve().parent.parent / "ranks" / "kappa" / "outputs"


def _compute_dimension_reliability(aligned: np.ndarray) -> np.ndarray:
    """Per-dimension reliability from aligned embeddings.

    For each dimension, computes pairwise Pearson correlations across all
    run pairs, then averages via Fisher-z transform.

    Parameters
    ----------
    aligned : (n_runs, n_samples, rank) array
        Hungarian-aligned embeddings from AlignedConsensus.

    Returns
    -------
    reliability : (rank,) array
        Mean pairwise correlation per dimension.
    """
    n_runs, n_samples, rank = aligned.shape
    reliability = np.zeros(rank)

    for d in range(rank):
        r_pairs = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                r = np.corrcoef(aligned[i, :, d], aligned[j, :, d])[0, 1]
                if np.isfinite(r):
                    r_pairs.append(r)
        if r_pairs:
            z = np.arctanh(np.clip(r_pairs, -0.999, 0.999))
            reliability[d] = float(np.tanh(np.mean(z)))

    return reliability


def _load_kappa_rank(dataset_name: str, subject_id: int | None) -> int:
    """Load k* from kappa outputs."""
    name = dataset_name.replace("-", "_")
    if subject_id is not None:
        name = f"{name}_subj{subject_id:02d}"
    path = KAPPA_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Kappa results not found: {path}\n"
            f"Run kappa first: ./scripts/submit experiments/datasets/ranks/kappa/run.py --bg"
        )
    with open(path) as f:
        data = json.load(f)
    return data.get("k_star") or data.get("k_star_kappa")


def run(cfg: DictConfig) -> None:
    subject_id = cfg.get("subject_id")

    ds_name = cfg.dataset.name
    output_dir = Path.cwd() / ds_name
    if subject_id is not None:
        output_dir = output_dir / f"subj{subject_id:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    optimal_rank = _load_kappa_rank(cfg.dataset.name, subject_id)
    log.info(f"Kappa k*: {optimal_rank}")

    log.info(f"Building similarity matrix for {cfg.dataset.name}...")
    similarity = build_similarity(cfg.dataset, subject_id=subject_id)
    n_samples = similarity.shape[0]
    log.info(f"Similarity matrix shape: ({n_samples}, {n_samples})")

    aggregation = cfg.get("aggregation", "select")
    n_runs = cfg.generate.n_stable_runs
    log.info(f"Running {n_runs} SRF fits, aggregation={aggregation}...")

    srf_kwargs = {}
    if "srf" in cfg:
        srf_kwargs["max_outer"] = cfg.srf.get("max_outer", 30)
        srf_kwargs["max_inner"] = cfg.srf.get("max_inner", 20)

    pipeline = Pipeline([
        ("ensemble", EnsembleEmbedding(
            SRF(rank=optimal_rank, random_state=cfg.common.random_state, **srf_kwargs),
            n_runs=n_runs,
            random_state=cfg.common.random_state,
            n_jobs=cfg.common.n_jobs,
        )),
        ("consensus", AlignedConsensus(
            rank=optimal_rank,
            aggregation=aggregation,
        )),
    ])

    pipeline.fit(similarity)
    embedding = pipeline.transform(similarity)

    ensemble = pipeline.named_steps["ensemble"]
    consensus = pipeline.named_steps["consensus"]

    log.info(f"Consensus embedding shape: {embedding.shape}")
    log.info(f"Selected run index: {consensus.selected_run_idx_}")
    log.info(f"Centrality scores (min/max): "
             f"{consensus.centrality_scores_.min():.4f} / "
             f"{consensus.centrality_scores_.max():.4f}")

    # Quality metrics
    recon = embedding @ embedding.T
    mask = np.isfinite(similarity)
    np.fill_diagonal(mask, False)
    if mask.any():
        recon_error = float(np.sqrt(np.mean((similarity[mask] - recon[mask])**2)))
        recon_r = float(np.corrcoef(similarity[mask].ravel(), recon[mask].ravel())[0, 1])
    else:
        recon_error = float("nan")
        recon_r = float("nan")

    sparsity = float((embedding == 0).mean())
    purity = float((embedding.max(axis=1) / (embedding.sum(axis=1) + 1e-10)).mean())

    # Per-dimension reliability across aligned runs
    aligned = consensus.aligned_embeddings_
    n_runs_actual = aligned.shape[0]
    reliability = _compute_dimension_reliability(aligned)
    mean_reliability = float(np.mean(reliability))

    log.info(f"Reconstruction RMSE: {recon_error:.4f}, r: {recon_r:.4f}")
    log.info(f"Sparsity: {sparsity:.3f}, Purity: {purity:.3f}")
    log.info(f"Dimension reliability: mean={mean_reliability:.3f}, "
             f"min={reliability.min():.3f}, max={reliability.max():.3f}")

    # Save
    np.save(output_dir / "embedding.npy", embedding)
    np.save(output_dir / "runs.npy", consensus.aligned_embeddings_)
    np.save(output_dir / "reliability.npy", reliability)

    summary = {
        "dataset": cfg.dataset.name,
        "subject_id": subject_id,
        "n_samples": n_samples,
        "rank": optimal_rank,
        "n_runs": n_runs,
        "aggregation": aggregation,
        "selected_run_idx": int(consensus.selected_run_idx_),
        "reconstruction_rmse": recon_error,
        "reconstruction_r": recon_r,
        "sparsity": sparsity,
        "purity": purity,
        "mean_reliability": mean_reliability,
        "min_reliability": float(reliability.min()),
        "max_reliability": float(reliability.max()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"Saved to {output_dir}")
