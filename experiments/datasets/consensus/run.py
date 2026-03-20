"""Generate consensus embeddings using pre-computed optimal rank.

Requires ranks/cv/ to have been run first for the dataset.

Usage:
    ./scripts/submit experiments/datasets/consensus/run.py dataset=mur92
    ./scripts/submit experiments/datasets/consensus/run.py dataset=nsd subject_id=1

Outputs:
    embedding.npy  - Final consensus embedding (n_samples, rank)
    runs.npy       - All ensemble run embeddings (n_runs, n_samples, rank)
    summary.json   - Metadata and quality metrics
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from pysrf import SRF
from pysrf.consensus import ClusterEmbedding, EnsembleEmbedding
from sklearn.pipeline import Pipeline

from similarity import build_similarity

log = logging.getLogger(__name__)

CV_DIR = Path(__file__).resolve().parent.parent / "ranks" / "cv" / "outputs"


def _load_rank_estimation(dataset_name: str, subject_id: int | None) -> dict:
    """Load rank from CV outputs. Path matches cv/run.py output structure."""
    base = CV_DIR / dataset_name
    if subject_id is not None:
        path = base / f"subj{subject_id:02d}" / "rank_estimation.json"
    else:
        path = base / "rank_estimation.json"
    if not path.exists():
        raise FileNotFoundError(
            f"CV results not found at {path}\n"
            f"Run CV first: ./scripts/submit experiments/datasets/ranks/cv/run.py dataset={dataset_name}"
        )
    log.info(f"Loading rank estimation from {path}")
    with open(path) as f:
        return json.load(f)


def run(cfg: DictConfig) -> None:
    subject_id = cfg.get("subject_id")

    ds_name = cfg.dataset.name
    output_dir = Path.cwd() / ds_name
    if subject_id is not None:
        output_dir = output_dir / f"subj{subject_id:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rank_est = _load_rank_estimation(cfg.dataset.name, subject_id)
    optimal_rank = rank_est["optimal_rank"]
    log.info(f"Optimal rank: {optimal_rank}")

    log.info(f"Building similarity matrix for {cfg.dataset.name}...")
    similarity = build_similarity(cfg.dataset, subject_id=subject_id)
    n_samples = similarity.shape[0]
    log.info(f"Similarity matrix shape: ({n_samples}, {n_samples})")

    n_runs = cfg.generate.n_stable_runs
    log.info(f"Running {n_runs} SRF fits for consensus...")

    pipeline = Pipeline([
        ("ensemble", EnsembleEmbedding(
            SRF(rank=optimal_rank, random_state=cfg.common.random_state),
            n_runs=n_runs,
            random_state=cfg.common.random_state,
            n_jobs=cfg.common.n_jobs,
        )),
        ("cluster", ClusterEmbedding(
            min_clusters=optimal_rank,
            max_clusters=optimal_rank,
            random_state=cfg.common.random_state,
            n_jobs=cfg.common.n_jobs,
        )),
    ])

    pipeline.fit(similarity)
    embedding = pipeline.transform(similarity)

    ensemble = pipeline.named_steps["ensemble"]
    cluster = pipeline.named_steps["cluster"]

    # Reshape ensemble embeddings: (n_samples, rank*n_runs) -> (n_runs, n_samples, rank)
    stacked = ensemble.embeddings_
    runs = stacked.reshape(n_samples, n_runs, optimal_rank).transpose(1, 0, 2)

    log.info(f"Consensus embedding shape: {embedding.shape}")
    log.info(f"Cluster k: {cluster.best_k_}")

    # Quality metrics
    recon = embedding @ embedding.T
    recon_error = np.linalg.norm(similarity - recon, "fro") / np.linalg.norm(similarity, "fro")
    sparsity = float((embedding == 0).mean())
    purity = float((embedding.max(axis=1) / (embedding.sum(axis=1) + 1e-10)).mean())

    log.info(f"Reconstruction error: {recon_error:.4f}")
    log.info(f"Sparsity: {sparsity:.3f}")
    log.info(f"Purity: {purity:.3f}")

    np.save(output_dir / "embedding.npy", embedding)
    np.save(output_dir / "runs.npy", runs)

    summary = {
        "dataset": cfg.dataset.name,
        "subject_id": subject_id,
        "n_samples": n_samples,
        "rank": optimal_rank,
        "n_runs": n_runs,
        "cluster_k": int(cluster.best_k_),
        "reconstruction_error": float(recon_error),
        "sparsity": sparsity,
        "purity": purity,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"Saved to {output_dir}")
