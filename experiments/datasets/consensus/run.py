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
from pysrf.consensus import AlignedConsensus, EnsembleEmbedding
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
    """Generate consensus embedding using pre-computed optimal rank."""
    subject_id = cfg.get("subject_id")

    output_dir = Path.cwd()
    if subject_id is not None:
        output_dir = output_dir / f"subj{subject_id:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-computed rank (required)
    rank_est = _load_rank_estimation(cfg.dataset.name, subject_id)
    optimal_rank = rank_est["optimal_rank"]
    log.info(f"Optimal rank: {optimal_rank}")

    # Build similarity matrix
    log.info(f"Building similarity matrix for {cfg.dataset.name}...")
    similarity = build_similarity(cfg.dataset, subject_id=subject_id)
    n_samples = similarity.shape[0]
    log.info(f"Similarity matrix shape: ({n_samples}, {n_samples})")

    # Fit ensemble and consensus
    n_runs = cfg.generate.n_stable_runs
    log.info(f"Running {n_runs} SRF fits for consensus...")

    pipeline = Pipeline(
        [
            (
                "ensemble",
                EnsembleEmbedding(
                    SRF(rank=optimal_rank, random_state=cfg.common.random_state),
                    n_runs=n_runs,
                    random_state=cfg.common.random_state,
                    n_jobs=cfg.common.n_jobs,
                ),
            ),
            ("consensus", AlignedConsensus(rank=optimal_rank, aggregation="select")),
        ]
    )

    pipeline.fit(similarity)
    embedding = pipeline.transform(similarity)

    # Extract results
    ensemble = pipeline.named_steps["ensemble"]
    consensus = pipeline.named_steps["consensus"]
    ensemble_embeddings = ensemble.embeddings_

    log.info(f"Consensus embedding shape: {embedding.shape}")
    log.info(f"Selected run: {consensus.selected_run_idx_}")
    log.info(
        f"Agreement: {consensus.agreement_scores_.mean():.3f} ± {consensus.agreement_scores_.std():.3f}"
    )

    # Quality metrics
    recon = embedding @ embedding.T
    recon_error = np.linalg.norm(similarity - recon, "fro") / np.linalg.norm(
        similarity, "fro"
    )
    sparsity = (embedding == 0).mean()
    purity = (embedding.max(axis=1) / (embedding.sum(axis=1) + 1e-10)).mean()

    log.info(f"Reconstruction error: {recon_error:.4f}")
    log.info(f"Sparsity: {sparsity:.3f}")
    log.info(f"Purity: {purity:.3f}")

    # Save outputs
    np.save(output_dir / "embedding.npy", embedding)
    np.save(output_dir / "runs.npy", ensemble_embeddings)

    summary = {
        "dataset": cfg.dataset.name,
        "subject_id": subject_id,
        "n_samples": n_samples,
        "rank": optimal_rank,
        "n_runs": n_runs,
        "selected_run_idx": int(consensus.selected_run_idx_),
        "agreement_scores": consensus.agreement_scores_.tolist(),
        "centrality_scores": consensus.centrality_scores_.tolist(),
        "reconstruction_error": float(recon_error),
        "sparsity": float(sparsity),
        "purity": float(purity),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    log.info(f"Saved to {output_dir}")
