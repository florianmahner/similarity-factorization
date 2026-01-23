"""Generate consensus embeddings using pre-computed optimal rank.

Requires estimate_rank to have been run first for the dataset.

Usage:
    ./scripts/submit experiments/consensus.py dataset=nsd subject_id=1
    ./scripts/submit experiments/consensus.py dataset=swow

Outputs:
    embedding.npy          - Final consensus embedding (n_samples, rank)
    ensemble_embeddings.npy - All run embeddings (n_runs, n_samples, rank)
    pipeline.joblib        - Full fitted pipeline for later analysis
    summary.json           - Metadata and quality metrics
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from omegaconf import DictConfig
from pysrf import SRF
from pysrf.consensus import AlignedConsensus, EnsembleEmbedding
from sklearn.pipeline import Pipeline

from similarity import build_similarity

log = logging.getLogger(__name__)


def _get_path_with_fallback(
    base_dir: Path, subject_id: int | None, filename: str
) -> Path:
    """Get path with subj0X format, falling back to subject_X for legacy data."""
    if subject_id is None:
        return base_dir / filename
    primary = base_dir / f"subj{subject_id:02d}" / filename
    fallback = base_dir / f"subject_{subject_id}" / filename
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary


def _load_rank_estimation(cfg: DictConfig, subject_id: int | None) -> dict:
    """Load pre-computed rank estimation. Raises if not found."""
    rank_dir = (
        Path(cfg.project_root)
        / "outputs"
        / "experiments"
        / "estimate_rank"
        / cfg.dataset.name
    )
    path = _get_path_with_fallback(rank_dir, subject_id, "rank_estimation.json")

    if not path.exists():
        subj_str = f" subject_id={subject_id}" if subject_id else ""
        raise FileNotFoundError(
            f"Rank estimation not found at {path}\n"
            f"Run estimate_rank first:\n"
            f"  ./scripts/submit experiments/estimate_rank.py dataset={cfg.dataset.name}{subj_str}"
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
    rank_est = _load_rank_estimation(cfg, subject_id)
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
    np.save(output_dir / "ensemble_embeddings.npy", ensemble_embeddings)
    joblib.dump(pipeline, output_dir / "pipeline.joblib")

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
