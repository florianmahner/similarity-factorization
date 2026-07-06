"""Generate consensus embeddings using pre-computed optimal rank.

Uses AlignedConsensus with Hungarian alignment to produce a principled
consensus: aligns all runs via optimal permutation matching, then selects
the most central run (preserving valid factorization WW^T ~ S).

Usage:
    poetry run python experiments/datasets/consensus/run.py dataset=mur92
    poetry run python experiments/datasets/consensus/run.py dataset=nsd subject_id=1

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
from pysrf.consensus import AlignedConsensus, EnsembleFit
from sklearn.pipeline import Pipeline

from similarity import build_similarity
from tools.stats import dimension_reliability

log = logging.getLogger(__name__)

DIMENSIONALITY_DIR = (
    Path(__file__).resolve().parent.parent / "dimensionality" / "outputs"
)


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


def _load_optimal_rank(
    dataset_name: str, subject_id: int | None, kind: str = "argmin"
) -> int:
    """Load optimal rank from the dimensionality CV outputs.

    Reads ``experiments/datasets/dimensionality/outputs/<name>[/subj{NN}]/cross_validation.json``
    and returns ``payload["validations"][payload["primary_variant"]][f"{kind}_rank"]``.

    Parameters
    ----------
    dataset_name : str
        Dataset name (e.g. ``"things_behavior"``, ``"nsd"``).
    subject_id : int or None
        Subject id (for per-subject datasets like NSD).
    kind : {"argmin", "one_se"}
        Which rank to load from the primary CV variant.
    """
    name = dataset_name.replace("-", "_")
    if subject_id is not None:
        name = f"{name}_subj{subject_id:02d}"
    path = DIMENSIONALITY_DIR / name / "cross_validation.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Dimensionality CV results not found: {path}\n"
            f"Run first: poetry run python experiments/datasets/dimensionality/run.py "
            f"dataset={dataset_name}"
        )
    with open(path) as f:
        payload = json.load(f)
    primary = payload["primary_variant"]
    return int(payload["validations"][primary][f"{kind}_rank"])


def run(cfg: DictConfig) -> None:
    subject_id = cfg.get("subject_id")

    ds_name = cfg.dataset.name
    output_dir = Path.cwd() / ds_name
    if subject_id is not None:
        output_dir = output_dir / f"subj{subject_id:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rank_kind = cfg.get("rank_kind", "argmin")
    optimal_rank = _load_optimal_rank(cfg.dataset.name, subject_id, kind=rank_kind)
    log.info(f"Optimal rank ({rank_kind}): {optimal_rank}")

    log.info(f"Building similarity matrix for {cfg.dataset.name}...")
    similarity = build_similarity(cfg.dataset, subject_id=subject_id)
    n_samples = similarity.shape[0]
    log.info(f"Similarity matrix shape: ({n_samples}, {n_samples})")

    n_runs = cfg.generate.n_stable_runs
    log.info(f"Running {n_runs} SRF fits...")

    srf_kwargs = {}
    if "srf" in cfg:
        srf_kwargs["max_outer"] = cfg.srf.get("max_outer", 30)
        srf_kwargs["max_inner"] = cfg.srf.get("max_inner", 20)

    pipeline = Pipeline([
        ("ensemble", EnsembleFit(
            SRF(rank=optimal_rank, random_state=cfg.common.random_state, **srf_kwargs),
            n_runs=n_runs,
            random_state=cfg.common.random_state,
            n_jobs=cfg.common.n_jobs,
        )),
        ("consensus", AlignedConsensus(rank=optimal_rank)),
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
    naive_reliability = _compute_dimension_reliability(aligned)
    cv_reliability = dimension_reliability(aligned)
    mean_cv = float(np.mean(cv_reliability))

    log.info(f"Reconstruction RMSE: {recon_error:.4f}, r: {recon_r:.4f}")
    log.info(f"Sparsity: {sparsity:.3f}, Purity: {purity:.3f}")
    log.info(f"CV reliability: mean={mean_cv:.3f}, "
             f"min={cv_reliability.min():.3f}, max={cv_reliability.max():.3f}")

    # Save
    np.save(output_dir / "embedding.npy", embedding)
    np.save(output_dir / "runs.npy", consensus.aligned_embeddings_)
    np.save(output_dir / "reliability.npy", naive_reliability)
    np.save(output_dir / "cv_reliability.npy", cv_reliability)

    summary = {
        "dataset": cfg.dataset.name,
        "subject_id": subject_id,
        "n_samples": n_samples,
        "rank": optimal_rank,
        "rank_kind": rank_kind,
        "n_runs": n_runs,
        "selected_run_idx": int(consensus.selected_run_idx_),
        "reconstruction_rmse": recon_error,
        "reconstruction_r": recon_r,
        "sparsity": sparsity,
        "purity": purity,
        "naive_reliability_mean": float(np.mean(naive_reliability)),
        "cv_reliability_mean": mean_cv,
        "cv_reliability_min": float(cv_reliability.min()),
        "cv_reliability_max": float(cv_reliability.max()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"Saved to {output_dir}")


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path=".", config_name="config")
    def _main(cfg: DictConfig) -> None:
        run(cfg)

    _main()


if __name__ == "__main__":
    main()
