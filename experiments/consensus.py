from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from pysrf import SRF, cross_val_score
from pysrf.bounds import estimate_sampling_bounds_ultra
from pysrf.consensus import AlignedConsensus, EnsembleEmbedding
from sklearn.pipeline import Pipeline

from similarity import build_similarity, get_rank_grid
from src.utils.figure_theme import CMAP, GRAY, create_figure, despine, save_figure

log = logging.getLogger(__name__)


def _load_bounds(bounds_path: Path) -> tuple[float, float]:
    """Load pre-computed bounds from JSON file."""
    with open(bounds_path) as f:
        bounds = json.load(f)
    return bounds["pmin"], bounds["pmax"]


def _get_sampling_fraction(pmin: float, pmax: float, selection: str = "mean") -> float:
    """Select sampling fraction from bounds."""
    options = {"min": pmin, "mean": 0.5 * (pmin + pmax), "max": pmax}
    if selection not in options:
        raise ValueError(f"sampling_selection must be one of {list(options.keys())}")
    return options[selection]


def _get_bounds_path(cfg: DictConfig, subject_id: int | None) -> Path:
    """Derive bounds path from dataset config using convention."""
    bounds_task = cfg.dataset.get("bounds_task", cfg.dataset.name)
    bounds_dir = (
        Path(cfg.project_root) / "outputs" / "experiments" / "bounds" / bounds_task
    )
    if subject_id is not None:
        new_path = bounds_dir / f"subject_{subject_id}" / "bounds.json"
        old_path = bounds_dir / f"subj{subject_id:02d}" / "bounds.json"
        if new_path.exists():
            return new_path
        if old_path.exists():
            return old_path
        return new_path
    return bounds_dir / "bounds.json"


def _get_rank_estimation_path(cfg: DictConfig, subject_id: int | None) -> Path:
    """Get path to pre-computed rank estimation."""
    rank_dir = Path(cfg.project_root) / "outputs" / "experiments" / "estimate_rank" / cfg.dataset.name
    if subject_id is not None:
        return rank_dir / f"subject_{subject_id}" / "rank_estimation.json"
    return rank_dir / "rank_estimation.json"


def _load_rank_estimation(path: Path) -> dict:
    """Load pre-computed rank estimation results."""
    with open(path) as f:
        return json.load(f)


def _plot_cv_results(
    output_dir: Path, cv_scores: dict[int, dict], optimal_rank: int
) -> None:
    """Plot cross-validation results."""
    fig, ax = create_figure("single")

    ranks = sorted(cv_scores.keys())
    means = [cv_scores[r]["mean"] for r in ranks]
    stds = [cv_scores[r]["std"] for r in ranks]

    ax.errorbar(
        ranks, means, yerr=stds, fmt="o-", color=CMAP[1], capsize=3, markersize=4
    )
    ax.axvline(
        optimal_rank,
        color=CMAP[0],
        linestyle="--",
        linewidth=1.5,
        label=f"Optimal: {optimal_rank}",
    )
    ax.set_xlabel("Rank")
    ax.set_ylabel("CV Score (MSE)")
    ax.legend(fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "cv_results.pdf")
    plt.close(fig)


def run(cfg: DictConfig) -> None:
    """Run consensus embedding generation."""
    subject_id = cfg.get("subject_id")
    output_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load similarity matrix
    log.info(f"Building similarity matrix for {cfg.dataset.name}...")
    similarity = build_similarity(cfg.dataset, subject_id=subject_id)
    log.info(f"Similarity matrix shape: {similarity.shape}")

    # Get rank grid
    rank_grid = get_rank_grid(cfg.dataset)
    log.info(f"Rank grid: {rank_grid}")

    # Load or estimate bounds
    bounds_file = _get_bounds_path(cfg, subject_id)
    if bounds_file.exists():
        log.info(f"Loading pre-computed bounds from {bounds_file}")
        pmin, pmax = _load_bounds(bounds_file)
    else:
        log.info(f"Bounds not found at {bounds_file}, estimating...")
        pmin, pmax, _ = estimate_sampling_bounds_ultra(
            similarity,
            random_state=cfg.common.random_state,
            verbose=True,
        )

    sampling_selection = cfg.generate.get("sampling_selection", "mean")
    sampling_fraction = _get_sampling_fraction(pmin, pmax, sampling_selection)
    log.info(f"Bounds: pmin={pmin:.4f}, pmax={pmax:.4f}")
    log.info(
        f"Using sampling_selection='{sampling_selection}' -> fraction={sampling_fraction:.4f}"
    )

    # Load pre-computed rank or run CV
    rank_estimation_path = _get_rank_estimation_path(cfg, subject_id)
    if rank_estimation_path.exists():
        log.info(f"Loading pre-computed rank from {rank_estimation_path}")
        rank_est = _load_rank_estimation(rank_estimation_path)
        optimal_rank = rank_est["optimal_rank"]
        cv_scores = rank_est.get("cv_scores", {})
        best_cv_score = rank_est.get("best_cv_score", 0.0)
        log.info(f"Optimal rank: {optimal_rank} (from rank estimation)")
    else:
        log.info(f"No rank estimation found, running CV with {cfg.generate.n_cv_repeats} repeats...")
        cv_result = cross_val_score(
            similarity,
            param_grid={"rank": rank_grid},
            n_repeats=cfg.generate.n_cv_repeats,
            sampling_fraction=sampling_fraction,
            estimate_sampling_fraction=False,
            fit_final_estimator=False,
            random_state=cfg.common.random_state,
            n_jobs=cfg.common.n_jobs,
            verbose=1,
        )
        optimal_rank = cv_result.best_params_["rank"]
        best_cv_score = cv_result.best_score_
        cv_results_df = cv_result.cv_results_
        cv_scores = {
            int(rank): {"mean": float(group["score"].mean()), "std": float(group["score"].std())}
            for rank, group in cv_results_df.groupby("rank")
        }
        log.info(f"Optimal rank: {optimal_rank} (CV score: {best_cv_score:.6f})")

    # Fit consensus embedding
    log.info(f"Running consensus embedding with {cfg.generate.n_stable_runs} runs...")
    pipeline = Pipeline(
        [
            (
                "ensemble",
                EnsembleEmbedding(
                    SRF(rank=optimal_rank, random_state=cfg.common.random_state),
                    n_runs=cfg.generate.n_stable_runs,
                    random_state=cfg.common.random_state,
                    n_jobs=cfg.common.n_jobs,
                ),
            ),
            (
                "consensus",
                AlignedConsensus(
                    rank=optimal_rank,
                    aggregation="select",
                ),
            ),
        ]
    )
    pipeline.fit(similarity)
    embedding = pipeline.transform(similarity)

    consensus = pipeline.named_steps["consensus"]
    log.info(f"Consensus embedding shape: {embedding.shape}")
    log.info(f"Selected run index: {consensus.selected_run_idx_}")

    # Compute quality metrics
    recon = embedding @ embedding.T
    recon_error = np.linalg.norm(similarity - recon, "fro") / np.linalg.norm(
        similarity, "fro"
    )
    purity = (embedding.max(axis=1) / embedding.sum(axis=1)).mean()
    log.info(f"Reconstruction error: {recon_error:.4f}")
    log.info(f"Mean purity: {purity:.3f}")

    # Save outputs
    np.save(output_dir / "embedding.npy", embedding)
    joblib.dump(pipeline, output_dir / "pipeline.joblib")

    summary = {
        "dataset": cfg.dataset.get("name", "unknown"),
        "n_samples": int(embedding.shape[0]),
        "optimal_rank": int(optimal_rank),
        "best_cv_score": float(best_cv_score),
        "reconstruction_error": float(recon_error),
        "mean_purity": float(purity),
        "selected_run_idx": int(consensus.selected_run_idx_),
        "agreement_mean": float(consensus.agreement_scores_.mean()),
        "agreement_std": float(consensus.agreement_scores_.std()),
        "centrality_mean": float(consensus.centrality_scores_.mean()),
        "centrality_std": float(consensus.centrality_scores_.std()),
        "n_runs": int(cfg.generate.n_stable_runs),
        "pmin": float(pmin),
        "pmax": float(pmax),
        "sampling_fraction": float(sampling_fraction),
        "cv_scores": cv_scores,
    }
    if subject_id is not None:
        summary["subject_id"] = int(subject_id)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if cv_scores:
        _plot_cv_results(output_dir, cv_scores, optimal_rank)

    log.info(f"Results saved to {output_dir}")
