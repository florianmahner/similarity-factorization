"""
Plot observed vs predicted RSM comparison.

Compare the original similarity matrix with the SRF reconstruction (W @ W.T).

Usage:
    ./scripts/submit experiments/plot_rsm.py dataset=things_behavior
    ./scripts/submit experiments/plot_rsm.py dataset=nsd subject_id=1
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from similarity import build_similarity
from src.tools.rsa import correlate_rsms
from src.colors import setup_style

log = logging.getLogger(__name__)


def _get_consensus_path(cfg: DictConfig, subject_id: int | None) -> Path:
    """Get path to consensus outputs."""
    consensus_dir = (
        Path(cfg.project_root) / "outputs" / "experiments" / "consensus" / cfg.dataset.name
    )
    if subject_id is None:
        return consensus_dir
    primary = consensus_dir / f"subj{subject_id:02d}"
    fallback = consensus_dir / f"subject_{subject_id}"
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary


def _load_embedding(consensus_path: Path) -> np.ndarray:
    """Load consensus embedding."""
    embedding_path = consensus_path / "embedding.npy"
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding not found: {embedding_path}")
    return np.load(embedding_path)


def _plot_rsm_comparison(
    observed: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
    dataset_name: str | None = None,
) -> float:
    """Plot observed vs predicted RSM side by side."""
    setup_style()

    # Zero out diagonals for display
    obs = observed.copy()
    pred = predicted.copy()
    np.fill_diagonal(obs, np.nan)
    np.fill_diagonal(pred, np.nan)

    # Compute correlation on upper triangular (excluding diagonal)
    r = correlate_rsms(observed, predicted)

    # Shared color limits (excluding diagonal)
    vmin = np.nanmin([obs, pred])
    vmax = np.nanmax([obs, pred])

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    im0 = axes[0].imshow(obs, cmap="Blues", vmin=vmin, vmax=vmax)
    axes[0].set_title("Observed", fontsize=11)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for spine in axes[0].spines.values():
        spine.set_visible(False)

    im1 = axes[1].imshow(pred, cmap="Blues", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Predicted (r = {r:.3f})", fontsize=11)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    for spine in axes[1].spines.values():
        spine.set_visible(False)

    # Colorbar on the far right
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("Similarity", fontsize=10)

    # Dataset title
    if dataset_name:
        title = dataset_name.replace("-", " ").replace("_", " ").title()
        fig.suptitle(title, fontsize=13, y=1.02)

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return r


def run(cfg: DictConfig) -> None:
    """Plot observed vs predicted RSM comparison."""
    subject_id = cfg.get("subject_id")
    output_dir = Path.cwd()

    # Load consensus embedding
    consensus_path = _get_consensus_path(cfg, subject_id)
    log.info(f"Loading embedding from {consensus_path}")
    embedding = _load_embedding(consensus_path)
    log.info(f"Embedding shape: {embedding.shape}")

    # Build observed similarity matrix
    log.info(f"Building similarity matrix for {cfg.dataset.name}...")
    observed = build_similarity(cfg.dataset, subject_id=subject_id)
    log.info(f"Observed RSM shape: {observed.shape}")

    # Compute predicted RSM
    predicted = embedding @ embedding.T

    # Plot comparison
    log.info("Plotting RSM comparison...")
    r = _plot_rsm_comparison(
        observed, predicted, output_dir / "rsm_comparison.png", dataset_name=cfg.dataset.name
    )
    log.info(f"RSM correlation: r = {r:.4f}")

    log.info(f"Results saved to {output_dir}")
