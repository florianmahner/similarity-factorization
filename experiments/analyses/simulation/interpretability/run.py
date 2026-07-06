"""Interpretability and stability analysis for SRF factors.

This task evaluates symmetric NMF (SRF) performance across varying difficulty
regimes characterized by cluster overlap and noise levels.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pysrf import SRF
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon, pdist, squareform
from scipy.stats import entropy

from tools.metrics import compute_similarity
from utils.helpers import add_noise_with_snr
from utils.simulation import simulation_dirichlet

log = logging.getLogger(__name__)


def _compute_difficulty(alpha: float) -> float:
    """Compute difficulty metric from Dirichlet concentration parameter."""
    return float(np.log(alpha + 1))


def _normalize_factors(factors: np.ndarray) -> np.ndarray:
    """Normalize factor matrix so rows sum to 1."""
    return factors / (factors.sum(axis=1, keepdims=True) + 1e-10)


def _align_factors(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align target factors to reference using Hungarian algorithm."""
    ref_norm = _normalize_factors(reference)
    target_norm = _normalize_factors(target)
    similarity = ref_norm.T @ target_norm
    _, col_ind = linear_sum_assignment(-similarity)
    return target[:, col_ind]


def _compute_consensus_metrics(
    similarity: np.ndarray, rank: int, n_runs: int, rng: np.random.Generator
) -> dict:
    """Compute consensus clustering metrics."""
    n = similarity.shape[0]
    consensus_sum = np.zeros((n, n), dtype=float)

    for _ in range(n_runs):
        seed = int(rng.integers(0, 1_000_000))
        factors = SRF(rank=rank, random_state=seed).fit_transform(similarity)
        assignments = np.argmax(_normalize_factors(factors), axis=1)
        connectivity = (assignments[:, None] == assignments[None, :]).astype(float)
        consensus_sum += connectivity

    consensus = consensus_sum / n_runs
    np.fill_diagonal(consensus, 1.0)

    dispersion = 1.0 - 4.0 * np.mean(consensus * (1.0 - consensus))
    consensus_dist = 1.0 - consensus
    condensed = squareform(consensus_dist, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    cophenetic_corr, _ = cophenet(linkage_matrix, condensed)

    return {
        "consensus": consensus,
        "dispersion": float(dispersion),
        "cophenetic_corr": float(cophenetic_corr),
    }


def _compute_interpretability_metrics(
    true_factors: np.ndarray, learned_factors: np.ndarray, similarity: np.ndarray
) -> dict:
    """Compute all interpretability metrics."""
    aligned = _align_factors(true_factors, learned_factors)

    # Divergence metrics
    kl_scores = [
        float(np.sum(jensenshannon(true_factors[i], aligned[i], base=2) ** 2))
        for i in range(len(true_factors))
    ]

    # Entropy and sparsity
    entropy_learned = np.array([entropy(row) for row in aligned])
    sparsity_learned = np.mean(aligned < 0.01, axis=1)

    # Reconstruction
    reconstructed = learned_factors @ learned_factors.T
    r2 = float(np.corrcoef(similarity.flatten(), reconstructed.flatten())[0, 1] ** 2)

    return {
        "kl_mean": float(np.mean(kl_scores)),
        "entropy_mean": float(np.mean(entropy_learned)),
        "entropy_std": float(np.std(entropy_learned)),
        "sparsity_mean": float(np.mean(sparsity_learned)),
        "reconstruction_r2": r2,
        "aligned_factors": aligned,
    }


def _run_single_condition(
    alpha: float, snr: float, cfg: DictConfig, rng: np.random.Generator
) -> dict:
    """Run analysis for a single (alpha, SNR) condition."""
    # Generate data
    true_factors = simulation_dirichlet(n=cfg.n, k=cfg.k, alpha=alpha, rng=rng)

    if cfg.kernel == "linear":
        base_similarity = true_factors @ true_factors.T
        base_similarity = (base_similarity + base_similarity.T) * 0.5
    else:
        base_similarity = compute_similarity(
            true_factors, true_factors, metric=cfg.kernel
        )

    similarity = add_noise_with_snr(base_similarity, snr, rng)
    similarity = (similarity + similarity.T) * 0.5

    # Fit SRF
    learned_factors = SRF(
        rank=cfg.k, random_state=int(rng.integers(0, 1_000_000))
    ).fit_transform(similarity)

    # Compute metrics
    interp_metrics = _compute_interpretability_metrics(
        true_factors, learned_factors, similarity
    )
    consensus_metrics = _compute_consensus_metrics(similarity, cfg.k, cfg.ccc_runs, rng)

    return {
        "alpha": alpha,
        "snr": snr,
        "difficulty": _compute_difficulty(alpha),
        "cophenetic_corr": consensus_metrics["cophenetic_corr"],
        "consensus_dispersion": consensus_metrics["dispersion"],
        **{k: v for k, v in interp_metrics.items() if k != "aligned_factors"},
    }


def run(cfg: DictConfig) -> None:
    """Run interpretability analysis task."""
    log.info(f"Running interpretability analysis: n={cfg.n}, k={cfg.k}")
    log.info(f"Alpha grid: {list(cfg.alpha_grid)}")
    log.info(f"SNR values: {list(cfg.snrs)}")

    output_dir = Path(cfg.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    log.info("Running parameter sweep...")
    records = []
    for alpha in cfg.alpha_grid:
        for snr in cfg.snrs:
            record = _run_single_condition(alpha, snr, cfg, rng)
            records.append(record)
            log.info(
                f"  alpha={alpha:.1f}, snr={snr:.1f}: cophenetic_corr={record['cophenetic_corr']:.3f}"
            )

    df = pd.DataFrame(records)
    df["cluster_separation"] = df["alpha"].apply(lambda x: "low" if x < 1 else "high")
    df["cluster_overlap"] = df["difficulty"]

    csv_path = output_dir / "interpretability.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Saved {csv_path}")


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path=".", config_name="config")
    def _main(cfg: DictConfig) -> None:
        run(cfg)

    _main()


if __name__ == "__main__":
    main()
