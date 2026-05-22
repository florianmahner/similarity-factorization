"""Create publication-quality simulation plots for SRF.

Three key results:
1. Factor recovery - SRF recovers ground truth factors
2. Stability - Solutions are consistent across runs
3. Reconstruction - SRF faithfully reconstructs similarity

Usage:
    poetry run python sandbox/simulation_plots/plot.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.colors import ROSE, TEAL, CYAN, GRAY_LIGHT
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

PROJECT_ROOT = Path(__file__).parents[2]
DATA_DIR = PROJECT_ROOT / "outputs/experiments/simulation/data"
OUTPUT_DIR = get_output_dir()

# Log-spaced alpha values (matches _plot_alpha_max_membership)
ALPHAS = np.logspace(-1, 1.5, 25)  # 0.1 to ~30, 25 points


def _run_single_sim(alpha: float, n: int, k: int, seed: int) -> dict:
    """Run single simulation for factor recovery."""
    from pysrf import SRF
    from scipy.optimize import linear_sum_assignment
    from utils.simulation import simulation_dirichlet

    def normalize_factors(factors):
        return factors / (factors.sum(axis=1, keepdims=True) + 1e-10)

    def align_factors(reference, target):
        ref_norm = normalize_factors(reference)
        target_norm = normalize_factors(target)
        similarity = ref_norm.T @ target_norm
        _, col_ind = linear_sum_assignment(-similarity)
        return target[:, col_ind]

    def factor_correlation(true_factors, learned):
        aligned = align_factors(true_factors, learned)
        correlations = []
        for i in range(true_factors.shape[1]):
            corr = np.corrcoef(true_factors[:, i], aligned[:, i])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        return float(np.mean(correlations)) if correlations else 0.0

    rng = np.random.default_rng(seed)
    true_factors = simulation_dirichlet(n, k, alpha, rng)
    similarity = true_factors @ true_factors.T

    srf = SRF(rank=k, random_state=seed)
    srf.fit(similarity)

    recon = srf.reconstruct()
    recon_r2 = float(np.corrcoef(similarity.flatten(), recon.flatten())[0, 1] ** 2)

    return {
        "alpha": alpha,
        "factor_corr": factor_correlation(true_factors, srf.w_),
        "recon_r2": recon_r2,
    }


def _run_stability_sim(alpha: float, n: int, k: int, n_runs: int, seed: int) -> dict:
    """Run stability simulation (cophenetic correlation)."""
    from pysrf import SRF
    from scipy.cluster.hierarchy import cophenet, linkage
    from scipy.spatial.distance import squareform
    from utils.simulation import simulation_dirichlet

    def normalize_factors(factors):
        return factors / (factors.sum(axis=1, keepdims=True) + 1e-10)

    rng = np.random.default_rng(seed)
    true_factors = simulation_dirichlet(n, k, alpha, rng)
    similarity = true_factors @ true_factors.T

    consensus_sum = np.zeros((n, n), dtype=float)
    for i in range(n_runs):
        srf = SRF(rank=k, random_state=seed + i)
        factors = srf.fit_transform(similarity)
        assignments = np.argmax(normalize_factors(factors), axis=1)
        connectivity = (assignments[:, None] == assignments[None, :]).astype(float)
        consensus_sum += connectivity

    consensus = consensus_sum / n_runs
    np.fill_diagonal(consensus, 1.0)

    consensus_dist = 1.0 - consensus
    condensed = squareform(consensus_dist, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    cophenetic_corr, _ = cophenet(linkage_matrix, condensed)

    return {"alpha": alpha, "cophenetic_corr": float(cophenetic_corr)}


def compute_all_metrics() -> pd.DataFrame:
    """Compute all metrics in parallel."""
    n, k = 64, 8
    n_reps = 50
    n_stability_runs = 20

    # Factor recovery and reconstruction
    tasks = [(alpha, n, k, rep) for alpha in ALPHAS for rep in range(n_reps)]
    results = Parallel(n_jobs=-1)(
        delayed(_run_single_sim)(alpha, n, k, seed)
        for alpha, n, k, seed in tasks
    )
    df_main = pd.DataFrame(results)

    # Stability (fewer reps, more expensive)
    stability_tasks = [(alpha, n, k, n_stability_runs, int(alpha * 1000) + rep)
                       for alpha in ALPHAS for rep in range(20)]
    stability_results = Parallel(n_jobs=-1)(
        delayed(_run_stability_sim)(alpha, n, k, n_runs, seed)
        for alpha, n, k, n_runs, seed in stability_tasks
    )
    df_stability = pd.DataFrame(stability_results)

    # Aggregate
    agg_main = df_main.groupby("alpha").agg(
        factor_corr_mean=("factor_corr", "mean"),
        factor_corr_sem=("factor_corr", "sem"),
        recon_r2_mean=("recon_r2", "mean"),
        recon_r2_sem=("recon_r2", "sem"),
    ).reset_index()

    agg_stability = df_stability.groupby("alpha").agg(
        cophenetic_mean=("cophenetic_corr", "mean"),
        cophenetic_sem=("cophenetic_corr", "sem"),
    ).reset_index()

    return agg_main.merge(agg_stability, on="alpha")


def plot_factor_recovery(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot factor recovery vs alpha."""
    fig, ax = create_figure("single")

    ax.plot(df["alpha"], df["factor_corr_mean"], "o-", color=ROSE,
            markersize=4, linewidth=1.5)
    ax.fill_between(
        df["alpha"],
        df["factor_corr_mean"] - df["factor_corr_sem"],
        df["factor_corr_mean"] + df["factor_corr_sem"],
        color=ROSE, alpha=0.2, linewidth=0,
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Factor recovery")
    ax.set_xlim(0.1, 30)
    ax.set_ylim(0, 1.02)

    despine(ax)
    save_figure(fig, output_dir / "factor_recovery.pdf")


def plot_stability(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot stability vs alpha."""
    fig, ax = create_figure("single")

    ax.plot(df["alpha"], df["cophenetic_mean"], "o-", color=TEAL,
            markersize=4, linewidth=1.5)
    ax.fill_between(
        df["alpha"],
        df["cophenetic_mean"] - df["cophenetic_sem"],
        df["cophenetic_mean"] + df["cophenetic_sem"],
        color=TEAL, alpha=0.2, linewidth=0,
    )

    ax.axhline(1.0, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Stability")
    ax.set_xlim(0.1, 30)
    ax.set_ylim(0, 1.02)

    despine(ax)
    save_figure(fig, output_dir / "stability.pdf")


def plot_reconstruction(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot reconstruction R² vs alpha."""
    fig, ax = create_figure("single")

    ax.plot(df["alpha"], df["recon_r2_mean"], "o-", color=CYAN,
            markersize=4, linewidth=1.5)
    ax.fill_between(
        df["alpha"],
        df["recon_r2_mean"] - df["recon_r2_sem"],
        df["recon_r2_mean"] + df["recon_r2_sem"],
        color=CYAN, alpha=0.2, linewidth=0,
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"Reconstruction $R^2$")
    ax.set_xlim(0.1, 30)
    ax.set_ylim(0, 1.02)

    despine(ax)
    save_figure(fig, output_dir / "reconstruction.pdf")


def plot_reconstruction_stability(df: pd.DataFrame, output_dir: Path) -> None:
    """2-panel plot: reconstruction (left) and stability (right)."""
    fig, axes = create_figure("full_width", nrows=1, ncols=2)

    # Left: Reconstruction R²
    ax = axes[0]
    ax.plot(df["alpha"], df["recon_r2_mean"], "o-", color=ROSE,
            markersize=4, linewidth=1.5)
    ax.fill_between(
        df["alpha"],
        df["recon_r2_mean"] - df["recon_r2_sem"],
        df["recon_r2_mean"] + df["recon_r2_sem"],
        color=ROSE, alpha=0.2, linewidth=0,
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"Reconstruction $R^2$")
    ax.set_xlim(0.1, 30)
    ax.set_ylim(0, 1.02)
    despine(ax)

    # Right: Stability (cophenetic correlation)
    ax = axes[1]
    ax.plot(df["alpha"], df["cophenetic_mean"], "o-", color=TEAL,
            markersize=4, linewidth=1.5)
    ax.fill_between(
        df["alpha"],
        df["cophenetic_mean"] - df["cophenetic_sem"],
        df["cophenetic_mean"] + df["cophenetic_sem"],
        color=TEAL, alpha=0.2, linewidth=0,
    )
    ax.axhline(1.0, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Cophenetic correlation")
    ax.set_xlim(0.1, 30)
    ax.set_ylim(0, 1.02)
    despine(ax)

    save_figure(fig, output_dir / "reconstruction_stability.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Computing metrics (this may take a few minutes)...")
    df = compute_all_metrics()

    # Save data
    df.to_csv(OUTPUT_DIR / "simulation_metrics.csv", index=False)
    print("  simulation_metrics.csv")

    print("Generating plots...")
    plot_reconstruction_stability(df, OUTPUT_DIR)
    print("  reconstruction_stability.pdf")

    plot_factor_recovery(df, OUTPUT_DIR)
    print("  factor_recovery.pdf")

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
