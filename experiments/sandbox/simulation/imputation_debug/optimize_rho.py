"""Find optimal rho for different retention rates.

The goal is to understand the rho-retention relationship and find
the best settings for each sparsity level.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.utils import get_output_dir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from pysrf import SRF
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from tools.metrics import compute_similarity
from utils.simulation import simulation_dirichlet

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()


def generate_dataset(n: int, k: int, kernel: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = simulation_dirichlet(n=n, k=k, rng=rng, alpha=1.0)
    similarity = compute_similarity(w, w, kernel)
    return similarity


def mask_similarity(
    matrix: np.ndarray, fraction_retained: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n = matrix.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)
    total = triu_i.shape[0]
    n_missing = int((1 - fraction_retained) * total)

    mask = np.zeros_like(matrix, dtype=bool)
    if n_missing > 0:
        choice = rng.choice(total, size=n_missing, replace=False)
        mask[triu_i[choice], triu_j[choice]] = True
        mask[triu_j[choice], triu_i[choice]] = True

    observed = matrix.copy()
    observed[mask] = np.nan
    np.fill_diagonal(observed, matrix.diagonal())
    return observed, mask


def impute_symmetric(imputer, observed: np.ndarray, original: np.ndarray) -> np.ndarray:
    n = observed.shape[0]

    imputer.fit(observed)
    imputed = imputer.reconstruct()

    triu_i, triu_j = np.triu_indices(n, k=1)
    result = imputed.copy()
    result[triu_i, triu_j] = (imputed[triu_i, triu_j] + imputed[triu_j, triu_i]) / 2
    result[triu_j, triu_i] = result[triu_i, triu_j]
    np.fill_diagonal(result, np.diag(original))
    return result


def evaluate_rho(
    similarity: np.ndarray,
    fraction_retained: float,
    rho: float,
    rank: int,
    seed: int,
) -> dict:
    """Evaluate SRF with specific rho."""
    mask_rng = np.random.default_rng(seed)
    observed, mask = mask_similarity(similarity, fraction_retained, mask_rng)

    srf = SRF(
        rank=rank,
        random_state=seed + 5,
        max_outer=150,
        max_inner=500,
        tol=1e-4,
        init="random_sqrt",
        verbose=0,
        bounds=(0.0, 1.0),
        missing_values=np.nan,
        rho=rho,
    )

    try:
        filled = impute_symmetric(srf, observed, similarity)
        r2 = float(r2_score(similarity[mask], filled[mask]))
        n_iter = srf.n_iter_
    except Exception as e:
        r2 = np.nan
        n_iter = -1

    return {
        "fraction_retained": fraction_retained,
        "rho": rho,
        "r2": r2,
        "n_iter": n_iter,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n = 200
    k = 5
    kernel = "gaussian_kernel"
    seed = 42

    # Test rho values
    rho_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    fractions = [0.05, 0.10, 0.15, 0.20, 0.40]
    n_replicates = 5

    log.info("=" * 60)
    log.info("RHO OPTIMIZATION: Finding best rho for each retention rate")
    log.info("=" * 60)

    rng = np.random.default_rng(seed)

    # Generate dataset
    log.info("Generating dataset...")
    similarity = generate_dataset(n, k, kernel, seed)

    # Build tasks
    tasks = []
    for frac in fractions:
        for rho in rho_values:
            for rep in range(n_replicates):
                tasks.append((similarity, frac, rho, k, seed + rep * 100))

    # Run evaluations
    log.info(f"Running {len(tasks)} evaluations...")
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_rho)(sim, frac, rho, rank, s)
        for sim, frac, rho, rank, s in tasks
    )

    df = pd.DataFrame(results)

    # Find optimal rho for each fraction
    summary = df.groupby(["fraction_retained", "rho"])["r2"].agg(["mean", "std"]).reset_index()
    summary.columns = ["fraction_retained", "rho", "r2_mean", "r2_std"]

    # Get best rho for each fraction
    log.info("\n" + "=" * 60)
    log.info("OPTIMAL RHO BY RETENTION RATE")
    log.info("=" * 60)

    best_rho = summary.loc[summary.groupby("fraction_retained")["r2_mean"].idxmax()]
    print(best_rho[["fraction_retained", "rho", "r2_mean", "r2_std"]].to_string(index=False))

    # Create heatmap
    log.info("\nCreating visualizations...")
    pivot = summary.pivot(index="rho", columns="fraction_retained", values="r2_mean")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap
    ax1 = axes[0]
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        ax=ax1,
        cbar_kws={"label": "R² Score"},
    )
    ax1.set_title("R² Score by rho and Retention Rate", fontsize=12)
    ax1.set_xlabel("Fraction Retained", fontsize=11)
    ax1.set_ylabel("rho", fontsize=11)

    # Line plot
    ax2 = axes[1]
    for frac in fractions:
        frac_data = summary[summary["fraction_retained"] == frac]
        ax2.plot(frac_data["rho"], frac_data["r2_mean"], marker="o", label=f"{frac:.0%}")
        ax2.fill_between(
            frac_data["rho"],
            frac_data["r2_mean"] - frac_data["r2_std"],
            frac_data["r2_mean"] + frac_data["r2_std"],
            alpha=0.2,
        )

    ax2.set_xscale("log")
    ax2.set_xlabel("rho (log scale)", fontsize=11)
    ax2.set_ylabel("R² Score", fontsize=11)
    ax2.set_title("R² vs rho for Different Retention Rates", fontsize=12)
    ax2.legend(title="Retention")
    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rho_optimization.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save data
    summary.to_csv(OUTPUT_DIR / "rho_optimization_results.csv", index=False)

    log.info(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
