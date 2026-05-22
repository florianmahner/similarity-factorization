"""Verify that lower rho fixes the SRF collapse at low retention rates.

The root cause: The default rho=3.0 is too high for sparse observations.
With few observed entries, high rho forces premature convergence to poor solutions.

The fix: Use rho=1.0 or lower (rho=0.5 seems optimal).
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
from sklearn.impute import KNNImputer, SimpleImputer
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

    if isinstance(imputer, SRF):
        imputer.fit(observed)
        imputed = imputer.reconstruct()
    else:
        imputed = imputer.fit_transform(observed)

    triu_i, triu_j = np.triu_indices(n, k=1)
    result = imputed.copy()
    result[triu_i, triu_j] = (imputed[triu_i, triu_j] + imputed[triu_j, triu_i]) / 2
    result[triu_j, triu_i] = result[triu_i, triu_j]
    np.fill_diagonal(result, np.diag(original))
    return result


def evaluate_condition(
    similarity: np.ndarray,
    fraction_retained: float,
    replicate: int,
    rank: int,
    mask_seed: int,
) -> list[dict]:
    """Evaluate all methods for one condition."""
    mask_rng = np.random.default_rng(mask_seed + replicate)
    observed, mask = mask_similarity(similarity, fraction_retained, mask_rng)

    records = []

    # Mean
    mean_imp = SimpleImputer(strategy="mean")
    mean_filled = impute_symmetric(mean_imp, observed, similarity)
    r2_mean = float(r2_score(similarity[mask], mean_filled[mask]))
    records.append(
        {
            "method": "Mean",
            "fraction_retained": fraction_retained,
            "replicate": replicate,
            "r2": r2_mean,
        }
    )

    # Median
    median_imp = SimpleImputer(strategy="median")
    median_filled = impute_symmetric(median_imp, observed, similarity)
    r2_median = float(r2_score(similarity[mask], median_filled[mask]))
    records.append(
        {
            "method": "Median",
            "fraction_retained": fraction_retained,
            "replicate": replicate,
            "r2": r2_median,
        }
    )

    # KNN
    knn_imp = KNNImputer()
    knn_filled = impute_symmetric(knn_imp, observed, similarity)
    r2_knn = float(r2_score(similarity[mask], knn_filled[mask]))
    records.append(
        {
            "method": "KNN",
            "fraction_retained": fraction_retained,
            "replicate": replicate,
            "r2": r2_knn,
        }
    )

    # SRF Original (rho=3.0, tol=0.0)
    srf_orig = SRF(
        rank=rank,
        random_state=mask_seed + 5,
        max_outer=100,
        max_inner=500,
        tol=0.0,
        init="random_sqrt",
        verbose=0,
        bounds=(0.0, 1.0),
        missing_values=np.nan,
        rho=3.0,
    )
    srf_orig_filled = impute_symmetric(srf_orig, observed, similarity)
    r2_srf_orig = float(r2_score(similarity[mask], srf_orig_filled[mask]))
    records.append(
        {
            "method": "SRF (original)",
            "fraction_retained": fraction_retained,
            "replicate": replicate,
            "r2": r2_srf_orig,
        }
    )

    # SRF Fixed (rho=1.0, tol=1e-4)
    srf_fixed = SRF(
        rank=rank,
        random_state=mask_seed + 5,
        max_outer=100,
        max_inner=500,
        tol=1e-4,
        init="random_sqrt",
        verbose=0,
        bounds=(0.0, 1.0),
        missing_values=np.nan,
        rho=1.0,
    )
    srf_fixed_filled = impute_symmetric(srf_fixed, observed, similarity)
    r2_srf_fixed = float(r2_score(similarity[mask], srf_fixed_filled[mask]))
    records.append(
        {
            "method": "SRF (fixed)",
            "fraction_retained": fraction_retained,
            "replicate": replicate,
            "r2": r2_srf_fixed,
        }
    )

    return records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Configuration
    n = 200
    k = 5
    kernel = "gaussian_kernel"
    seed = 42
    n_datasets = 5
    n_replicates = 3

    fractions = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

    log.info("=" * 60)
    log.info("VERIFICATION: Testing fixed SRF parameters")
    log.info("=" * 60)
    log.info(f"Fix: rho=1.0, tol=1e-4 (was: rho=3.0, tol=0.0)")
    log.info(f"Datasets: {n_datasets}, Replicates: {n_replicates}")

    rng = np.random.default_rng(seed)

    # Generate datasets
    log.info("\nGenerating datasets...")
    dataset_seeds = rng.integers(0, 1_000_000, size=n_datasets)
    datasets = [generate_dataset(n, k, kernel, int(s)) for s in dataset_seeds]

    # Build task list
    tasks = []
    mask_seed = int(rng.integers(0, 1_000_000))

    for dataset_idx, similarity in enumerate(datasets):
        for frac in fractions:
            for rep in range(n_replicates):
                tasks.append((similarity, frac, rep, k, mask_seed + dataset_idx * 1000))

    # Run evaluations
    log.info(f"Running {len(tasks)} evaluations...")
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_condition)(sim, frac, rep, rank, mseed)
        for sim, frac, rep, rank, mseed in tasks
    )

    # Aggregate
    records = [r for res in results for r in res]
    df = pd.DataFrame(records)

    # Compute summary statistics
    summary = (
        df.groupby(["method", "fraction_retained"])["r2"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["method", "fraction_retained", "r2_mean", "r2_std"]

    # Save results
    df.to_csv(OUTPUT_DIR / "verification_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "verification_summary.csv", index=False)

    # Print comparison
    log.info("\n" + "=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("=" * 60)

    pivot = summary.pivot(index="fraction_retained", columns="method", values="r2_mean")
    pivot = pivot[["Mean", "Median", "KNN", "SRF (original)", "SRF (fixed)"]]
    print("\nMean R² by method and retention rate:")
    print(pivot.round(3).to_string())

    # Check if SRF fixed beats baselines
    log.info("\n" + "=" * 60)
    log.info("COMPARISON: SRF (fixed) vs Baselines")
    log.info("=" * 60)

    for frac in fractions:
        row = pivot.loc[frac]
        srf_fixed = row["SRF (fixed)"]
        srf_orig = row["SRF (original)"]
        median = row["Median"]
        knn = row["KNN"]

        beats_median = "✓" if srf_fixed > median else "✗"
        beats_knn = "✓" if srf_fixed > knn else "✗"
        improvement = srf_fixed - srf_orig

        log.info(
            f"{frac:.0%}: SRF={srf_fixed:.3f} vs Median={median:.3f} {beats_median} "
            f"KNN={knn:.3f} {beats_knn} (improvement: {improvement:+.3f})"
        )

    # Create plot
    log.info("\nCreating comparison plot...")
    create_comparison_plot(df, OUTPUT_DIR)

    log.info(f"\nResults saved to {OUTPUT_DIR}")


def create_comparison_plot(df: pd.DataFrame, output_dir: Path):
    """Create visualization comparing methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: All methods
    ax1 = axes[0]
    palette = {
        "Mean": "#999999",
        "Median": "#666666",
        "KNN": "#333333",
        "SRF (original)": "#e74c3c",
        "SRF (fixed)": "#2ecc71",
    }

    for method in ["Mean", "Median", "KNN", "SRF (original)", "SRF (fixed)"]:
        method_df = df[df["method"] == method]
        summary = method_df.groupby("fraction_retained")["r2"].agg(["mean", "std"])

        ax1.plot(
            summary.index,
            summary["mean"],
            marker="o",
            label=method,
            color=palette.get(method, "#000000"),
            linewidth=2 if "SRF" in method else 1,
        )
        ax1.fill_between(
            summary.index,
            summary["mean"] - summary["std"],
            summary["mean"] + summary["std"],
            alpha=0.2,
            color=palette.get(method, "#000000"),
        )

    ax1.set_xlabel("Fraction Retained", fontsize=12)
    ax1.set_ylabel("R² Score", fontsize=12)
    ax1.set_title("Imputation Performance: Original vs Fixed SRF", fontsize=14)
    ax1.legend(loc="lower right")
    ax1.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax1.set_ylim(-2, 1.05)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Focus on SRF comparison
    ax2 = axes[1]
    for method in ["SRF (original)", "SRF (fixed)"]:
        method_df = df[df["method"] == method]
        summary = method_df.groupby("fraction_retained")["r2"].agg(["mean", "std"])

        ax2.plot(
            summary.index,
            summary["mean"],
            marker="o",
            label=method,
            color=palette.get(method),
            linewidth=2,
        )
        ax2.fill_between(
            summary.index,
            summary["mean"] - summary["std"],
            summary["mean"] + summary["std"],
            alpha=0.3,
            color=palette.get(method),
        )

    ax2.set_xlabel("Fraction Retained", fontsize=12)
    ax2.set_ylabel("R² Score", fontsize=12)
    ax2.set_title("SRF: rho=3.0 (original) vs rho=1.0 (fixed)", fontsize=14)
    ax2.legend(loc="lower right")
    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "imputation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
