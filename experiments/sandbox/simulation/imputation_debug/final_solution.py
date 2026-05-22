"""Final solution: Adaptive rho for SRF imputation.

Key finding: The optimal ADMM penalty parameter (rho) scales with the
fraction of observed data. At low retention rates, lower rho allows
the algorithm to find better solutions.

Recommended settings:
- rho = max(0.01, min(1.0, fraction_retained))
- tol = 1e-4 (allow early convergence)
- max_outer = 200 (more iterations for sparse data)
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


def get_adaptive_rho(fraction_retained: float) -> float:
    """Compute adaptive rho based on retention rate.

    Empirically optimized mapping:
    - 1-5%: rho = 0.01 (very weak constraint)
    - 5-10%: rho = 0.01-0.05 (weak constraint)
    - 10-20%: rho = 0.05-0.5 (moderate constraint)
    - 20%+: rho = 0.5-1.0 (standard constraint)
    """
    if fraction_retained <= 0.05:
        rho = 0.01
    elif fraction_retained <= 0.10:
        # Linear interpolation from 0.01 to 0.05
        rho = 0.01 + (fraction_retained - 0.05) / 0.05 * 0.04
    elif fraction_retained <= 0.20:
        # Linear interpolation from 0.05 to 0.5
        rho = 0.05 + (fraction_retained - 0.10) / 0.10 * 0.45
    else:
        # Linear interpolation from 0.5 to 1.0, capped at 1.0
        rho = min(1.0, 0.5 + (fraction_retained - 0.20) / 0.80 * 0.5)
    return rho


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

    # Median
    median_imp = SimpleImputer(strategy="median")
    median_filled = impute_symmetric(median_imp, observed, similarity)
    r2_median = float(r2_score(similarity[mask], median_filled[mask]))
    records.append({
        "method": "Median",
        "fraction_retained": fraction_retained,
        "replicate": replicate,
        "r2": r2_median,
    })

    # KNN
    knn_imp = KNNImputer()
    knn_filled = impute_symmetric(knn_imp, observed, similarity)
    r2_knn = float(r2_score(similarity[mask], knn_filled[mask]))
    records.append({
        "method": "KNN",
        "fraction_retained": fraction_retained,
        "replicate": replicate,
        "r2": r2_knn,
    })

    # SRF with adaptive rho and iterations
    adaptive_rho = get_adaptive_rho(fraction_retained)
    # More iterations for sparse data
    max_outer = 300 if fraction_retained <= 0.10 else 200
    srf = SRF(
        rank=rank,
        random_state=mask_seed + 5,
        max_outer=max_outer,
        max_inner=500,
        tol=1e-5,
        init="random_sqrt",
        verbose=0,
        bounds=(0.0, 1.0),
        missing_values=np.nan,
        rho=adaptive_rho,
    )
    srf_filled = impute_symmetric(srf, observed, similarity)
    r2_srf = float(r2_score(similarity[mask], srf_filled[mask]))
    records.append({
        "method": "SRF",
        "fraction_retained": fraction_retained,
        "replicate": replicate,
        "r2": r2_srf,
    })

    return records


def create_publication_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create publication-quality imputation comparison plot."""
    # Set publication style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    # Aggregate data
    agg = (
        df.groupby(["fraction_retained", "method"])["r2"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg["pct"] = agg["fraction_retained"] * 100

    # Colors - colorblind-friendly palette
    colors = {
        "Median": "#D55E00",  # Vermillion
        "KNN": "#0072B2",     # Blue
        "SRF": "#009E73",     # Bluish green
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 4))

    fractions = sorted(agg["pct"].unique())
    x = np.arange(len(fractions))

    # Plot each method
    for method in ["Median", "KNN", "SRF"]:
        data = agg[agg["method"] == method].sort_values("pct")
        c = colors[method]

        # Line with markers
        line = ax.plot(
            x,
            data["mean"],
            label=method,
            color=c,
            marker="o",
            lw=2.5 if method == "SRF" else 1.8,
            markersize=8 if method == "SRF" else 6,
            markerfacecolor="white",
            markeredgewidth=2 if method == "SRF" else 1.5,
            markeredgecolor=c,
            zorder=10 if method == "SRF" else 5,
        )

        # Confidence band
        ax.fill_between(
            x,
            data["mean"] - data["std"],
            data["mean"] + data["std"],
            color=c,
            alpha=0.15,
            zorder=1,
        )

    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(f)}%" for f in fractions])
    ax.set_xlabel("Fraction of observed entries", fontsize=11)
    ax.set_ylabel(r"Imputation $R^2$", fontsize=11)

    # Add horizontal line at 0
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, lw=0.8, zorder=0)

    # Legend
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="none",
        fancybox=False,
    )
    legend.get_frame().set_linewidth(0)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Grid
    ax.yaxis.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Set y-axis limits
    ax.set_ylim(-0.5, 1.05)

    plt.tight_layout()

    # Save in multiple formats
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(
            output_dir / f"imputation_r2.{ext}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )

    plt.close()
    log.info(f"Saved publication plot to {output_dir}/imputation_r2.[pdf/png/svg]")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Configuration
    n = 200
    k = 5
    kernel = "gaussian_kernel"
    seed = 42
    n_datasets = 10
    n_replicates = 5

    # Include very sparse fractions
    fractions = [0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90]

    log.info("=" * 60)
    log.info("FINAL SOLUTION: Adaptive rho SRF imputation")
    log.info("=" * 60)
    log.info(f"Datasets: {n_datasets}, Replicates: {n_replicates}")
    log.info(f"Fractions: {fractions}")

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

    # Save results
    df.to_csv(OUTPUT_DIR / "final_results.csv", index=False)

    # Print summary
    log.info("\n" + "=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("=" * 60)

    summary = df.groupby(["method", "fraction_retained"])["r2"].agg(["mean", "std"]).reset_index()
    pivot = summary.pivot(index="fraction_retained", columns="method", values="mean")
    pivot = pivot[["Median", "KNN", "SRF"]]
    print("\nMean R² by method:")
    print(pivot.round(3).to_string())

    # Show improvement
    log.info("\n" + "=" * 60)
    log.info("SRF vs Baselines (adaptive rho)")
    log.info("=" * 60)

    for frac in fractions:
        row = pivot.loc[frac]
        srf = row["SRF"]
        median = row["Median"]
        knn = row["KNN"]

        beats_median = ">" if srf > median else "<"
        beats_knn = ">" if srf > knn else "<"

        log.info(
            f"{frac:5.0%}: SRF={srf:6.3f} {beats_median} Median={median:6.3f}, "
            f"SRF {beats_knn} KNN={knn:6.3f}"
        )

    # Create publication plot
    log.info("\nCreating publication-quality plot...")
    create_publication_plot(df, OUTPUT_DIR)

    log.info(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
