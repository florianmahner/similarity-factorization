"""Publication-quality figure for SRF imputation superiority.

Principled demonstration:
- Use n=1000 so that even 1% retention provides sufficient observations
- Multiple independent datasets to show robustness
- Clean, Nature-quality visualization
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.utils import get_output_dir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    """Empirically optimized rho based on retention rate."""
    if fraction_retained <= 0.05:
        rho = 0.01
    elif fraction_retained <= 0.10:
        rho = 0.01 + (fraction_retained - 0.05) / 0.05 * 0.04
    elif fraction_retained <= 0.20:
        rho = 0.05 + (fraction_retained - 0.10) / 0.10 * 0.45
    elif fraction_retained <= 0.50:
        rho = 0.5 + (fraction_retained - 0.20) / 0.30 * 1.5  # Up to 2.0
    else:
        rho = 2.0 + (fraction_retained - 0.50) / 0.50 * 1.0  # Up to 3.0
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

    # SRF with adaptive rho
    adaptive_rho = get_adaptive_rho(fraction_retained)
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


def create_publication_figure(df: pd.DataFrame, output_dir: Path) -> None:
    """Create Nature-quality single panel figure."""

    # Aggregate data
    agg = (
        df.groupby(["fraction_retained", "method"])["r2"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["pct"] = agg["fraction_retained"] * 100
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])

    # Set up publication style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
    })

    # Colorblind-friendly, publication-quality colors
    colors = {
        "Median": "#CC79A7",  # Reddish purple
        "KNN": "#56B4E9",     # Sky blue
        "SRF": "#009E73",     # Bluish green
    }

    markers = {
        "Median": "s",  # Square
        "KNN": "^",     # Triangle
        "SRF": "o",     # Circle
    }

    # Create figure - single panel, Nature-style dimensions
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    fractions = sorted(agg["pct"].unique())
    x = np.arange(len(fractions))

    # Plot each method
    for method in ["Median", "KNN", "SRF"]:
        data = agg[agg["method"] == method].sort_values("pct")
        c = colors[method]
        m = markers[method]

        # Emphasize SRF
        lw = 2.0 if method == "SRF" else 1.2
        ms = 7 if method == "SRF" else 5
        zorder = 10 if method == "SRF" else 5

        # Plot line with markers
        ax.plot(
            x,
            data["mean"],
            label=method,
            color=c,
            marker=m,
            lw=lw,
            markersize=ms,
            markerfacecolor="white",
            markeredgewidth=1.5 if method == "SRF" else 1.0,
            markeredgecolor=c,
            zorder=zorder,
        )

        # Error band (SEM)
        ax.fill_between(
            x,
            data["mean"] - data["sem"],
            data["mean"] + data["sem"],
            color=c,
            alpha=0.15,
            zorder=1,
            linewidth=0,
        )

    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(f)}" for f in fractions])
    ax.set_xlabel("Observed entries (%)", fontsize=10)
    ax.set_ylabel(r"Imputation $R^2$", fontsize=10)

    # Reference line at 0
    ax.axhline(y=0, color="#888888", linestyle="--", alpha=0.5, lw=0.6, zorder=0)

    # Legend - inside plot, lower right
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="none",
        handlelength=1.5,
        handletextpad=0.5,
    )
    legend.get_frame().set_linewidth(0)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    # Subtle grid
    ax.yaxis.grid(True, alpha=0.3, linestyle="-", linewidth=0.4, color="#cccccc")
    ax.set_axisbelow(True)

    # Y-axis limits
    ax.set_ylim(-0.3, 1.05)

    plt.tight_layout()

    # Save in multiple formats
    for ext in ["pdf", "png", "svg"]:
        dpi = 600 if ext == "png" else None
        fig.savefig(
            output_dir / f"imputation_comparison.{ext}",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            transparent=False,
        )

    plt.close()
    log.info(f"Saved publication figure to {output_dir}/imputation_comparison.[pdf/png/svg]")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Configuration - larger n for principled demonstration
    # n=2000 gives 2M upper triangle entries, so 1% = 20,000 observations
    # vs 10,000 DoF (n*k) = 2x overdetermined even at 1%
    n = 2000
    k = 5
    kernel = "gaussian_kernel"
    seed = 42
    n_datasets = 3  # Fewer datasets due to larger n
    n_replicates = 3

    # Retention fractions including very sparse
    fractions = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80]

    log.info("=" * 60)
    log.info("PUBLICATION FIGURE: SRF Imputation Superiority")
    log.info("=" * 60)
    log.info(f"Matrix size: {n}x{n}")
    log.info(f"True rank: {k}")
    log.info(f"Upper triangle entries: {n*(n-1)//2:,}")
    log.info(f"At 1% retention: {int(0.01 * n*(n-1)//2):,} observations")
    log.info(f"Degrees of freedom: {n*k:,}")
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

    # Save results
    df.to_csv(OUTPUT_DIR / "publication_results.csv", index=False)

    # Print summary
    log.info("\n" + "=" * 60)
    log.info("RESULTS")
    log.info("=" * 60)

    summary = df.groupby(["method", "fraction_retained"])["r2"].agg(["mean", "std"]).reset_index()
    pivot = summary.pivot(index="fraction_retained", columns="method", values="mean")
    pivot = pivot[["Median", "KNN", "SRF"]]

    print("\nMean R² by method:")
    print(pivot.round(3).to_string())

    # Check SRF superiority
    log.info("\n" + "=" * 60)
    log.info("SRF SUPERIORITY CHECK")
    log.info("=" * 60)

    all_better = True
    for frac in fractions:
        row = pivot.loc[frac]
        srf_better_median = row["SRF"] > row["Median"]
        srf_better_knn = row["SRF"] > row["KNN"]
        status = "✓" if (srf_better_median and srf_better_knn) else "✗"
        all_better = all_better and srf_better_median and srf_better_knn
        log.info(f"{frac:5.0%}: SRF={row['SRF']:.3f} vs Median={row['Median']:.3f}, KNN={row['KNN']:.3f} {status}")

    if all_better:
        log.info("\n✓ SRF outperforms all baselines at every retention rate!")
    else:
        log.info("\n✗ SRF does not outperform at all retention rates")

    # Create publication figure
    log.info("\nCreating publication figure...")
    create_publication_figure(df, OUTPUT_DIR)

    log.info(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
