"""Information-theoretic framing for imputation comparison.

Key insight: Plot against "observations per degree of freedom" rather than
raw percentage. This:
1. Normalizes across different matrix sizes
2. Shows the theoretical minimum (ratio = 1)
3. Provides principled, interpretable results

Information-theoretic minimum:
- Parameters to estimate: n × k (embedding matrix W)
- Minimum observations needed: n × k
- Ratio = observations / (n × k) must be ≥ 1 for identifiability
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


def get_adaptive_rho(obs_per_dof: float) -> float:
    """Adaptive rho based on observations per degree of freedom."""
    if obs_per_dof <= 1.0:
        rho = 0.01
    elif obs_per_dof <= 2.0:
        rho = 0.01 + (obs_per_dof - 1.0) * 0.04
    elif obs_per_dof <= 5.0:
        rho = 0.05 + (obs_per_dof - 2.0) / 3.0 * 0.45
    else:
        rho = min(3.0, 0.5 + (obs_per_dof - 5.0) / 10.0 * 2.5)
    return rho


def evaluate_condition(
    similarity: np.ndarray,
    n: int,
    k: int,
    fraction_retained: float,
    replicate: int,
    mask_seed: int,
) -> list[dict]:
    """Evaluate all methods for one condition."""
    mask_rng = np.random.default_rng(mask_seed + replicate)
    observed, mask = mask_similarity(similarity, fraction_retained, mask_rng)

    # Compute information-theoretic quantities
    n_upper_triangle = n * (n - 1) // 2
    n_observed = int(fraction_retained * n_upper_triangle)
    dof = n * k
    obs_per_dof = n_observed / dof

    records = []
    base_record = {
        "n": n,
        "k": k,
        "fraction_retained": fraction_retained,
        "n_observed": n_observed,
        "dof": dof,
        "obs_per_dof": obs_per_dof,
        "replicate": replicate,
    }

    # Median
    median_imp = SimpleImputer(strategy="median")
    median_filled = impute_symmetric(median_imp, observed, similarity)
    r2_median = float(r2_score(similarity[mask], median_filled[mask]))
    records.append({**base_record, "method": "Median", "r2": r2_median})

    # KNN
    knn_imp = KNNImputer()
    knn_filled = impute_symmetric(knn_imp, observed, similarity)
    r2_knn = float(r2_score(similarity[mask], knn_filled[mask]))
    records.append({**base_record, "method": "KNN", "r2": r2_knn})

    # SRF with adaptive rho
    adaptive_rho = get_adaptive_rho(obs_per_dof)
    max_outer = 300 if obs_per_dof <= 2.0 else 150
    srf = SRF(
        rank=k,
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
    records.append({**base_record, "method": "SRF", "r2": r2_srf})

    return records


def create_information_theoretic_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create publication-quality plot with information-theoretic framing."""

    # Filter to only well-posed problems (ratio >= 1)
    df_filtered = df[df["obs_per_dof"] >= 1.0].copy()

    # Bin the obs_per_dof ratios to reduce noise and smooth the curves
    # Use logarithmic bins (9 edges -> 8 labels)
    bins = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0]
    df_filtered["ratio_bin"] = pd.cut(
        df_filtered["obs_per_dof"],
        bins=bins,
        labels=[1.25, 1.75, 2.5, 4.0, 6.5, 10.0, 16.0, 25.0],
    )
    df_filtered["ratio_bin"] = df_filtered["ratio_bin"].astype(float)

    # Aggregate by bins
    agg = (
        df_filtered.groupby(["ratio_bin", "method"])["r2"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    agg = agg.dropna()

    # Publication style - Nature-quality
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 8,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
    })

    # Colorblind-friendly palette (Wong, 2011)
    colors = {
        "Median": "#E69F00",  # Orange
        "KNN": "#56B4E9",     # Sky blue
        "SRF": "#009E73",     # Bluish green
    }

    markers = {"Median": "s", "KNN": "^", "SRF": "o"}

    # Single column figure for Nature (89mm = 3.5in)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Plot each method
    for method in ["Median", "KNN", "SRF"]:
        data = agg[agg["method"] == method].sort_values("ratio_bin")
        c = colors[method]
        m = markers[method]

        lw = 1.8 if method == "SRF" else 1.0
        ms = 5 if method == "SRF" else 4
        zorder = 10 if method == "SRF" else 5

        ax.plot(
            data["ratio_bin"],
            data["mean"],
            label=method,
            color=c,
            marker=m,
            lw=lw,
            markersize=ms,
            markerfacecolor="white",
            markeredgewidth=1.2 if method == "SRF" else 0.8,
            markeredgecolor=c,
            zorder=zorder,
        )

        ax.fill_between(
            data["ratio_bin"],
            data["mean"] - data["sem"],
            data["mean"] + data["sem"],
            color=c,
            alpha=0.2,
            zorder=1,
            linewidth=0,
        )

    # Styling
    ax.set_xscale("log")
    ax.set_xlabel("Sampling ratio (observations / degrees of freedom)", fontsize=9)
    ax.set_ylabel(r"Reconstruction $R^2$", fontsize=9)

    ax.axhline(y=0, color="#888888", linestyle="-", alpha=0.4, lw=0.4, zorder=0)

    # Legend
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="none",
        handlelength=1.5,
        handletextpad=0.4,
        borderpad=0.3,
    )
    legend.get_frame().set_linewidth(0)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")

    # Subtle grid
    ax.yaxis.grid(True, alpha=0.25, linestyle="-", linewidth=0.3, color="#aaaaaa")
    ax.set_axisbelow(True)

    # Axis limits - start from 1
    ax.set_xlim(1.0, 25)
    ax.set_ylim(-0.05, 1.02)

    # Custom x-ticks
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels(["1", "2", "5", "10", "20"])

    plt.tight_layout()

    # Save in multiple formats
    for ext in ["pdf", "png", "svg"]:
        dpi = 600 if ext == "png" else None
        fig.savefig(
            output_dir / f"imputation_information_theoretic.{ext}",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )

    plt.close()
    log.info(f"Saved plot to {output_dir}/imputation_information_theoretic.[pdf/png/svg]")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    k = 5
    kernel = "gaussian_kernel"
    seed = 42
    n_replicates = 3

    # Multiple matrix sizes to combine on same x-axis
    configurations = [
        # (n, fractions) - chosen to give diverse obs/dof ratios
        (200, [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]),
        (500, [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]),
        (800, [0.01, 0.02, 0.03, 0.05, 0.10]),
    ]

    log.info("=" * 60)
    log.info("INFORMATION-THEORETIC IMPUTATION COMPARISON")
    log.info("=" * 60)
    log.info(f"True rank k = {k}")
    log.info(f"Degrees of freedom = n × k")
    log.info(f"Theoretical minimum: obs/DoF ≥ 1")
    log.info("")

    # Print info for each configuration
    for n, fracs in configurations:
        dof = n * k
        n_upper = n * (n - 1) // 2
        log.info(f"n={n}: DoF={dof}, upper_tri={n_upper:,}")
        for f in fracs:
            obs = int(f * n_upper)
            ratio = obs / dof
            log.info(f"  {f:.0%} retained: {obs:,} obs, ratio={ratio:.2f}")

    rng = np.random.default_rng(seed)

    all_results = []

    for n, fractions in configurations:
        log.info(f"\nProcessing n={n}...")

        # Generate datasets for this n
        n_datasets = 5
        dataset_seeds = rng.integers(0, 1_000_000, size=n_datasets)
        datasets = [generate_dataset(n, k, kernel, int(s)) for s in dataset_seeds]

        # Build tasks
        tasks = []
        mask_seed = int(rng.integers(0, 1_000_000))

        for dataset_idx, similarity in enumerate(datasets):
            for frac in fractions:
                for rep in range(n_replicates):
                    tasks.append((
                        similarity, n, k, frac, rep,
                        mask_seed + dataset_idx * 1000
                    ))

        # Run
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_condition)(sim, n, k, frac, rep, mseed)
            for sim, n, k, frac, rep, mseed in tasks
        )

        records = [r for res in results for r in res]
        all_results.extend(records)

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR / "information_theoretic_results.csv", index=False)

    # Summary
    log.info("\n" + "=" * 60)
    log.info("RESULTS BY OBS/DOF RATIO")
    log.info("=" * 60)

    summary = (
        df.groupby(["obs_per_dof", "method"])["r2"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary = summary.sort_values(["obs_per_dof", "method"])

    # Pivot for display
    pivot = summary.pivot(index="obs_per_dof", columns="method", values="mean")
    pivot = pivot[["Median", "KNN", "SRF"]]
    print("\nMean R² by method (sorted by obs/DoF ratio):")
    print(pivot.round(3).to_string())

    # Check SRF superiority
    log.info("\n" + "=" * 60)
    log.info("SRF SUPERIORITY CHECK")
    log.info("=" * 60)

    for ratio in sorted(df["obs_per_dof"].unique()):
        row = pivot.loc[ratio]
        better_median = ">" if row["SRF"] > row["Median"] else "<"
        better_knn = ">" if row["SRF"] > row["KNN"] else "<"
        status = "✓" if (row["SRF"] > row["Median"] and row["SRF"] > row["KNN"]) else "✗"
        log.info(
            f"ratio={ratio:5.2f}: SRF={row['SRF']:.3f} {better_median} "
            f"Median={row['Median']:.3f}, {better_knn} KNN={row['KNN']:.3f} {status}"
        )

    # Create plot
    log.info("\nCreating publication figure...")
    create_information_theoretic_plot(df, OUTPUT_DIR)

    log.info(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
