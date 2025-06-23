"""
Dimension estimation experiment for SPoSE data using cross-validation on observed entries.

This module estimates the optimal dimensionality for similarity data derived from 
triplet behavioral ratings. It uses cross-validation only on the observed entries
(those with behavioral ratings) to find the optimal rank, avoiding the assumption
that missing entries are zero.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from tqdm import tqdm

from cross_validation import cross_validate_admm
from experiments.spose import compute_similarity_matrix, load_shared_data
from models.admm import ADMM


def estimate_dimensions_from_triplets(
    triplets: np.ndarray,
    n_items: int,
    rank_range: range = None,
    n_repeats: int = 3,
    train_ratio: float = 0.8,
    n_jobs: int = -1,
    verbose: int = 1,
    random_state: int = 42,
) -> Dict:
    """
    Estimate optimal dimensionality from triplet data using cross-validation.

    This function:
    1. Converts triplets to similarity matrix (with many missing entries)
    2. Uses cross-validation ONLY on observed entries to find optimal rank
    3. Returns dimension estimation results

    Args:
        triplets: Array of triplets (i, j, k) where i is closer to j than k
        n_items: Total number of items/concepts
        rank_range: Range of ranks to test (default: 2 to 50)
        n_repeats: Number of CV repeats
        train_ratio: Ratio of observed entries to use for training
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        random_state: Random seed

    Returns:
        Dictionary with estimation results including best rank, CV curves, etc.
    """

    if rank_range is None:
        rank_range = range(2, min(51, n_items // 2))

    print(f"Estimating dimensions from {len(triplets)} triplets for {n_items} items")

    # Convert triplets to similarity matrix with missing entries
    similarity, mask = compute_similarity_matrix(n_items, triplets)

    # Count observed entries
    observed_entries = np.sum(mask)
    total_entries = n_items * n_items - n_items  # Exclude diagonal
    coverage = observed_entries / total_entries

    print(
        f"Similarity matrix coverage: {observed_entries}/{total_entries} "
        f"({coverage*100:.1f}%) entries observed from behavioral ratings"
    )

    # Replace unobserved entries with NaN for proper cross-validation
    similarity_with_nan = similarity.copy().astype(float)
    similarity_with_nan[mask == 0] = np.nan
    np.fill_diagonal(similarity_with_nan, 1.0)  # Keep diagonal as 1

    # Create parameter grid
    param_grid = {
        "rank": list(rank_range),
        "max_outer": [20],
        "max_inner": [5],
        "tol": [0.0],
        "rho": [1.0],
    }

    print(f"Testing ranks: {min(rank_range)} to {max(rank_range)}")

    # Run cross-validation on observed entries only
    cv_results = cross_validate_admm(
        similarity_matrix=similarity_with_nan,
        param_grid=param_grid,
        n_repeats=n_repeats,
        train_ratio=train_ratio,
        random_state=random_state,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    best_rank = cv_results.best_params_["rank"]
    best_rmse = cv_results.best_score_

    print(f"Optimal rank estimation: {best_rank} (CV RMSE: {best_rmse:.4f})")

    return {
        "best_rank": best_rank,
        "best_rmse": best_rmse,
        "cv_results": cv_results.cv_results_,
        "grid_search": cv_results,
        "similarity_matrix": similarity_with_nan,
        "mask": mask,
        "coverage": coverage,
        "n_triplets": len(triplets),
        "n_items": n_items,
        "rank_range": rank_range,
    }


def dimension_estimation_experiment(
    triplets_path: Path,
    n_items: int = 1854,
    rank_range: range = None,
    output_path: Path = None,
    **cv_kwargs,
) -> Dict:
    """
    Run dimension estimation experiment on SPoSE triplet data.

    Args:
        triplets_path: Path to triplet data file
        n_items: Number of items/concepts
        rank_range: Range of ranks to test
        output_path: Where to save results
        **cv_kwargs: Additional arguments for cross_validate_admm

    Returns:
        Dictionary with results
    """

    if output_path is None:
        output_path = Path("results/spose/dimension_estimation.csv")

    if rank_range is None:
        rank_range = range(2, 51)  # Test ranks 2-50

    print(f"Loading triplets from: {triplets_path}")
    triplets = np.loadtxt(triplets_path).astype(int)

    # Run dimension estimation
    results = estimate_dimensions_from_triplets(
        triplets=triplets, n_items=n_items, rank_range=rank_range, **cv_kwargs
    )

    # Save detailed results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results["cv_results"].to_csv(output_path, index=False)
    print(f"Detailed CV results saved to: {output_path}")

    # Create summary
    summary = {
        "dataset": triplets_path.name,
        "n_triplets": results["n_triplets"],
        "n_items": results["n_items"],
        "coverage": results["coverage"],
        "best_rank": results["best_rank"],
        "best_rmse": results["best_rmse"],
        "rank_range_min": min(rank_range),
        "rank_range_max": max(rank_range),
    }

    # Save summary
    summary_path = (
        output_path.parent / f"dimension_estimation_summary_{triplets_path.stem}.csv"
    )
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    return results


def plot_dimension_estimation_results(
    results: Dict, save_path: Path = None, show_plot: bool = True
):
    """
    Plot dimension estimation results.

    Args:
        results: Results from dimension_estimation_experiment
        save_path: Where to save the plot
        show_plot: Whether to display the plot
    """

    cv_results = results["cv_results"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: CV curve
    ranks = cv_results["param_rank"].values
    scores = cv_results["mean_test_score"].values
    stds = cv_results["std_test_score"].values

    ax1.errorbar(
        ranks,
        scores,
        yerr=stds,
        marker="o",
        capsize=4,
        linewidth=2,
        markersize=6,
        color="steelblue",
    )
    ax1.axvline(
        results["best_rank"],
        color="red",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label=f'Best rank ({results["best_rank"]})',
    )
    ax1.set_xlabel("Rank", fontsize=14)
    ax1.set_ylabel("Cross-validation RMSE", fontsize=14)
    ax1.set_title(
        f'Dimension estimation from {results["n_triplets"]} triplets\n'
        f'Coverage: {results["coverage"]*100:.1f}% of entries observed',
        fontsize=16,
    )
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Coverage visualization
    similarity = results["similarity_matrix"]
    observed_mask = ~np.isnan(similarity)

    # Create a visualization matrix: 1 for observed, 0 for missing, NaN for diagonal
    vis_matrix = np.zeros_like(similarity)
    vis_matrix[observed_mask] = 1
    vis_matrix[~observed_mask] = 0
    np.fill_diagonal(vis_matrix, np.nan)  # Diagonal as NaN for different color

    im = ax2.imshow(vis_matrix, cmap="RdYlBu_r", vmin=0, vmax=1)
    ax2.set_title(
        f"Similarity matrix coverage\n"
        f'{results["n_items"]}×{results["n_items"]} items, {results["coverage"]*100:.1f}% observed',
        fontsize=16,
    )
    ax2.set_xlabel("Item index", fontsize=14)
    ax2.set_ylabel("Item index", fontsize=14)

    # Add colorbar with custom labels
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Missing", "Observed"])
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    if show_plot:
        plt.show()

    return fig


def compare_multiple_datasets(
    triplet_paths: List[Path],
    n_items: int = 1854,
    rank_range: range = None,
    output_dir: Path = None,
    **cv_kwargs,
) -> Dict:
    """
    Compare dimension estimation across multiple triplet datasets.

    Args:
        triplet_paths: List of paths to triplet data files
        n_items: Number of items/concepts
        rank_range: Range of ranks to test
        output_dir: Directory to save results
        **cv_kwargs: Additional arguments for cross_validate_admm

    Returns:
        Dictionary with results for all datasets
    """

    if output_dir is None:
        output_dir = Path("results/spose/dimension_comparison")

    if rank_range is None:
        rank_range = range(2, 51)

    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    summary_data = []

    print(f"Comparing dimension estimation across {len(triplet_paths)} datasets")
    print("=" * 60)

    for triplet_path in tqdm(triplet_paths, desc="Processing datasets"):
        print(f"\nProcessing: {triplet_path.name}")

        try:
            # Run dimension estimation for this dataset
            results = dimension_estimation_experiment(
                triplets_path=triplet_path,
                n_items=n_items,
                rank_range=rank_range,
                output_path=output_dir / f"cv_results_{triplet_path.stem}.csv",
                **cv_kwargs,
            )

            all_results[triplet_path.stem] = results

            # Add to summary
            summary_data.append(
                {
                    "dataset": triplet_path.stem,
                    "n_triplets": results["n_triplets"],
                    "coverage": results["coverage"],
                    "best_rank": results["best_rank"],
                    "best_rmse": results["best_rmse"],
                }
            )

            # Create individual plot
            plot_path = output_dir / f"dimension_estimation_{triplet_path.stem}.png"
            plot_dimension_estimation_results(
                results, save_path=plot_path, show_plot=False
            )

        except Exception as e:
            print(f"Error processing {triplet_path.name}: {e}")
            continue

    # Save comparison summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / "dimension_comparison_summary.csv", index=False)

    # Create comparison plot
    plot_comparison_results(
        all_results, save_path=output_dir / "dimension_comparison.png"
    )

    print(f"\nComparison complete! Results saved to: {output_dir}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))

    return all_results


def plot_comparison_results(
    all_results: Dict, save_path: Path = None, show_plot: bool = True
):
    """
    Plot comparison of dimension estimation across multiple datasets.

    Args:
        all_results: Dictionary of results from compare_multiple_datasets
        save_path: Where to save the plot
        show_plot: Whether to display the plot
    """

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    colors = plt.cm.Set1(np.linspace(0, 1, len(all_results)))

    # Plot 1: CV curves for all datasets
    for i, (dataset_name, results) in enumerate(all_results.items()):
        cv_results = results["cv_results"]
        ranks = cv_results["param_rank"].values
        scores = cv_results["mean_test_score"].values

        ax1.plot(
            ranks,
            scores,
            marker="o",
            label=f'{dataset_name} (best: {results["best_rank"]})',
            color=colors[i],
            linewidth=2,
            markersize=4,
        )

    ax1.set_xlabel("Rank", fontsize=14)
    ax1.set_ylabel("Cross-validation RMSE", fontsize=14)
    ax1.set_title("Dimension estimation comparison across datasets", fontsize=16)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Best rank vs coverage
    coverages = [results["coverage"] for results in all_results.values()]
    best_ranks = [results["best_rank"] for results in all_results.values()]
    dataset_names = list(all_results.keys())

    scatter = ax2.scatter(
        coverages, best_ranks, c=colors[: len(all_results)], s=100, alpha=0.7
    )
    for i, name in enumerate(dataset_names):
        ax2.annotate(
            name,
            (coverages[i], best_ranks[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
        )

    ax2.set_xlabel("Coverage (% of entries observed)", fontsize=14)
    ax2.set_ylabel("Best estimated rank", fontsize=14)
    ax2.set_title("Best rank vs data coverage", fontsize=16)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Number of triplets vs best RMSE
    n_triplets = [results["n_triplets"] for results in all_results.values()]
    best_rmses = [results["best_rmse"] for results in all_results.values()]

    scatter = ax3.scatter(
        n_triplets, best_rmses, c=colors[: len(all_results)], s=100, alpha=0.7
    )
    for i, name in enumerate(dataset_names):
        ax3.annotate(
            name,
            (n_triplets[i], best_rmses[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
        )

    ax3.set_xlabel("Number of triplets", fontsize=14)
    ax3.set_ylabel("Best CV RMSE", fontsize=14)
    ax3.set_title("Reconstruction quality vs dataset size", fontsize=16)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale("log")

    # Plot 4: Summary statistics
    summary_data = pd.DataFrame(
        {
            "Dataset": dataset_names,
            "Coverage": [f"{c*100:.1f}%" for c in coverages],
            "N_Triplets": [f"{n:,}" for n in n_triplets],
            "Best_Rank": best_ranks,
            "Best_RMSE": [f"{r:.4f}" for r in best_rmses],
        }
    )

    ax4.axis("tight")
    ax4.axis("off")
    table = ax4.table(
        cellText=summary_data.values,
        colLabels=summary_data.columns,
        cellLoc="center",
        loc="center",
        fontsize=10,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title("Summary statistics", fontsize=16, pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved to: {save_path}")

    if show_plot:
        plt.show()

    return fig


def run_spose_dimension_analysis():
    """
    Run SPoSE dimension estimation analysis on the training dataset only.
    """

    # Define paths
    data_dir = Path("data/spose_triplets")
    output_dir = Path("results/spose/dimension_estimation")

    # Only analyze the training set
    triplet_path = data_dir / "trainset.txt"

    if not triplet_path.exists():
        raise FileNotFoundError(f"Training set not found: {triplet_path}")

    print("Running SPoSE dimension estimation analysis")
    print("=" * 50)
    print(f"Analyzing dataset: {triplet_path.name}")

    # Run dimension estimation for training set only
    results = dimension_estimation_experiment(
        triplets_path=triplet_path,
        n_items=1854,  # Number of items in THINGS dataset
        rank_range=range(5, 105, 5),  # Test wider range of ranks
        output_path=output_dir / f"cv_results_{triplet_path.stem}.csv",
        n_repeats=5,
        train_ratio=0.4,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    # Create plot for the training set results
    plot_path = output_dir / f"dimension_estimation_{triplet_path.stem}.png"
    plot_dimension_estimation_results(results, save_path=plot_path, show_plot=False)

    print(f"\nAnalysis complete! Results saved to: {output_dir}")

    # Print key findings for training set
    print("\nKey findings:")
    print("-" * 30)
    print(f"Training set ({triplet_path.name}):")
    print(f"  Best rank: {results['best_rank']}")
    print(f"  Coverage: {results['coverage']*100:.1f}%")
    print(f"  N triplets: {results['n_triplets']:,}")
    print(f"  CV RMSE: {results['best_rmse']:.4f}")

    return results


if __name__ == "__main__":
    # Run the complete analysis
    results = run_spose_dimension_analysis()

    # Print key findings
    print("\nKey findings:")
    print("-" * 30)
    for dataset_name, result in results.items():
        print(f"{dataset_name}:")
        print(f"  Best rank: {result['best_rank']}")
        print(f"  Coverage: {result['coverage']*100:.1f}%")
        print(f"  N triplets: {result['n_triplets']:,}")
        print(f"  CV RMSE: {result['best_rmse']:.4f}")
        print()
