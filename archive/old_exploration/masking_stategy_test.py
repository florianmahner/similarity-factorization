import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest
from pathlib import Path
from sklearn.utils import check_random_state
import seaborn as sns


def mask_missing_entries_random(
    x: np.ndarray,
    observed_fraction: float,
    rng: np.random.RandomState,
    missing_values: float | None = np.nan,
) -> np.ndarray:
    observed_mask = ~np.isnan(x) if missing_values is np.nan else x != missing_values

    triu_i, triu_j = np.triu_indices_from(x, k=1)
    triu_observed = observed_mask[triu_i, triu_j]
    valid_positions = np.where(triu_observed)[0]

    n_to_keep = int(observed_fraction * len(valid_positions))
    if n_to_keep == 0:
        return np.ones_like(x, dtype=bool)

    # Subsample from valid upper triangular positions to keep as observed
    keep_positions = rng.choice(valid_positions, size=n_to_keep, replace=False)

    # Create missing mask directly - start with all missing
    missing_mask = np.ones_like(x, dtype=bool)

    # Set kept positions as observed (False = not missing)
    keep_i = triu_i[keep_positions]
    keep_j = triu_j[keep_positions]
    missing_mask[keep_i, keep_j] = False
    missing_mask[keep_j, keep_i] = False

    # Diagonal is always observed
    np.fill_diagonal(missing_mask, False)
    return missing_mask


def mask_missing_entries_balanced_random(
    x: np.ndarray,
    observed_fraction: float,
    rng: np.random.RandomState,
    missing_values: float | None = np.nan,
) -> np.ndarray:
    """Ultra-fast balanced random masking with weighted row sampling."""
    n = x.shape[0]
    observed_mask = ~np.isnan(x) if missing_values is np.nan else x != missing_values
    triu_i, triu_j = np.triu_indices_from(x, k=1)
    n_valid = np.sum(observed_mask[triu_i, triu_j])

    if n_valid == 0:
        return np.ones_like(x, dtype=bool)

    n_target = max(1, int(round(observed_fraction * n_valid)))

    # Start with all masked except diagonal
    missing_mask = np.ones_like(x, dtype=bool)
    np.fill_diagonal(missing_mask, False)
    row_counts = np.ones(n)

    for _ in range(n_target * 2):  # Try up to 2x target attempts
        if np.sum(~missing_mask[triu_i, triu_j]) >= n_target:
            break

        # Weight rows inversely by current count
        weights = 1.0 / row_counts
        probs = weights / weights.sum()

        # Sample two rows with replacement, then ensure different
        rows = rng.choice(n, size=2, p=probs, replace=True)
        i, j = min(rows), max(rows)

        if i != j and observed_mask[i, j] and missing_mask[i, j]:
            missing_mask[i, j] = missing_mask[j, i] = False
            row_counts[i] += 1
            row_counts[j] += 1

    return missing_mask


import numpy as np


def mask_missing_entries_balanced_random(
    x: np.ndarray,
    observed_fraction: float,
    rng: np.random.RandomState,
    missing_values: float | None = np.nan,
) -> np.ndarray:
    """Randomly mask entries while keeping row/col coverage roughly even."""
    if not (0 < observed_fraction <= 1):
        raise ValueError("`observed_fraction` must lie in (0, 1].")
    n = x.shape[0]
    if x.shape[1] != n:
        raise ValueError("`x` must be square.")

    observed = ~np.isnan(x) if missing_values is np.nan else x != missing_values
    i, j = np.triu_indices(n, 1)
    valid = observed[i, j]
    if not np.any(valid):
        return np.ones_like(x, bool)

    i, j = i[valid], j[valid]
    target_pairs = max(1, int(round(observed_fraction * len(i))))
    target_per_row = 2 * target_pairs / n

    mask = np.ones_like(x, bool)
    np.fill_diagonal(mask, False)
    row_count = np.zeros(n, dtype=int)

    order = rng.permutation(len(i))
    max_allowed = target_per_row + 2  # small slack avoids stalling
    selected = 0

    for idx in order:
        if selected == target_pairs:
            break
        a, b = i[idx], j[idx]
        if row_count[a] < max_allowed and row_count[b] < max_allowed:
            mask[a, b] = mask[b, a] = False
            row_count[a] += 1
            row_count[b] += 1
            selected += 1

    if selected < target_pairs:  # finish with most under-represented rows
        remaining = np.argsort(row_count[i] + row_count[j])
        for idx in remaining:
            if mask[i[idx], j[idx]]:
                mask[i[idx], j[idx]] = mask[j[idx], i[idx]] = False
                selected += 1
                if selected == target_pairs:
                    break
    return mask


# def mask_missing_entries_balanced_random(
#     x: np.ndarray,
#     observed_fraction: float,
#     rng: np.random.RandomState,
#     missing_values: float | None = np.nan,
# ) -> np.ndarray:
#     """
#     Randomly select entries while ensuring balanced row/column coverage.

#     This function randomly selects entries to observe while ensuring each row/column
#     has approximately the same number of observed entries. Unlike diagonal masking,
#     this maintains randomness while achieving balance.

#     Parameters
#     ----------
#     x : np.ndarray
#         Square, symmetric similarity matrix.
#     observed_fraction : float
#         Fraction of the valid upper-triangular entries to keep observed.
#     rng : np.random.RandomState
#         Random number generator.
#     missing_values : float | None, optional
#         Value that marks a pre-existing missing entry (default: np.nan).

#     Returns
#     -------
#     missing_mask : np.ndarray (bool)
#         True → entry is masked/held-out, False → entry is observed
#     """
#     if not (0.0 < observed_fraction <= 1.0):
#         raise ValueError("`observed_fraction` must lie in (0, 1].")

#     n = x.shape[0]
#     if x.shape[1] != n:
#         raise ValueError("`x` must be square.")

#     # Identify already-valid similarities in upper triangle
#     observed_mask = ~np.isnan(x) if missing_values is np.nan else x != missing_values
#     triu_i, triu_j = np.triu_indices_from(x, k=1)
#     valid_pairs_mask = observed_mask[triu_i, triu_j]

#     if not np.any(valid_pairs_mask):
#         return np.ones_like(x, dtype=bool)

#     # Get valid upper triangle positions
#     valid_i = triu_i[valid_pairs_mask]
#     valid_j = triu_j[valid_pairs_mask]
#     n_valid = len(valid_i)

#     # Target number of entries to keep
#     n_target = max(1, int(round(observed_fraction * n_valid)))

#     # Calculate target observations per row (excluding diagonal)
#     # Each off-diagonal entry contributes to 2 rows when symmetric
#     target_per_row = (2 * n_target) / n

#     # Start with all entries masked except diagonal
#     missing_mask = np.ones_like(x, dtype=bool)
#     np.fill_diagonal(missing_mask, False)

#     # Track current observation count per row (diagonal already counted)
#     row_counts = np.ones(n)  # Start with 1 for diagonal

#     # Create randomized list of valid positions
#     position_indices = np.arange(n_valid)
#     rng.shuffle(position_indices)

#     selected_count = 0

#     for idx in position_indices:
#         if selected_count >= n_target:
#             break

#         i, j = valid_i[idx], valid_j[idx]

#         # Check if both rows can accept another observation
#         # Allow some flexibility to achieve target coverage
#         max_allowed = target_per_row + 2  # Allow some imbalance for better coverage

#         if row_counts[i] < max_allowed and row_counts[j] < max_allowed:
#             # Select this entry
#             missing_mask[i, j] = False
#             missing_mask[j, i] = False
#             row_counts[i] += 1
#             row_counts[j] += 1
#             selected_count += 1

#     # If we haven't reached target due to balance constraints,
#     # add remaining entries with preference for underrepresented rows
#     if selected_count < n_target:
#         remaining_positions = []
#         for idx in position_indices:
#             i, j = valid_i[idx], valid_j[idx]
#             if missing_mask[i, j]:  # Still masked
#                 remaining_positions.append((i, j, row_counts[i] + row_counts[j]))

#         # Sort by sum of row counts (prefer rows with fewer observations)
#         remaining_positions.sort(key=lambda x: x[2])

#         for i, j, _ in remaining_positions:
#             if selected_count >= n_target:
#                 break
#             missing_mask[i, j] = False
#             missing_mask[j, i] = False
#             row_counts[i] += 1
#             row_counts[j] += 1
#             selected_count += 1

#     return missing_mask


def compute_row_statistics(missing_mask: np.ndarray) -> dict:
    """Compute statistics for row observation counts."""
    observed_counts = np.sum(~missing_mask, axis=1)
    return {
        "min_count": observed_counts.min(),
        "max_count": observed_counts.max(),
        "mean_count": observed_counts.mean(),
        "std_count": observed_counts.std(),
        "cv_count": (
            observed_counts.std() / observed_counts.mean()
            if observed_counts.mean() > 0
            else np.inf
        ),
        "range_count": observed_counts.max() - observed_counts.min(),
    }


def visualize_masking_comparison(test_matrices, output_dir="results/masking_analysis"):
    """Generate comprehensive visualizations comparing random vs balanced random masking."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    rng = check_random_state(42)
    sizes = list(test_matrices.keys())
    keep_fractions = [0.4, 0.6, 0.8]
    strategies = {
        "Random": mask_missing_entries_random,
        "Balanced Random": mask_missing_entries_balanced_random,
    }

    # Collect results
    results = []
    for size in sizes:
        matrix = test_matrices[size]
        for fraction in keep_fractions:
            for strategy_name, strategy_func in strategies.items():
                for trial in range(5):
                    mask = strategy_func(matrix, fraction, rng)
                    stats = compute_row_statistics(mask)
                    results.append(
                        {
                            "size": size,
                            "keep_fraction": fraction,
                            "strategy": strategy_name,
                            "trial": trial,
                            **stats,
                        }
                    )

    df = pd.DataFrame(results)

    # Create comprehensive overview plot
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # 1. CV comparison by size
    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(data=df, x="size", y="cv_count", hue="strategy", ax=ax1)
    ax1.set_title("Coefficient of Variation by Matrix Size", fontsize=12, pad=15)
    ax1.set_xlabel("Matrix Size")
    ax1.set_ylabel("CV of Row Counts")

    # 2. CV comparison by fraction
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(data=df, x="keep_fraction", y="cv_count", hue="strategy", ax=ax2)
    ax2.set_title("CV by Keep Fraction", fontsize=12, pad=15)
    ax2.set_xlabel("Keep Fraction")
    ax2.set_ylabel("CV of Row Counts")

    # 3. Standard deviation comparison
    ax3 = fig.add_subplot(gs[0, 2])
    sns.boxplot(data=df, x="strategy", y="std_count", ax=ax3)
    ax3.set_title("Standard Deviation of Row Counts", fontsize=12, pad=15)
    ax3.set_xlabel("Strategy")
    ax3.set_ylabel("Std of Row Counts")

    # 4. Range comparison
    ax4 = fig.add_subplot(gs[0, 3])
    sns.boxplot(data=df, x="strategy", y="range_count", ax=ax4)
    ax4.set_title("Range of Row Counts", fontsize=12, pad=15)
    ax4.set_xlabel("Strategy")
    ax4.set_ylabel("Max - Min Row Counts")

    # 5-7. Standard deviation curves by size
    for i, size in enumerate(sizes):
        ax = fig.add_subplot(gs[1, i])
        for strategy in strategies.keys():
            subset = df[(df["size"] == size) & (df["strategy"] == strategy)]
            mean_std = subset.groupby("keep_fraction")["std_count"].mean()
            ax.plot(
                mean_std.index,
                mean_std.values,
                marker="o",
                label=strategy,
                linewidth=3,
                markersize=8,
            )

        ax.set_title(f"Row Count Std vs Keep Fraction\n(n={size})", fontsize=11)
        ax.set_xlabel("Keep Fraction")
        ax.set_ylabel("Std of Row Counts")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # 8. Example patterns visualization
    demo_size = 100
    demo_fraction = 0.6
    if demo_size in test_matrices:
        matrix = test_matrices[demo_size]
        mask_random = mask_missing_entries_random(matrix, demo_fraction, rng)
        mask_balanced_rand = mask_missing_entries_balanced_random(
            matrix, demo_fraction, rng
        )

        # Show only upper triangular part for clarity
        mask_random_display = mask_random.copy()
        mask_balanced_rand_display = mask_balanced_rand.copy()

        tril_indices = np.tril_indices(demo_size, k=-1)
        mask_random_display[tril_indices] = 0.5
        mask_balanced_rand_display[tril_indices] = 0.5

        ax8 = fig.add_subplot(gs[2, 0])
        im1 = ax8.imshow(mask_random_display, cmap="RdYlBu_r", aspect="equal")
        ax8.set_title("Random Masking Pattern", fontsize=11)
        ax8.set_xlabel("Object Index")
        ax8.set_ylabel("Object Index")

        ax9 = fig.add_subplot(gs[2, 1])
        im2 = ax9.imshow(mask_balanced_rand_display, cmap="RdYlBu_r", aspect="equal")
        ax9.set_title("Balanced Random Masking Pattern", fontsize=11)
        ax9.set_xlabel("Object Index")

        # 9. Row count distributions
        ax10 = fig.add_subplot(gs[2, 2:])
        rows = np.arange(demo_size)
        random_counts = np.sum(~mask_random, axis=1)
        balanced_rand_counts = np.sum(~mask_balanced_rand, axis=1)

        width = 0.35
        ax10.bar(
            rows - width / 2,
            random_counts,
            width,
            alpha=0.8,
            label="Random",
            color="C0",
        )
        ax10.bar(
            rows + width / 2,
            balanced_rand_counts,
            width,
            alpha=0.8,
            label="Balanced Random",
            color="C1",
        )

        ax10.set_title(
            f"Row Count Distributions (n={demo_size}, keep {demo_fraction*100:.0f}%)",
            fontsize=11,
        )
        ax10.set_xlabel("Object Index")
        ax10.set_ylabel("Observed Count")
        ax10.legend()
        ax10.set_ylim(0, demo_size)

    plt.suptitle(
        "Masking Strategy Comparison: Random vs Balanced Random", fontsize=16, y=0.98
    )
    plt.savefig(
        output_dir / "masking_comparison_overview.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Save summary statistics
    summary_stats = (
        df.groupby(["strategy", "size", "keep_fraction"])
        .agg(
            {
                "mean_count": "mean",
                "std_count": "mean",
                "cv_count": "mean",
                "range_count": "mean",
            }
        )
        .round(3)
    )
    summary_stats.to_csv(output_dir / "masking_results.csv")
    df.to_csv(output_dir / "masking_detailed_results.csv", index=False)

    # Print concise comparison
    print("\nMasking Strategy Comparison")
    print("=" * 30)
    comparison = (
        df.groupby("strategy")
        .agg({"cv_count": "mean", "std_count": "mean", "range_count": "mean"})
        .round(3)
    )

    for metric in ["cv_count", "std_count", "range_count"]:
        print(f"{metric.upper().replace('_', ' ')}:")
        for strategy in comparison.index:
            print(f"  {strategy}: {comparison.loc[strategy, metric]:.3f}")
        better = comparison[metric].idxmin()
        print(f"  Better: {better}\n")

    print(f"Results saved to {output_dir}")
    return df


if __name__ == "__main__":
    # Generate test matrices
    rng = np.random.RandomState(42)
    test_matrices = {}

    for size in [50, 100, 300, 1000]:
        A = rng.rand(size, size)
        matrix = A @ A.T
        test_matrices[size] = matrix

    # Run visualization
    results_df = visualize_masking_comparison(test_matrices)
