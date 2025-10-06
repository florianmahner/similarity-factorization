# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from typing import Optional


# %%
def detect_gradient_saturation(
    x: np.ndarray,
    y: np.ndarray,
    window_length: int = 5,
    poly_order: int = 3,
    tolerance_factor: float = 0.5,
    min_region_length: int = 3,
) -> tuple[float, dict]:
    """
    Detect gradient saturation regions and return median dimension.
    """

    if len(x) != len(y):
        raise ValueError("x and y must have same length")

    # Compute gradient using Savitzky-Golay filter
    dx = np.mean(np.diff(x))
    gradient = savgol_filter(
        y,
        window_length=window_length,
        polyorder=poly_order,
        deriv=1,
        delta=dx,
        mode="interp",
    )

    # Define saturation tolerance
    gradient_std = np.nanstd(gradient)
    tolerance = tolerance_factor * gradient_std

    # Identify flat (saturated) regions
    is_flat = np.abs(gradient) < tolerance

    # Find contiguous saturation regions
    saturation_regions = _find_contiguous_regions(is_flat, x, y, min_region_length)

    if not saturation_regions:
        return None, {
            "status": "no_saturation",
            "n_regions": 0,
            "gradient_std": gradient_std,
            "tolerance": tolerance,
            "tolerance_factor": tolerance_factor,
            "min_region_length": min_region_length,
        }

    # Calculate recommended rank as median of region medians
    region_medians = [region["y_median"] for region in saturation_regions]
    recommended_rank = np.median(region_medians)

    # Compile analysis information
    analysis_info = {
        "status": "success",
        "recommended_rank": recommended_rank,
        "n_regions": len(saturation_regions),
        "region_medians": region_medians,
        "gradient_std": gradient_std,
        "tolerance": tolerance,
        "tolerance_factor": tolerance_factor,
        "min_region_length": min_region_length,
        "regions": saturation_regions,
        "gradient": gradient,
        "is_flat": is_flat,
        "x": x,
        "y": y,
    }

    return recommended_rank, analysis_info


def _find_contiguous_regions(
    flat_mask: np.ndarray, x: np.ndarray, y: np.ndarray, min_length: int
) -> list[dict]:
    """
    Find contiguous flat regions meeting minimum length requirement.

    This is the core logic: scan through the boolean mask and group
    consecutive True values into regions.
    """
    regions = []
    in_region = False
    start_idx = None

    for i, is_flat in enumerate(flat_mask):
        if is_flat and not in_region:
            # Start new region
            start_idx = i
            in_region = True
        elif not is_flat and in_region:
            # End current region
            end_idx = i - 1
            if end_idx - start_idx + 1 >= min_length:  # Check minimum length
                regions.append(_create_region_info(start_idx, end_idx, x, y))
            in_region = False

    # Handle region that ends at data boundary
    if in_region and start_idx is not None:
        end_idx = len(flat_mask) - 1
        if end_idx - start_idx + 1 >= min_length:
            regions.append(_create_region_info(start_idx, end_idx, x, y))

    return regions


def _create_region_info(
    start_idx: int, end_idx: int, x: np.ndarray, y: np.ndarray
) -> dict:
    """Create region information dictionary."""
    region_y = y[start_idx : end_idx + 1]
    return {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "x_range": (x[start_idx], x[end_idx]),
        "y_median": np.median(region_y),
        "y_mean": np.mean(region_y),
        "y_std": np.std(region_y),
        "length": end_idx - start_idx + 1,
    }


def plot_algorithm_steps(analysis_info: dict, recommended_rank: Optional[float] = None):
    """Show the 6 key steps of the algorithm clearly."""

    x = analysis_info["x"]
    y = analysis_info["y"]
    gradient = analysis_info["gradient"]
    is_flat = analysis_info["is_flat"]
    tolerance = analysis_info["tolerance"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Gradient saturation detection algorithm", fontsize=16, fontweight="bold"
    )

    # Step 1: Original data
    ax = axes[0, 0]
    ax.scatter(x, y, alpha=0.7, s=40, color="steelblue")
    ax.set_xlabel("Observed fraction")
    ax.set_ylabel("Best rank")
    ax.set_title("1. Original data", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Step 2: Gradient computation
    ax = axes[0, 1]
    raw_gradient = np.gradient(y, x)
    ax.plot(x, raw_gradient, alpha=0.4, color="gray", linewidth=1, label="Raw gradient")
    ax.plot(x, gradient, color="orange", linewidth=2, label="Smoothed gradient")
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax.set_xlabel("Observed fraction")
    ax.set_ylabel("Gradient (dy/dx)")
    ax.set_title("2. Compute smoothed gradient", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Step 3: Tolerance calculation
    ax = axes[0, 2]
    ax.hist(gradient, bins=20, alpha=0.7, color="lightblue", density=True)
    ax.axvline(0, color="black", linestyle="-", linewidth=1, label="Zero")
    ax.axvline(
        tolerance,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Tolerance: ±{tolerance:.4f}",
    )
    ax.axvline(-tolerance, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Gradient value")
    ax.set_ylabel("Density")
    ax.set_title(
        f"3. Set tolerance = {analysis_info['tolerance_factor']} × std(gradient)",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Step 4: Identify flat points
    ax = axes[1, 0]
    ax.plot(x, gradient, color="orange", linewidth=2, alpha=0.7, label="Gradient")
    ax.fill_between(x, -tolerance, tolerance, alpha=0.3, color="red", label="Flat zone")
    ax.scatter(
        x[is_flat],
        gradient[is_flat],
        color="red",
        s=60,
        alpha=0.9,
        label=f"Flat points: {np.sum(is_flat)}",
        zorder=5,
    )
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax.axhline(y=tolerance, color="red", linestyle=":", alpha=0.8)
    ax.axhline(y=-tolerance, color="red", linestyle=":", alpha=0.8)
    ax.set_xlabel("Observed fraction")
    ax.set_ylabel("Gradient")
    ax.set_title("4. Find points where |gradient| < tolerance", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Step 5: Group into contiguous regions
    ax = axes[1, 1]
    ax.scatter(x, y, alpha=0.3, s=20, color="lightgray", label="All data")

    if analysis_info["regions"]:
        colors = plt.cm.Set3(np.linspace(0, 1, len(analysis_info["regions"])))
        for i, (region, color) in enumerate(zip(analysis_info["regions"], colors)):
            start, end = region["start_idx"], region["end_idx"]
            ax.scatter(
                x[start : end + 1],
                y[start : end + 1],
                color=color,
                s=60,
                alpha=0.9,
                label=f'Region {i+1} ({region["length"]} pts)',
                edgecolors="black",
                linewidth=0.5,
            )

        # Show the contiguous grouping logic
        ax.text(
            0.02,
            0.98,
            f"Algorithm:\n1. Scan flat_mask left→right\n2. Group consecutive True values\n3. Keep regions ≥ {analysis_info['min_region_length']} points",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

    ax.set_xlabel("Observed fraction")
    ax.set_ylabel("Best rank")
    ax.set_title("5. Group consecutive flat points into regions", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Step 6: Final recommendation
    ax = axes[1, 2]
    ax.scatter(x, y, alpha=0.4, s=30, color="lightgray", label="All data")
    ax.scatter(
        x[is_flat], y[is_flat], alpha=0.7, s=40, color="red", label="Flat regions"
    )

    if recommended_rank is not None:
        ax.axhline(
            y=recommended_rank,
            color="black",
            linestyle="-",
            linewidth=4,
            label=f"Recommendation: {recommended_rank:.3f}",
        )

        # Show how we got the recommendation
        if analysis_info["regions"]:
            region_medians = [r["y_median"] for r in analysis_info["regions"]]
            ax.text(
                0.02,
                0.02,
                f"Region medians: {[f'{m:.2f}' for m in region_medians]}\nFinal = median({region_medians}) = {recommended_rank:.3f}",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            )

    ax.set_xlabel("Observed fraction")
    ax.set_ylabel("Best rank")
    ax.set_title("6. Recommendation = median of region medians", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# %%
# Load and analyze data
df = pd.read_csv("./results/cross_validation/rank_experiment_simulation_test.csv")
df = df.iloc[::5]

x = df.observed_fraction.values
y = df.best_rank.values.astype(float)

# Detect gradient saturation
recommended_rank, analysis_info = detect_gradient_saturation(
    x, y, window_length=5, poly_order=3, tolerance_factor=0.5, min_region_length=3
)

# %%

if recommended_rank is not None:
    print(f"Result: {recommended_rank:.3f}")
    print(
        f"Found {analysis_info['n_regions']} regions with medians: {[f'{m:.3f}' for m in analysis_info['region_medians']]}"
    )

    fig = plot_algorithm_steps(analysis_info, recommended_rank)

    results_dir = Path("./results/saturation_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(results_dir / "algorithm_steps.png", dpi=300, bbox_inches="tight")

else:
    print("No saturation regions found")
    print(f"Try: reducing tolerance_factor or min_region_length")
