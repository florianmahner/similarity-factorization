import numpy as np
from scipy.signal import savgol_filter


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
