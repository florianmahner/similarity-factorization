"""Coherence-based rank estimation for real datasets.

Uses projected incremental coherence I^proj_k(p) to estimate the
dimensionality of similarity matrices. Compares multiple estimation
methods against existing CV-based results.

Theory: signal dimensions activate at low p (their eigenvectors survive
heavy masking), noise dimensions at high p or never. The transition
gives the rank estimate.

Usage:
    ./scripts/submit sandbox/coherence/estimate_rank/run.py --bg
    ./scripts/submit sandbox/coherence/estimate_rank/run.py dataset=peterson-animals --bg
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

from pysrf import SRF
from pysrf.cross_validation import cross_val_score

from src.colors import ROSE, TEAL, CYAN, GRAY, GRAY_LIGHT
from src.utils import get_output_dir
from src.utils.figure_theme import create_figure, despine, save_figure

OUTPUT_DIR = get_output_dir()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "run.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Dataset loading (without Hydra)
# ---------------------------------------------------------------------------


def load_similarity(dataset_name: str) -> np.ndarray:
    """Load a similarity matrix by dataset name."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from datasets import load_dataset

    dataset_paths = {
        "mur92": "/SSD/datasets/similarity_datasets/mur92",
        "peterson-animals": "/SSD/datasets/similarity_datasets/peterson",
        "peterson-various": "/SSD/datasets/similarity_datasets/peterson",
    }

    path = dataset_paths.get(dataset_name)
    if path is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_paths)}")

    if dataset_name == "peterson-various":
        ds = load_dataset("peterson-various", root=path)
    else:
        ds = load_dataset(dataset_name, root=path)

    return ds.rsm


def load_existing_bounds(dataset_name: str) -> dict | None:
    """Load pre-computed bounds if available."""
    bounds_path = PROJECT_ROOT / "outputs" / "experiments" / "bounds" / dataset_name / "bounds.json"
    if bounds_path.exists():
        with open(bounds_path) as f:
            return json.load(f)
    return None


def load_existing_rank_estimate(dataset_name: str) -> dict | None:
    """Load existing CV-based rank estimate if available."""
    rank_path = PROJECT_ROOT / "outputs" / "experiments" / "estimate_rank" / dataset_name / "rank_estimation.json"
    if rank_path.exists():
        with open(rank_path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Projected coherence (import from kachun)
# ---------------------------------------------------------------------------


def _import_coherence():
    """Import coherence functions from the kachun sandbox."""
    sys.path.insert(0, str(PROJECT_ROOT / "sandbox" / "coherence" / "kachun"))
    from run import compute_projected_coherence, baseline_correct

    return compute_projected_coherence, baseline_correct


# ---------------------------------------------------------------------------
# Rank estimation methods from coherence
# ---------------------------------------------------------------------------


def estimate_rank_gap(activation_p: np.ndarray, k_list: np.ndarray) -> dict:
    """Estimate rank by finding the largest gap in activation points.

    Signal dimensions activate at low p, noise at high p. The largest
    gap in sorted activation points separates the two.
    """
    valid = np.isfinite(activation_p)
    n_valid = int(np.sum(valid))

    if n_valid <= 1:
        return {"rank": n_valid, "method": "gap", "gap_size": 0.0, "gap_location": 0}

    # Sort valid activation points keeping track of which k they belong to
    valid_k = k_list[valid]
    valid_p = activation_p[valid]
    order = np.argsort(valid_p)
    sorted_p = valid_p[order]
    sorted_k = valid_k[order]

    gaps = np.diff(sorted_p)
    if len(gaps) == 0:
        return {"rank": n_valid, "method": "gap", "gap_size": 0.0, "gap_location": 0}

    gap_idx = int(np.argmax(gaps))
    rank = gap_idx + 1

    return {
        "rank": int(rank),
        "method": "gap",
        "gap_size": float(gaps[gap_idx]),
        "gap_location": int(gap_idx),
        "sorted_p": sorted_p.tolist(),
        "sorted_k": sorted_k.tolist(),
    }


def estimate_rank_threshold(
    activation_p: np.ndarray, p_threshold: float = 0.5
) -> dict:
    """Estimate rank as number of dimensions activating before p_threshold."""
    valid = np.isfinite(activation_p)
    early = valid & (activation_p <= p_threshold)
    return {
        "rank": int(np.sum(early)),
        "method": f"threshold_p{p_threshold}",
        "p_threshold": p_threshold,
    }


def estimate_rank_plateau(
    x_mean: np.ndarray, k_list: np.ndarray, threshold: float = 0.5
) -> dict:
    """Estimate rank from coherence values at high p.

    Signal dimensions have x_k(p→1) → 1, noise dimensions stay near 0.
    """
    # Use the last column (highest p)
    x_at_max_p = x_mean[:, -1]
    above = x_at_max_p > threshold
    rank = int(np.sum(above))
    return {
        "rank": rank,
        "method": f"plateau_{threshold}",
        "threshold": threshold,
        "values_at_max_p": x_at_max_p.tolist(),
    }


def estimate_rank_slope(
    x_mean: np.ndarray, p_list: np.ndarray, k_list: np.ndarray, slope_threshold: float = 0.1
) -> dict:
    """Estimate rank from slope of I^proj curves at high p.

    Signal dimensions have positive slope at high p (still gaining coherence),
    while noise dimensions are flat near zero. We look at the slope in the
    upper half of the p range.
    """
    # Use the upper half of the p-range
    mid = len(p_list) // 2
    p_upper = p_list[mid:]
    x_upper = x_mean[:, mid:]

    slopes = np.zeros(len(k_list))
    for i in range(len(k_list)):
        if len(p_upper) >= 2:
            slopes[i] = np.polyfit(p_upper, x_upper[i], 1)[0]

    # Also consider the mean value at high p
    x_high = x_mean[:, -3:].mean(axis=1) if x_mean.shape[1] >= 3 else x_mean[:, -1]

    # A dimension is "signal" if it has positive slope AND appreciable value
    signal = (slopes > slope_threshold) | (x_high > 0.2)
    rank = int(np.sum(signal))

    return {
        "rank": rank,
        "method": "slope",
        "slopes": slopes.tolist(),
        "x_high": x_high.tolist(),
    }


def estimate_rank_cumulative_elbow(
    activation_p: np.ndarray, k_list: np.ndarray, p_list: np.ndarray
) -> dict:
    """Estimate rank from the elbow of the cumulative activation curve.

    N_act(p) = #{k : p_act(k) <= p} should flatten after all signal
    dimensions have activated.
    """
    valid = np.isfinite(activation_p)

    n_act = np.zeros(len(p_list), int)
    for j, p in enumerate(p_list):
        n_act[j] = int(np.sum(valid & (activation_p <= p)))

    # Find elbow: largest second derivative (change in slope)
    if len(n_act) < 3:
        return {"rank": int(n_act[-1]), "method": "elbow", "elbow_p": float(p_list[-1])}

    # Use discrete second difference
    d2 = np.diff(n_act, n=2)
    # The elbow is where acceleration is most negative (flattening)
    if np.all(d2 >= 0):
        rank = int(n_act[-1])
        elbow_idx = len(p_list) - 1
    else:
        elbow_idx = int(np.argmin(d2)) + 1  # +1 for offset from diff
        rank = int(n_act[elbow_idx])

    return {
        "rank": rank,
        "method": "elbow",
        "elbow_p": float(p_list[elbow_idx]),
        "n_act_curve": n_act.tolist(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_iproj_curves(
    result: dict, output_path: Path, max_k: int = 30
) -> None:
    """Plot baseline-corrected I^proj_k(p) curves."""
    p_list = result["p"]
    k_list = result["k_list"]
    x_mean = result["x_mean"]
    x_ci_lo = result["x_ci_lo"]
    x_ci_hi = result["x_ci_hi"]
    activation_idx = result["activation_idx"]

    n_k = min(len(k_list), max_k)
    fig, ax = create_figure("wide")
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_k))

    for i in range(n_k):
        k = k_list[i]
        ax.plot(p_list, x_mean[i], color=colors[i], linewidth=1.2, label=f"k={k}")
        ax.fill_between(p_list, x_ci_lo[i], x_ci_hi[i], color=colors[i], alpha=0.08)

        if activation_idx[i] >= 0:
            j = activation_idx[i]
            ax.scatter([p_list[j]], [x_mean[i, j]], color=colors[i], s=25, zorder=5)

    ax.axhline(0, color=GRAY, linestyle="-", linewidth=0.5)
    ax.set_xlabel("p (masking probability)")
    ax.set_ylabel("x_k(p) (baseline-corrected)")
    ax.set_xlim(0, 1)
    despine(ax)
    save_figure(fig, output_path)
    plt.close(fig)


def plot_activation_points(
    result: dict,
    rank_estimates: dict,
    output_path: Path,
    cv_rank: int | None = None,
) -> None:
    """Plot p_act(k) with rank estimates marked."""
    k_list = result["k_list"]
    activation_p = result["activation_p"]
    valid = np.isfinite(activation_p)

    fig, ax = create_figure("single")

    ax.scatter(k_list[valid], activation_p[valid], color=TEAL, s=30, zorder=3)
    ax.plot(k_list[valid], activation_p[valid], color=TEAL, linewidth=1, alpha=0.7)

    if np.any(~valid):
        ax.scatter(
            k_list[~valid], np.ones(np.sum(~valid)) * 1.05,
            color=GRAY_LIGHT, marker="x", s=25,
        )

    # Mark rank estimates
    gap_rank = rank_estimates.get("gap", {}).get("rank")
    if gap_rank is not None and gap_rank > 0:
        ax.axvline(gap_rank + 0.5, color=ROSE, linestyle="--", linewidth=1.5,
                    label=f"gap: k={gap_rank}")

    if cv_rank is not None:
        ax.axvline(cv_rank + 0.5, color=CYAN, linestyle=":", linewidth=1.5,
                    label=f"CV: k={cv_rank}")

    ax.set_xlabel("k (rank)")
    ax.set_ylabel("p_act(k)")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=8)
    despine(ax)
    save_figure(fig, output_path)
    plt.close(fig)


def plot_cumulative_activations(
    result: dict,
    rank_estimates: dict,
    output_path: Path,
) -> None:
    """Plot cumulative activation N_act(p) = #{k: p_act(k) <= p}."""
    p_list = result["p"]
    activation_p = result["activation_p"]
    valid = np.isfinite(activation_p)

    n_act = np.zeros(len(p_list), int)
    for j, p in enumerate(p_list):
        n_act[j] = int(np.sum(valid & (activation_p <= p)))

    fig, ax = create_figure("single")
    ax.plot(p_list, n_act, color=TEAL, linewidth=2, marker="o", markersize=4)

    gap_rank = rank_estimates.get("gap", {}).get("rank")
    if gap_rank is not None:
        ax.axhline(gap_rank, color=ROSE, linestyle="--", linewidth=1,
                    label=f"gap estimate: {gap_rank}")

    ax.set_xlabel("p (masking probability)")
    ax.set_ylabel("N_act(p) = cumulative activations")
    ax.legend(frameon=False, fontsize=8)
    despine(ax)
    save_figure(fig, output_path)
    plt.close(fig)


def choose_cv_sampling_fraction(
    pmin: float, pmax: float, activation_p: np.ndarray | None = None
) -> float:
    """Choose sampling fraction for CV.

    Uses the mean of bounds (same as the standard pipeline). The coherence
    activation points are logged for diagnostics but don't override the
    bounds-based selection.
    """
    return 0.5 * (pmin + pmax)


def run_cv(
    s: np.ndarray,
    sampling_fraction: float,
    rank_grid: list[int],
    n_repeats: int = 10,
    n_jobs: int = -1,
    random_state: int = 0,
) -> dict:
    """Run cross-validation at the coherence-informed sampling fraction."""
    grid = cross_val_score(
        s,
        estimator=SRF(random_state=random_state),
        param_grid={"rank": rank_grid},
        n_repeats=n_repeats,
        sampling_fraction=sampling_fraction,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
    )

    cv_results = grid.cv_results_
    mean_scores = cv_results.groupby("rank")["score"].agg(["mean", "std"])
    mean_scores = mean_scores.reset_index()

    return {
        "best_rank": int(grid.best_params_["rank"]),
        "best_score": float(grid.best_score_),
        "sampling_fraction": sampling_fraction,
        "ranks": mean_scores["rank"].tolist(),
        "mean_mse": mean_scores["mean"].tolist(),
        "std_mse": mean_scores["std"].tolist(),
        "n_repeats": n_repeats,
    }


def plot_cv_curve(
    cv_result: dict,
    rank_estimates: dict,
    output_path: Path,
    n_activating: int | None = None,
) -> None:
    """Plot CV error curve with coherence-based rank estimates."""
    ranks = np.array(cv_result["ranks"])
    mean_mse = np.array(cv_result["mean_mse"])
    std_mse = np.array(cv_result["std_mse"])
    se = std_mse / np.sqrt(cv_result["n_repeats"])

    fig, ax = create_figure("wide")
    ax.plot(ranks, mean_mse, color=TEAL, linewidth=1.5, marker="o", markersize=4)
    ax.fill_between(ranks, mean_mse - se, mean_mse + se, color=TEAL, alpha=0.15)

    # Mark CV best
    best_idx = np.argmin(mean_mse)
    ax.axvline(ranks[best_idx], color=TEAL, linestyle=":", linewidth=1,
               label=f"CV best: k={ranks[best_idx]}")

    # Mark coherence gap estimate
    gap_rank = rank_estimates.get("gap", {}).get("rank")
    if gap_rank is not None and gap_rank > 0:
        ax.axvline(gap_rank, color=ROSE, linestyle="--", linewidth=1,
                   label=f"coherence gap: k={gap_rank}")

    # Mark number of activating dimensions
    n_act = n_activating
    if n_act is not None and n_act > 0:
        ax.axvline(n_act, color=CYAN, linestyle="--", linewidth=1,
                   label=f"# activating: k={n_act}")

    ax.set_xlabel("Rank k")
    ax.set_ylabel("CV MSE")
    ax.set_title(f"p={cv_result['sampling_fraction']:.2f} (bounds mean)")
    ax.legend(frameon=False, fontsize=7)
    despine(ax)
    save_figure(fig, output_path)
    plt.close(fig)


def plot_cv_sweep(
    cv_results_all: dict[float, dict],
    rank_estimates: dict,
    output_path: Path,
    n_activating: int | None = None,
) -> None:
    """Plot CV curves at multiple sampling fractions."""
    n_p = len(cv_results_all)
    fig, axes = plt.subplots(1, n_p, figsize=(3.2 * n_p, 2.5), sharey=False)
    if n_p == 1:
        axes = [axes]

    for ax, (p_val, cv_res) in zip(axes, sorted(cv_results_all.items())):
        ranks = np.array(cv_res["ranks"])
        mean_mse = np.array(cv_res["mean_mse"])
        std_mse = np.array(cv_res["std_mse"])
        se = std_mse / np.sqrt(cv_res["n_repeats"])

        ax.plot(ranks, mean_mse, color=TEAL, linewidth=1.5, marker="o", markersize=3)
        ax.fill_between(ranks, mean_mse - se, mean_mse + se, color=TEAL, alpha=0.15)

        best_idx = np.argmin(mean_mse)
        ax.axvline(ranks[best_idx], color=TEAL, linestyle=":", linewidth=1, alpha=0.7)
        ax.set_title(f"p={p_val:.2f}  →  k*={ranks[best_idx]}", fontsize=9)
        ax.set_xlabel("Rank k", fontsize=8)

        if n_activating is not None and n_activating > 0:
            ax.axvline(n_activating, color=CYAN, linestyle="--", linewidth=1, alpha=0.5)

        despine(ax)

    axes[0].set_ylabel("CV MSE", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_spectrum_with_rank(
    s: np.ndarray,
    rank_estimates: dict,
    output_path: Path,
    cv_rank: int | None = None,
    max_k: int = 30,
) -> None:
    """Plot eigenvalue spectrum with rank estimates."""
    evals = la.eigvalsh(s)[::-1][:max_k]

    fig, ax = create_figure("single")
    k_range = np.arange(1, len(evals) + 1)
    ax.bar(k_range, evals, color=TEAL, alpha=0.7, width=0.8)

    gap_rank = rank_estimates.get("gap", {}).get("rank")
    if gap_rank is not None and gap_rank > 0:
        ax.axvline(gap_rank + 0.5, color=ROSE, linestyle="--", linewidth=1.5,
                    label=f"coherence gap: {gap_rank}")

    if cv_rank is not None:
        ax.axvline(cv_rank + 0.5, color=CYAN, linestyle=":", linewidth=1.5,
                    label=f"CV: {cv_rank}")

    ax.set_xlabel("k")
    ax.set_ylabel("Eigenvalue")
    ax.legend(frameon=False, fontsize=8)
    despine(ax)
    save_figure(fig, output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mur92")
    parser.add_argument("--max-k", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--n-p", type=int, default=40)
    parser.add_argument("--alpha-tau", type=float, default=0.90,
                        help="Null quantile (lower = more permissive)")
    parser.add_argument("--ci-level", type=float, default=0.90,
                        help="Bootstrap CI level (lower = wider window for activation)")
    parser.add_argument("--n-cv-repeats", type=int, default=10,
                        help="Number of CV repeats")
    args, _ = parser.parse_known_args()

    dataset_name = args.dataset
    log.info(f"=== Coherence-based rank estimation: {dataset_name} ===")

    # Load data
    log.info(f"Loading {dataset_name}...")
    s = load_similarity(dataset_name)
    n = s.shape[0]
    log.info(f"Similarity matrix: {n}x{n}")

    # Default max-k from dataset configs
    rank_ranges = {"mur92": 20, "peterson-animals": 50, "peterson-various": 50}
    max_k = args.max_k or rank_ranges.get(dataset_name, min(30, n // 3))

    k_list = list(range(1, max_k + 1))
    p_list = np.linspace(0.05, 0.95, args.n_p)

    log.info(f"k_list: 1..{max_k}, p_list: {len(p_list)} values from {p_list[0]:.2f} to {p_list[-1]:.2f}")
    log.info(f"Bootstrap: {args.n_boot} realizations")

    # Run projected coherence
    compute_projected_coherence, _ = _import_coherence()

    result = compute_projected_coherence(
        s, k_list, p_list,
        n_boot=args.n_boot,
        alpha_tau=args.alpha_tau,
        ci_level=args.ci_level,
        use_baseline_correction=True,
        clip_baseline=False,
        n_jobs=-1,
    )

    log.info(f"alpha_tau={args.alpha_tau}, ci_level={args.ci_level}")

    # Estimate rank with multiple methods
    log.info("--- Rank estimation ---")
    rank_estimates = {}

    # Method 1: Gap detection
    gap_est = estimate_rank_gap(result["activation_p"], result["k_list"])
    rank_estimates["gap"] = gap_est
    log.info(f"Gap method: rank={gap_est['rank']} (gap_size={gap_est['gap_size']:.3f})")

    # Method 2: Threshold at various p
    for p_thr in [0.3, 0.5, 0.7]:
        thr_est = estimate_rank_threshold(result["activation_p"], p_threshold=p_thr)
        rank_estimates[f"threshold_{p_thr}"] = thr_est
        log.info(f"Threshold p<={p_thr}: rank={thr_est['rank']}")

    # Method 3: Plateau at high p
    for thr in [0.3, 0.5]:
        plat_est = estimate_rank_plateau(result["x_mean"], result["k_list"], threshold=thr)
        rank_estimates[f"plateau_{thr}"] = plat_est
        log.info(f"Plateau (x>{thr} at max p): rank={plat_est['rank']}")

    # Method 4: Slope-based
    slope_est = estimate_rank_slope(result["x_mean"], p_list, result["k_list"])
    rank_estimates["slope"] = slope_est
    log.info(f"Slope method: rank={slope_est['rank']}")

    # Method 5: Cumulative elbow
    elbow_est = estimate_rank_cumulative_elbow(
        result["activation_p"], result["k_list"], p_list
    )
    rank_estimates["elbow"] = elbow_est
    log.info(f"Cumulative elbow: rank={elbow_est['rank']} (elbow_p={elbow_est['elbow_p']:.3f})")

    # Track number of activating dimensions (stored separately, not in rank_estimates dict)
    n_activating = int(np.sum(np.isfinite(result["activation_p"])))

    # Load existing results for comparison
    cv_rank = None
    existing_rank = load_existing_rank_estimate(dataset_name)
    if existing_rank is not None:
        cv_rank = existing_rank["optimal_rank"]
        log.info(f"Existing CV rank estimate: {cv_rank}")
    else:
        log.info("No existing CV rank estimate found")

    existing_bounds = load_existing_bounds(dataset_name)
    if existing_bounds is not None:
        log.info(f"Existing bounds: pmin={existing_bounds['pmin']:.4f}, pmax={existing_bounds['pmax']:.4f}")

    # --- Cross-validation at multiple sampling fractions ---
    if existing_bounds is not None:
        pmin_b, pmax_b = existing_bounds["pmin"], existing_bounds["pmax"]
    else:
        from pysrf.bounds import estimate_sampling_bounds_fast
        pmin_b, pmax_b, _ = estimate_sampling_bounds_fast(s, random_state=0, n_jobs=-1)

    p_mean = 0.5 * (pmin_b + pmax_b)
    # Sweep: bounds mean, plus a few higher values up to 0.85
    p_sweep = sorted(set([round(p_mean, 2)] + [0.5, 0.65, 0.8]))
    p_sweep = [p for p in p_sweep if 0.2 <= p <= 0.85]

    max_cv_rank = max(n_activating * 3, 25)
    max_cv_rank = min(max_cv_rank, max_k, n // 3)
    rank_grid = list(range(1, max_cv_rank + 1))

    log.info(f"\n--- Cross-validation sweep (p={p_sweep}) ---")
    log.info(f"Rank grid: {rank_grid[0]}..{rank_grid[-1]}")

    cv_results_all = {}
    for p_cv in p_sweep:
        log.info(f"  p={p_cv:.2f}...")
        cv_res = run_cv(
            s,
            sampling_fraction=p_cv,
            rank_grid=rank_grid,
            n_repeats=args.n_cv_repeats,
            n_jobs=-1,
        )
        cv_results_all[p_cv] = cv_res
        log.info(f"    -> best rank={cv_res['best_rank']} (MSE={cv_res['best_score']:.6f})")

    # Primary CV result uses bounds mean
    cv_result = cv_results_all[round(p_mean, 2)]
    cv_rank = cv_result["best_rank"]

    # Save results
    summary = {
        "dataset": dataset_name,
        "n": n,
        "max_k": max_k,
        "n_boot": args.n_boot,
        "rank_estimates": {k: {kk: vv for kk, vv in v.items() if kk != "values_at_max_p"} for k, v in rank_estimates.items()},
        "n_activating": n_activating,
        "cv_rank": cv_rank,
        "cv_results_sweep": {str(p): r for p, r in cv_results_all.items()},
        "bounds": existing_bounds,
    }

    # Activation table
    log.info("\nActivation table:")
    log.info(f"{'k':>3}  {'p_act':>8}  {'x(p_max)':>10}")
    for i, k in enumerate(result["k_list"][:max_k]):
        p_act = result["activation_p"][i]
        x_max = result["x_mean"][i, -1]
        p_str = f"{p_act:.3f}" if np.isfinite(p_act) else "never"
        log.info(f"{k:3d}  {p_str:>8}  {x_max:10.4f}")

    (OUTPUT_DIR / "rank_estimation.json").write_text(json.dumps(summary, indent=2))

    np.savez(
        OUTPUT_DIR / "coherence_results.npz",
        p=result["p"],
        k_list=result["k_list"],
        x_mean=result["x_mean"],
        x_ci_lo=result["x_ci_lo"],
        x_ci_hi=result["x_ci_hi"],
        tau_kp=result["tau_kp"],
        activation_p=result["activation_p"],
        evals_ref=result["evals_ref"],
    )

    # Plots
    plot_iproj_curves(result, OUTPUT_DIR / "iproj_curves.png", max_k=max_k)
    plot_activation_points(result, rank_estimates, OUTPUT_DIR / "activation_points.png", cv_rank=cv_rank)
    plot_cumulative_activations(result, rank_estimates, OUTPUT_DIR / "cumulative_activations.png")
    plot_spectrum_with_rank(s, rank_estimates, OUTPUT_DIR / "spectrum_with_rank.png", cv_rank=cv_rank, max_k=max_k)
    plot_cv_curve(cv_result, rank_estimates, OUTPUT_DIR / "cv_curve.png", n_activating=n_activating)
    plot_cv_sweep(cv_results_all, rank_estimates, OUTPUT_DIR / "cv_sweep.png", n_activating=n_activating)

    log.info(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
