"""Coherence-based rank estimation for Samuel mathematical concepts.

The Samuel dataset is 429x429 with only 4.3% observed entries. Coherence
analysis requires a complete matrix, so we must impute first. But imputation
itself imposes rank structure — if we impute at rank=20, coherence will find
~20 signal dimensions.

Strategy:
1. Check imputation sensitivity: run coherence at multiple imputation ranks
2. Run CV-based rank selection (SRF handles NaN natively — no imputation needed)
3. Apply all rank estimation methods from the coherence framework
4. Compare and summarize

Usage:
    ./scripts/submit sandbox/samuel/coherence/run.py --bg
"""

from __future__ import annotations

import json
import logging
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pysrf import SRF
from pysrf.cross_validation import cross_val_score
from src.tools.coherence import impute_similarity_matrix
from sandbox.coherence.projected_iproj.run import (
    compute_projected_coherence,
    estimate_rank_from_activations,
    plot_iproj_curves,
    plot_activation_summary,
)
from src.utils import get_output_dir
from src.colors import ROSE, TEAL, CYAN, SAND, GRAY_LIGHT
from src.utils.figure_theme import create_figure, despine, save_figure

OUTPUT_DIR = get_output_dir()
DATA_PATH = Path("data/samuel/similarityMatrix.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "run.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def load_and_preprocess() -> tuple[np.ndarray, list[str]]:
    """Load, symmetrize, and set diagonal."""
    import pandas as pd

    df = pd.read_csv(DATA_PATH, index_col=0)
    labels = df.index.tolist()
    s = df.values.astype(float)
    n = s.shape[0]

    st = s.T
    has_s, has_st = ~np.isnan(s), ~np.isnan(st)
    sym = np.full((n, n), np.nan)
    sym[has_s & has_st] = (s[has_s & has_st] + st[has_s & has_st]) / 2
    sym[has_s & ~has_st] = s[has_s & ~has_st]
    sym[~has_s & has_st] = st[~has_s & has_st]

    max_val = np.nanmax(sym)
    np.fill_diagonal(sym, max_val)

    n_obs = np.sum(~np.isnan(sym))
    n_total = n * n
    log.info(f"Loaded {n} concepts, {n_obs} observed ({100*n_obs/n_total:.1f}%), diagonal={max_val:.2f}")
    return sym, labels


# ---------------------------------------------------------------------------
# Rank estimation methods (from estimate_rank sandbox)
# ---------------------------------------------------------------------------


def estimate_rank_gap(activation_p: np.ndarray, k_list: np.ndarray) -> dict:
    """Largest gap in sorted activation points separates signal from noise."""
    valid = np.isfinite(activation_p)
    n_valid = int(np.sum(valid))

    if n_valid <= 1:
        return {"rank": n_valid, "method": "gap", "gap_size": 0.0}

    valid_p = activation_p[valid]
    order = np.argsort(valid_p)
    sorted_p = valid_p[order]

    gaps = np.diff(sorted_p)
    if len(gaps) == 0:
        return {"rank": n_valid, "method": "gap", "gap_size": 0.0}

    gap_idx = int(np.argmax(gaps))
    return {
        "rank": int(gap_idx + 1),
        "method": "gap",
        "gap_size": float(gaps[gap_idx]),
    }


def estimate_rank_threshold(activation_p: np.ndarray, p_threshold: float) -> dict:
    """Count dimensions activating before p_threshold."""
    valid = np.isfinite(activation_p)
    early = valid & (activation_p <= p_threshold)
    return {"rank": int(np.sum(early)), "method": f"threshold_{p_threshold}"}


def estimate_rank_plateau(x_mean: np.ndarray, threshold: float = 0.5) -> dict:
    """Signal dimensions have x_k(p->1) -> 1, noise stays near 0."""
    x_at_max_p = x_mean[:, -1]
    above = x_at_max_p > threshold
    return {"rank": int(np.sum(above)), "method": f"plateau_{threshold}"}


def estimate_rank_slope(
    x_mean: np.ndarray, p_list: np.ndarray, slope_threshold: float = 0.1
) -> dict:
    """Positive slope at high p = signal dimension."""
    mid = len(p_list) // 2
    p_upper = p_list[mid:]
    x_upper = x_mean[:, mid:]
    k = x_mean.shape[0]

    slopes = np.zeros(k)
    for i in range(k):
        if len(p_upper) >= 2:
            slopes[i] = np.polyfit(p_upper, x_upper[i], 1)[0]

    x_high = x_mean[:, -3:].mean(axis=1) if x_mean.shape[1] >= 3 else x_mean[:, -1]
    signal = (slopes > slope_threshold) | (x_high > 0.2)
    return {"rank": int(np.sum(signal)), "method": "slope", "slopes": slopes.tolist()}


def estimate_rank_cumulative_elbow(
    activation_p: np.ndarray, p_list: np.ndarray
) -> dict:
    """Elbow of N_act(p) = #{k : p_act(k) <= p}."""
    valid = np.isfinite(activation_p)

    n_act = np.zeros(len(p_list), int)
    for j, p in enumerate(p_list):
        n_act[j] = int(np.sum(valid & (activation_p <= p)))

    if len(n_act) < 3:
        return {"rank": int(n_act[-1]), "method": "elbow", "elbow_p": float(p_list[-1])}

    d2 = np.diff(n_act, n=2)
    if np.all(d2 >= 0):
        rank = int(n_act[-1])
        elbow_idx = len(p_list) - 1
    else:
        elbow_idx = int(np.argmin(d2)) + 1
        rank = int(n_act[elbow_idx])

    return {"rank": rank, "method": "elbow", "elbow_p": float(p_list[elbow_idx])}


def run_all_rank_estimates(result: dict, p_list: np.ndarray) -> dict:
    """Run all rank estimation methods on coherence result."""
    estimates = {}
    ap = result["activation_p"]
    kl = result["k_list"]
    xm = result["x_mean"]

    estimates["gap"] = estimate_rank_gap(ap, kl)
    for p_thr in [0.3, 0.5, 0.7]:
        estimates[f"threshold_{p_thr}"] = estimate_rank_threshold(ap, p_thr)
    for thr in [0.3, 0.5]:
        estimates[f"plateau_{thr}"] = estimate_rank_plateau(xm, thr)
    estimates["slope"] = estimate_rank_slope(xm, p_list)
    estimates["elbow"] = estimate_rank_cumulative_elbow(ap, p_list)

    return estimates


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_imputation_sensitivity(
    imputation_results: dict[int, dict],
    output_path: Path,
) -> None:
    """Plot activation points for different imputation ranks side by side."""
    ranks = sorted(imputation_results.keys())
    n = len(ranks)

    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 2.8), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, imp_rank in zip(axes, ranks):
        result = imputation_results[imp_rank]
        k_list = result["k_list"]
        activation_p = result["activation_p"]
        valid = np.isfinite(activation_p)

        ax.scatter(k_list[valid], activation_p[valid], color=TEAL, s=20, zorder=3)
        ax.plot(k_list[valid], activation_p[valid], color=TEAL, linewidth=0.8, alpha=0.7)

        if np.any(~valid):
            ax.scatter(k_list[~valid], np.ones(np.sum(~valid)) * 1.05,
                       color=GRAY_LIGHT, marker="x", s=20)

        estimates = run_all_rank_estimates(result, result["p"])
        gap_rank = estimates["gap"]["rank"]
        thr_rank = estimates["threshold_0.5"]["rank"]
        ax.axvline(gap_rank + 0.5, color=ROSE, linestyle="--", linewidth=1, alpha=0.7)

        n_act = int(np.sum(valid))
        ax.set_title(f"imp_rank={imp_rank}\ngap={gap_rank}, thr₅₀={thr_rank}, n_act={n_act}",
                     fontsize=8)
        ax.set_xlabel("k")
        ax.set_ylim(0, 1.15)
        despine(ax)

    axes[0].set_ylabel("p_act(k)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cv_results(
    cv_results: dict[float, dict],
    rank_estimates: dict,
    output_path: Path,
) -> None:
    """Plot CV curves at multiple sampling fractions."""
    n_p = len(cv_results)
    fig, axes = plt.subplots(1, n_p, figsize=(3.2 * n_p, 2.5), sharey=False)
    if n_p == 1:
        axes = [axes]

    for ax, (p_val, cv_res) in zip(axes, sorted(cv_results.items())):
        ranks = np.array(cv_res["ranks"])
        mean_mse = np.array(cv_res["mean_mse"])
        std_mse = np.array(cv_res["std_mse"])
        se = std_mse / np.sqrt(cv_res["n_repeats"])

        ax.plot(ranks, mean_mse, color=TEAL, linewidth=1.5, marker="o", markersize=3)
        ax.fill_between(ranks, mean_mse - se, mean_mse + se, color=TEAL, alpha=0.15)

        best_idx = np.argmin(mean_mse)
        ax.axvline(ranks[best_idx], color=TEAL, linestyle=":", linewidth=1, alpha=0.7)

        gap_rank = rank_estimates.get("gap", {}).get("rank")
        if gap_rank and gap_rank > 0:
            ax.axvline(gap_rank, color=ROSE, linestyle="--", linewidth=1, alpha=0.5)

        ax.set_title(f"p={p_val:.2f} → k*={ranks[best_idx]}", fontsize=9)
        ax.set_xlabel("Rank k", fontsize=8)
        despine(ax)

    axes[0].set_ylabel("CV MSE", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_activation_with_estimates(
    result: dict,
    rank_estimates: dict,
    cv_rank: int | None,
    output_path: Path,
) -> None:
    """Plot p_act(k) with all rank estimates marked."""
    k_list = result["k_list"]
    activation_p = result["activation_p"]
    valid = np.isfinite(activation_p)

    fig, ax = create_figure("wide")

    ax.scatter(k_list[valid], activation_p[valid], color=TEAL, s=30, zorder=3)
    ax.plot(k_list[valid], activation_p[valid], color=TEAL, linewidth=1, alpha=0.7)

    if np.any(~valid):
        ax.scatter(k_list[~valid], np.ones(np.sum(~valid)) * 1.05,
                   color=GRAY_LIGHT, marker="x", s=25)

    gap_rank = rank_estimates.get("gap", {}).get("rank")
    if gap_rank and gap_rank > 0:
        ax.axvline(gap_rank + 0.5, color=ROSE, linestyle="--", linewidth=1.5,
                   label=f"gap: k={gap_rank}")

    if cv_rank is not None:
        ax.axvline(cv_rank + 0.5, color=CYAN, linestyle=":", linewidth=1.5,
                   label=f"CV: k={cv_rank}")

    elbow_rank = rank_estimates.get("elbow", {}).get("rank")
    if elbow_rank and elbow_rank > 0 and elbow_rank != gap_rank:
        ax.axvline(elbow_rank + 0.5, color=SAND, linestyle="-.", linewidth=1.5,
                   label=f"elbow: k={elbow_rank}")

    ax.set_xlabel("k (rank)")
    ax.set_ylabel("p_act(k)")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=8)
    despine(ax)
    save_figure(fig, output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    log.info("=" * 60)
    log.info("Samuel Mathematical Concepts — Coherence Rank Estimation")
    log.info("=" * 60)

    sym, labels = load_and_preprocess()
    n = sym.shape[0]

    # Coherence grid
    max_k = 40
    k_list = list(range(1, max_k + 1))
    p_list = np.linspace(0.05, 0.95, 30)
    n_boot = 80

    # ===================================================================
    # Part 1: Imputation sensitivity — try multiple imputation ranks
    # ===================================================================
    log.info("\n=== Part 1: Imputation sensitivity ===")
    imputation_ranks = [5, 10, 15, 20, 30]
    imputation_results = {}

    for imp_rank in imputation_ranks:
        log.info(f"\n--- Imputing with rank={imp_rank} ---")
        s_imp = impute_similarity_matrix(sym, rank=imp_rank, max_outer=50, tol=1e-4)

        result = compute_projected_coherence(
            s_imp, k_list, p_list, b=n_boot,
            random_state=42, use_baseline_correction=True,
            clip_baseline=False, n_jobs=-1,
        )
        imputation_results[imp_rank] = result

        estimates = run_all_rank_estimates(result, p_list)
        n_act = int(np.sum(np.isfinite(result["activation_p"])))

        log.info(f"  imp_rank={imp_rank}: n_activating={n_act}")
        for name, est in estimates.items():
            log.info(f"    {name}: rank={est['rank']}")

        # Save individual coherence curves
        plot_iproj_curves(result, OUTPUT_DIR / f"iproj_imp{imp_rank}.png", show_ci=True)
        plot_activation_summary(result, OUTPUT_DIR / f"activation_imp{imp_rank}.png")

    # Sensitivity summary plot
    plot_imputation_sensitivity(imputation_results, OUTPUT_DIR / "imputation_sensitivity.png")
    log.info("Saved imputation_sensitivity.png")

    # ===================================================================
    # Part 2: CV-based rank selection (no imputation needed)
    # ===================================================================
    log.info("\n=== Part 2: CV-based rank selection ===")

    # Use a range of sampling fractions since we don't have bounds
    # for a matrix this sparse. Try conservative and aggressive values.
    p_sweep = [0.3, 0.5, 0.65, 0.8]
    rank_grid = list(range(1, 31))
    n_cv_repeats = 10

    cv_results = {}
    for p_cv in p_sweep:
        log.info(f"  CV at p={p_cv:.2f}...")
        grid = cross_val_score(
            sym,
            estimator=SRF(random_state=0, missing_values=np.nan),
            param_grid={"rank": rank_grid},
            n_repeats=n_cv_repeats,
            sampling_fraction=p_cv,
            random_state=0,
            n_jobs=-1,
            verbose=0,
        )

        cv_res = grid.cv_results_
        mean_scores = cv_res.groupby("rank")["score"].agg(["mean", "std"]).reset_index()

        cv_results[p_cv] = {
            "best_rank": int(grid.best_params_["rank"]),
            "best_score": float(grid.best_score_),
            "sampling_fraction": p_cv,
            "ranks": mean_scores["rank"].tolist(),
            "mean_mse": mean_scores["mean"].tolist(),
            "std_mse": mean_scores["std"].tolist(),
            "n_repeats": n_cv_repeats,
        }
        log.info(f"    -> best rank={grid.best_params_['rank']} (MSE={grid.best_score_:.6f})")

    # ===================================================================
    # Part 3: Combine results
    # ===================================================================
    log.info("\n=== Part 3: Summary ===")

    # Use a middle imputation rank for the "primary" coherence estimate
    primary_imp_rank = 10
    primary_result = imputation_results[primary_imp_rank]
    primary_estimates = run_all_rank_estimates(primary_result, p_list)

    # Primary CV result at p=0.5
    primary_cv = cv_results[0.5]
    cv_rank = primary_cv["best_rank"]

    log.info(f"\nPrimary coherence (imp_rank={primary_imp_rank}):")
    for name, est in primary_estimates.items():
        log.info(f"  {name}: rank={est['rank']}")

    log.info(f"\nCV ranks across sampling fractions:")
    for p_cv, res in sorted(cv_results.items()):
        log.info(f"  p={p_cv:.2f}: best_rank={res['best_rank']} (MSE={res['best_score']:.6f})")

    # Activation table for primary
    log.info(f"\nActivation table (imp_rank={primary_imp_rank}):")
    log.info(f"{'k':>3}  {'p_act':>8}  {'x(p_max)':>10}")
    for i, k in enumerate(primary_result["k_list"][:max_k]):
        p_act = primary_result["activation_p"][i]
        x_max = primary_result["x_mean"][i, -1]
        p_str = f"{p_act:.3f}" if np.isfinite(p_act) else "never"
        log.info(f"{k:3d}  {p_str:>8}  {x_max:10.4f}")

    # Imputation sensitivity summary
    log.info("\nImputation sensitivity summary:")
    log.info(f"{'imp_rank':>10}  {'n_act':>6}  {'gap':>5}  {'thr_0.5':>8}  {'elbow':>6}")
    for imp_rank in imputation_ranks:
        result = imputation_results[imp_rank]
        estimates = run_all_rank_estimates(result, p_list)
        n_act = int(np.sum(np.isfinite(result["activation_p"])))
        log.info(
            f"{imp_rank:10d}  {n_act:6d}  {estimates['gap']['rank']:5d}  "
            f"{estimates['threshold_0.5']['rank']:8d}  {estimates['elbow']['rank']:6d}"
        )

    # ===================================================================
    # Plots
    # ===================================================================
    plot_cv_results(cv_results, primary_estimates, OUTPUT_DIR / "cv_sweep.png")
    plot_activation_with_estimates(
        primary_result, primary_estimates, cv_rank, OUTPUT_DIR / "activation_with_estimates.png"
    )

    # ===================================================================
    # Save all results
    # ===================================================================
    summary = {
        "dataset": "samuel_math_concepts",
        "n": n,
        "n_observed": int(np.sum(~np.isnan(sym))),
        "obs_fraction": float(np.sum(~np.isnan(sym)) / (n * n)),
        "coherence_params": {
            "max_k": max_k,
            "n_boot": n_boot,
            "n_p": len(p_list),
        },
        "imputation_sensitivity": {},
        "cv_results": {str(p): r for p, r in cv_results.items()},
        "primary_estimates": {k: v for k, v in primary_estimates.items()},
        "primary_imp_rank": primary_imp_rank,
    }

    for imp_rank in imputation_ranks:
        result = imputation_results[imp_rank]
        estimates = run_all_rank_estimates(result, p_list)
        n_act = int(np.sum(np.isfinite(result["activation_p"])))
        summary["imputation_sensitivity"][str(imp_rank)] = {
            "n_activating": n_act,
            "rank_estimates": {k: {"rank": v["rank"]} for k, v in estimates.items()},
        }

    (OUTPUT_DIR / "rank_estimation.json").write_text(json.dumps(summary, indent=2))

    # Save primary coherence result
    np.savez(
        OUTPUT_DIR / "coherence_results.npz",
        p=primary_result["p"],
        k_list=primary_result["k_list"],
        x_mean=primary_result["x_mean"],
        x_ci_lo=primary_result["x_ci_lo"],
        x_ci_hi=primary_result["x_ci_hi"],
        tau_kp=primary_result["tau_kp"],
        activation_p=primary_result["activation_p"],
    )

    log.info(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
