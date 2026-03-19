"""Test improvements for higher rank detection.

Explores:
1. Larger sample size n
2. Finer rank grid
3. More CV repeats
4. Different simulation parameters
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from pysrf import SRF
from pysrf.cross_validation import cross_val_score

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def add_positive_noise_with_snr(
    x: np.ndarray, snr: float, rng: np.random.Generator
) -> np.ndarray:
    signal_power = np.mean(x**2)
    noise_power = signal_power / snr if snr > 0 else 0
    noise_std = np.sqrt(noise_power)
    noise = rng.normal(0, noise_std, x.shape)
    return np.maximum(x + noise, 0)


def simulation(
    n: int, k: int,
    primary_concentration: float = 5.0,
    base_concentration: float = 1.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    m = np.zeros((n, k))
    cluster_sizes = [n // k] * k
    leftover = n - sum(cluster_sizes)
    for i in range(leftover):
        cluster_sizes[i] += 1

    start_idx = 0
    primary_cluster = np.zeros(n, dtype=int)
    for c in range(k):
        end_idx = start_idx + cluster_sizes[c]
        m[start_idx:end_idx, c] = 1.0
        primary_cluster[start_idx:end_idx] = c
        start_idx = end_idx

    soft_m = np.zeros_like(m, dtype=float)
    for i in range(n):
        alphas = np.ones(k) * base_concentration
        alphas[primary_cluster[i]] = primary_concentration
        soft_m[i, :] = rng.dirichlet(alphas)
    return soft_m


def make_similarity(
    n: int, k: int, snr: float, seed: int,
    primary_concentration: float = 2.0,
    base_concentration: float = 0.1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = simulation(n, k, rng=rng,
                   primary_concentration=primary_concentration,
                   base_concentration=base_concentration)
    similarity = w @ w.T
    similarity = add_positive_noise_with_snr(similarity, snr, rng)
    return (similarity + similarity.T) * 0.5


def evaluate_condition(
    n: int, true_rank: int, snr: float, seed: int,
    grid_step: int = 5, grid_span: int = 10,
    cv_repeats: int = 5,
    primary_concentration: float = 2.0,
    base_concentration: float = 0.1,
) -> dict:
    """Evaluate single condition."""
    similarity = make_similarity(n, true_rank, snr, seed,
                                 primary_concentration, base_concentration)

    lower = max(2, true_rank - grid_span)
    upper = true_rank + grid_span
    candidate_ranks = list(range(lower, upper + 1, grid_step))
    if true_rank not in candidate_ranks:
        candidate_ranks.append(true_rank)
    candidate_ranks = sorted(set(candidate_ranks))

    cv_results = cross_val_score(
        similarity,
        estimator=SRF(init="random_sqrt", random_state=seed, max_outer=30, max_inner=20),
        param_grid={"rank": candidate_ranks},
        n_repeats=cv_repeats,
        estimate_sampling_fraction=True,
        sampling_selection="mean",
        random_state=seed,
        verbose=0,
        n_jobs=1,
        fit_final_estimator=False,
    )

    cv_df = cv_results.cv_results_
    mean_scores = cv_df.groupby("rank")["score"].mean()

    best_rank = int(mean_scores.idxmin())
    scores_sorted = mean_scores.sort_values()
    second_best = int(scores_sorted.index[1]) if len(scores_sorted) > 1 else best_rank
    score_gap = float(scores_sorted.iloc[1] - scores_sorted.iloc[0]) if len(scores_sorted) > 1 else 0

    return {
        "n": n,
        "true_rank": true_rank,
        "snr": snr,
        "seed": seed,
        "grid_step": grid_step,
        "cv_repeats": cv_repeats,
        "primary_conc": primary_concentration,
        "base_conc": base_concentration,
        "selected_rank": best_rank,
        "abs_error": abs(best_rank - true_rank),
        "is_correct": best_rank == true_rank,
        "score_gap": score_gap,
    }


def test_sample_size_effect():
    """Test effect of sample size n on rank detection."""
    log.info("\n=== Testing Sample Size Effect ===")

    true_rank = 30
    snr = 0.8
    n_values = [100, 200, 300, 400, 500]
    n_seeds = 10

    conditions = [
        (n, true_rank, snr, seed)
        for n in n_values
        for seed in range(n_seeds)
    ]

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(evaluate_condition)(n, tr, snr, seed, grid_step=5, cv_repeats=5)
        for n, tr, snr, seed in conditions
    )

    df = pd.DataFrame(results)
    summary = df.groupby("n").agg(
        accuracy=("is_correct", "mean"),
        mae=("abs_error", "mean"),
    ).reset_index()

    log.info("\nSample size effect (true_rank=30, SNR=0.8):")
    for _, row in summary.iterrows():
        log.info(f"  n={row['n']:3.0f}: Accuracy={row['accuracy']:.0%}, MAE={row['mae']:.1f}")

    return df, summary


def test_grid_resolution():
    """Test effect of finer rank grid."""
    log.info("\n=== Testing Grid Resolution ===")

    n = 200
    true_rank = 30
    snr = 0.8
    grid_steps = [5, 3, 2, 1]
    n_seeds = 10

    conditions = [
        (n, true_rank, snr, seed, step)
        for step in grid_steps
        for seed in range(n_seeds)
    ]

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(evaluate_condition)(n, tr, snr, seed, grid_step=step, cv_repeats=5)
        for n, tr, snr, seed, step in conditions
    )

    df = pd.DataFrame(results)
    summary = df.groupby("grid_step").agg(
        accuracy=("is_correct", "mean"),
        mae=("abs_error", "mean"),
    ).reset_index()

    log.info("\nGrid resolution effect (n=200, true_rank=30, SNR=0.8):")
    for _, row in summary.iterrows():
        log.info(f"  step={row['grid_step']}: Accuracy={row['accuracy']:.0%}, MAE={row['mae']:.1f}")

    return df, summary


def test_cv_repeats():
    """Test effect of more CV repeats."""
    log.info("\n=== Testing CV Repeats ===")

    n = 200
    true_rank = 30
    snr = 0.8
    cv_repeat_values = [3, 5, 10, 15, 20]
    n_seeds = 10

    conditions = [
        (n, true_rank, snr, seed, cv_reps)
        for cv_reps in cv_repeat_values
        for seed in range(n_seeds)
    ]

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(evaluate_condition)(n, tr, snr, seed, grid_step=5, cv_repeats=cv_reps)
        for n, tr, snr, seed, cv_reps in conditions
    )

    df = pd.DataFrame(results)
    summary = df.groupby("cv_repeats").agg(
        accuracy=("is_correct", "mean"),
        mae=("abs_error", "mean"),
    ).reset_index()

    log.info("\nCV repeats effect (n=200, true_rank=30, SNR=0.8):")
    for _, row in summary.iterrows():
        log.info(f"  cv_repeats={row['cv_repeats']:2.0f}: Accuracy={row['accuracy']:.0%}, MAE={row['mae']:.1f}")

    return df, summary


def test_simulation_params():
    """Test effect of simulation parameters (cluster separability)."""
    log.info("\n=== Testing Simulation Parameters ===")

    n = 200
    true_rank = 30
    snr = 0.8
    param_sets = [
        (2.0, 0.1),   # current (moderate)
        (5.0, 0.1),   # more concentrated
        (10.0, 0.1),  # highly concentrated
        (2.0, 0.5),   # more overlap
        (5.0, 0.5),   # moderate with overlap
    ]
    n_seeds = 10

    conditions = [
        (n, true_rank, snr, seed, pc, bc)
        for pc, bc in param_sets
        for seed in range(n_seeds)
    ]

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(evaluate_condition)(
            n, tr, snr, seed, grid_step=5, cv_repeats=5,
            primary_concentration=pc, base_concentration=bc
        )
        for n, tr, snr, seed, pc, bc in conditions
    )

    df = pd.DataFrame(results)
    df["params"] = df.apply(lambda x: f"({x['primary_conc']}, {x['base_conc']})", axis=1)
    summary = df.groupby("params").agg(
        accuracy=("is_correct", "mean"),
        mae=("abs_error", "mean"),
    ).reset_index()

    log.info("\nSimulation params effect (n=200, true_rank=30, SNR=0.8):")
    for _, row in summary.iterrows():
        log.info(f"  (prim, base)={row['params']}: Accuracy={row['accuracy']:.0%}, MAE={row['mae']:.1f}")

    return df, summary


def plot_improvement_summary(results: dict, output_dir: Path) -> None:
    """Create summary plot of all improvement tests."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    ax = axes[0, 0]
    df = results["sample_size"]
    ax.plot(df["n"], df["accuracy"], "o-", color="steelblue", markersize=6)
    ax.set_xlabel("Sample Size (n)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) Sample Size Effect", loc="left", fontsize=10, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.5)
    sns.despine(ax=ax)

    ax = axes[0, 1]
    df = results["grid_step"]
    ax.plot(df["grid_step"], df["accuracy"], "s-", color="darkorange", markersize=6)
    ax.set_xlabel("Grid Step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) Grid Resolution Effect", loc="left", fontsize=10, fontweight="bold")
    ax.invert_xaxis()
    ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.5)
    sns.despine(ax=ax)

    ax = axes[1, 0]
    df = results["cv_repeats"]
    ax.plot(df["cv_repeats"], df["accuracy"], "^-", color="forestgreen", markersize=6)
    ax.set_xlabel("CV Repeats")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("(c) CV Repeats Effect", loc="left", fontsize=10, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.5)
    sns.despine(ax=ax)

    ax = axes[1, 1]
    df = results["sim_params"]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(df)))
    bars = ax.barh(range(len(df)), df["accuracy"], color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["params"])
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1.05)
    ax.set_title("(d) Simulation Parameters", loc="left", fontsize=10, fontweight="bold")
    ax.axvline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.5)
    sns.despine(ax=ax)

    plt.tight_layout()
    fig.savefig(output_dir / "improvement_summary.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: improvement_summary.pdf")


def main():
    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(exist_ok=True)

    _, sample_size_summary = test_sample_size_effect()
    _, grid_step_summary = test_grid_resolution()
    _, cv_repeats_summary = test_cv_repeats()
    _, sim_params_summary = test_simulation_params()

    results = {
        "sample_size": sample_size_summary,
        "grid_step": grid_step_summary,
        "cv_repeats": cv_repeats_summary,
        "sim_params": sim_params_summary,
    }

    plot_improvement_summary(results, output_dir)

    log.info("\n=== Key Findings ===")
    log.info("For improving rank detection at higher ranks:")
    log.info(f"  1. Sample size: n=500 gives {sample_size_summary[sample_size_summary['n']==500]['accuracy'].iloc[0]:.0%} (vs {sample_size_summary[sample_size_summary['n']==200]['accuracy'].iloc[0]:.0%} at n=200)")
    log.info(f"  2. Grid step: step=1 gives {grid_step_summary[grid_step_summary['grid_step']==1]['accuracy'].iloc[0]:.0%} (vs {grid_step_summary[grid_step_summary['grid_step']==5]['accuracy'].iloc[0]:.0%} at step=5)")
    log.info(f"  3. CV repeats: 20 repeats gives {cv_repeats_summary[cv_repeats_summary['cv_repeats']==20]['accuracy'].iloc[0]:.0%} (vs {cv_repeats_summary[cv_repeats_summary['cv_repeats']==5]['accuracy'].iloc[0]:.0%} at 5)")


if __name__ == "__main__":
    main()
