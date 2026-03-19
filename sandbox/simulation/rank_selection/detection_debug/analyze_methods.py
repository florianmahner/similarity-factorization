"""Compare different rank detection methods.

Tests multiple approaches:
1. Minimum score (current)
2. Elbow detection (second derivative)
3. Relative improvement threshold
4. BIC-like penalty
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
from scipy.interpolate import UnivariateSpline

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


def make_similarity(n: int, k: int, snr: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = simulation(n, k, rng=rng, primary_concentration=2.0, base_concentration=0.1)
    similarity = w @ w.T
    similarity = add_positive_noise_with_snr(similarity, snr, rng)
    return (similarity + similarity.T) * 0.5


def select_rank_minimum(ranks: np.ndarray, scores: np.ndarray) -> int:
    """Select rank with minimum score."""
    return int(ranks[np.argmin(scores)])


def select_rank_elbow(ranks: np.ndarray, scores: np.ndarray) -> int:
    """Select rank using elbow detection (max curvature)."""
    if len(ranks) < 3:
        return select_rank_minimum(ranks, scores)

    sorted_idx = np.argsort(ranks)
    r_sorted = ranks[sorted_idx]
    s_sorted = scores[sorted_idx]

    s_norm = (s_sorted - s_sorted.min()) / (s_sorted.max() - s_sorted.min() + 1e-10)

    first_deriv = np.gradient(s_norm, r_sorted)
    second_deriv = np.gradient(first_deriv, r_sorted)

    elbow_idx = np.argmax(second_deriv)
    return int(r_sorted[elbow_idx])


def select_rank_relative_threshold(
    ranks: np.ndarray, scores: np.ndarray, threshold: float = 0.01
) -> int:
    """Select smallest rank where relative improvement drops below threshold."""
    sorted_idx = np.argsort(ranks)
    r_sorted = ranks[sorted_idx]
    s_sorted = scores[sorted_idx]

    for i in range(len(r_sorted) - 1):
        improvement = (s_sorted[i] - s_sorted[i + 1]) / (s_sorted[i] + 1e-10)
        if improvement < threshold:
            return int(r_sorted[i])

    return int(r_sorted[np.argmin(s_sorted)])


def select_rank_bic_like(
    ranks: np.ndarray, scores: np.ndarray, n: int, penalty_weight: float = 0.1
) -> int:
    """Select rank using BIC-like penalty for model complexity."""
    penalty = penalty_weight * np.log(n) * ranks / n
    penalized_scores = scores + penalty
    return int(ranks[np.argmin(penalized_scores)])


def evaluate_condition(
    n: int, true_rank: int, snr: float, seed: int,
    cv_repeats: int = 10, n_jobs_inner: int = 1,
) -> dict:
    """Evaluate single condition with multiple selection methods."""
    rng = np.random.default_rng(seed)
    similarity = make_similarity(n, true_rank, snr, seed)

    grid_span = 15
    lower = max(2, true_rank - grid_span)
    upper = true_rank + grid_span
    candidate_ranks = list(range(lower, upper + 1, 3))
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
        n_jobs=n_jobs_inner,
        fit_final_estimator=False,
    )

    cv_df = cv_results.cv_results_
    mean_scores = cv_df.groupby("rank")["score"].mean()
    std_scores = cv_df.groupby("rank")["score"].std()

    ranks = np.array(mean_scores.index)
    scores = np.array(mean_scores.values)

    selected_min = select_rank_minimum(ranks, scores)
    selected_elbow = select_rank_elbow(ranks, scores)
    selected_rel = select_rank_relative_threshold(ranks, scores, threshold=0.02)
    selected_bic = select_rank_bic_like(ranks, scores, n, penalty_weight=0.05)

    return {
        "true_rank": true_rank,
        "snr": snr,
        "seed": seed,
        "selected_min": selected_min,
        "selected_elbow": selected_elbow,
        "selected_rel": selected_rel,
        "selected_bic": selected_bic,
        "error_min": abs(selected_min - true_rank),
        "error_elbow": abs(selected_elbow - true_rank),
        "error_rel": abs(selected_rel - true_rank),
        "error_bic": abs(selected_bic - true_rank),
        "correct_min": selected_min == true_rank,
        "correct_elbow": selected_elbow == true_rank,
        "correct_rel": selected_rel == true_rank,
        "correct_bic": selected_bic == true_rank,
        "ranks": list(ranks),
        "scores": list(scores),
        "stds": list(std_scores.values),
    }


def plot_method_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Compare accuracy across methods."""
    methods = ["min", "elbow", "rel", "bic"]
    method_labels = {
        "min": "Minimum",
        "elbow": "Elbow",
        "rel": "Rel. Threshold",
        "bic": "BIC-like",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    summary_data = []
    for method in methods:
        for tr in df["true_rank"].unique():
            acc = df[df["true_rank"] == tr][f"correct_{method}"].mean()
            summary_data.append({
                "method": method_labels[method],
                "true_rank": tr,
                "accuracy": acc
            })

    summary_df = pd.DataFrame(summary_data)

    ax = axes[0]
    for method in methods:
        subset = summary_df[summary_df["method"] == method_labels[method]].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["accuracy"], marker="o", label=method_labels[method], markersize=6)

    ax.set_xlabel("True Rank", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.5)
    sns.despine(ax=ax)
    ax.set_title("Accuracy vs True Rank", fontsize=11)

    ax = axes[1]
    overall_acc = [df[f"correct_{m}"].mean() for m in methods]
    colors = plt.cm.tab10(np.linspace(0, 0.4, len(methods)))
    bars = ax.bar([method_labels[m] for m in methods], overall_acc, color=colors)
    ax.set_ylabel("Overall Accuracy", fontsize=10)
    ax.set_ylim(0, 1.05)
    sns.despine(ax=ax)
    ax.set_title("Overall Accuracy by Method", fontsize=11)

    for bar, acc in zip(bars, overall_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{acc:.1%}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "method_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cv_curves_grid(results: list[dict], output_dir: Path) -> None:
    """Plot CV curves with selection markers for multiple conditions."""
    true_ranks_to_show = [5, 15, 25, 30]
    snr = 0.6

    samples = [r for r in results if r["snr"] == snr and r["true_rank"] in true_ranks_to_show and r["seed"] == 0]
    samples = sorted(samples, key=lambda x: x["true_rank"])

    fig, axes = plt.subplots(1, len(samples), figsize=(12, 3), sharey=False)

    for ax, result in zip(axes, samples):
        ranks = np.array(result["ranks"])
        scores = np.array(result["scores"])
        stds = np.array(result["stds"])

        ax.errorbar(ranks, scores, yerr=stds, marker="o", markersize=4, capsize=2, color="steelblue", alpha=0.8)

        ax.axvline(result["true_rank"], color="red", linestyle="--", lw=1.5, label="True", alpha=0.8)
        ax.axvline(result["selected_min"], color="green", linestyle=":", lw=1.5, label="Min")
        ax.axvline(result["selected_elbow"], color="orange", linestyle="-.", lw=1.5, label="Elbow")

        ax.set_title(f"True k={result['true_rank']}", fontsize=10)
        ax.set_xlabel("Rank", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("CV Score (MSE)", fontsize=9)
        sns.despine(ax=ax)

        if ax == axes[-1]:
            ax.legend(fontsize=7, frameon=False, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / "cv_curves_grid.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_error_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Violin plot of errors by method and true rank."""
    methods = ["min", "elbow", "rel", "bic"]
    method_labels = {
        "min": "Minimum",
        "elbow": "Elbow",
        "rel": "Rel. Threshold",
        "bic": "BIC-like",
    }

    error_data = []
    for _, row in df.iterrows():
        for method in methods:
            error_data.append({
                "method": method_labels[method],
                "true_rank": row["true_rank"],
                "error": row[f"error_{method}"],
            })

    error_df = pd.DataFrame(error_data)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        data=error_df,
        x="true_rank",
        y="error",
        hue="method",
        ax=ax,
        palette="Set2",
    )

    ax.set_xlabel("True Rank", fontsize=10)
    ax.set_ylabel("Absolute Error", fontsize=10)
    ax.legend(title="Method", frameon=False, fontsize=8, loc="upper left")
    sns.despine()

    plt.tight_layout()
    fig.savefig(output_dir / "error_distribution.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(exist_ok=True)

    n = 200
    true_ranks = [5, 10, 15, 20, 25, 30]
    snrs = [0.4, 0.6, 0.8, 1.0]
    n_seeds = 5
    cv_repeats = 10

    conditions = [
        (tr, snr, seed)
        for tr in true_ranks
        for snr in snrs
        for seed in range(n_seeds)
    ]

    log.info(f"Running {len(conditions)} conditions...")

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(evaluate_condition)(n, tr, snr, seed, cv_repeats)
        for tr, snr, seed in conditions
    )

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ["ranks", "scores", "stds"]} for r in results])
    df.to_csv(output_dir / "method_comparison_results.csv", index=False)

    log.info("\nCreating plots...")
    plot_method_comparison(df, output_dir)
    plot_cv_curves_grid(results, output_dir)
    plot_error_distribution(df, output_dir)

    log.info("\n=== Results Summary ===")
    for method in ["min", "elbow", "rel", "bic"]:
        acc = df[f"correct_{method}"].mean()
        mae = df[f"error_{method}"].mean()
        log.info(f"{method:8s}: Accuracy={acc:.1%}, MAE={mae:.2f}")

    log.info("\nAccuracy by true rank (Minimum method):")
    for tr in true_ranks:
        acc = df[df["true_rank"] == tr]["correct_min"].mean()
        log.info(f"  k={tr:2d}: {acc:.0%}")

    log.info(f"\nOutputs saved to {output_dir}")


if __name__ == "__main__":
    main()
