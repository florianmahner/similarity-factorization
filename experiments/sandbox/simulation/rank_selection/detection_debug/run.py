"""Debug rank detection: visualizations and method improvements.

Explores better visualizations and rank detection methods for higher ranks.
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
    """Add noise to array while keeping values positive."""
    signal_power = np.mean(x**2)
    noise_power = signal_power / snr if snr > 0 else 0
    noise_std = np.sqrt(noise_power)
    noise = rng.normal(0, noise_std, x.shape)
    noisy = x + noise
    return np.maximum(noisy, 0)


def simulation(
    n: int,
    k: int,
    primary_concentration: float = 5.0,
    base_concentration: float = 1.0,
    rng: np.random.Generator = np.random.default_rng(42),
) -> np.ndarray:
    """Generate soft cluster memberships."""
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
    """Generate synthetic similarity matrix."""
    rng = np.random.default_rng(seed)
    w = simulation(n, k, rng=rng, primary_concentration=2.0, base_concentration=0.1)
    similarity = w @ w.T
    similarity = add_positive_noise_with_snr(similarity, snr, rng)
    return (similarity + similarity.T) * 0.5


def rank_grid(true_rank: int, grid_span: int, step: int = 5) -> list[int]:
    """Generate candidate rank grid."""
    lower = max(1, true_rank - grid_span)
    upper = true_rank + grid_span
    start = lower if lower % step == 0 else lower + (step - lower % step)
    grid = list(range(start, upper + 1, step))
    grid.extend([lower, upper, true_rank])
    return sorted(set(r for r in grid if r >= 1))


def evaluate_single(
    n: int,
    true_rank: int,
    snr: float,
    seed: int,
    grid_span: int,
    cv_repeats: int,
    n_jobs_inner: int = 1,
) -> dict:
    """Run CV for a single condition, returning full score curve."""
    similarity = make_similarity(n, true_rank, snr, seed)
    candidate_ranks = rank_grid(true_rank, grid_span)

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
    mean_scores = cv_df.groupby("rank")["score"].mean().sort_index()
    std_scores = cv_df.groupby("rank")["score"].std().sort_index()

    best_rank = int(mean_scores.idxmin())
    best_score = float(mean_scores.loc[best_rank])

    scores_sorted = mean_scores.sort_values()
    second_rank = int(scores_sorted.index[1]) if len(scores_sorted) > 1 else best_rank
    second_score = float(scores_sorted.iloc[1]) if len(scores_sorted) > 1 else best_score

    return {
        "true_rank": true_rank,
        "snr": snr,
        "seed": seed,
        "selected_rank": best_rank,
        "best_score": best_score,
        "second_rank": second_rank,
        "second_score": second_score,
        "score_gap": second_score - best_score,
        "abs_error": abs(best_rank - true_rank),
        "signed_error": best_rank - true_rank,
        "is_correct": best_rank == true_rank,
        "candidate_ranks": candidate_ranks,
        "mean_scores": mean_scores.to_dict(),
        "std_scores": std_scores.to_dict(),
    }


def save_figure(fig, path: Path, filename: str) -> None:
    fig.savefig(path / filename, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap: true_rank x SNR -> accuracy."""
    summary = df.groupby(["true_rank", "snr"])["is_correct"].mean().reset_index()
    pivot = summary.pivot(index="true_rank", columns="snr", values="is_correct")
    pivot = pivot.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".1%",
        annot_kws={"size": 8},
        cbar_kws={"label": "Accuracy", "shrink": 0.8},
    )
    ax.set_xlabel("SNR", fontsize=10)
    ax.set_ylabel("True Rank", fontsize=10)
    plt.tight_layout()
    save_figure(fig, output_dir, "accuracy_heatmap.pdf")


def plot_boxplot_by_rank(df: pd.DataFrame, output_dir: Path) -> None:
    """Box plots of selected rank by true rank, faceted by SNR."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    snrs = sorted(df["snr"].unique())

    for ax, snr in zip(axes, snrs):
        subset = df[df["snr"] == snr]
        sns.boxplot(
            data=subset, x="true_rank", y="selected_rank", ax=ax, color="steelblue"
        )
        true_ranks = sorted(subset["true_rank"].unique())
        ax.plot(range(len(true_ranks)), true_ranks, "--", color="red", lw=1.5, label="Ideal")
        ax.set_title(f"SNR = {snr}", fontsize=10)
        ax.set_xlabel("True Rank", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("Selected Rank", fontsize=9)
        else:
            ax.set_ylabel("")
        sns.despine(ax=ax)

    plt.tight_layout()
    save_figure(fig, output_dir, "boxplot_by_snr.pdf")


def plot_score_gap_vs_rank(df: pd.DataFrame, output_dir: Path) -> None:
    """Line plot of score gap vs true rank, colored by SNR."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    summary = df.groupby(["true_rank", "snr"])["score_gap"].agg(["mean", "std"]).reset_index()

    for snr in sorted(df["snr"].unique()):
        subset = summary[summary["snr"] == snr].sort_values("true_rank")
        ax.semilogy(
            subset["true_rank"], subset["mean"], marker="o", label=f"SNR={snr}", markersize=5
        )

    ax.set_xlabel("True Rank", fontsize=10)
    ax.set_ylabel("Score Gap (log scale)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    sns.despine()
    plt.tight_layout()
    save_figure(fig, output_dir, "score_gap_vs_rank.pdf")


def plot_confusion_matrix(df: pd.DataFrame, output_dir: Path) -> None:
    """Confusion-style heatmap: true_rank vs selected_rank."""
    true_ranks = sorted(df["true_rank"].unique())

    conf = pd.crosstab(df["true_rank"], df["selected_rank"], normalize="index")
    conf = conf.reindex(index=true_ranks, fill_value=0)
    conf = conf.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        conf,
        ax=ax,
        cmap="Blues",
        annot=True,
        fmt=".0%",
        annot_kws={"size": 7},
        cbar_kws={"label": "Proportion"},
    )
    ax.set_xlabel("Selected Rank", fontsize=10)
    ax.set_ylabel("True Rank", fontsize=10)
    plt.tight_layout()
    save_figure(fig, output_dir, "confusion_matrix.pdf")


def plot_cv_curves(results: list[dict], output_dir: Path) -> None:
    """Plot CV score curves for a subset of conditions."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()

    samples = [r for r in results if r["seed"] == 0][:6]

    for ax, result in zip(axes, samples):
        ranks = list(result["mean_scores"].keys())
        scores = [result["mean_scores"][r] for r in ranks]
        stds = [result["std_scores"].get(r, 0) for r in ranks]

        ax.errorbar(ranks, scores, yerr=stds, marker="o", markersize=4, capsize=3)
        ax.axvline(result["true_rank"], color="red", linestyle="--", lw=1, label="True")
        ax.axvline(result["selected_rank"], color="green", linestyle=":", lw=1, label="Selected")
        ax.set_title(f"k={result['true_rank']}, SNR={result['snr']}", fontsize=9)
        ax.set_xlabel("Rank", fontsize=8)
        ax.set_ylabel("CV Score (MSE)", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        sns.despine(ax=ax)

    plt.tight_layout()
    save_figure(fig, output_dir, "cv_curves.pdf")


def plot_scatter_improved(df: pd.DataFrame, output_dir: Path) -> None:
    """Improved scatter plot with density and error highlighting."""
    rng = np.random.default_rng(42)
    jitter = 0.6
    df = df.copy()
    df["true_j"] = df["true_rank"] + rng.uniform(-jitter, jitter, len(df))
    df["sel_j"] = df["selected_rank"] + rng.uniform(-jitter, jitter, len(df))
    df["correct"] = df["is_correct"].map({True: "Correct", False: "Error"})

    fig, ax = plt.subplots(figsize=(4.5, 4))

    colors = {"Correct": "#2ecc71", "Error": "#e74c3c"}
    for label, color in colors.items():
        subset = df[df["correct"] == label]
        ax.scatter(
            subset["true_j"], subset["sel_j"], c=color, alpha=0.6, s=30,
            label=label, edgecolors="white", linewidths=0.3
        )

    min_r, max_r = df["true_rank"].min(), df["true_rank"].max()
    ax.plot([min_r, max_r], [min_r, max_r], "--", color="gray", lw=1.5, zorder=0)

    ax.set_xlabel("True Rank", fontsize=10)
    ax.set_ylabel("Selected Rank", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(min_r - 2, max_r + 2)
    ax.set_ylim(min_r - 8, max_r + 8)
    sns.despine()
    plt.tight_layout()
    save_figure(fig, output_dir, "scatter_improved.pdf")


def plot_accuracy_line(df: pd.DataFrame, output_dir: Path) -> None:
    """Line plot of accuracy vs true rank, colored by SNR."""
    fig, ax = plt.subplots(figsize=(4.5, 3))

    summary = df.groupby(["true_rank", "snr"])["is_correct"].mean().reset_index()

    for snr in sorted(df["snr"].unique()):
        subset = summary[summary["snr"] == snr].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["is_correct"], marker="o", label=f"SNR={snr}", markersize=6)

    ax.set_xlabel("True Rank", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.5)
    sns.despine()
    plt.tight_layout()
    save_figure(fig, output_dir, "accuracy_line.pdf")


def run_experiment(
    n: int = 200,
    true_ranks: list[int] = None,
    snrs: list[float] = None,
    n_seeds: int = 5,
    grid_span: int = 10,
    cv_repeats: int = 5,
    n_jobs: int = -1,
    output_dir: Path = None,
) -> pd.DataFrame:
    """Run rank detection experiment."""
    if true_ranks is None:
        true_ranks = [5, 10, 15, 20, 25, 30]
    if snrs is None:
        snrs = [0.4, 0.6, 0.8, 1.0]
    if output_dir is None:
        output_dir = Path.cwd() / "outputs"

    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = [
        (tr, snr, seed)
        for tr in true_ranks
        for snr in snrs
        for seed in range(n_seeds)
    ]

    log.info(f"Running {len(conditions)} conditions with n={n}...")

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(evaluate_single)(n, tr, snr, seed, grid_span, cv_repeats)
        for tr, snr, seed in conditions
    )

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ["candidate_ranks", "mean_scores", "std_scores"]} for r in results])

    df.to_csv(output_dir / "results.csv", index=False)
    log.info(f"Saved results to {output_dir / 'results.csv'}")

    return df, results


def main():
    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(exist_ok=True)

    existing_csv = Path("/LOCAL/fmahner/similarity-factorization/outputs/experiments/simulation/rank_detection/rank_detection_results.csv")

    if existing_csv.exists():
        log.info(f"Loading existing results from {existing_csv}")
        df = pd.read_csv(existing_csv)
        results = None
    else:
        log.info("Running new experiment...")
        df, results = run_experiment(
            n=200,
            true_ranks=[5, 10, 15, 20, 25, 30],
            snrs=[0.4, 0.6, 0.8, 1.0],
            n_seeds=5,
            grid_span=10,
            cv_repeats=5,
            n_jobs=-1,
            output_dir=output_dir,
        )

    log.info("Creating visualizations...")

    plot_accuracy_heatmap(df, output_dir)
    log.info("Created: accuracy_heatmap.pdf")

    plot_boxplot_by_rank(df, output_dir)
    log.info("Created: boxplot_by_snr.pdf")

    plot_score_gap_vs_rank(df, output_dir)
    log.info("Created: score_gap_vs_rank.pdf")

    plot_confusion_matrix(df, output_dir)
    log.info("Created: confusion_matrix.pdf")

    plot_scatter_improved(df, output_dir)
    log.info("Created: scatter_improved.pdf")

    plot_accuracy_line(df, output_dir)
    log.info("Created: accuracy_line.pdf")

    summary = df.groupby(["true_rank", "snr"]).agg(
        accuracy=("is_correct", "mean"),
        mean_error=("abs_error", "mean"),
        score_gap=("score_gap", "median"),
    ).reset_index()

    log.info("\nSummary by condition:")
    log.info(summary.to_string())

    log.info(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
