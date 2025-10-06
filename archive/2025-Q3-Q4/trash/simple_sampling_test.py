#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed
from scipy.signal import savgol_filter

from cross_validation import mask_missing_entries, fit_and_score
from models.admm import ADMM
from tools.rsa import compute_similarity
from utils.simulation import SimulationParams, generate_simulation_data
from cv_heuristic import detect_gradient_saturation


class AdaptiveSampler:

    def __init__(self, eval_func, x_range=(0.01, 1.0)):
        self.eval_func = eval_func
        self.x_min, self.x_max = x_range
        self.x_samples = []
        self.y_samples = []

    def adaptive_sequential_sampling(
        self, initial_points=6, max_points=20, convergence_threshold=0.3
    ):

        x_initial = np.logspace(
            np.log10(self.x_min), np.log10(self.x_max), initial_points
        )
        y_initial = [self.eval_func(x) for x in x_initial]

        self.x_samples = list(x_initial)
        self.y_samples = list(y_initial)

        for iteration in range(max_points - initial_points):
            if len(self.x_samples) < 5:
                break

            x_sorted, y_sorted = self._get_sorted_samples()

            try:
                recommended_rank, analysis_info = detect_gradient_saturation(
                    np.array(x_sorted), np.array(y_sorted)
                )

                if recommended_rank is not None:
                    gradient_std = analysis_info["gradient_std"]
                    if gradient_std < convergence_threshold:
                        break

            except:
                pass

            gradient = self._compute_gradient(x_sorted, y_sorted)
            high_gradient_idx = np.argmax(np.abs(gradient))

            if high_gradient_idx < len(x_sorted) - 1:
                new_x = (
                    x_sorted[high_gradient_idx] + x_sorted[high_gradient_idx + 1]
                ) / 2
            else:
                new_x = (
                    x_sorted[high_gradient_idx]
                    + (self.x_max - x_sorted[high_gradient_idx]) / 3
                )

            if (
                self.x_min <= new_x <= self.x_max
                and min(np.abs(np.array(self.x_samples) - new_x)) > 0.005
            ):

                new_y = self.eval_func(new_x)
                self.x_samples.append(new_x)
                self.y_samples.append(new_y)

        return self._get_sorted_samples()

    def uncertainty_driven_sampling(self, initial_points=6, max_points=20):

        x_initial = np.logspace(
            np.log10(self.x_min), np.log10(self.x_max), initial_points
        )
        y_initial = [self.eval_func(x) for x in x_initial]

        self.x_samples = list(x_initial)
        self.y_samples = list(y_initial)

        for iteration in range(max_points - initial_points):
            if len(self.x_samples) < 5:
                break

            x_sorted, y_sorted = self._get_sorted_samples()

            gap_sizes = np.diff(x_sorted)
            gradient = self._compute_gradient(x_sorted, y_sorted)

            uncertainty_score = gap_sizes * np.abs(gradient[:-1])
            max_uncertainty_idx = np.argmax(uncertainty_score)

            new_x = (
                x_sorted[max_uncertainty_idx] + x_sorted[max_uncertainty_idx + 1]
            ) / 2

            if min(np.abs(np.array(self.x_samples) - new_x)) > 0.005:
                new_y = self.eval_func(new_x)
                self.x_samples.append(new_x)
                self.y_samples.append(new_y)

        return self._get_sorted_samples()

    def early_stopping_sampling(
        self, initial_points=6, max_points=25, stability_checks=3
    ):

        x_initial = np.logspace(
            np.log10(self.x_min), np.log10(self.x_max), initial_points
        )
        y_initial = [self.eval_func(x) for x in x_initial]

        self.x_samples = list(x_initial)
        self.y_samples = list(y_initial)

        stable_recommendations = []

        for iteration in range(max_points - initial_points):
            if len(self.x_samples) < 5:
                continue

            x_sorted, y_sorted = self._get_sorted_samples()

            try:
                recommended_rank, analysis_info = detect_gradient_saturation(
                    np.array(x_sorted), np.array(y_sorted)
                )

                if recommended_rank is not None:
                    stable_recommendations.append(recommended_rank)

                    if len(stable_recommendations) >= stability_checks:
                        recent_recs = stable_recommendations[-stability_checks:]
                        if np.std(recent_recs) < 0.5:
                            break

            except:
                pass

            gradient = self._compute_gradient(x_sorted, y_sorted)
            high_gradient_idx = np.argmax(np.abs(gradient))

            if high_gradient_idx < len(x_sorted) - 1:
                new_x = (
                    x_sorted[high_gradient_idx] + x_sorted[high_gradient_idx + 1]
                ) / 2
            else:
                new_x = (
                    x_sorted[high_gradient_idx]
                    + (self.x_max - x_sorted[high_gradient_idx]) / 3
                )

            if (
                self.x_min <= new_x <= self.x_max
                and min(np.abs(np.array(self.x_samples) - new_x)) > 0.005
            ):

                new_y = self.eval_func(new_x)
                self.x_samples.append(new_x)
                self.y_samples.append(new_y)

        return self._get_sorted_samples()

    def _get_sorted_samples(self):
        indices = np.argsort(self.x_samples)
        return [self.x_samples[i] for i in indices], [
            self.y_samples[i] for i in indices
        ]

    def _compute_gradient(self, x, y):
        x, y = np.array(x), np.array(y)
        if len(x) >= 5:
            return savgol_filter(
                y,
                window_length=min(5, len(x)),
                polyorder=min(3, len(x) - 1),
                deriv=1,
                delta=np.mean(np.diff(x)),
                mode="interp",
            )
        else:
            return np.gradient(y, x)


def simulate_similarity_matrix(n_objects=100, true_rank=10, seed=42):
    simulation_params = SimulationParams(
        n=n_objects,
        k=true_rank,
        p=100,
        snr=1.0,
        rng_state=seed,
        primary_concentration=5.0,
        base_concentration=0.1,
        sparsity=0.8,
    )
    X = generate_simulation_data(simulation_params)[0]
    return compute_similarity(X, X, "linear")


def create_evaluation_function(s_matrix):
    def evaluate_observed_fraction(observed_fraction, trial_id=0):
        rng = np.random.default_rng(trial_id + 1000)
        val_mask = mask_missing_entries(s_matrix, observed_fraction, rng)

        best_score = float("inf")
        best_rank = 1

        for rank in range(1, 26):
            estimator = ADMM(random_state=trial_id + 2000)
            params = {
                "rank": rank,
                "max_outer": 30,
                "max_inner": 20,
                "rho": 1.0,
                "tol": 1e-4,
            }

            result = fit_and_score(estimator, s_matrix, val_mask, params, split_idx=0)

            if result["score"] < best_score:
                best_score = result["score"]
                best_rank = rank

        return best_rank

    return evaluate_observed_fraction


def test_adaptive_strategies():

    s_matrix = simulate_similarity_matrix()
    eval_func = create_evaluation_function(s_matrix)

    strategies = {
        "uniform_15": lambda: list(np.linspace(0.01, 1.0, 15)),
        "adaptive_sequential": lambda: AdaptiveSampler(
            eval_func
        ).adaptive_sequential_sampling()[0],
        "uncertainty_driven": lambda: AdaptiveSampler(
            eval_func
        ).uncertainty_driven_sampling()[0],
        "early_stopping": lambda: AdaptiveSampler(eval_func).early_stopping_sampling()[
            0
        ],
    }

    results = []

    for strategy_name, strategy_func in strategies.items():
        print(f"Testing {strategy_name}...")

        observed_fractions = strategy_func()
        y_values = [eval_func(x) for x in observed_fractions]

        try:
            recommended_rank, analysis_info = detect_gradient_saturation(
                np.array(observed_fractions), np.array(y_values)
            )

            error = (
                abs(recommended_rank - 10)
                if recommended_rank is not None
                else float("inf")
            )

        except:
            recommended_rank = None
            error = float("inf")

        results.append(
            {
                "strategy": strategy_name,
                "n_points": len(observed_fractions),
                "detected_rank": recommended_rank,
                "error": error,
                "x_points": observed_fractions,
                "y_points": y_values,
            }
        )

        print(
            f"  Points: {len(observed_fractions)}, Detected: {recommended_rank}, Error: {error:.2f}"
        )

    return results


def plot_adaptive_comparison(results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    colors = ["blue", "red", "green", "orange"]

    for i, result in enumerate(results):
        ax = axes[i]

        x = result["x_points"]
        y = result["y_points"]

        ax.scatter(x, y, alpha=0.7, s=60, color=colors[i], label=result["strategy"])
        ax.axhline(
            y=10, color="red", linestyle="--", linewidth=2, label="True rank = 10"
        )

        if result["detected_rank"] is not None:
            ax.axhline(
                y=result["detected_rank"],
                color="black",
                linestyle="-",
                linewidth=3,
                label=f"Detected: {result['detected_rank']:.1f}",
            )

            ax.text(
                0.02,
                0.98,
                f"Detected: {result['detected_rank']:.1f}\nError: {result['error']:.1f}\nPoints: {result['n_points']}",
                transform=ax.transAxes,
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
            )
        else:
            ax.text(
                0.02,
                0.98,
                f"No detection\nPoints: {result['n_points']}",
                transform=ax.transAxes,
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.8),
            )

        ax.set_xlabel("Observed fraction")
        ax.set_ylabel("Best rank")
        ax.set_title(f"{result['strategy']}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 25)

    plt.suptitle(
        "Adaptive vs Fixed Sampling Strategies", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    return fig


def main():
    print("=== ADAPTIVE SAMPLING TEST ===")
    print("Comparing fixed uniform sampling vs adaptive data-driven approaches")
    print()

    results = test_adaptive_strategies()

    print("\n=== COMPARISON RESULTS ===")
    print("Strategy              | Points | Detected | Error  | Efficiency")
    print("-" * 65)

    for result in results:
        if result["detected_rank"] is not None:
            efficiency = result["error"] / result["n_points"]  # Lower is better
            print(
                f"{result['strategy']:20} | {result['n_points']:6} | {result['detected_rank']:8.1f} | {result['error']:6.1f} | {efficiency:.3f}"
            )
        else:
            print(
                f"{result['strategy']:20} | {result['n_points']:6} | {'None':>8} | {'inf':>6} | inf"
            )

    results_dir = Path("./results/adaptive_sampling")
    results_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_adaptive_comparison(results)
    fig.savefig(results_dir / "adaptive_comparison.png", dpi=300, bbox_inches="tight")

    results_df = pd.DataFrame(
        [
            {
                "strategy": r["strategy"],
                "n_points": r["n_points"],
                "detected_rank": r["detected_rank"],
                "error": r["error"],
            }
            for r in results
        ]
    )

    results_df.to_csv(results_dir / "adaptive_results.csv", index=False)

    print(f"\nResults saved to: {results_dir}")

    best_result = min(
        [r for r in results if r["detected_rank"] is not None],
        key=lambda x: x["error"] + 0.1 * x["n_points"],
    )
    print(
        f"Best strategy: {best_result['strategy']} (Error: {best_result['error']:.1f}, Points: {best_result['n_points']})"
    )


if __name__ == "__main__":
    main()
