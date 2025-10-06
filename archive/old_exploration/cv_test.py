"""
Test sklearn-compatible cross-validation for ADMM matrix factorization.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pysrf import cross_val_score
from pysrf.bounds import estimate_p_bound_fast
from matplotlib.colors import LogNorm


def simulate_data(n_samples: int = 10, rank: int = 5, seed: int = None):

    rng = np.random.default_rng(seed)
    w_true = rng.random(size=(n_samples, rank))
    x_true = w_true @ w_true.T
    return x_true


def main(
    n_samples: int = 200,
    rank: int = 10,
    seed: int = 0,
):

    x_true = simulate_data(n_samples, rank)
    rank_range = list(range(max(1, rank - 10), rank + 10))

    p_min, p_max, _ = estimate_p_bound_fast(x_true, random_state=seed)
    print("Average p bounds: ", 0.5 * (p_min + p_max))

    scorer = cross_val_score(
        x_true,
        param_grid={
            "rank": rank_range,
            "tol": [0.0],
            "max_outer": [50],
            "max_inner": [30],
            "init": ["random_sqrt"],
        },
        n_repeats=10,
        sampling_fraction=0.5 * (p_min + p_max),
        n_jobs=-1,
        random_state=seed,
        verbose=0,
        missing_values=np.nan,
        fit_final_estimator=False,
    )

    # plot the results
    plt.figure(figsize=(4, 3), dpi=300)
    sns.lineplot(x="rank", y="score", data=scorer.cv_results_, marker="o")
    plt.axvline(
        x=scorer.cv_results_.loc[scorer.cv_results_["score"].idxmin(), "rank"],
        color="red",
        linestyle="--",
        alpha=0.7,
    )
    plt.xlabel("Rank")
    plt.ylabel("Score")
    # plt.yscale("log")
    plt.title("Cross-validation results")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./tests/cv_test.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test cross-validation for ADMM matrix factorization"
    )
    parser.add_argument("--n-samples", type=int, default=200, help="Number of samples")
    parser.add_argument("--rank", type=int, default=10, help="True rank of the matrix")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()
    main(
        n_samples=args.n_samples,
        rank=args.rank,
        seed=args.seed,
    )
