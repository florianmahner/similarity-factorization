"""
Test sklearn-compatible cross-validation for ADMM matrix factorization.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cross_validation import cross_val_score


def simulate_data(n_samples: int = 10, rank: int = 5, seed: int = None):

    rng = np.random.default_rng(seed)

    w_true = rng.random(size=(n_samples, rank))
    x_true = w_true @ w_true.T
    return x_true


def main(
    n_samples: int = 200,
    rank: int = 10,
    seed: int = 0,
    observed: float = 0.3,
    rho: float = 1.0,
):

    x_true = simulate_data(n_samples, rank)
    rank_range = list(range(max(1, rank - 5), rank + 6))

    scorer = cross_val_score(
        x_true,
        param_grid={
            "rank": rank_range,
            "max_outer": [200],
            "max_inner": [50],
            "rho": [rho],
            "tol": [0.0],
            "init": ["random_sqrt"],
        },
        n_repeats=5,
        observed_fraction=observed,
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
    parser.add_argument(
        "--observed",
        type=float,
        default=0.3,
        help="Fraction of observed entries (0, 1]",
    )

    parser.add_argument(
        "--rho",
        type=float,
        default=10.0,
        help="Regularization parameter",
    )

    args = parser.parse_args()
    main(
        n_samples=args.n_samples,
        rank=args.rank,
        seed=args.seed,
        observed=args.observed,
        rho=args.rho,
    )
