"""
Test sklearn-compatible cross-validation for ADMM matrix factorization.
"""

import numpy as np
import matplotlib.pyplot as plt

from cross_validation import cross_val_score


def simulate_data(n_samples: int = 10, rank: int = 5, seed: int = 42):

    rng = np.random.default_rng(seed)

    w_true = rng.random(size=(n_samples, rank))
    x_true = w_true @ w_true.T
    return x_true


def main(n_samples: int = 215, rank: int = 10, seed: int = 42):

    x_true = simulate_data(n_samples, rank, seed)
    rank_range = list(range(max(1, rank - 2), rank + 3))

    scorer = cross_val_score(
        x_true,
        param_grid={
            "rank": rank_range,
            "max_outer": [100],
            "max_inner": [100],
        },
        n_repeats=20,
        observed_fraction=0.8,
        n_jobs=10,
        random_state=seed,
        verbose=0,
        missing_values=np.nan,
        fit_final_estimator=False,
    )

    cv_results = scorer.cv_results_.groupby("rank")["score"].mean()

    # plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(cv_results.index, cv_results.values, marker="o")
    plt.axvline(x=cv_results.idxmin(), color="red", linestyle="--", alpha=0.7)
    plt.xlabel("Rank")
    plt.ylabel("Score")
    plt.yscale("log")
    plt.title("Cross-validation results")
    plt.grid(True)
    plt.savefig("./tests/cv_test.png")


if __name__ == "__main__":
    main()
