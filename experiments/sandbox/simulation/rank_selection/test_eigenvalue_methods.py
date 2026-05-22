"""Test eigenvalue-based rank selection methods.

Methods that use only the similarity matrix (no CV):
1. Elbow: Max curvature in scree plot
2. Variance threshold: Cumulative variance > 90%
3. Kaiser: Eigenvalues > mean(eigenvalues)
4. BIC: Reconstruction error + log(n)*k penalty
5. AIC: Reconstruction error + 2*k penalty
"""

from __future__ import annotations

import numpy as np
from pysrf import SRF

from src.utils.simulation import simulation_dirichlet


def select_rank_elbow(eigvals: np.ndarray, max_rank: int = 50) -> int:
    """Select rank using elbow detection (max second derivative)."""
    eigvals = eigvals[:max_rank]
    if len(eigvals) < 3:
        return 1

    # Normalize eigenvalues
    eigvals_norm = eigvals / eigvals[0]

    # Compute second derivative
    first_deriv = np.diff(eigvals_norm)
    second_deriv = np.diff(first_deriv)

    # Elbow is where second derivative is maximum (most positive = biggest bend)
    elbow_idx = np.argmax(second_deriv) + 1  # +1 because of diff
    return int(elbow_idx + 1)  # +1 for 1-indexed rank


def select_rank_variance(eigvals: np.ndarray, threshold: float = 0.90) -> int:
    """Select smallest rank where cumulative variance explained > threshold."""
    total_var = np.sum(eigvals ** 2)
    cumvar = np.cumsum(eigvals ** 2) / total_var

    idx = np.searchsorted(cumvar, threshold)
    return int(idx + 1)  # 1-indexed


def select_rank_kaiser(eigvals: np.ndarray) -> int:
    """Kaiser criterion: keep eigenvalues > mean(eigenvalues)."""
    mean_eig = np.mean(eigvals)
    n_above = np.sum(eigvals > mean_eig)
    return max(1, int(n_above))


def select_rank_bic(
    similarity: np.ndarray,
    candidate_ranks: list[int],
    random_state: int = 42,
) -> tuple[int, dict]:
    """Select rank using BIC (Bayesian Information Criterion).

    BIC = n * log(MSE) + k * log(n)
    where k = number of parameters = n * rank (for embedding matrix)
    """
    n = similarity.shape[0]
    n_entries = n * (n + 1) // 2  # Upper triangle entries

    results = {}
    for rank in candidate_ranks:
        model = SRF(rank=rank, random_state=random_state, max_outer=50)
        model.fit(similarity)
        recon = model.reconstruct()

        mse = np.mean((similarity - recon) ** 2)
        n_params = n * rank  # Embedding matrix has n*rank parameters

        bic = n_entries * np.log(mse + 1e-10) + n_params * np.log(n_entries)
        results[rank] = {"mse": mse, "bic": bic, "n_params": n_params}

    best_rank = min(results, key=lambda r: results[r]["bic"])
    return best_rank, results


def select_rank_aic(
    similarity: np.ndarray,
    candidate_ranks: list[int],
    random_state: int = 42,
) -> tuple[int, dict]:
    """Select rank using AIC (Akaike Information Criterion).

    AIC = n * log(MSE) + 2 * k
    where k = number of parameters = n * rank
    """
    n = similarity.shape[0]
    n_entries = n * (n + 1) // 2

    results = {}
    for rank in candidate_ranks:
        model = SRF(rank=rank, random_state=random_state, max_outer=50)
        model.fit(similarity)
        recon = model.reconstruct()

        mse = np.mean((similarity - recon) ** 2)
        n_params = n * rank

        aic = n_entries * np.log(mse + 1e-10) + 2 * n_params
        results[rank] = {"mse": mse, "aic": aic, "n_params": n_params}

    best_rank = min(results, key=lambda r: results[r]["aic"])
    return best_rank, results


def main():
    print("=" * 60)
    print("Testing eigenvalue-based rank selection methods")
    print("=" * 60)

    # Generate test data
    n = 200
    true_k = 15
    alpha = 0.1

    print(f"\nSimulation: n={n}, k={true_k}, alpha={alpha}")

    rng = np.random.default_rng(42)
    W = simulation_dirichlet(n=n, k=true_k, alpha=alpha, rng=rng)
    S = W @ W.T

    # Compute eigenvalues
    eigvals = np.linalg.eigvalsh(S)[::-1]

    print(f"\nTop 10 eigenvalues: {eigvals[:10].round(2)}")
    print(f"Spectral gap at k={true_k}: {eigvals[true_k-1] - eigvals[true_k]:.3f}")

    # Test each method
    print("\n" + "-" * 40)
    print("EIGENVALUE-BASED METHODS (no model fitting)")
    print("-" * 40)

    rank_elbow = select_rank_elbow(eigvals)
    print(f"Elbow method:      k={rank_elbow} (true={true_k}, error={abs(rank_elbow - true_k)})")

    rank_var90 = select_rank_variance(eigvals, threshold=0.90)
    rank_var95 = select_rank_variance(eigvals, threshold=0.95)
    print(f"Variance 90%:      k={rank_var90} (true={true_k}, error={abs(rank_var90 - true_k)})")
    print(f"Variance 95%:      k={rank_var95} (true={true_k}, error={abs(rank_var95 - true_k)})")

    rank_kaiser = select_rank_kaiser(eigvals)
    print(f"Kaiser criterion:  k={rank_kaiser} (true={true_k}, error={abs(rank_kaiser - true_k)})")

    print("\n" + "-" * 40)
    print("MODEL-BASED METHODS (fit SRF at each rank)")
    print("-" * 40)

    candidate_ranks = list(range(2, 30, 2))

    rank_bic, bic_results = select_rank_bic(S, candidate_ranks)
    print(f"BIC method:        k={rank_bic} (true={true_k}, error={abs(rank_bic - true_k)})")

    rank_aic, aic_results = select_rank_aic(S, candidate_ranks)
    print(f"AIC method:        k={rank_aic} (true={true_k}, error={abs(rank_aic - true_k)})")

    print("\n" + "-" * 40)
    print("BIC values by rank:")
    for r in sorted(bic_results.keys()):
        marker = " <-- selected" if r == rank_bic else ""
        marker = " <-- TRUE" if r == true_k else marker
        print(f"  k={r:2d}: BIC={bic_results[r]['bic']:.1f}, MSE={bic_results[r]['mse']:.6f}{marker}")

    print("\n" + "=" * 60)
    print("Testing with harder case: alpha=2.0")
    print("=" * 60)

    alpha = 2.0
    W2 = simulation_dirichlet(n=n, k=true_k, alpha=alpha, rng=np.random.default_rng(42))
    S2 = W2 @ W2.T
    eigvals2 = np.linalg.eigvalsh(S2)[::-1]

    print(f"\nSpectral gap at k={true_k}: {eigvals2[true_k-1] - eigvals2[true_k]:.3f}")

    rank_elbow2 = select_rank_elbow(eigvals2)
    rank_var90_2 = select_rank_variance(eigvals2, threshold=0.90)
    rank_kaiser2 = select_rank_kaiser(eigvals2)

    print(f"Elbow method:      k={rank_elbow2} (error={abs(rank_elbow2 - true_k)})")
    print(f"Variance 90%:      k={rank_var90_2} (error={abs(rank_var90_2 - true_k)})")
    print(f"Kaiser criterion:  k={rank_kaiser2} (error={abs(rank_kaiser2 - true_k)})")

    rank_bic2, _ = select_rank_bic(S2, candidate_ranks)
    rank_aic2, _ = select_rank_aic(S2, candidate_ranks)
    print(f"BIC method:        k={rank_bic2} (error={abs(rank_bic2 - true_k)})")
    print(f"AIC method:        k={rank_aic2} (error={abs(rank_aic2 - true_k)})")


if __name__ == "__main__":
    main()
