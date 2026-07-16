"""End-to-end SRF demo on a small simulated similarity matrix.

Loads ``demo_similarity.npz`` (100 objects, ground-truth rank 6) and runs the
full pipeline: estimate the number of dimensions from the similarity matrix
alone, fit SRF at that rank, score recovery against the ground truth, and
refit with 40% of the entries hidden to show that SRF recovers held-out
similarities without imputation.

Usage
-----
    python demo/run_demo.py

Self-contained and deterministic; no external data or network access is
required. Expected run time is a few seconds on a laptop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from pysrf import SRF, estimate_rank

DATA = Path(__file__).resolve().parent / "demo_similarity.npz"
SEED = 0


def correlate(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def dimension_recovery(w_true: np.ndarray, w_hat: np.ndarray) -> float:
    """Mean per-dimension correlation after optimally matching columns.

    SRF recovers dimensions up to a permutation, so recovered dimensions are
    matched to ground-truth ones by Hungarian assignment on |correlation|.
    """
    k = w_true.shape[1]
    corr = np.abs(np.corrcoef(w_true.T, w_hat.T)[:k, k:])
    row, col = linear_sum_assignment(-corr)
    return float(corr[row, col].mean())


def main() -> None:
    data = np.load(DATA)
    similarity = data["similarity"]
    embedding_true = data["embedding"]
    true_rank = int(data["n_dims"])
    n = similarity.shape[0]
    off_diagonal = ~np.eye(n, dtype=bool)

    print("=" * 60)
    print("SRF demo")
    print("=" * 60)
    print(f"similarity matrix : {similarity.shape}")
    print(f"true rank         : {true_rank}")
    print()

    print("[1/4] estimating rank (bootstrap eigenspace coherence)...")
    result = estimate_rank(similarity, k_max=12, n_boot=20, random_state=SEED)
    k_star = int(result["k_star"])
    print(f"      estimated rank k* = {k_star} (true rank = {true_rank})")
    print()

    print(f"[2/4] fitting SRF at rank {k_star}...")
    model = SRF(rank=k_star, random_state=SEED, max_outer=200)
    w_hat = model.fit_transform(similarity)
    print(f"      converged in {model.n_iter_} iterations")
    print()

    print("[3/4] scoring...")
    reconstruction_r = correlate(
        similarity[off_diagonal], model.reconstruct()[off_diagonal]
    )
    recovery_r = dimension_recovery(embedding_true, w_hat)
    print(f"      similarity reconstruction r  = {reconstruction_r:.3f}")
    print(f"      mean dimension recovery r    = {recovery_r:.3f}")
    print()

    # SRF fits observed entries only (no imputation), so the entries hidden
    # here are never seen during training and test held-out reconstruction.
    print("[4/4] refitting with 40% of entries hidden (missing-data mode)...")
    rng = np.random.default_rng(SEED)
    hidden = np.triu(rng.random((n, n)) < 0.4, k=1)
    hidden |= hidden.T
    model_missing = SRF(
        rank=k_star, random_state=SEED, max_outer=200, missing_values=np.nan
    )
    model_missing.fit(np.where(hidden, np.nan, similarity))
    held_out_r = correlate(similarity[hidden], model_missing.reconstruct()[hidden])
    print(f"      fraction hidden              = {hidden[off_diagonal].mean():.2f}")
    print(f"      held-out reconstruction r    = {held_out_r:.3f}")
    print()

    passed = (
        k_star == true_rank
        and reconstruction_r > 0.95
        and recovery_r > 0.8
        and held_out_r > 0.9
    )
    print("=" * 60)
    print("DEMO PASSED" if passed else "DEMO FINISHED (check values above)")
    print("=" * 60)


if __name__ == "__main__":
    main()
