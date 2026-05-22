"""Iterative softmax bias correction of count-based RSM.

The count RSM estimates E[P(choose i,j | random k)], not s_ij.
Given current s, compute predicted P_hat by averaging the softmax over
all contexts k, then adjust s to make P_hat match P_observed.
Entirely matrix-level -- no individual triplets needed.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.special import logit

from src.utils import get_output_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "things" / "triplets_47"
N = 1854


def load_triplets(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float).astype(int)


def triplet_accuracy(s: np.ndarray, triplets: np.ndarray) -> float:
    sij = s[triplets[:, 0], triplets[:, 1]]
    sik = s[triplets[:, 0], triplets[:, 2]]
    sjk = s[triplets[:, 1], triplets[:, 2]]
    return float(np.mean((sij > sik) & (sij > sjk)))


def build_counts_shown(triplets: np.ndarray, n: int):
    ii, jj, kk = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))
    np.add.at(counts, (ii, jj), 1)
    np.add.at(counts, (jj, ii), 1)
    for a, b in [(ii, jj), (ii, kk), (jj, kk)]:
        np.add.at(shown, (a, b), 1)
        np.add.at(shown, (b, a), 1)
    return counts, shown


def compute_predicted_probabilities(s: np.ndarray) -> np.ndarray:
    """P_hat[i,j] = mean_{k != i,j} softmax(s_ij, s_ik, s_jk)[0].

    Vectorized: for each k, compute 1/(1 + exp(s_ik-s_ij) + exp(s_jk-s_ij))
    for all (i,j) simultaneously. O(n^2) per k, O(n^3) total.
    """
    n = len(s)
    p_hat = np.zeros((n, n))

    for k in range(n):
        s_ik = s[:, k][:, None]  # (n, 1)
        s_jk = s[:, k][None, :]  # (1, n)
        d_ik = np.clip(s_ik - s, -50, 50)
        d_jk = np.clip(s_jk - s, -50, 50)
        p_hat += 1.0 / (1.0 + np.exp(d_ik) + np.exp(d_jk))

    # Subtract self-comparisons (k=i and k=j) and average over n-2 valid contexts
    diag = np.diag(s)

    # k=i: s_ik=s[i,i], s_jk=s[j,i]=s[i,j] (symmetric)
    d_ii = np.clip(diag[:, None] - s, -50, 50)  # s[i,i] - s[i,j]
    d_ji = np.zeros_like(s)  # s[j,i] - s[i,j] = 0 for symmetric
    p_hat -= 1.0 / (1.0 + np.exp(d_ii) + np.exp(d_ji))

    # k=j: s_ik=s[i,j], s_jk=s[j,j]
    d_ij = np.zeros_like(s)  # s[i,j] - s[i,j] = 0
    d_jj = np.clip(diag[None, :] - s, -50, 50)  # s[j,j] - s[i,j]
    p_hat -= 1.0 / (1.0 + np.exp(d_ij) + np.exp(d_jj))

    p_hat /= (n - 2)
    return p_hat


def denoise_hard_rank(s: np.ndarray, rank: int) -> np.ndarray:
    eigenvalues, eigvecs = np.linalg.eigh(s)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigvecs = eigenvalues[idx], eigvecs[:, idx]
    eig_trunc = np.maximum(eigenvalues[:rank], 0)
    s_hat = eigvecs[:, :rank] @ np.diag(eig_trunc) @ eigvecs[:, :rank].T
    d = np.sqrt(np.diag(s_hat))
    d[d == 0] = 1
    return s_hat / np.outer(d, d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iters", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log.info("Loading triplets...")
    train = load_triplets(DATA_DIR / "train_90.txt")
    test = load_triplets(DATA_DIR / "test_10.txt")
    log.info(f"Train: {len(train):,}, Test: {len(test):,}")

    counts, shown = build_counts_shown(train, N)
    observed = shown > 0

    p_obs = np.divide(counts + 1, shown + 2, out=np.full((N, N), 0.5), where=observed)
    np.fill_diagonal(p_obs, 1.0)

    # =========================================================================
    # Baselines
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("BASELINES")
    log.info("=" * 70)

    log.info(f"  Count RSM raw:      train={triplet_accuracy(p_obs, train):.4f}, "
             f"test={triplet_accuracy(p_obs, test):.4f}")
    log.info(f"  Count rank-35:      test={triplet_accuracy(denoise_hard_rank(p_obs, 35), test):.4f}")

    spose_path = PROJECT_ROOT / "data" / "things" / "spose_embedding_66d.txt"
    if spose_path.exists():
        spose = np.maximum(np.loadtxt(spose_path), 0)
        log.info(f"  SPoSE:              test={triplet_accuracy(spose @ spose.T, test):.4f}")

    vice_path = PROJECT_ROOT / "data" / "things" / "vice_embedding_66d.txt"
    if vice_path.exists():
        vice = np.maximum(np.loadtxt(vice_path), 0)
        log.info(f"  VICE:               test={triplet_accuracy(vice @ vice.T, test):.4f}")

    # =========================================================================
    # Iterative correction for multiple eta values
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("ITERATIVE SOFTMAX BIAS CORRECTION")
    log.info("=" * 70)

    p_clipped = np.clip(p_obs, 0.01, 0.99)

    for eta in [0.5, 1.0, 2.0, 5.0]:
        log.info(f"\n  --- eta = {eta} ---")
        s = logit(p_clipped).copy()
        np.fill_diagonal(s, np.max(s))

        for it in range(args.n_iters):
            p_hat = compute_predicted_probabilities(s)

            residual = p_obs - p_hat
            residual[~observed] = 0
            np.fill_diagonal(residual, 0)
            rmse = np.sqrt(np.mean(residual[observed] ** 2))

            s += eta * residual
            s = (s + s.T) / 2
            np.fill_diagonal(s, np.max(s))

            test_raw = triplet_accuracy(s, test)
            s_n = (s - s.min()) / (s.max() - s.min())
            np.fill_diagonal(s_n, 1.0)
            test_r35 = triplet_accuracy(denoise_hard_rank(s_n, 35), test)

            log.info(f"    iter {it:2d}: rmse={rmse:.6f}, "
                     f"raw test={test_raw:.4f}, rank-35 test={test_r35:.4f}")

            if rmse < 1e-6:
                log.info("    Converged!")
                break

        # Save best corrected matrix
        np.save(OUTPUT_DIR / f"s_corrected_eta{eta}.npy", s)

    log.info(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
