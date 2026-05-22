"""Iterative softmax bias correction with low-rank constraint.

v1 failed because corrections went into the noise subspace (rank > 35).
v2 alternates between:
  1. Bias correction: s += eta * (P_obs - P_hat(s))
  2. Low-rank projection: s = rank_k(s)

This forces the correction to only modify the signal-relevant structure.
Parameterize s = L @ L.T directly and compute gradients through P_hat.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.special import logit
from joblib import Parallel, delayed

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


def _p_hat_chunk(s: np.ndarray, k_indices: np.ndarray) -> np.ndarray:
    """Compute partial P_hat sum for a chunk of k values."""
    n = len(s)
    partial = np.zeros((n, n))
    for k in k_indices:
        d_ik = np.clip(s[:, k][:, None] - s, -50, 50)
        d_jk = np.clip(s[:, k][None, :] - s, -50, 50)
        partial += 1.0 / (1.0 + np.exp(d_ik) + np.exp(d_jk))
    return partial


def compute_p_hat(s: np.ndarray, n_jobs: int = -1) -> np.ndarray:
    """P_hat[i,j] = mean_{k!=i,j} softmax(s_ij, s_ik, s_jk)[0]. Parallelized."""
    n = len(s)
    diag = np.diag(s)

    chunks = np.array_split(np.arange(n), max(1, abs(n_jobs) if n_jobs != -1 else 16))
    partials = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_p_hat_chunk)(s, chunk) for chunk in chunks
    )
    p_hat = sum(partials)

    # Subtract k=i and k=j self-comparisons
    d_ii = np.clip(diag[:, None] - s, -50, 50)
    p_hat -= 1.0 / (1.0 + np.exp(d_ii) + np.ones_like(s))

    d_jj = np.clip(diag[None, :] - s, -50, 50)
    p_hat -= 1.0 / (1.0 + np.ones_like(s) + np.exp(d_jj))

    p_hat /= (n - 2)
    return p_hat


def low_rank_project(s: np.ndarray, rank: int) -> np.ndarray:
    """Project to rank-k PSD, normalize diagonal to max."""
    eigenvalues, eigvecs = np.linalg.eigh(s)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigvecs = eigenvalues[idx], eigvecs[:, idx]
    eig_trunc = np.maximum(eigenvalues[:rank], 0)
    return eigvecs[:, :rank] @ np.diag(eig_trunc) @ eigvecs[:, :rank].T


def normalize_diag(s: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.abs(np.diag(s)))
    d[d == 0] = 1
    return s / np.outer(d, d)


def compute_gradient_L(l: np.ndarray, p_obs: np.ndarray, p_hat: np.ndarray,
                       observed: np.ndarray) -> np.ndarray:
    """Gradient of ||P_obs - P_hat||^2 w.r.t. L where s = LL^T.

    Uses the chain rule: dL/dL_ia = sum_jb (dLoss/ds_jb) * (ds_jb/dL_ia)
    where ds_jb/dL_ia = delta(j,i)*L_ba + delta(b,i)*L_ja
    so dL/dL_ia = 2 * sum_j (dLoss/ds_ij) * L_ja = 2 * G @ L
    where G_ij = dLoss/ds_ij = -2 * (P_obs - P_hat) * (dP_hat/ds_ij)

    For the softmax model, dP_hat[i,j]/ds[i,j] involves all contexts k.
    This is complex. Instead, use a simpler approach: treat the residual
    P_obs - P_hat as an approximate gradient for s, then project to L-space.
    """
    residual = np.zeros_like(p_obs)
    residual[observed] = (p_obs - p_hat)[observed]
    np.fill_diagonal(residual, 0)
    residual = (residual + residual.T) / 2
    # grad_L = residual @ L (chain rule for s = LL^T)
    return 2.0 * residual @ l


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iters", type=int, default=20)
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

    log.info(f"  Count RSM raw:      test={triplet_accuracy(p_obs, test):.4f}")

    p_r35 = normalize_diag(low_rank_project(p_obs, 35))
    log.info(f"  Count rank-35:      test={triplet_accuracy(p_r35, test):.4f}")

    spose_path = PROJECT_ROOT / "data" / "things" / "spose_embedding_66d.txt"
    if spose_path.exists():
        spose = np.maximum(np.loadtxt(spose_path), 0)
        log.info(f"  SPoSE:              test={triplet_accuracy(spose @ spose.T, test):.4f}")

    vice_path = PROJECT_ROOT / "data" / "things" / "vice_embedding_66d.txt"
    if vice_path.exists():
        vice = np.maximum(np.loadtxt(vice_path), 0)
        log.info(f"  VICE:               test={triplet_accuracy(vice @ vice.T, test):.4f}")

    # =========================================================================
    # Approach 1: Alternating correction + rank projection (lean grid)
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("APPROACH 1: Alternating correction + rank projection")
    log.info("=" * 70)

    p_clipped = np.clip(p_obs, 0.01, 0.99)

    for rank, eta in [(35, 3.0), (35, 10.0), (50, 5.0)]:
        log.info(f"\n  rank={rank}, eta={eta}:")
        s = logit(p_clipped).copy()
        s = low_rank_project(s, rank)

        for it in range(10):
            p_hat = compute_p_hat(s)
            residual = p_obs - p_hat
            residual[~observed] = 0
            np.fill_diagonal(residual, 0)
            rmse = np.sqrt(np.mean(residual[observed] ** 2))

            s += eta * residual
            s = (s + s.T) / 2
            s = low_rank_project(s, rank)

            s_eval = normalize_diag(s)
            test_acc = triplet_accuracy(s_eval, test)
            log.info(f"    iter {it:2d}: rmse={rmse:.6f}, test={test_acc:.4f}")

    # =========================================================================
    # Approach 2: Gradient descent on L (s = LL^T), lean grid
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("APPROACH 2: Gradient descent on L (s = LL^T)")
    log.info("=" * 70)

    for rank, lr in [(35, 0.05), (50, 0.05)]:
        log.info(f"\n  rank={rank}, lr={lr}:")

        s_init = logit(p_clipped)
        eigenvalues, eigvecs = np.linalg.eigh(s_init)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigvecs = eigenvalues[idx], eigvecs[:, idx]
        eig_pos = np.maximum(eigenvalues[:rank], 1e-6)
        l = eigvecs[:, :rank] * np.sqrt(eig_pos)[None, :]

        m = np.zeros_like(l)
        v = np.zeros_like(l)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for it in range(10):
            s = l @ l.T
            p_hat = compute_p_hat(s)
            residual = p_obs - p_hat
            residual[~observed] = 0
            np.fill_diagonal(residual, 0)
            rmse = np.sqrt(np.mean(residual[observed] ** 2))

            grad = compute_gradient_L(l, p_obs, p_hat, observed)

            t = it + 1
            m = beta1 * m + (1 - beta1) * grad
            v_adam = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v_adam / (1 - beta2 ** t)
            l += lr * m_hat / (np.sqrt(v_hat) + eps)
            v = v_adam

            s_eval = normalize_diag(l @ l.T)
            test_acc = triplet_accuracy(s_eval, test)
            log.info(f"    iter {it:2d}: rmse={rmse:.6f}, test={test_acc:.4f}")

    log.info(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
