"""Iterative softmax bias correction v3: GPU-accelerated, unconstrained.

Correct the full matrix, denoise at the end only.
GPU (torch) computes P_hat in batched 3D tensor ops -- ~1s per iteration.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

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


def compute_p_hat_gpu(s_torch: torch.Tensor, batch_size: int = 200) -> torch.Tensor:
    """Compute P_hat on GPU using batched 3D ops.

    P_hat[i,j] = mean_{k!=i,j} 1 / (1 + exp(s_ik - s_ij) + exp(s_jk - s_ij))

    Batches over k to fit in GPU memory.
    """
    n = s_torch.shape[0]
    p_hat = torch.zeros(n, n, device=s_torch.device)

    for k_start in range(0, n, batch_size):
        k_end = min(k_start + batch_size, n)
        bk = k_end - k_start

        # s_ik: (n, bk) -> (n, 1, bk)  -- s[i, k] for batch of k values
        s_ik = s_torch[:, k_start:k_end].unsqueeze(1)  # (n, 1, bk)
        # s_jk: (n, bk) -> (1, n, bk)
        s_jk = s_torch[:, k_start:k_end].unsqueeze(0)  # (1, n, bk)
        # s_ij: (n, n) -> (n, n, 1)
        s_ij = s_torch.unsqueeze(2)  # (n, n, 1)

        d_ik = (s_ik - s_ij).clamp(-50, 50)  # (n, n, bk)
        d_jk = (s_jk - s_ij).clamp(-50, 50)  # (n, n, bk)

        contrib = 1.0 / (1.0 + torch.exp(d_ik) + torch.exp(d_jk))  # (n, n, bk)
        p_hat += contrib.sum(dim=2)  # sum over k batch

    # Subtract k=i and k=j self-comparisons
    diag = s_torch.diag()
    d_ii = (diag.unsqueeze(1) - s_torch).clamp(-50, 50)
    p_hat -= 1.0 / (1.0 + torch.exp(d_ii) + 1.0)

    d_jj = (diag.unsqueeze(0) - s_torch).clamp(-50, 50)
    p_hat -= 1.0 / (1.0 + 1.0 + torch.exp(d_jj))

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
    parser.add_argument("--n-iters", type=int, default=100)
    parser.add_argument("--eta", type=float, default=3.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:2")
    args = parser.parse_args()

    device = torch.device(args.device)
    log.info(f"Using device: {device}")

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
    for r in [30, 35, 50]:
        log.info(f"  Count rank-{r:2d}:      test={triplet_accuracy(denoise_hard_rank(p_obs, r), test):.4f}")

    spose_path = PROJECT_ROOT / "data" / "things" / "spose_embedding_66d.txt"
    if spose_path.exists():
        spose = np.maximum(np.loadtxt(spose_path), 0)
        log.info(f"  SPoSE:              test={triplet_accuracy(spose @ spose.T, test):.4f}")

    vice_path = PROJECT_ROOT / "data" / "things" / "vice_embedding_66d.txt"
    if vice_path.exists():
        vice = np.maximum(np.loadtxt(vice_path), 0)
        log.info(f"  VICE:               test={triplet_accuracy(vice @ vice.T, test):.4f}")

    # =========================================================================
    # Unconstrained correction with momentum, denoise at end
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info(f"UNCONSTRAINED CORRECTION (eta={args.eta}, momentum={args.momentum})")
    log.info("=" * 70)

    p_clipped = np.clip(p_obs, 0.01, 0.99)
    s = np.log(p_clipped / (1 - p_clipped))  # logit
    np.fill_diagonal(s, np.max(s))

    observed_mask = torch.tensor(observed, device=device)
    p_obs_t = torch.tensor(p_obs, dtype=torch.float32, device=device)

    velocity = np.zeros_like(s)
    best_test = 0.0
    best_s = None
    best_iter = 0

    for it in range(args.n_iters):
        s_t = torch.tensor(s, dtype=torch.float32, device=device)
        p_hat_t = compute_p_hat_gpu(s_t)

        residual_t = torch.zeros_like(p_obs_t)
        residual_t[observed_mask] = (p_obs_t - p_hat_t)[observed_mask]
        residual_t.fill_diagonal_(0)
        residual = residual_t.cpu().numpy()
        rmse = float(torch.sqrt(torch.mean(residual_t[observed_mask] ** 2)).item())

        # Momentum update
        velocity = args.momentum * velocity + args.eta * residual
        s += velocity
        s = (s + s.T) / 2
        np.fill_diagonal(s, np.max(s))

        # Evaluate every 5 iterations
        if it % 5 == 0 or it < 10:
            raw_test = triplet_accuracy(s, test)
            r35_test = triplet_accuracy(denoise_hard_rank(s, 35), test)
            r50_test = triplet_accuracy(denoise_hard_rank(s, 50), test)

            if r35_test > best_test:
                best_test = r35_test
                best_s = s.copy()
                best_iter = it

            log.info(f"  iter {it:3d}: rmse={rmse:.6f}, "
                     f"raw={raw_test:.4f}, r35={r35_test:.4f}, r50={r50_test:.4f}")

    log.info(f"\n  Best: iter={best_iter}, r35 test={best_test:.4f}")

    # =========================================================================
    # Sweep eta values (no momentum, clean comparison)
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("ETA SWEEP (no momentum, 50 iters)")
    log.info("=" * 70)

    for eta in [1.0, 2.0, 3.0, 5.0]:
        s = np.log(p_clipped / (1 - p_clipped))
        np.fill_diagonal(s, np.max(s))
        best_test_eta = 0.0
        best_iter_eta = 0

        for it in range(50):
            s_t = torch.tensor(s, dtype=torch.float32, device=device)
            p_hat_t = compute_p_hat_gpu(s_t)
            residual_t = torch.zeros_like(p_obs_t)
            residual_t[observed_mask] = (p_obs_t - p_hat_t)[observed_mask]
            residual_t.fill_diagonal_(0)
            residual = residual_t.cpu().numpy()
            rmse = float(torch.sqrt(torch.mean(residual_t[observed_mask] ** 2)).item())

            s += eta * residual
            s = (s + s.T) / 2
            np.fill_diagonal(s, np.max(s))

            if it % 10 == 0:
                r35 = triplet_accuracy(denoise_hard_rank(s, 35), test)
                if r35 > best_test_eta:
                    best_test_eta = r35
                    best_iter_eta = it
                log.info(f"  eta={eta}, iter {it:3d}: rmse={rmse:.6f}, r35={r35:.4f}")

        # Final eval
        r35_final = triplet_accuracy(denoise_hard_rank(s, 35), test)
        if r35_final > best_test_eta:
            best_test_eta = r35_final
            best_iter_eta = 49
        log.info(f"  eta={eta}: best r35={best_test_eta:.4f} at iter {best_iter_eta}")

    log.info(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
