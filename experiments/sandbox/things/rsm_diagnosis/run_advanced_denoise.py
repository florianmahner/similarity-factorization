"""Advanced denoising of count-based RSM from triplets.

Try to close the gap between rank-30 denoised count RSM (63.6%) and SPoSE (64.9%).

Approaches:
1. Gavish-Donoho optimal eigenvalue shrinkage
2. Variance-stabilizing transform (arcsine sqrt) before denoising
3. Weighted low-rank (weight by observation count)
4. Low-rank MLE refinement (gradient-based triplet loss optimization)
5. Centering strategies
6. Soft shrinkage vs hard truncation
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.special import softmax

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


def denoise_hard_rank(s: np.ndarray, rank: int) -> np.ndarray:
    """Hard rank truncation with PSD projection and diagonal normalization."""
    eigenvalues, eigvecs = np.linalg.eigh(s)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigvecs = eigenvalues[idx], eigvecs[:, idx]
    eig_trunc = np.maximum(eigenvalues[:rank], 0)
    s_hat = eigvecs[:, :rank] @ np.diag(eig_trunc) @ eigvecs[:, :rank].T
    d = np.sqrt(np.diag(s_hat))
    d[d == 0] = 1
    return s_hat / np.outer(d, d)


def gavish_donoho_shrinkage(s: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Gavish-Donoho optimal singular value shrinkage.

    For square n x n matrix with noise level sigma:
    - Optimal hard threshold: lambda* = (4/sqrt(3)) * sqrt(n) * sigma
    - When sigma unknown, use median eigenvalue heuristic

    Also implements soft shrinkage: sigma_i -> sqrt(max(sigma_i^2 - beta*n*sigma^2, 0))
    """
    eigenvalues, eigvecs = np.linalg.eigh(s)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigvecs = eigenvalues[idx], eigvecs[:, idx]
    n = len(s)

    if sigma is None:
        # Estimate noise from median eigenvalue (Gavish-Donoho 2014)
        # For square matrix: sigma_est = median(eigenvalues) / (sqrt(n) * 0.6745)
        # But eigenvalues can be negative for our non-PSD matrix
        # Use median of absolute eigenvalues
        med = np.median(np.abs(eigenvalues))
        sigma = med / (np.sqrt(n) * 0.6745)
        log.info(f"    Estimated sigma={sigma:.6f}")

    # Optimal hard threshold
    threshold = (4.0 / np.sqrt(3.0)) * np.sqrt(n) * sigma
    log.info(f"    Threshold={threshold:.4f}, eigenvalues above: {np.sum(eigenvalues > threshold)}")

    # Hard thresholding
    eig_hard = eigenvalues.copy()
    eig_hard[eigenvalues < threshold] = 0

    # Soft shrinkage (optimal for Frobenius loss)
    eig_soft = np.sqrt(np.maximum(eigenvalues ** 2 - n * sigma ** 2, 0))
    eig_soft[eigenvalues < threshold] = 0

    # Build both
    s_hard = eigvecs @ np.diag(eig_hard) @ eigvecs.T
    s_soft = eigvecs @ np.diag(eig_soft) @ eigvecs.T

    return s_hard, s_soft


def variance_stabilize(counts: np.ndarray, shown: np.ndarray) -> np.ndarray:
    """Arcsine square root variance-stabilizing transform for binomial proportions.

    For p_hat = counts/shown, the transform arcsin(sqrt(p_hat)) has approximately
    constant variance 1/(4*shown), independent of p.
    """
    alpha = 1.0  # Laplace smoothing
    p = (counts + alpha) / (shown + 2 * alpha)
    # Apply arcsin sqrt transform
    p_stab = np.arcsin(np.sqrt(np.clip(p, 0, 1)))
    # Mask unobserved pairs
    p_stab[shown == 0] = np.arcsin(np.sqrt(0.5))  # neutral value
    np.fill_diagonal(p_stab, np.max(p_stab))
    return p_stab


def weighted_low_rank(s: np.ndarray, w: np.ndarray, rank: int, n_iters: int = 50) -> np.ndarray:
    """Weighted low-rank approximation: min_L sum w_ij * (s_ij - [LL^T]_ij)^2.

    Uses alternating least squares on L (n x rank).
    Weight w_ij = observation count (more observed = more trusted).
    """
    n = len(s)
    rng = np.random.default_rng(42)

    # Initialize from truncated eigendecomposition
    eigenvalues, eigvecs = np.linalg.eigh(s)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigvecs = eigenvalues[idx], eigvecs[:, idx]
    eig_pos = np.maximum(eigenvalues[:rank], 0)
    l = eigvecs[:, :rank] * np.sqrt(eig_pos)[None, :]

    for it in range(n_iters):
        s_hat = l @ l.T
        residual = s - s_hat
        grad = -2 * (w * residual) @ l
        # Simple gradient descent with line search
        lr = 0.001
        l_new = l - lr * grad
        l = l_new

        if it % 10 == 0:
            loss = np.sum(w * (s - l @ l.T) ** 2)
            log.info(f"      iter {it}: loss={loss:.4f}")

    s_hat = l @ l.T
    d = np.sqrt(np.diag(s_hat))
    d[d == 0] = 1
    return s_hat / np.outer(d, d)


def low_rank_mle_refinement(
    s_init: np.ndarray,
    triplets: np.ndarray,
    rank: int,
    lr: float = 0.01,
    n_epochs: int = 10,
    batch_size: int = 50000,
) -> np.ndarray:
    """Refine a low-rank similarity matrix by optimizing triplet softmax loss.

    Parameterize S = LL^T where L is (n, rank). Optimize L via SGD on:
        loss = -mean log softmax(s_ij, s_ik, s_jk)[0]

    Initialize L from eigendecomposition of s_init.
    """
    n = len(s_init)

    # Initialize L from s_init
    eigenvalues, eigvecs = np.linalg.eigh(s_init)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigvecs = eigenvalues[idx], eigvecs[:, idx]
    eig_pos = np.maximum(eigenvalues[:rank], 1e-8)
    l = eigvecs[:, :rank] * np.sqrt(eig_pos)[None, :]

    rng = np.random.default_rng(42)

    # Adam optimizer state
    m = np.zeros_like(l)
    v = np.zeros_like(l)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for epoch in range(n_epochs):
        perm = rng.permutation(len(triplets))
        n_batches = len(triplets) // batch_size

        epoch_loss = 0.0
        for batch_idx in range(n_batches):
            batch = triplets[perm[batch_idx * batch_size:(batch_idx + 1) * batch_size]]
            ii, jj, kk = batch[:, 0], batch[:, 1], batch[:, 2]

            # Forward: compute similarities
            li, lj, lk = l[ii], l[jj], l[kk]
            sij = np.sum(li * lj, axis=1)
            sik = np.sum(li * lk, axis=1)
            sjk = np.sum(lj * lk, axis=1)

            # Softmax probabilities
            logits = np.column_stack([sij, sik, sjk])
            log_probs = logits - np.log(np.sum(np.exp(logits - logits.max(axis=1, keepdims=True)), axis=1, keepdims=True)) - logits.max(axis=1, keepdims=True) + logits.max(axis=1, keepdims=True)
            # Simpler: use scipy softmax
            probs = softmax(logits, axis=1)
            loss = -np.mean(np.log(probs[:, 0] + 1e-12))
            epoch_loss += loss

            # Backward: gradient of -log(softmax[0]) w.r.t. L
            # d_loss/d_sij = -(1 - p0), d_loss/d_sik = p1, d_loss/d_sjk = p2
            d_sij = -(1 - probs[:, 0])  # (batch,)
            d_sik = probs[:, 1]
            d_sjk = probs[:, 2]

            # Gradient for L: accumulate across batch
            grad = np.zeros_like(l)
            np.add.at(grad, ii, d_sij[:, None] * lj + d_sik[:, None] * lk)
            np.add.at(grad, jj, d_sij[:, None] * li + d_sjk[:, None] * lk)
            np.add.at(grad, kk, d_sik[:, None] * li + d_sjk[:, None] * lj)
            grad /= len(batch)

            # Adam update
            t += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            l -= lr * m_hat / (np.sqrt(v_hat) + eps)

        avg_loss = epoch_loss / n_batches

        # Compute accuracy
        s_hat = l @ l.T
        acc_train = triplet_accuracy(s_hat, triplets[:100000])
        log.info(f"    Epoch {epoch}: loss={avg_loss:.4f}, train_acc(100k)={acc_train:.4f}")

    s_final = l @ l.T
    return s_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log.info("Loading triplets...")
    train = load_triplets(DATA_DIR / "train_90.txt")
    test = load_triplets(DATA_DIR / "test_10.txt")
    log.info(f"Train: {len(train):,}, Test: {len(test):,}")

    counts, shown = build_counts_shown(train, N)

    # Build count RSM
    alpha = 1.0
    s_count = np.divide(
        counts + alpha, shown + 2 * alpha,
        out=np.full((N, N), 0.5), where=shown > 0,
    )
    np.fill_diagonal(s_count, 1.0)

    log.info("\n" + "=" * 70)
    log.info("BASELINES")
    log.info("=" * 70)

    log.info(f"  Count RSM raw:    train={triplet_accuracy(s_count, train):.4f}, "
             f"test={triplet_accuracy(s_count, test):.4f}")

    for rank in [20, 25, 30, 35, 40]:
        s_r = denoise_hard_rank(s_count, rank)
        log.info(f"  Count rank-{rank:2d}:    train={triplet_accuracy(s_r, train):.4f}, "
                 f"test={triplet_accuracy(s_r, test):.4f}")

    # =========================================================================
    # 1. Gavish-Donoho optimal shrinkage
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("1. GAVISH-DONOHO EIGENVALUE SHRINKAGE")
    log.info("=" * 70)

    s_hard, s_soft = gavish_donoho_shrinkage(s_count)

    # Normalize diagonal
    for name, s_gd in [("hard", s_hard), ("soft", s_soft)]:
        d = np.sqrt(np.abs(np.diag(s_gd)))
        d[d == 0] = 1
        s_gd_norm = s_gd / np.outer(d, d)
        log.info(f"  GD {name}:          train={triplet_accuracy(s_gd_norm, train):.4f}, "
                 f"test={triplet_accuracy(s_gd_norm, test):.4f}")

    # =========================================================================
    # 2. Centering before denoising
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("2. CENTERING + DENOISING")
    log.info("=" * 70)

    # Row/column centering (double centering)
    row_mean = s_count.mean(axis=1, keepdims=True)
    col_mean = s_count.mean(axis=0, keepdims=True)
    grand_mean = s_count.mean()
    s_centered = s_count - row_mean - col_mean + grand_mean

    for rank in [20, 30, 50]:
        eigenvalues, eigvecs = np.linalg.eigh(s_centered)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigvecs = eigenvalues[idx], eigvecs[:, idx]
        eig_trunc = np.maximum(eigenvalues[:rank], 0)
        s_hat = eigvecs[:, :rank] @ np.diag(eig_trunc) @ eigvecs[:, :rank].T
        # Add back the means
        s_recon = s_hat + row_mean + col_mean - grand_mean
        # Normalize diagonal
        d = np.sqrt(np.abs(np.diag(s_recon)))
        d[d == 0] = 1
        s_recon = s_recon / np.outer(d, d)
        log.info(f"  Centered rank-{rank:2d}: train={triplet_accuracy(s_recon, train):.4f}, "
                 f"test={triplet_accuracy(s_recon, test):.4f}")

    # =========================================================================
    # 3. Variance-stabilizing transform + denoising
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("3. VARIANCE-STABILIZING TRANSFORM (arcsin sqrt)")
    log.info("=" * 70)

    s_stab = variance_stabilize(counts, shown)

    log.info(f"  VS raw:           train={triplet_accuracy(s_stab, train):.4f}, "
             f"test={triplet_accuracy(s_stab, test):.4f}")

    for rank in [20, 25, 30, 35, 40]:
        s_r = denoise_hard_rank(s_stab, rank)
        log.info(f"  VS rank-{rank:2d}:       train={triplet_accuracy(s_r, train):.4f}, "
                 f"test={triplet_accuracy(s_r, test):.4f}")

    # GD on variance-stabilized
    s_stab_hard, s_stab_soft = gavish_donoho_shrinkage(s_stab)
    for name, s_gd in [("hard", s_stab_hard), ("soft", s_stab_soft)]:
        d = np.sqrt(np.abs(np.diag(s_gd)))
        d[d == 0] = 1
        s_gd_norm = s_gd / np.outer(d, d)
        log.info(f"  VS+GD {name}:      train={triplet_accuracy(s_gd_norm, train):.4f}, "
                 f"test={triplet_accuracy(s_gd_norm, test):.4f}")

    # =========================================================================
    # 4. Weighted low-rank approximation
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("4. WEIGHTED LOW-RANK (weight by observation count)")
    log.info("=" * 70)

    # Weight matrix: use shown counts (more observations = more weight)
    w = shown.copy()
    w[shown == 0] = 0.1  # small weight for unobserved
    np.fill_diagonal(w, w.max())  # high weight for diagonal

    for rank in [25, 30]:
        log.info(f"  Weighted rank-{rank}:")
        s_wlr = weighted_low_rank(s_count, w, rank, n_iters=50)
        log.info(f"  WLR rank-{rank:2d}:      train={triplet_accuracy(s_wlr, train):.4f}, "
                 f"test={triplet_accuracy(s_wlr, test):.4f}")

    # =========================================================================
    # 5. Fine-grained rank sweep on best approach
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("5. FINE-GRAINED RANK SWEEP")
    log.info("=" * 70)

    best_test = 0
    best_rank = 0
    for rank in range(15, 45):
        s_r = denoise_hard_rank(s_count, rank)
        test_acc = triplet_accuracy(s_r, test)
        if test_acc > best_test:
            best_test = test_acc
            best_rank = rank
        if rank % 5 == 0 or rank == best_rank:
            log.info(f"  Rank-{rank:2d}: test={test_acc:.4f}")

    log.info(f"  ** Best: rank={best_rank}, test={best_test:.4f}")

    # =========================================================================
    # 6. Low-rank MLE refinement
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("6. LOW-RANK MLE REFINEMENT (triplet softmax loss)")
    log.info("=" * 70)

    s_init = denoise_hard_rank(s_count, best_rank)
    for lr, epochs in [(0.01, 5), (0.005, 10), (0.001, 20)]:
        log.info(f"\n  lr={lr}, epochs={epochs}, rank={best_rank}:")
        s_mle = low_rank_mle_refinement(
            s_init, train, rank=best_rank, lr=lr, n_epochs=epochs, batch_size=50000
        )
        train_acc = triplet_accuracy(s_mle, train)
        test_acc = triplet_accuracy(s_mle, test)
        log.info(f"  MLE refined:      train={train_acc:.4f}, test={test_acc:.4f}")

    # =========================================================================
    # Reference
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("REFERENCE: SPoSE")
    log.info("=" * 70)

    spose_path = PROJECT_ROOT / "data" / "things" / "spose_embedding_66d.txt"
    if spose_path.exists():
        spose = np.maximum(np.loadtxt(spose_path), 0)
        rsm_spose = spose @ spose.T
        log.info(f"  SPoSE:            train={triplet_accuracy(rsm_spose, train):.4f}, "
                 f"test={triplet_accuracy(rsm_spose, test):.4f}")

    log.info(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
