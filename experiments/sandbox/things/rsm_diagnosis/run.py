"""Diagnose count-based RSM from triplets: baseline, structural analysis, improvements.

Goal: Build RSM from train_90.txt using compute_similarity_matrix_from_triplets,
evaluate odd-one-out prediction on test_10.txt, then try to improve it.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.special import softmax

from src.utils import get_output_dir
from src.utils.helpers import compute_similarity_matrix_from_triplets

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "things" / "triplets_47"
N = 1854


def load_triplets(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float).astype(int)


def triplet_accuracy(s: np.ndarray, triplets: np.ndarray) -> float:
    """Predict odd-one-out from similarity matrix (argmax over 3 pairs)."""
    sij = s[triplets[:, 0], triplets[:, 1]]
    sik = s[triplets[:, 0], triplets[:, 2]]
    sjk = s[triplets[:, 1], triplets[:, 2]]
    sims = np.column_stack([sij, sik, sjk])
    return float(np.mean(np.argmax(sims, axis=1) == 0))


def build_counts_shown(triplets: np.ndarray, n: int):
    """Vectorized computation of counts and shown matrices."""
    ii, jj, kk = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))
    np.add.at(counts, (ii, jj), 1)
    np.add.at(counts, (jj, ii), 1)
    for a, b in [(ii, jj), (ii, kk), (jj, kk)]:
        np.add.at(shown, (a, b), 1)
        np.add.at(shown, (b, a), 1)
    return counts, shown


def denoise_low_rank(s: np.ndarray, rank: int) -> np.ndarray:
    """Low-rank PSD projection: keep top-k positive eigenvalues, renormalize."""
    eigenvalues, eigenvectors = np.linalg.eigh(s)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues_trunc = np.maximum(eigenvalues[:rank], 0)
    s_hat = eigenvectors[:, :rank] @ np.diag(eigenvalues_trunc) @ eigenvectors[:, :rank].T
    d = np.sqrt(np.diag(s_hat))
    d[d == 0] = 1
    s_hat = s_hat / np.outer(d, d)
    return s_hat


def compute_log_odds_rsm(counts: np.ndarray, shown: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Log-odds transform to undo softmax nonlinearity."""
    prob = np.divide(
        counts + alpha,
        shown + 2 * alpha,
        out=np.full_like(counts, np.nan),
        where=shown > 0,
    )
    eps = 1e-8
    lo = np.log(prob + eps) - np.log(1 - prob + eps)
    np.fill_diagonal(lo, np.nanmax(lo))
    lo_min, lo_max = np.nanmin(lo), np.nanmax(lo)
    lo = (lo - lo_min) / (lo_max - lo_min)
    lo[np.isnan(lo)] = 0.5
    np.fill_diagonal(lo, 1.0)
    return lo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    log.info("Loading triplets...")
    train = load_triplets(DATA_DIR / "train_90.txt")
    test = load_triplets(DATA_DIR / "test_10.txt")
    log.info(f"Train: {len(train):,}, Test: {len(test):,}")

    if args.subsample < 1.0:
        n_sub = int(len(train) * args.subsample)
        idx = rng.choice(len(train), size=n_sub, replace=False)
        train = train[idx]
        log.info(f"Subsampled train to {len(train):,}")

    counts, shown = build_counts_shown(train, N)

    # =========================================================================
    # Step 1: Baseline -- count-based RSM with different alpha values
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("STEP 1: Count-based RSM (compute_similarity_matrix_from_triplets)")
    log.info("=" * 70)

    for alpha in [0.0, 0.5, 1.0, 2.0, 5.0]:
        s = compute_similarity_matrix_from_triplets(N, train, alpha=alpha)
        s_pred = s.copy()
        s_pred[np.isnan(s_pred)] = 0.5
        train_acc = triplet_accuracy(s_pred, train)
        test_acc = triplet_accuracy(s_pred, test)
        log.info(f"  alpha={alpha:<4.1f}: train={train_acc:.4f}, test={test_acc:.4f}")

    # =========================================================================
    # Step 2: Structural analysis
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("STEP 2: Structural analysis of count-based RSM (alpha=1)")
    log.info("=" * 70)

    s = compute_similarity_matrix_from_triplets(N, train, alpha=1.0)
    nan_frac = np.isnan(s).mean()
    s_clean = s.copy()
    s_clean[np.isnan(s_clean)] = 0.5

    total_pairs = N * (N - 1) // 2
    observed_mask = shown[np.triu_indices(N, k=1)] > 0
    observed_pairs = observed_mask.sum()
    mean_shown = shown[shown > 0].mean()
    median_shown = np.median(shown[shown > 0])

    log.info(f"  NaN fraction: {nan_frac:.6f}")
    log.info(f"  Observed pairs: {observed_pairs:,} / {total_pairs:,} ({observed_pairs/total_pairs:.4f})")
    log.info(f"  Mean/median times shown: {mean_shown:.1f} / {median_shown:.1f}")

    eigenvalues = np.linalg.eigvalsh(s_clean)[::-1]
    n_pos = np.sum(eigenvalues > 0)
    n_neg = np.sum(eigenvalues < 0)
    cumvar = np.cumsum(eigenvalues[eigenvalues > 0]) / np.sum(np.abs(eigenvalues))

    log.info(f"  Eigenvalues: {n_pos} positive, {n_neg} negative")
    log.info(f"  Top-10: {eigenvalues[:10].round(2)}")
    log.info(f"  Variance explained: top-10={cumvar[9]:.4f}, top-30={cumvar[29]:.4f}, top-50={cumvar[49]:.4f}, top-100={cumvar[99]:.4f}")

    # Off-diagonal distribution
    off_diag = s_clean[~np.eye(N, dtype=bool)]
    log.info(f"  Off-diag: mean={off_diag.mean():.4f}, std={off_diag.std():.4f}, "
             f"min={off_diag.min():.4f}, max={off_diag.max():.4f}")

    # =========================================================================
    # Step 3: Log-odds transform
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("STEP 3: Log-odds RSM")
    log.info("=" * 70)

    s_lo = compute_log_odds_rsm(counts, shown, alpha=1.0)
    train_acc = triplet_accuracy(s_lo, train)
    test_acc = triplet_accuracy(s_lo, test)
    log.info(f"  Log-odds: train={train_acc:.4f}, test={test_acc:.4f}")

    # =========================================================================
    # Step 4: Low-rank PSD denoising on count-based RSM
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("STEP 4: Low-rank PSD denoising of count RSM")
    log.info("=" * 70)

    for rank in [10, 20, 30, 50, 80, 100, 150, 200, 500]:
        s_denoised = denoise_low_rank(s_clean, rank)
        train_acc = triplet_accuracy(s_denoised, train)
        test_acc = triplet_accuracy(s_denoised, test)
        log.info(f"  Rank-{rank:3d}: train={train_acc:.4f}, test={test_acc:.4f}")

    # =========================================================================
    # Step 5: Low-rank denoising on log-odds RSM
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("STEP 5: Low-rank PSD denoising of log-odds RSM")
    log.info("=" * 70)

    for rank in [10, 20, 30, 50, 80, 100, 150, 200, 500]:
        s_denoised = denoise_low_rank(s_lo, rank)
        train_acc = triplet_accuracy(s_denoised, train)
        test_acc = triplet_accuracy(s_denoised, test)
        log.info(f"  LogOdds+Rank-{rank:3d}: train={train_acc:.4f}, test={test_acc:.4f}")

    # =========================================================================
    # Step 6: NaN impact and difficulty analysis
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("STEP 6: NaN impact and difficulty analysis")
    log.info("=" * 70)

    s_nan = compute_similarity_matrix_from_triplets(N, train, alpha=1.0)
    has_nan = (
        np.isnan(s_nan[test[:, 0], test[:, 1]]) |
        np.isnan(s_nan[test[:, 0], test[:, 2]]) |
        np.isnan(s_nan[test[:, 1], test[:, 2]])
    )
    log.info(f"  Test triplets with NaN pair: {has_nan.sum():,} / {len(test):,} ({has_nan.mean():.4f})")

    if has_nan.sum() > 0:
        acc_no_nan = triplet_accuracy(s_clean, test[~has_nan])
        acc_has_nan = triplet_accuracy(s_clean, test[has_nan])
        log.info(f"  Accuracy (no NaN pairs): {acc_no_nan:.4f}")
        log.info(f"  Accuracy (has NaN pair): {acc_has_nan:.4f}")

    # Difficulty by similarity margin
    sij = s_clean[test[:, 0], test[:, 1]]
    sik = s_clean[test[:, 0], test[:, 2]]
    sjk = s_clean[test[:, 1], test[:, 2]]
    margin = sij - np.maximum(sik, sjk)
    correct = (sij > sik) & (sij > sjk)

    log.info(f"\n  Difficulty breakdown (margin = s_correct - max(s_incorrect)):")
    for lo, hi in [(-1, -0.05), (-0.05, 0.0), (0.0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 1)]:
        mask = (margin >= lo) & (margin < hi)
        if mask.sum() > 0:
            log.info(f"    [{lo:+.2f}, {hi:+.2f}): n={mask.sum():>6,}, acc={correct[mask].mean():.4f}")

    # =========================================================================
    # Step 7: Compare with SPoSE baseline
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("STEP 7: SPoSE baseline comparison")
    log.info("=" * 70)

    spose_path = PROJECT_ROOT / "data" / "things" / "spose_embedding_66d.txt"
    if spose_path.exists():
        spose = np.maximum(np.loadtxt(spose_path), 0)
        rsm_spose = spose @ spose.T
        # normalize diagonal to 1
        d = np.sqrt(np.diag(rsm_spose))
        d[d == 0] = 1
        rsm_spose_norm = rsm_spose / np.outer(d, d)

        spose_train = triplet_accuracy(rsm_spose_norm, train)
        spose_test = triplet_accuracy(rsm_spose_norm, test)
        log.info(f"  SPoSE RSM (dot-product, normalized): train={spose_train:.4f}, test={spose_test:.4f}")

        # Also test raw dot product (without normalization)
        spose_train_raw = triplet_accuracy(rsm_spose, train)
        spose_test_raw = triplet_accuracy(rsm_spose, test)
        log.info(f"  SPoSE RSM (raw dot-product):         train={spose_train_raw:.4f}, test={spose_test_raw:.4f}")

    # =========================================================================
    # Step 8: Double-centering + low-rank
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("STEP 8: Double-centering variants")
    log.info("=" * 70)

    # Convert similarity to distance, double-center, then denoise
    # This is what MDS does: G = -0.5 * H * D * H where H = I - 1/n * 11^T
    d_clean = 1.0 - s_clean
    h = np.eye(N) - np.ones((N, N)) / N
    gram = -0.5 * h @ d_clean @ h

    for rank in [30, 50, 80, 100]:
        s_dc = denoise_low_rank(gram, rank)
        # Gram matrix entries are inner products, not similarities in [0,1]
        # But argmax should still work for prediction
        train_acc = triplet_accuracy(s_dc, train)
        test_acc = triplet_accuracy(s_dc, test)
        log.info(f"  DoubleCenter+Rank-{rank:3d}: train={train_acc:.4f}, test={test_acc:.4f}")

    log.info(f"\nAll done. Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
