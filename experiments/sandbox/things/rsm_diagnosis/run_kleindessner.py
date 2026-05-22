"""Implement Kleindessner & von Luxburg (NeurIPS 2017) kernel functions from triplets.

k1: For each object a, build a feature vector indexed by all pairs (i,j),
    where the entry encodes whether a ranks i closer than j or vice versa.
    K1(a,b) = normalized_phi_a^T . normalized_phi_b
    = "Do a and b rank the rest of the world similarly?"

k2: For each object a, feature vector indexed by (anchor, reference) pairs,
    encoding whether a is closer to the anchor than the reference is.
    K2(a,b) = normalized_phi_a^T . normalized_phi_b
    = "Are a and b ranked similarly by all other objects?"

Key insight vs count-based RSM: Count-based uses ~6.5 observations per pair.
Kleindessner kernels use ALL triplets involving each object (~4000 per object).
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from src.utils import get_output_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "things" / "triplets_47"
N = 1854


def load_triplets(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float).astype(int)


def pair_index(i: int, j: int, n: int) -> int:
    """Map ordered pair (i<j) to linear index in [0, C(n,2))."""
    return i * (2 * n - i - 1) // 2 + (j - i - 1)


def triplet_accuracy(s: np.ndarray, triplets: np.ndarray) -> float:
    sij = s[triplets[:, 0], triplets[:, 1]]
    sik = s[triplets[:, 0], triplets[:, 2]]
    sjk = s[triplets[:, 1], triplets[:, 2]]
    return float(np.mean((sij > sik) & (sij > sjk)))


def build_k1_kernel(triplets: np.ndarray, n: int) -> np.ndarray:
    """Build k1 kernel (Kendall tau ranking similarity). Fully sparse."""
    n_pairs = n * (n - 1) // 2
    log.info(f"  k1: n={n}, feature_dim={n_pairs:,}, triplets={len(triplets):,}")

    # Each triplet (i,j,k) with (i,j) similar pair gives:
    #   Anchor i: d(i,j) < d(i,k) -> for pair (min(j,k), max(j,k)):
    #     +1 if closer to the smaller index, -1 if closer to the larger
    #   Anchor j: d(j,i) < d(j,k) -> same logic for pair (min(i,k), max(i,k))

    rows = np.empty(2 * len(triplets), dtype=np.int32)
    cols = np.empty(2 * len(triplets), dtype=np.int32)
    vals = np.empty(2 * len(triplets), dtype=np.float32)

    for idx in range(len(triplets)):
        i, j, k = triplets[idx]

        # Anchor i: closer to j than k
        if j < k:
            rows[2 * idx] = i
            cols[2 * idx] = pair_index(j, k, n)
            vals[2 * idx] = 1.0  # closer to smaller index
        else:
            rows[2 * idx] = i
            cols[2 * idx] = pair_index(k, j, n)
            vals[2 * idx] = -1.0  # closer to larger index

        # Anchor j: closer to i than k
        if i < k:
            rows[2 * idx + 1] = j
            cols[2 * idx + 1] = pair_index(i, k, n)
            vals[2 * idx + 1] = 1.0
        else:
            rows[2 * idx + 1] = j
            cols[2 * idx + 1] = pair_index(k, i, n)
            vals[2 * idx + 1] = -1.0

        if idx % 1000000 == 0:
            log.info(f"    {idx:,}/{len(triplets):,}")

    log.info(f"    Building sparse phi...")
    # sum_mat: sum of +1/-1 per (anchor, pair)
    sum_mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n_pairs), dtype=np.float32)
    # count_mat: total observations per (anchor, pair)
    count_mat = sparse.csr_matrix(
        (np.ones(len(vals), dtype=np.float32), (rows, cols)), shape=(n, n_pairs), dtype=np.float32
    )

    # phi = sum / count element-wise (sparse)
    count_mat.data = 1.0 / count_mat.data
    phi = sum_mat.multiply(count_mat).tocsr()

    # Normalize rows to unit length
    phi = normalize(phi, norm="l2", axis=1)

    log.info(f"    phi: nnz={phi.nnz:,}, density={phi.nnz / (n * n_pairs):.6f}")
    log.info(f"    Computing K1 = Phi @ Phi.T ...")
    k1 = (phi @ phi.T).toarray()

    return k1


def build_k2_kernel(triplets: np.ndarray, n: int) -> np.ndarray:
    """Build k2 kernel (ranked similarly by others). Fully sparse.

    Feature indexed by (anchor, reference) = flat index anchor*n + reference.
    From triplet (i,j,k):
      - d(i,j)<d(i,k): object j is closer to anchor i than reference k is
        -> phi[j][i*n + k] += 1
      - d(i,k)>d(i,j): object k is farther from anchor i than reference j
        -> phi[k][i*n + j] -= 1
      - d(j,i)<d(j,k): object i is closer to anchor j than reference k is
        -> phi[i][j*n + k] += 1
      - d(j,k)>d(j,i): object k is farther from anchor j than reference i
        -> phi[k][j*n + i] -= 1
    """
    feat_dim = n * n
    log.info(f"  k2: n={n}, feature_dim={feat_dim:,}, triplets={len(triplets):,}")

    rows = np.empty(4 * len(triplets), dtype=np.int32)
    cols = np.empty(4 * len(triplets), dtype=np.int32)
    vals = np.empty(4 * len(triplets), dtype=np.float32)

    for idx in range(len(triplets)):
        i, j, k = triplets[idx]
        base = 4 * idx

        # From anchor i: j is closer, k is farther
        rows[base] = j
        cols[base] = i * n + k
        vals[base] = 1.0

        rows[base + 1] = k
        cols[base + 1] = i * n + j
        vals[base + 1] = -1.0

        # From anchor j: i is closer, k is farther
        rows[base + 2] = i
        cols[base + 2] = j * n + k
        vals[base + 2] = 1.0

        rows[base + 3] = k
        cols[base + 3] = j * n + i
        vals[base + 3] = -1.0

        if idx % 1000000 == 0:
            log.info(f"    {idx:,}/{len(triplets):,}")

    log.info(f"    Building sparse phi...")
    sum_mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n, feat_dim), dtype=np.float32)
    count_mat = sparse.csr_matrix(
        (np.ones(len(vals), dtype=np.float32), (rows, cols)), shape=(n, feat_dim), dtype=np.float32
    )

    count_mat.data = 1.0 / count_mat.data
    phi = sum_mat.multiply(count_mat).tocsr()
    phi = normalize(phi, norm="l2", axis=1)

    log.info(f"    phi: nnz={phi.nnz:,}")
    log.info(f"    Computing K2 = Phi @ Phi.T ...")
    k2 = (phi @ phi.T).toarray()

    return k2


def fix_diagonal_dominance(k: np.ndarray) -> np.ndarray:
    """K -> K - lambda_min * I (Section 2.1 of paper)."""
    eigenvalues = np.linalg.eigvalsh(k)
    lam_min = eigenvalues[0]
    if lam_min < 0:
        return k - lam_min * np.eye(len(k))
    return k.copy()


def denoise_low_rank(s: np.ndarray, rank: int) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(s)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues_trunc = np.maximum(eigenvalues[:rank], 0)
    s_hat = eigenvectors[:, :rank] @ np.diag(eigenvalues_trunc) @ eigenvectors[:, :rank].T
    d = np.sqrt(np.diag(s_hat))
    d[d == 0] = 1
    return s_hat / np.outer(d, d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-k2", action="store_true")
    args = parser.parse_args()

    log.info("Loading triplets...")
    train = load_triplets(DATA_DIR / "train_90.txt")
    test = load_triplets(DATA_DIR / "test_10.txt")
    log.info(f"Train: {len(train):,}, Test: {len(test):,}")

    # =========================================================================
    # Baseline
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("BASELINE: Count-based RSM")
    log.info("=" * 70)

    from src.utils.helpers import compute_similarity_matrix_from_triplets
    s_count = compute_similarity_matrix_from_triplets(N, train, alpha=1.0)
    s_count[np.isnan(s_count)] = 0.5

    log.info(f"  Count RSM:        train={triplet_accuracy(s_count, train):.4f}, "
             f"test={triplet_accuracy(s_count, test):.4f}")

    s_count_r30 = denoise_low_rank(s_count, 30)
    log.info(f"  Count RSM rank30: train={triplet_accuracy(s_count_r30, train):.4f}, "
             f"test={triplet_accuracy(s_count_r30, test):.4f}")

    # =========================================================================
    # K1
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("KERNEL K1 (Kendall tau ranking similarity)")
    log.info("=" * 70)

    k1 = build_k1_kernel(train, N)
    np.save(OUTPUT_DIR / "k1_raw.npy", k1)

    log.info(f"  K1 raw:           train={triplet_accuracy(k1, train):.4f}, "
             f"test={triplet_accuracy(k1, test):.4f}")

    k1_fixed = fix_diagonal_dominance(k1)
    log.info(f"  K1 diag-fixed:    train={triplet_accuracy(k1_fixed, train):.4f}, "
             f"test={triplet_accuracy(k1_fixed, test):.4f}")

    for rank in [10, 20, 30, 50, 80, 100]:
        k1_lr = denoise_low_rank(k1, rank)
        log.info(f"  K1 rank-{rank:3d}:      train={triplet_accuracy(k1_lr, train):.4f}, "
                 f"test={triplet_accuracy(k1_lr, test):.4f}")

    eig_k1 = np.linalg.eigvalsh(k1)[::-1]
    log.info(f"\n  K1 structure:")
    log.info(f"    Positive eigenvalues: {np.sum(eig_k1 > 0)}")
    log.info(f"    Negative eigenvalues: {np.sum(eig_k1 < 0)}")
    log.info(f"    Top-10: {eig_k1[:10].round(4)}")
    off = k1[~np.eye(N, dtype=bool)]
    log.info(f"    Off-diag range: [{off.min():.4f}, {off.max():.4f}], mean={off.mean():.4f}")

    # =========================================================================
    # K2
    # =========================================================================
    if not args.skip_k2:
        log.info("\n" + "=" * 70)
        log.info("KERNEL K2 (ranked similarly by others)")
        log.info("=" * 70)

        k2 = build_k2_kernel(train, N)
        np.save(OUTPUT_DIR / "k2_raw.npy", k2)

        log.info(f"  K2 raw:           train={triplet_accuracy(k2, train):.4f}, "
                 f"test={triplet_accuracy(k2, test):.4f}")

        k2_fixed = fix_diagonal_dominance(k2)
        log.info(f"  K2 diag-fixed:    train={triplet_accuracy(k2_fixed, train):.4f}, "
                 f"test={triplet_accuracy(k2_fixed, test):.4f}")

        for rank in [10, 20, 30, 50, 80, 100]:
            k2_lr = denoise_low_rank(k2, rank)
            log.info(f"  K2 rank-{rank:3d}:      train={triplet_accuracy(k2_lr, train):.4f}, "
                     f"test={triplet_accuracy(k2_lr, test):.4f}")

    # =========================================================================
    # Hybrids
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("HYBRID: count RSM + K1")
    log.info("=" * 70)

    k1_norm = (k1 - k1.min()) / (k1.max() - k1.min())
    s_count_norm = (s_count - s_count.min()) / (s_count.max() - s_count.min())

    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        s_hybrid = alpha * s_count_norm + (1 - alpha) * k1_norm
        log.info(f"  {alpha:.1f}*count + {1-alpha:.1f}*K1: "
                 f"train={triplet_accuracy(s_hybrid, train):.4f}, "
                 f"test={triplet_accuracy(s_hybrid, test):.4f}")

    log.info("\n  Low-rank hybrid:")
    k1_r30 = denoise_low_rank(k1, 30)
    k1_r30_n = (k1_r30 - k1_r30.min()) / (k1_r30.max() - k1_r30.min())
    s_c30_n = (s_count_r30 - s_count_r30.min()) / (s_count_r30.max() - s_count_r30.min())

    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        s_hybrid = alpha * s_c30_n + (1 - alpha) * k1_r30_n
        log.info(f"  {alpha:.1f}*countR30 + {1-alpha:.1f}*K1R30: "
                 f"train={triplet_accuracy(s_hybrid, train):.4f}, "
                 f"test={triplet_accuracy(s_hybrid, test):.4f}")

    # =========================================================================
    # SPoSE reference
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
