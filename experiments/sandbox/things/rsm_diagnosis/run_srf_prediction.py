"""Test SRF on count-based RSM for triplet prediction.

The key question: does the pipeline
  triplets -> count RSM -> SRF -> embedding -> dot-product prediction
match SPoSE's triplet prediction accuracy?

If yes, the argument is: the count-based RSM is a sufficient representation
of the triplet data, and SRF recovers competitive embeddings without ever
optimizing the triplet loss directly.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from pysrf import SRF

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


def triplet_accuracy_matrix(s: np.ndarray, triplets: np.ndarray) -> float:
    """Predict from full similarity matrix."""
    sij = s[triplets[:, 0], triplets[:, 1]]
    sik = s[triplets[:, 0], triplets[:, 2]]
    sjk = s[triplets[:, 1], triplets[:, 2]]
    return float(np.mean((sij > sik) & (sij > sjk)))


def triplet_accuracy_embedding(w: np.ndarray, triplets: np.ndarray) -> float:
    """Predict from embedding via dot product."""
    ei, ej, ek = w[triplets[:, 0]], w[triplets[:, 1]], w[triplets[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def fit_srf(rsm, rank, seed):
    model = SRF(rank=rank, random_state=seed, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
    return model.fit_transform(rsm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log.info("Loading triplets...")
    train = load_triplets(DATA_DIR / "train_90.txt")
    test = load_triplets(DATA_DIR / "test_10.txt")
    log.info(f"Train: {len(train):,}, Test: {len(test):,}")

    # Build count RSM
    log.info("Building count RSM...")
    s = compute_similarity_matrix_from_triplets(N, train, alpha=1.0)
    s[np.isnan(s)] = 0.5

    # =========================================================================
    # SRF on raw count RSM at different ranks
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("SRF ON RAW COUNT RSM")
    log.info("=" * 70)

    ranks = [20, 30, 40, 50, 66]
    log.info(f"Fitting SRF at ranks {ranks} in parallel...")

    embeddings = Parallel(n_jobs=len(ranks), verbose=10)(
        delayed(fit_srf)(s, r, args.seed) for r in ranks
    )

    for rank, w in zip(ranks, embeddings):
        rsm_srf = w @ w.T
        train_mat = triplet_accuracy_matrix(rsm_srf, train)
        test_mat = triplet_accuracy_matrix(rsm_srf, test)
        train_emb = triplet_accuracy_embedding(w, train)
        test_emb = triplet_accuracy_embedding(w, test)
        log.info(f"  SRF rank-{rank:2d}: "
                 f"RSM train={train_mat:.4f} test={test_mat:.4f} | "
                 f"embed train={train_emb:.4f} test={test_emb:.4f}")

    # =========================================================================
    # SRF on denoised count RSM
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("SRF ON DENOISED COUNT RSM (rank-35 PSD projection first)")
    log.info("=" * 70)

    # Denoise first
    eigenvalues, eigvecs = np.linalg.eigh(s)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigvecs = eigenvalues[idx], eigvecs[:, idx]
    eig_trunc = np.maximum(eigenvalues[:35], 0)
    s_denoised = eigvecs[:, :35] @ np.diag(eig_trunc) @ eigvecs[:, :35].T
    d = np.sqrt(np.diag(s_denoised))
    d[d == 0] = 1
    s_denoised = s_denoised / np.outer(d, d)

    log.info(f"  Denoised RSM:    train={triplet_accuracy_matrix(s_denoised, train):.4f}, "
             f"test={triplet_accuracy_matrix(s_denoised, test):.4f}")

    ranks_d = [20, 30, 35]
    embeddings_d = Parallel(n_jobs=len(ranks_d), verbose=10)(
        delayed(fit_srf)(s_denoised, r, args.seed) for r in ranks_d
    )

    for rank, w in zip(ranks_d, embeddings_d):
        rsm_srf = w @ w.T
        train_mat = triplet_accuracy_matrix(rsm_srf, train)
        test_mat = triplet_accuracy_matrix(rsm_srf, test)
        train_emb = triplet_accuracy_embedding(w, train)
        test_emb = triplet_accuracy_embedding(w, test)
        log.info(f"  SRF(denoised) rank-{rank:2d}: "
                 f"RSM train={train_mat:.4f} test={test_mat:.4f} | "
                 f"embed train={train_emb:.4f} test={test_emb:.4f}")

    # =========================================================================
    # Multiple SRF runs for stability (consensus-like)
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("MULTIPLE SRF RUNS (stability check)")
    log.info("=" * 70)

    rank = 40
    seeds = list(range(5))
    embeddings_multi = Parallel(n_jobs=len(seeds), verbose=10)(
        delayed(fit_srf)(s, rank, seed) for seed in seeds
    )

    for seed, w in zip(seeds, embeddings_multi):
        test_acc = triplet_accuracy_embedding(w, test)
        log.info(f"  SRF rank-{rank} seed-{seed}: test={test_acc:.4f}")

    # Average RSM across runs
    rsm_avg = np.mean([w @ w.T for w in embeddings_multi], axis=0)
    test_avg = triplet_accuracy_matrix(rsm_avg, test)
    log.info(f"  Average RSM (5 runs):      test={test_avg:.4f}")

    # =========================================================================
    # References
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("REFERENCES")
    log.info("=" * 70)

    spose_path = PROJECT_ROOT / "data" / "things" / "spose_embedding_66d.txt"
    if spose_path.exists():
        spose = np.maximum(np.loadtxt(spose_path), 0)
        rsm_spose = spose @ spose.T
        log.info(f"  SPoSE embed:      train={triplet_accuracy_embedding(spose, train):.4f}, "
                 f"test={triplet_accuracy_embedding(spose, test):.4f}")
        log.info(f"  SPoSE RSM:        train={triplet_accuracy_matrix(rsm_spose, train):.4f}, "
                 f"test={triplet_accuracy_matrix(rsm_spose, test):.4f}")

    vice_path = PROJECT_ROOT / "data" / "things" / "vice_embedding_66d.txt"
    if vice_path.exists():
        vice = np.maximum(np.loadtxt(vice_path), 0)
        log.info(f"  VICE embed:       train={triplet_accuracy_embedding(vice, train):.4f}, "
                 f"test={triplet_accuracy_embedding(vice, test):.4f}")

    log.info(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
