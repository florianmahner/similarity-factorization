"""Step 2: Evaluate SRF on corrected vs count RSM. Compare with VICE."""
import logging
from pathlib import Path

import numpy as np
from pysrf import SRF

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "things" / "triplets_47"
OUTPUT_DIR = Path(__file__).parent / "outputs"
N = 1854
RANK = 24  # from lowdata 100% rank selection


def triplet_accuracy(w: np.ndarray, triplets: np.ndarray) -> float:
    ei, ej, ek = w[triplets[:, 0]], w[triplets[:, 1]], w[triplets[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def main():
    train = np.loadtxt(DATA_DIR / "train_90.txt", dtype=float).astype(int)
    test = np.loadtxt(DATA_DIR / "test_10.txt", dtype=float).astype(int)

    s_corrected = np.load(OUTPUT_DIR / "s_corrected_rsm.npy")
    s_count = np.load(OUTPUT_DIR / "count_rsm.npy")

    log.info(f"Rank={RANK}")

    for name, rsm in [("count RSM", s_count), ("corrected RSM", s_corrected)]:
        log.info(f"\n  SRF on {name}...")
        model = SRF(rank=RANK, random_state=42, max_outer=500, max_inner=30, tol=1e-4, verbose=0)
        w = model.fit_transform(rsm)
        train_acc = triplet_accuracy(w, train[:200000])
        test_acc = triplet_accuracy(w, test)
        log.info(f"    train={train_acc:.4f}, test={test_acc:.4f}")

    # VICE baseline
    vice_path = PROJECT_ROOT / "data" / "things" / "vice_embedding_66d.txt"
    if vice_path.exists():
        vice = np.maximum(np.loadtxt(vice_path), 0)
        log.info(f"\n  VICE: test={triplet_accuracy(vice, test):.4f}")

    spose_path = PROJECT_ROOT / "data" / "things" / "spose_embedding_66d.txt"
    if spose_path.exists():
        spose = np.maximum(np.loadtxt(spose_path), 0)
        log.info(f"  SPoSE: test={triplet_accuracy(spose, test):.4f}")


if __name__ == "__main__":
    main()
