"""Compute PCT ranks for THINGS 10% and 20%."""

import logging
import time
from pathlib import Path

import numpy as np

from src.coherence import permutation_coherence_test
from src.utils import get_output_dir
from src.utils.helpers import compute_similarity_matrix_from_triplets

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

OUTPUT_DIR = get_output_dir()
DATA_DIR = Path("data/things")
N = 1854


def main():
    for pct in [10, 20]:
        path = DATA_DIR / "partitions" / f"{pct}pct_part0" / "train_90.txt"
        triplets = np.loadtxt(path).astype(int)
        s = compute_similarity_matrix_from_triplets(N, triplets, alpha=0)

        t0 = time.time()
        result = permutation_coherence_test(
            s, k_max=60, p_list=np.linspace(0.3, 0.95, 10),
            B=15, J=50, alpha=0.05, random_state=42, show_progress=True,
        )
        log.info(f"THINGS {pct}%: k*={result['k_star']}, time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
