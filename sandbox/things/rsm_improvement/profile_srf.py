"""Profile the SRF missing-data path to find bottlenecks."""

import cProfile
import io
import logging
import pstats
from pathlib import Path

import numpy as np
from pysrf import SRF

from src.utils import get_output_dir
from src.utils.helpers import compute_similarity_matrix_from_triplets
from src.utils.io import load_triplets

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
DATA_DIR = Path("data/things")


def main():
    train, _ = load_triplets(DATA_DIR)
    s = compute_similarity_matrix_from_triplets(1854, train, alpha=0.0)

    # Mask 28% near 1/3 (matches thresh=0.10 from earlier)
    s_masked = s.copy()
    diag = np.eye(1854, dtype=bool)
    nan_mask = np.isnan(s)
    near_chance = (np.abs(s_masked - 1 / 3) < 0.10) & ~diag & ~nan_mask
    s_masked[near_chance] = np.nan
    log.info("Total NaN entries: %.1f%%", 100 * np.isnan(s_masked).mean())

    # === Profile complete-data path ===
    log.info("\n=== Complete-data path (raw RSM) ===")
    pr = cProfile.Profile()
    pr.enable()
    SRF(rank=45, random_state=0, max_outer=10, max_inner=50, tol=1e-4, verbose=0).fit(s)
    pr.disable()

    sio = io.StringIO()
    ps = pstats.Stats(pr, stream=sio).sort_stats("cumulative")
    ps.print_stats(20)
    log.info(sio.getvalue())
    (OUTPUT_DIR / "profile_complete.txt").write_text(sio.getvalue())

    # === Profile missing-data path ===
    log.info("\n=== Missing-data path (masked RSM) ===")
    pr = cProfile.Profile()
    pr.enable()
    SRF(rank=45, random_state=0, max_outer=10, max_inner=50, tol=1e-4, verbose=0).fit(s_masked)
    pr.disable()

    sio = io.StringIO()
    ps = pstats.Stats(pr, stream=sio).sort_stats("cumulative")
    ps.print_stats(25)
    log.info(sio.getvalue())
    (OUTPUT_DIR / "profile_missing.txt").write_text(sio.getvalue())

    # Tottime view too
    sio2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=sio2).sort_stats("tottime")
    ps2.print_stats(25)
    log.info("\n=== Missing-data path sorted by TOTTIME ===\n%s", sio2.getvalue())


if __name__ == "__main__":
    main()
