"""B-sweep: does our pysrf (post-edits) converge to rank=26 with more bootstraps?

Tests current pysrf.estimate_rank on THINGS at n_bootstrap in {20, 50, 100, 200, 400}.

If the rank converges to 26 (the reference value) with enough B, then
bumping B is the principled fix. If it sticks at 25 forever, there's a
genuine systematic offset between SeedSequence-style seeds and the
reference seeds that more reps cannot wash out.

Run:
    ./scripts/submit sandbox/dimensionality/b_sweep/run.py --bg
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from pysrf import estimate_rank
from src.utils import get_output_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = get_output_dir()


def load_things() -> np.ndarray:
    path = PROJECT_ROOT / "update_pysrf" / "data" / "things_behavior_similarity.npy"
    a = np.load(path)
    a = (a + a.T) / 2.0
    diag = np.eye(a.shape[0], dtype=bool)
    med = float(np.nanmedian(a[~diag]))
    return np.where(np.isnan(a), med, a)


def main() -> None:
    log.info("=" * 70)
    log.info("B-sweep on THINGS via current pysrf.estimate_rank")
    log.info("=" * 70)
    s = load_things()
    log.info(f"shape={s.shape}")

    results = []
    for B in [20, 50, 100, 200, 400]:
        log.info(f"\nn_bootstrap={B}...")
        t0 = time.time()
        est = estimate_rank(s, n_bootstrap=B, random_state=0)
        elapsed = time.time() - t0
        leak = est.leakage[:30]
        log.info(f"  rank={est.rank}  p*={est.sampling_fraction:.3f}  "
                 f"leak[23..26]=[{leak[22]:.3f}, {leak[23]:.3f}, {leak[24]:.3f}, {leak[25]:.3f}]  "
                 f"runtime={elapsed:.1f}s")
        results.append({
            "n_bootstrap": int(B),
            "rank": int(est.rank),
            "sampling_fraction": float(est.sampling_fraction),
            "leakage_top30": [float(v) for v in leak],
            "runtime_sec": round(elapsed, 1),
        })

    log.info("\n" + "=" * 70)
    log.info("Summary (target = reference rank = 26)")
    log.info("=" * 70)
    log.info(f"{'B':>4}  {'rank':>4}  {'p*':>6}  {'leak[24]':>9}  {'runtime':>7}")
    for r in results:
        log.info(f"{r['n_bootstrap']:>4}  {r['rank']:>4}  "
                 f"{r['sampling_fraction']:>6.3f}  {r['leakage_top30'][23]:>9.3f}  "
                 f"{r['runtime_sec']:>5.0f}s")

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(results, indent=2))
    log.info(f"\nSaved {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
