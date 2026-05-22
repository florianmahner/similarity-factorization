"""Compare our pysrf.estimate_rank p_star against update_pysrf's reference.

The dimensionality pipeline gives p_star=0.705 for Peterson-animals; the
sandbox spectrum_vs_cv saved p_star=0.697 (from before the recent
estimate_rank seed-pattern fix). To know which is correct, this script
runs update_pysrf's `spectral_pass` + `recipe_K` on the exact same
matrix and reports what THAT pipeline returns for p_star.

Run:
    ./scripts/submit sandbox/dimensionality/peterson_pstar/run.py --bg
"""

from __future__ import annotations

import logging
import os
import sys
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


def main() -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "update_pysrf" / "src"))
    from _common import spectral_pass, recipe_K  # noqa: E402

    cache = (
        PROJECT_ROOT / "experiments" / "figures" / "plot_embeddings"
        / "outputs" / ".cache" / "sim_peterson-animals.npy"
    )
    s = np.load(cache)
    log.info(f"shape={s.shape}")

    # Our pipeline
    est = estimate_rank(s)
    log.info(f"OUR  pysrf      : k_cut={est.rank}  "
             f"p_star={est.sampling_fraction:.6f}  "
             f"floor={est.detectability_floor:.6f}")

    # Reference pipeline (update_pysrf)
    spec = spectral_pass(s, B=20, smooth_window=10, show_progress=False)
    rec = recipe_K(spec, delta=0.10, k_cv=5, p_floor=0.5,
                    p_floor_mode="adaptive", M_min=2000)
    log.info(f"REF  update_pysrf: k_cut={int(spec['k_cut'])}  "
             f"p_star_raw={float(rec['p_star_raw']):.6f}  "
             f"p_star_recipe={float(rec['p_star']):.6f}  "
             f"p_cv={float(rec['p_cv']):.6f}")

    log.info("")
    log.info("=== Conclusion ===")
    if abs(est.sampling_fraction - float(rec["p_star"])) < 1e-4:
        log.info(f"MATCH: our p_star agrees with update_pysrf recipe_K p_star")
    elif abs(est.sampling_fraction - float(rec["p_star_raw"])) < 1e-4:
        log.info(f"MATCH: our p_star agrees with update_pysrf raw p_star "
                 f"(pre-recipe_K calibration)")
    else:
        log.info(f"MISMATCH: ours={est.sampling_fraction:.6f}, "
                 f"ref raw={float(rec['p_star_raw']):.6f}, "
                 f"ref recipe={float(rec['p_star']):.6f}")


if __name__ == "__main__":
    main()
