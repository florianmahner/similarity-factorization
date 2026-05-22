"""Upper half of the rank grid for things_behavior longer-training CV.

Runs ranks {55, 60, 70, 80, 100, 120, 150} in parallel with the lower half
running on precuneus. Writes to cv_results_upper.csv so the two processes
don't race on file writes. Merge after both finish.
"""

from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import pandas as pd

from pysrf import cross_val_score
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()
SIM_PATH = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/datasets/dimensionality/outputs/cache/things_behavior.npy")
CV_JSON = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/datasets/dimensionality/outputs/things_behavior/cross_validation.json")

RANKS = [55, 60, 70, 80, 100, 120, 150]
N_REPEATS = 10
MAX_OUTER = 200
N_JOBS = 50  # insula has 72 physical cores; 50 lets 1 wave fit (50 fits per rank)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sim = np.load(SIM_PATH).astype(np.float64)
    p_star = json.loads(CV_JSON.read_text())["validations"]["5fold"]["params"]["sampling_fraction"]
    log(f"loaded sim n={sim.shape[0]}, p*={p_star:.4f}")
    log(f"upper ranks={RANKS}, n_repeats={N_REPEATS}, max_outer={MAX_OUTER}, n_jobs={N_JOBS}")
    srf_kwargs = dict(rho=3.0, max_inner=30, max_outer=MAX_OUTER, tol=0.0, check_input=False)
    rows = []
    csv_path = OUTPUT_DIR / "cv_results_upper.csv"
    for rank in RANKS:
        t0 = time.time()
        curve = cross_val_score(
            sim, ranks=[rank], sampling_fraction=p_star,
            n_folds=5, n_repeats=N_REPEATS, random_state=42, n_jobs=N_JOBS,
            srf_kwargs=srf_kwargs,
        )
        mean = float(curve["val_mse"].mean())
        sem = float(curve["val_mse"].std(ddof=1) / np.sqrt(len(curve)))
        elapsed = time.time() - t0
        rows.append({"rank": rank, "val_mse_mean": mean, "val_mse_sem": sem,
                     "n_fits": len(curve), "elapsed_sec": elapsed})
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        log(f"  rank={rank:>3}: val_mse={mean:.6e} +/- {sem:.2e}  ({elapsed:.1f}s)")
    log("upper half done")


if __name__ == "__main__":
    main()
