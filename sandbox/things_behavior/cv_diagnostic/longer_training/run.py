"""Re-CV things_behavior with longer SRF training (max_outer=200).

Phase C showed that doubling max_outer from 50 -> 200 dropped val_mse at rank=50
by 9 SEMs. If the same drop happens at OTHER ranks but rank=50 catches up MORE
than rank=30 does, the curve will become cleaner U-shape with a real minimum
in the 30-50 range.

Grid: dense rank panel around the suspected elbow + a few high ranks for
the overfit tail.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pysrf import cross_val_score
from src.colors import INDIGO
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()
SIM_PATH = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/datasets/dimensionality/outputs/cache/things_behavior.npy")
CV_JSON = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/datasets/dimensionality/outputs/things_behavior/cross_validation.json")

RANKS = [10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 100, 120, 150]
N_REPEATS = 10
MAX_OUTER = 200
N_JOBS = 24


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sim = np.load(SIM_PATH).astype(np.float64)
    p_star = json.loads(CV_JSON.read_text())["validations"]["5fold"]["params"]["sampling_fraction"]
    log(f"loaded sim n={sim.shape[0]}, p*={p_star:.4f}")
    log(f"ranks={RANKS}, n_repeats={N_REPEATS}, max_outer={MAX_OUTER}, n_jobs={N_JOBS}")
    log(f"total fits: {len(RANKS) * 5 * N_REPEATS}")

    srf_kwargs = dict(rho=3.0, max_inner=30, max_outer=MAX_OUTER, tol=0.0, check_input=False)
    rows = []
    csv_path = OUTPUT_DIR / "cv_results.csv"

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
        pd.DataFrame(rows).to_csv(csv_path, index=False)  # incremental
        log(f"  rank={rank:>3}: val_mse={mean:.6e} ± {sem:.2e}  ({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    argmin_idx = int(df["val_mse_mean"].idxmin())
    log(f"argmin rank: {df.loc[argmin_idx, 'rank']} (val_mse={df.loc[argmin_idx, 'val_mse_mean']:.6e})")

    # Plot vs the old max_outer=50 curve for comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    old = json.loads(CV_JSON.read_text())["validations"]["5fold"]
    ox = np.array(old["ranks"])
    oy = np.array(old["val_mse_mean"])
    osem = np.array(old["val_mse_sem"])
    ax.errorbar(ox, oy, yerr=2 * osem, marker="o", linewidth=1.2, capsize=2,
                color="#999999", alpha=0.6, label="current: max_outer=50, n_repeats=20")
    x = df["rank"].to_numpy()
    y = df["val_mse_mean"].to_numpy()
    sem = df["val_mse_sem"].to_numpy()
    ax.errorbar(x, y, yerr=2 * sem, marker="o", linewidth=1.6, capsize=3,
                color=INDIGO, label=f"new: max_outer={MAX_OUTER}, n_repeats={N_REPEATS}")
    ax.set_xlabel("rank")
    ax.set_ylabel("Validation MSE")
    ax.set_title("things_behavior: longer SRF training reveals true U-shape?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cv_curve_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"saved {OUTPUT_DIR}/cv_curve_comparison.png + cv_results.csv")


if __name__ == "__main__":
    main()
