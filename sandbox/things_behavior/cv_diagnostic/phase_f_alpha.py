"""Phase F: does alpha=0 (no Laplace smoothing) give a sharper eigenspectrum / CV U-shape?

Builds the things_behavior similarity at alpha ∈ {1.0 (current), 0.1, 0.0} from
the raw 4.7M triplets, compares:
  1. eigenvalue spectra (sharper elbow with less smoothing?)
  2. fraction missing (alpha=0 leaves NaNs for unobserved pairs)
  3. quick CV at ranks {20, 30, 40, 50, 70, 100} on each variant

Per-config CSV + plots. Resume-safe (skips configs already in results).
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
from src.colors import CYCLE, INDIGO
from src.utils import get_output_dir
from utils.helpers import compute_similarity_matrix_from_triplets

OUTPUT_DIR = get_output_dir()
TRIPLET_PATH = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/data/things/triplets_47/trainset.txt")
CV_JSON = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/datasets/dimensionality/outputs/things_behavior/cross_validation.json")

ALPHAS = [1.0, 0.1, 0.0]
RANKS = [20, 30, 40, 50, 70, 100]
SRF_KWARGS = dict(rho=3.0, max_inner=30, max_outer=50, tol=0.0, check_input=False)
N_OBJECTS = 1854
N_JOBS = 16


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    triplets = np.loadtxt(TRIPLET_PATH, dtype=np.int32)
    log(f"loaded {len(triplets)} triplets")
    p_star = json.loads(CV_JSON.read_text())["validations"]["5fold"]["params"]["sampling_fraction"]
    log(f"using p*={p_star:.4f} from existing alpha=1.0 CV (same for all alpha variants)")

    spectra = {}
    miss_frac = {}
    cv_rows = []
    cv_csv = OUTPUT_DIR / "f_cv_alpha.csv"
    spec_csv = OUTPUT_DIR / "f_spectra_alpha.csv"

    for alpha in ALPHAS:
        tag = f"alpha={alpha}"
        log(f"=== {tag} ===")
        t0 = time.time()
        sim = compute_similarity_matrix_from_triplets(N_OBJECTS, triplets, alpha=alpha)
        n_missing = int(np.isnan(sim).sum())
        miss_frac[alpha] = n_missing / (sim.size)
        log(f"  built similarity in {time.time()-t0:.1f}s   "
            f"missing fraction: {miss_frac[alpha]:.4f}   "
            f"min={np.nanmin(sim):.4f} max={np.nanmax(sim):.4f}")

        # Spectrum (replace NaN with 0 for eigendecomp comparison)
        s_clean = np.nan_to_num(sim, nan=0.0)
        s_clean = 0.5 * (s_clean + s_clean.T)
        eigs = np.linalg.eigvalsh(s_clean)[::-1][:200]
        spectra[alpha] = eigs
        log(f"  top-5 eigenvalues: {eigs[:5]}")
        log(f"  eig at idx 30/50/100: {eigs[29]:.3f} / {eigs[49]:.3f} / {eigs[99]:.3f}")

        # Quick CV at the rank panel
        for rank in RANKS:
            t1 = time.time()
            curve = cross_val_score(
                sim.astype(np.float64),
                ranks=[rank],
                sampling_fraction=p_star,
                n_folds=5, n_repeats=5, random_state=42, n_jobs=N_JOBS,
                srf_kwargs=SRF_KWARGS,
            )
            mse_mean = float(curve["val_mse"].mean())
            mse_sem = float(curve["val_mse"].std(ddof=1) / np.sqrt(len(curve)))
            elapsed = time.time() - t1
            cv_rows.append({
                "alpha": alpha, "rank": rank,
                "val_mse_mean": mse_mean, "val_mse_sem": mse_sem,
                "missing_frac": miss_frac[alpha],
                "fit_sec": elapsed,
            })
            pd.DataFrame(cv_rows).to_csv(cv_csv, index=False)  # incremental save
            log(f"  alpha={alpha} rank={rank}: val_mse={mse_mean:.6e} +/- {mse_sem:.2e}  ({elapsed:.1f}s)")

    pd.DataFrame(spectra).to_csv(spec_csv, index_label="index")
    log("=== plotting ===")

    # Spectrum plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, alpha in enumerate(ALPHAS):
        axes[0].plot(np.arange(1, 201), spectra[alpha], marker=".", ms=2,
                     color=CYCLE[i % len(CYCLE)], label=f"alpha={alpha}")
        axes[1].semilogy(np.arange(1, 201), np.clip(spectra[alpha], 1e-12, None),
                         marker=".", ms=2, color=CYCLE[i % len(CYCLE)], label=f"alpha={alpha}")
    for ax in axes:
        ax.set_xlabel("eigenvalue index")
        ax.legend(frameon=False)
    axes[0].set_ylabel("eigenvalue (linear)")
    axes[1].set_ylabel("eigenvalue (log)")
    axes[0].set_title("F.1 eigenvalue spectrum vs alpha")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "f_spectra_alpha.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # CV curve plot
    df = pd.DataFrame(cv_rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, alpha in enumerate(ALPHAS):
        sub = df[df["alpha"] == alpha].sort_values("rank")
        x = sub["rank"].to_numpy()
        y = sub["val_mse_mean"].to_numpy()
        sem = sub["val_mse_sem"].to_numpy()
        ax.errorbar(x, y, yerr=2*sem, marker="o", color=CYCLE[i % len(CYCLE)],
                    label=f"alpha={alpha} (miss={miss_frac[alpha]:.1%})",
                    capsize=3, lw=1.5)
    ax.set_xlabel("rank")
    ax.set_ylabel("Validation MSE")
    ax.set_title("F.2 CV curve vs alpha (lower alpha = less smoothing)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "f_cv_alpha.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log("=== DONE ===")
    log(f"results: {cv_csv}, {spec_csv}, f_*.png")


if __name__ == "__main__":
    main()
