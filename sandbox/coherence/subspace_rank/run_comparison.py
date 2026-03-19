"""Compare subspace coherence S_k vs per-eigenvector I^proj across k_max values.

Tests stability: k* should not change when k_max varies from 80 to 200.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
import logging

from src.similarity import build_similarity
from src.colors import ROSE, TEAL, GRAY_LIGHT
from src.utils import get_output_dir
from src.utils.figure_theme import create_figure, despine
from omegaconf import OmegaConf

from _subspace_coherence import compute_subspace_coherence, select_rank

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()


def main():
    cfg = OmegaConf.create({
        "name": "things_behavior",
        "type": "triplet",
        "path": "data/things",
        "triplet_number": "4.7mio",
        "n_objects": 1854,
    })
    s = build_similarity(cfg)
    log.info(f"THINGS behavioral: {s.shape}")

    p_list = np.linspace(0.05, 0.95, 25)

    k_list_full = list(range(5, 201, 5))
    log.info(f"Running subspace coherence: k=5..200, B=50, B_null=20")

    result_full = compute_subspace_coherence(
        s, k_list_full, p_list,
        b=50, b_null=20,
        use_baseline_correction=True,
        n_jobs=None,
    )

    # Test stability across k_max
    k_max_values = [80, 100, 120, 140, 160, 180, 200]
    k_arr_full = result_full["k_list"]
    results_subspace = []

    for km in k_max_values:
        idx_end = int(np.searchsorted(k_arr_full, km, side="right"))
        sub = {
            key: result_full[key]
            for key in result_full
            if key not in ("delta_boot", "Iproj_boot", "null_delta_boot",
                           "delta_mean", "delta_ci_lo", "delta_ci_hi", "tau_kp", "k_list")
        }
        sub["k_list"] = k_arr_full[:idx_end]
        sub["delta_boot"] = result_full["delta_boot"][:idx_end]
        sub["Iproj_boot"] = result_full["Iproj_boot"][:idx_end]
        sub["delta_mean"] = result_full["delta_mean"][:idx_end]
        sub["delta_ci_lo"] = result_full["delta_ci_lo"][:idx_end]
        sub["delta_ci_hi"] = result_full["delta_ci_hi"][:idx_end]
        if result_full["null_delta_boot"] is not None:
            sub["null_delta_boot"] = result_full["null_delta_boot"][:idx_end]
        if result_full["tau_kp"] is not None:
            sub["tau_kp"] = result_full["tau_kp"][:idx_end]

        rank = select_rank(sub, fdr_q=0.05)
        results_subspace.append((km, rank.k_star, rank.p_star, rank.kappa_k_cut))

    log.info("\n=== STABILITY TABLE ===")
    log.info(f"{'k_max':>6} | {'S_k k*':>8} | {'S_k p*':>8} | {'Kappa k_cut':>12}")
    log.info("-" * 45)
    for km, ks, ps, kc in results_subspace:
        ps_str = f"{ps:.3f}" if ps is not None else "None"
        log.info(f"{km:>6} | {ks:>8} | {ps_str:>8} | {kc:>12}")

    # Plot: k* vs k_max
    fig, ax = create_figure("wide")
    k_max_arr = np.array([r[0] for r in results_subspace])
    k_star_arr = np.array([r[1] for r in results_subspace])
    kappa_arr = np.array([r[3] for r in results_subspace])

    ax.plot(k_max_arr, k_star_arr, marker="o", markersize=5, linewidth=1.5,
            color=TEAL, label="$S_k$ (subspace, FDR)")
    ax.plot(k_max_arr, kappa_arr, marker="s", markersize=5, linewidth=1.5,
            color=ROSE, label="Kappa changepoint")
    ax.plot(k_max_arr, k_max_arr, linestyle=":", linewidth=0.8, color=GRAY_LIGHT,
            label="$k^* = k_{max}$ (ceiling)")
    ax.set_xlabel("Maximum tested rank $k_{\\max}$")
    ax.set_ylabel("Estimated $k^*$")
    ax.set_title("Rank estimate stability across $k_{\\max}$")
    ax.legend(fontsize=6)
    despine(ax)
    fig.savefig(OUTPUT_DIR / "stability_kstar_vs_kmax.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    log.info(f"\nPlots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
