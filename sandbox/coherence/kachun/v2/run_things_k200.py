"""Count-based activation on THINGS up to k=200 to check if k*=100 is stable."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import logging

from src.similarity import build_similarity
from src.colors import ROSE, TEAL, GRAY, GRAY_LIGHT
from src.utils import get_output_dir
from src.utils.figure_theme import create_figure, despine
from omegaconf import OmegaConf

from _compute_coherence import (
    compute_incremental_coherence_multi_k_eig_anisotropic,
    analyze_iproj_signal_alignment_v3,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()


def _count_based_activation(tau_kp, x_ci_lo, k_arr, p_list, k_max):
    idx_end = np.searchsorted(k_arr, k_max, side="right")
    K = idx_end
    A = x_ci_lo[:K] >= tau_kp[:K]
    N = A.sum(axis=0)
    activation_p = np.full(K, np.nan, float)
    for r in range(K):
        idxs = np.where(N >= (r + 1))[0]
        if len(idxs) > 0:
            activation_p[r] = float(p_list[idxs[0]])
    valid = ~np.isnan(activation_p)
    k_star = int(k_arr[np.where(valid)[0][-1]]) if np.any(valid) else None
    return k_star


def main():
    cfg = OmegaConf.create({
        "name": "things_behavior",
        "type": "triplet",
        "path": "data/things",
        "triplet_number": "4.7mio",
        "n_objects": 1854,
    })
    s = build_similarity(cfg)
    log.info(f"THINGS behavioral: {s.shape}, NaN={np.sum(np.isnan(s))}")

    k_list_full = list(range(5, 201, 5))
    p_list = np.linspace(0.05, 0.95, 25)
    k_arr = np.array(k_list_full)

    log.info(f"Running coherence: k=5..200 (step 5), B=100, P=25")
    result = compute_incremental_coherence_multi_k_eig_anisotropic(
        s, k_list_full, p_list, B=100,
        use_baseline_correction=False, clip_baseline=False,
        plot_tau=False, mark_activation=True, compute_null=True,
        visualize=False,
    )
    log.info("Done.")

    diag = result["diagnostics"]
    tau_kp = diag["tau_kp"]
    x_ci_lo = diag["x_ci_lo"]
    i_boot = result["I_boot"]

    # Sweep k_max from 40 to 200 in steps of 20
    k_max_values = list(range(40, 201, 20))
    k_act_results = []
    k_kap_results = []

    for km in k_max_values:
        k_act = _count_based_activation(tau_kp, x_ci_lo, k_arr, p_list, km)
        k_act_results.append(k_act)

        idx_end = np.searchsorted(k_arr, km, side="right")
        analysis = analyze_iproj_signal_alignment_v3(
            Iproj_boot=i_boot[:idx_end],
            k_list=k_arr[:idx_end].tolist(),
            p_list=p_list,
            K_sig_mode="kappa_changepoint",
            signal_reference="boundary",
            smooth_window=3,
            make_plots=False,
        )
        k_kap_results.append(analysis.summary["K_sig_details"]["k_cut"])

    log.info("\n=== RESULTS ===")
    log.info(f"{'k_max':>6} | {'Activation k*':>15} | {'Kappa k*':>10}")
    log.info("-" * 40)
    for km, ka, kk in zip(k_max_values, k_act_results, k_kap_results):
        ka_str = str(ka) if ka is not None else "None"
        log.info(f"{km:>6} | {ka_str:>15} | {kk:>10}")

    # -- Plot --
    k_act_arr = np.array([v if v is not None else np.nan for v in k_act_results])
    k_kap_arr = np.array(k_kap_results)

    fig, ax = create_figure("wide")

    ax.fill_between(k_max_values, k_max_values, alpha=0.04, color=GRAY)
    ax.plot(k_max_values, k_max_values, linestyle=":", linewidth=0.8, color=GRAY_LIGHT,
            label="$k^* = k_{\\max}$")
    ax.plot(k_max_values, k_act_arr, marker="o", markersize=4, linewidth=1.5,
            color=ROSE, label="Null-calibrated activation", zorder=5)
    ax.plot(k_max_values, k_kap_arr, marker="s", markersize=4, linewidth=1.5,
            color=TEAL, label="Kappa changepoint", zorder=5)

    # Annotate plateau
    plateau_val = k_act_arr[~np.isnan(k_act_arr)][-1]
    ax.axhline(plateau_val, linestyle="--", linewidth=0.8, color=ROSE, alpha=0.4)
    ax.text(202, plateau_val, f"{int(plateau_val)}", fontsize=7, color=ROSE,
            va="center", ha="left")

    kap_val = k_kap_arr[-1]
    ax.axhline(kap_val, linestyle="--", linewidth=0.8, color=TEAL, alpha=0.4)
    ax.text(202, kap_val, f"{int(kap_val)}", fontsize=7, color=TEAL,
            va="center", ha="left")

    ax.set_xlabel("Maximum tested rank $k_{\\max}$")
    ax.set_ylabel("Estimated $k^*$")
    ax.set_title("THINGS behavioral: rank estimate vs. search range")
    ax.legend(fontsize=7, loc="center right")
    ax.set_xlim(35, 205)
    despine(ax)

    fig.savefig(OUTPUT_DIR / "activation_instability_k200.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    log.info(f"\nPlot saved to {OUTPUT_DIR / 'activation_instability_k200.png'}")


if __name__ == "__main__":
    main()
