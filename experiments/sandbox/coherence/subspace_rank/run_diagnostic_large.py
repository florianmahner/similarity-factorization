"""Diagnostic: per-component activation on SWOW (k_max=300) and NSD subject 1."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kachun" / "v2"))

import numpy as np
import logging

from src.similarity import build_similarity
from src.utils import get_output_dir
from omegaconf import OmegaConf

from _compute_coherence import compute_incremental_coherence_multi_k_eig_anisotropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()


def run_diagnostic(name, s, k_list, p_list, b=20, b_null=20):
    n = s.shape[0]
    log.info(f"\n{'='*60}")
    log.info(f"Dataset: {name} ({n}x{n}), NaN={np.isnan(s).sum()}")
    log.info(f"{'='*60}")

    k_list_valid = [k for k in k_list if k < n]
    if not k_list_valid:
        log.info(f"  SKIP: no valid k for n={n}")
        return
    k_arr = np.array(k_list_valid)
    k_count = len(k_list_valid)

    log.info(f"  k: {k_list_valid[0]}..{k_list_valid[-1]} ({k_count} values), B={b}, B_null={b_null}")
    result = compute_incremental_coherence_multi_k_eig_anisotropic(
        s, k_list_valid, p_list,
        B=b,
        compute_null=True,
        B_null=b_null,
        alpha_tau=0.95,
        ci_level=0.95,
        use_baseline_correction=False,
        visualize=False,
        mark_activation=True,
        plot_tau=False,
    )

    diag = result["diagnostics"]
    iproj_boot = result["Iproj_boot"]
    tau_kp = diag["tau_kp"]
    x_ci_lo = diag["x_ci_lo"]

    A = (x_ci_lo >= tau_kp)
    activation_p = np.full(k_count, np.nan, float)
    for kk in range(k_count):
        idxs = np.where(A[kk, :])[0]
        if len(idxs) > 0:
            activation_p[kk] = float(p_list[idxs[0]])

    frac_above = A.mean(axis=1)

    log.info(f"\n  --- Per-component activation ---")
    for i, k in enumerate(k_arr):
        p_act = activation_p[i]
        f = frac_above[i]
        if np.isnan(p_act):
            log.info(f"  k={k:4d}: never activated, frac={f:.3f}")
        else:
            log.info(f"  k={k:4d}: activation_p={p_act:.3f}, frac={f:.3f}")

    for p_test in [0.50, 0.80]:
        j = int(np.argmin(np.abs(p_list - p_test)))
        iproj_med = np.median(iproj_boot[:, j, :], axis=1)
        tau_p = tau_kp[:, j]
        excess = iproj_med - tau_p
        log.info(f"\n  --- Excess at p={p_list[j]:.2f} ---")
        for i, k in enumerate(k_arr):
            sig = "***" if x_ci_lo[i, j] > tau_kp[i, j] else ""
            log.info(f"  k={k:4d}: I^proj={iproj_med[i]:.4f}, tau={tau_p[i]:.4f}, "
                     f"excess={excess[i]:.4f} {sig}")

    np.savez(
        OUTPUT_DIR / f"diagnostic_{name}.npz",
        k_list=k_arr, p_list=p_list,
        activation_p=activation_p, frac_above=frac_above,
        iproj_boot=iproj_boot, tau_kp=tau_kp, x_ci_lo=x_ci_lo,
    )


def main():
    p_list = np.linspace(0.05, 0.95, 15)

    # SWOW: k_max=300 with coarser spacing at high k
    log.info("Loading SWOW...")
    cfg_swow = OmegaConf.create({
        "name": "swow",
        "type": "word_association",
        "path": "data/small-world-of-words",
        "similarity_method": "ppmi",
        "use_all_responses": True,
        "top_n_words": None,
        "min_word_length": 1,
        "symmetrization": "sum",
        "bidirectional_only": False,
    })
    s_swow = build_similarity(cfg_swow)
    k_swow = list(range(5, 101, 5)) + list(range(120, 301, 20))
    run_diagnostic("swow_k300", s_swow, k_swow, p_list, b=20, b_null=20)
    del s_swow

    # NSD subject 1
    log.info("\nLoading NSD subject 1...")
    cfg_nsd = OmegaConf.load("configs/dataset/nsd.yaml")
    s_nsd = build_similarity(cfg_nsd, subject_id=1)
    k_nsd = list(range(5, 101, 5))
    run_diagnostic("nsd_sub01", s_nsd, k_nsd, p_list, b=20, b_null=20)

    log.info(f"\nAll diagnostics saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
