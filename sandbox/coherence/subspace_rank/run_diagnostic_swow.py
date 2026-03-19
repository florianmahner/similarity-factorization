"""Diagnostic: per-component activation on SWOW (8647x8647).

Uses reduced parameters: B=20, P=15, k_max=100 to keep runtime manageable.
Expected runtime: ~2-4 hours.
"""

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


def main():
    cfg = OmegaConf.create({
        "name": "swow",
        "type": "word_association",
        "path": "data/small-world-of-words",
        "similarity_method": "rw",
        "alpha": 0.75,
        "use_all_responses": True,
        "top_n_words": None,
        "min_word_length": 1,
        "symmetrization": "sum",
        "bidirectional_only": False,
    })

    log.info("Loading SWOW...")
    s = build_similarity(cfg)
    n = s.shape[0]
    log.info(f"SWOW: {s.shape}, NaN={np.isnan(s).sum()}")

    k_list = list(range(5, 101, 5)) + list(range(120, 301, 20))
    p_list = np.linspace(0.05, 0.95, 15)

    log.info(f"Running v2 coherence: k={k_list[0]}..{k_list[-1]}, B=20, B_null=20, P={len(p_list)}")
    result = compute_incremental_coherence_multi_k_eig_anisotropic(
        s, k_list, p_list,
        B=20,
        compute_null=True,
        B_null=20,
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
    k_arr = np.array(k_list)
    k_count = len(k_list)

    A = (x_ci_lo >= tau_kp)
    activation_p_component = np.full(k_count, np.nan, float)
    for kk in range(k_count):
        idxs = np.where(A[kk, :])[0]
        if len(idxs) > 0:
            activation_p_component[kk] = float(p_list[idxs[0]])

    frac_above = A.mean(axis=1)

    log.info(f"\n=== PER-COMPONENT ACTIVATION (SWOW) ===")
    for i, k in enumerate(k_arr):
        p_act = activation_p_component[i]
        f = frac_above[i]
        if np.isnan(p_act):
            log.info(f"  k={k:3d}: never activated, frac={f:.3f}")
        else:
            log.info(f"  k={k:3d}: activation_p={p_act:.3f}, frac={f:.3f}")

    for p_test in [0.50, 0.80]:
        j = int(np.argmin(np.abs(p_list - p_test)))
        iproj_med = np.median(iproj_boot[:, j, :], axis=1)
        tau_p = tau_kp[:, j]
        excess = iproj_med - tau_p
        log.info(f"\n=== EXCESS at p={p_list[j]:.2f} ===")
        for i, k in enumerate(k_arr):
            sig = "***" if x_ci_lo[i, j] > tau_kp[i, j] else ""
            log.info(f"  k={k:3d}: I^proj={iproj_med[i]:.4f}, tau={tau_p[i]:.4f}, "
                     f"excess={excess[i]:.4f} {sig}")

    np.savez(
        OUTPUT_DIR / "diagnostic_swow.npz",
        k_list=k_arr,
        p_list=p_list,
        activation_p_component=activation_p_component,
        frac_above=frac_above,
        iproj_boot=iproj_boot,
        tau_kp=tau_kp,
        x_ci_lo=x_ci_lo,
    )

    log.info(f"\nDiagnostic saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
