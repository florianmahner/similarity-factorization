"""Diagnostic: extract per-component activation and raw values from v2 coherence.

Runs v2 coherence on THINGS behavioral and dumps ALL intermediate values
needed to understand why rank selection fails.
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
        "name": "things_behavior",
        "type": "triplet",
        "path": "data/things",
        "triplet_number": "4.7mio",
        "n_objects": 1854,
    })
    s = build_similarity(cfg)
    n = s.shape[0]
    log.info(f"THINGS behavioral: {s.shape}, NaN={np.sum(np.isnan(s))}")

    k_list = list(range(5, 121, 5))
    p_list = np.linspace(0.05, 0.95, 25)
    k_arr = np.array(k_list)

    log.info(f"Running v2 coherence: k={k_list[0]}..{k_list[-1]}, B=100, B_null=30")
    result = compute_incremental_coherence_multi_k_eig_anisotropic(
        s, k_list, p_list,
        B=100,
        compute_null=True,
        B_null=30,
        alpha_tau=0.95,
        ci_level=0.95,
        use_baseline_correction=False,
        visualize=False,
        mark_activation=True,
        plot_tau=False,
    )

    diag = result["diagnostics"]
    iproj_boot = result["Iproj_boot"]  # (K, P, B)
    tau_kp = diag["tau_kp"]             # (K, P)
    x_ci_lo = diag["x_ci_lo"]          # (K, P)
    x_ci_hi = diag["x_ci_hi"]          # (K, P)
    x_mean = diag["x_mean"]            # (K, P)

    k_count = len(k_list)

    # =====================================================================
    # Per-component activation (v2 computes but discards)
    # =====================================================================
    A = (x_ci_lo >= tau_kp)  # (K, P) boolean
    activation_p_component = np.full(k_count, np.nan, float)
    for kk in range(k_count):
        idxs = np.where(A[kk, :])[0]
        if len(idxs) > 0:
            activation_p_component[kk] = float(p_list[idxs[0]])

    log.info("\n=== PER-COMPONENT ACTIVATION ===")
    log.info("(First p where lower CI of I^proj_k exceeds null tau_k)")
    for i, k in enumerate(k_arr):
        p_act = activation_p_component[i]
        if np.isnan(p_act):
            log.info(f"  k={k:3d}: never activated")
        else:
            log.info(f"  k={k:3d}: activation_p = {p_act:.3f}")

    # =====================================================================
    # Raw values at multiple p slices
    # =====================================================================
    p_slices = [0.30, 0.50, 0.70, 0.85, 0.95]
    log.info("\n=== I^proj MEDIAN vs TAU at selected p values ===")
    header = f"{'k':>5}"
    for p in p_slices:
        header += f" | I({p:.2f})  tau({p:.2f})"
    log.info(header)
    log.info("-" * len(header))

    for i, k in enumerate(k_arr):
        row = f"{k:5d}"
        for p_target in p_slices:
            j = int(np.argmin(np.abs(p_list - p_target)))
            i_med = float(np.median(iproj_boot[i, j, :]))
            tau = float(tau_kp[i, j]) if tau_kp is not None else 0
            row += f" | {i_med:.4f}  {tau:.4f}"
        log.info(row)

    # =====================================================================
    # Activation profile: what fraction of p values does each k exceed null?
    # =====================================================================
    frac_above = A.mean(axis=1)  # (K,) fraction of p grid where CI > tau
    log.info("\n=== FRACTION OF P-GRID WHERE CI > TAU ===")
    for i, k in enumerate(k_arr):
        log.info(f"  k={k:3d}: {frac_above[i]:.3f} ({int(A[i].sum())}/{len(p_list)} p-values)")

    # =====================================================================
    # I^proj at p=0.50 and p=0.80 (where test should be discriminating)
    # =====================================================================
    for p_test in [0.50, 0.80]:
        j = int(np.argmin(np.abs(p_list - p_test)))
        iproj_at_p = np.median(iproj_boot[:, j, :], axis=1)
        tau_at_p = tau_kp[:, j] if tau_kp is not None else np.zeros(k_count)
        excess = iproj_at_p - tau_at_p

        log.info(f"\n=== EXCESS I^proj - tau at p={p_list[j]:.2f} ===")
        for i, k in enumerate(k_arr):
            sig = "***" if excess[i] > 0 and x_ci_lo[i, j] > tau_kp[i, j] else ""
            log.info(f"  k={k:3d}: I^proj={iproj_at_p[i]:.4f}, tau={tau_at_p[i]:.4f}, "
                     f"excess={excess[i]:.4f} {sig}")

    # Save everything for offline analysis
    np.savez(
        OUTPUT_DIR / "diagnostic_dump.npz",
        k_list=k_arr,
        p_list=p_list,
        iproj_boot=iproj_boot,
        tau_kp=tau_kp,
        x_ci_lo=x_ci_lo,
        x_ci_hi=x_ci_hi,
        x_mean=x_mean,
        activation_p_component=activation_p_component,
        frac_above=frac_above,
    )

    log.info(f"\nDiagnostic dump saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
