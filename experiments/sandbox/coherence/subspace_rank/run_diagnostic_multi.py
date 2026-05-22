"""Diagnostic: per-component activation across multiple datasets.

Runs v2 coherence on Peterson animals, Peterson various, and optionally SWOW,
extracting the activation_p profile and fraction-above-null for each.
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


DATASETS = {
    "peterson_animals": OmegaConf.load("configs/dataset/peterson_animals.yaml"),
    "peterson_various": OmegaConf.load("configs/dataset/peterson_various.yaml"),
}


def run_diagnostic(name, cfg, k_list, p_list, b=100, b_null=30):
    log.info(f"\n{'='*60}")
    log.info(f"Dataset: {name}")
    log.info(f"{'='*60}")

    s = build_similarity(cfg)
    n = s.shape[0]
    n_nan = np.isnan(s).sum()
    log.info(f"  Shape: {s.shape}, NaN={n_nan}")

    k_list_valid = [k for k in k_list if k < n]
    if not k_list_valid:
        log.info(f"  SKIP: no valid k for n={n}")
        return None
    k_arr = np.array(k_list_valid)

    log.info(f"  k_list: {k_list_valid[0]}..{k_list_valid[-1]}, B={b}, B_null={b_null}")
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
    k_count = len(k_list_valid)

    # Per-component activation
    A = (x_ci_lo >= tau_kp)
    activation_p_component = np.full(k_count, np.nan, float)
    for kk in range(k_count):
        idxs = np.where(A[kk, :])[0]
        if len(idxs) > 0:
            activation_p_component[kk] = float(p_list[idxs[0]])

    frac_above = A.mean(axis=1)

    log.info(f"\n  --- Per-component activation ---")
    for i, k in enumerate(k_arr):
        p_act = activation_p_component[i]
        f = frac_above[i]
        if np.isnan(p_act):
            log.info(f"  k={k:3d}: never activated, frac={f:.3f}")
        else:
            log.info(f"  k={k:3d}: activation_p={p_act:.3f}, frac={f:.3f}")

    # Excess at p=0.50 and p=0.80
    for p_test in [0.50, 0.80]:
        j = int(np.argmin(np.abs(p_list - p_test)))
        iproj_med = np.median(iproj_boot[:, j, :], axis=1)
        tau_p = tau_kp[:, j]
        excess = iproj_med - tau_p
        log.info(f"\n  --- Excess at p={p_list[j]:.2f} ---")
        for i, k in enumerate(k_arr):
            sig = "***" if x_ci_lo[i, j] > tau_kp[i, j] else ""
            log.info(f"  k={k:3d}: I^proj={iproj_med[i]:.4f}, tau={tau_p[i]:.4f}, "
                     f"excess={excess[i]:.4f} {sig}")

    # Save
    np.savez(
        OUTPUT_DIR / f"diagnostic_{name}.npz",
        k_list=k_arr,
        p_list=p_list,
        activation_p_component=activation_p_component,
        frac_above=frac_above,
        iproj_boot=iproj_boot,
        tau_kp=tau_kp,
        x_ci_lo=x_ci_lo,
    )

    return activation_p_component, frac_above


def main():
    p_list = np.linspace(0.05, 0.95, 25)

    for name, cfg in DATASETS.items():
        k_max = 60 if "peterson" in name else 80
        k_step = 2 if "peterson" in name else 5
        k_list = list(range(2, k_max + 1, k_step))
        run_diagnostic(name, cfg, k_list, p_list, b=100, b_null=30)

    log.info(f"\nAll diagnostics saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
