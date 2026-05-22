"""Subspace coherence rank selection on THINGS behavioral similarity."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import logging

from src.similarity import build_similarity
from src.utils import get_output_dir
from omegaconf import OmegaConf

from _subspace_coherence import compute_subspace_coherence, select_rank, plot_rank_selection

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

    log.info(f"Running subspace coherence: k={k_list[0]}..{k_list[-1]}, B=100, B_null=20")
    result = compute_subspace_coherence(
        s, k_list, p_list,
        b=100, b_null=20,
        use_baseline_correction=True,
        n_jobs=None,
    )

    rank = select_rank(result, fdr_q=0.10)

    log.info(f"\n=== RANK SELECTION ===")
    log.info(f"  k* (FDR q=0.10): {rank.k_star}")
    log.info(f"  p* (liftoff):    {rank.p_star}")
    log.info(f"  Kappa k_cut:     {rank.kappa_k_cut} (secondary)")

    log.info(f"\nPer-dimension results:")
    for i, k in enumerate(result["k_list"]):
        sig = "***" if rank.significant[i] else ""
        log.info(f"  k={k:3d}: p-value={rank.pvalues[i]:.4f} {sig}")

    saved = plot_rank_selection(result, rank, OUTPUT_DIR)
    log.info(f"\nPlots saved to {OUTPUT_DIR}")

    np.savez(
        OUTPUT_DIR / "subspace_coherence_results.npz",
        delta_boot=result["delta_boot"],
        Iproj_boot=result["Iproj_boot"],
        delta_mean=result["delta_mean"],
        k_list=result["k_list"],
        p_list=result["p"],
        k_star=rank.k_star,
        p_star=rank.p_star if rank.p_star is not None else np.nan,
        pvalues=rank.pvalues,
        significant=rank.significant,
        kappa_hat=rank.kappa_hat,
    )


if __name__ == "__main__":
    main()
