"""Diagnose the THINGS rank discrepancy.

Reference (update_pysrf) reports k_cut=26 on things_behavior; our
pipeline reports 23. This script isolates whether the cause is the
input matrix (data preprocessing) or the bootstrap algorithm.

Steps:
  1. Load both candidate THINGS matrices:
       S_ours = build_similarity(things_behavior_cfg) from raw triplets
       S_ref  = update_pysrf/data/things_behavior_similarity.npy (cached,
                NaN-imputed like the reference does)
  2. Quantify how different they are (Frobenius diff, off-diag correlation).
  3. Run pysrf.estimate_rank on both.
  4. Run update_pysrf's spectral_pass on both (if importable).

Output: outputs/summary.txt + outputs/diagnostic.json with all numbers.

Run:
  ./scripts/submit sandbox/dimensionality/things_diagnostic/run.py --bg
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from omegaconf import OmegaConf

from pysrf import estimate_rank
from src.utils import get_output_dir
from similarity import build_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = get_output_dir()


def load_ours() -> np.ndarray:
    cfg_path = PROJECT_ROOT / "configs" / "dataset" / "things_behavior.yaml"
    raw = OmegaConf.load(cfg_path)
    parent = OmegaConf.create({
        "paths": OmegaConf.load(PROJECT_ROOT / "configs" / "paths" / "local.yaml"),
        "dataset": raw,
        "project_root": str(PROJECT_ROOT),
    })
    OmegaConf.resolve(parent)
    return build_similarity(parent.dataset)


def load_reference() -> np.ndarray:
    """Load reference cached matrix; symmetrize and median-impute NaNs."""
    path = PROJECT_ROOT / "update_pysrf" / "data" / "things_behavior_similarity.npy"
    A = np.load(path)
    A = (A + A.T) / 2.0
    diag = np.eye(A.shape[0], dtype=bool)
    med = float(np.nanmedian(A[~diag]))
    n_nan = int(np.isnan(A).sum())
    A = np.where(np.isnan(A), med, A)
    log.info(f"  reference: shape={A.shape}, imputed {n_nan} NaNs with {med:.4f}")
    return A


def compare_matrices(s_ours: np.ndarray, s_ref: np.ndarray) -> dict:
    """Quantify how different two same-shaped similarity matrices are."""
    assert s_ours.shape == s_ref.shape, f"{s_ours.shape} vs {s_ref.shape}"
    n = s_ours.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)

    off_ours = s_ours[triu_i, triu_j]
    off_ref = s_ref[triu_i, triu_j]

    corr = float(np.corrcoef(off_ours, off_ref)[0, 1])
    diff = off_ours - off_ref
    return {
        "n": int(n),
        "ours_range": [float(off_ours.min()), float(off_ours.max())],
        "ref_range": [float(off_ref.min()), float(off_ref.max())],
        "ours_mean": float(off_ours.mean()),
        "ref_mean": float(off_ref.mean()),
        "ours_std": float(off_ours.std()),
        "ref_std": float(off_ref.std()),
        "off_diag_pearson_r": corr,
        "max_abs_diff": float(np.abs(diff).max()),
        "rms_diff": float(np.sqrt(np.mean(diff ** 2))),
        "frobenius_ratio": float(np.linalg.norm(s_ours - s_ref) / np.linalg.norm(s_ref)),
    }


def run_pysrf_estimate(s: np.ndarray, label: str) -> dict:
    log.info(f"  pysrf estimate_rank on {label}...")
    t0 = time.time()
    est = estimate_rank(s, n_bootstrap=20, random_state=0)
    elapsed = time.time() - t0
    return {
        "rank": int(est.rank),
        "sampling_fraction": float(est.sampling_fraction),
        "detectability_floor": float(est.detectability_floor),
        "eigenvalues_top10": [float(v) for v in est.eigenvalues[:10]],
        "leakage_top30": [float(v) for v in est.leakage[:30]],
        "runtime_sec": round(elapsed, 1),
    }


def try_run_reference(s: np.ndarray, label: str) -> dict | None:
    """Run update_pysrf's spectral_pass; returns None if import fails."""
    sys.path.insert(0, str(PROJECT_ROOT / "update_pysrf" / "src"))
    try:
        from _common import spectral_pass  # noqa: E402
    except Exception as e:
        log.warning(f"  update_pysrf import failed: {e}")
        return None
    log.info(f"  update_pysrf spectral_pass on {label}...")
    t0 = time.time()
    try:
        spec = spectral_pass(s, B=20, smooth_window=10, show_progress=False)
    except Exception as e:
        log.error(f"  spectral_pass failed: {e}")
        return {"error": str(e)}
    elapsed = time.time() - t0
    return {
        "k_cut_fstat": int(spec["k_cut"]),
        "k_smooth_legacy": int(spec.get("k_smooth_legacy", -1)),
        "k_cliff": int(spec["k_cliff"]) if spec.get("k_cliff") is not None else None,
        "evals_top10": [float(v) for v in np.asarray(spec["evals_ref"])[:10]],
        "kappa_top30": [float(v) for v in np.asarray(spec["kappa_hat"])[:30]],
        "runtime_sec": round(elapsed, 1),
    }


def main() -> None:
    log.info("=" * 70)
    log.info("THINGS rank diagnostic: our matrix vs reference cached matrix")
    log.info("=" * 70)

    log.info("[1/4] Loading our matrix (build_similarity from triplets)...")
    t0 = time.time()
    s_ours = load_ours()
    log.info(f"  shape={s_ours.shape} in {time.time() - t0:.1f}s")

    log.info("[2/4] Loading reference cached matrix...")
    s_ref = load_reference()

    log.info("[3/4] Comparing matrices...")
    diff = compare_matrices(s_ours, s_ref)
    log.info(f"  off-diag Pearson r = {diff['off_diag_pearson_r']:.6f}")
    log.info(f"  Frobenius ratio    = {diff['frobenius_ratio']:.6f}")
    log.info(f"  ours range = {diff['ours_range']}, ref range = {diff['ref_range']}")
    log.info(f"  max abs diff = {diff['max_abs_diff']:.4f}, rms diff = {diff['rms_diff']:.4f}")

    log.info("[4/4] Running estimate_rank on both...")
    results = {
        "diff": diff,
        "pysrf_ours": run_pysrf_estimate(s_ours, "ours"),
        "pysrf_ref": run_pysrf_estimate(s_ref, "reference"),
        "reference_ours": try_run_reference(s_ours, "ours"),
        "reference_ref": try_run_reference(s_ref, "reference"),
    }

    out_json = OUTPUT_DIR / "diagnostic.json"
    out_json.write_text(json.dumps(results, indent=2))
    log.info(f"\nSaved {out_json}")

    summary_path = OUTPUT_DIR / "summary.txt"
    with open(summary_path, "w") as f:
        def w(line=""):
            print(line)
            f.write(line + "\n")

        w("=" * 70)
        w("THINGS rank diagnostic summary")
        w("=" * 70)
        w()
        w("Matrix comparison:")
        w(f"  off-diag Pearson r = {diff['off_diag_pearson_r']:.6f}")
        w(f"  Frobenius ratio    = {diff['frobenius_ratio']:.6f}")
        w(f"  ours range = {diff['ours_range']}, mean={diff['ours_mean']:.4f}")
        w(f"  ref  range = {diff['ref_range']}, mean={diff['ref_mean']:.4f}")
        w()
        w("Rank estimates (n_bootstrap=20, random_state=0):")
        w(f"  pysrf on ours      : rank={results['pysrf_ours']['rank']}, "
          f"p*={results['pysrf_ours']['sampling_fraction']:.3f}")
        w(f"  pysrf on reference : rank={results['pysrf_ref']['rank']}, "
          f"p*={results['pysrf_ref']['sampling_fraction']:.3f}")
        if results["reference_ours"] is not None:
            r = results["reference_ours"]
            w(f"  reference on ours      : k_cut={r.get('k_cut_fstat')}, "
              f"k_smooth_legacy={r.get('k_smooth_legacy')}")
        if results["reference_ref"] is not None:
            r = results["reference_ref"]
            w(f"  reference on reference : k_cut={r.get('k_cut_fstat')}, "
              f"k_smooth_legacy={r.get('k_smooth_legacy')}")
        w()
        w("Interpretation:")
        if abs(diff["frobenius_ratio"]) > 0.01:
            w("  Matrices differ. The data preprocessing affects the rank estimate.")
        else:
            w("  Matrices nearly identical -- differences must be in the algorithm.")

    log.info(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
