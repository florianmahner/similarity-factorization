"""Audit pysrf.estimate_rank against update_pysrf.recipe_K on identical inputs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from pysrf import estimate_rank
from src.utils import get_output_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "update_pysrf" / "src"))
from sbm.datasets import clean_weighted_sbm  # noqa: E402
from _synthetic_extras import make_power_law  # noqa: E402
from _common import recipe_K, spectral_pass  # noqa: E402


log = logging.getLogger(__name__)
OUTPUT_DIR = get_output_dir()


SCALAR_TOL = 1e-9
CURVE_TOL = 1e-6  # ARPACK eigsh tol


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    finite = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0


def _audit_one(label: str, s: np.ndarray, k_cv: int = 5) -> dict:
    sp = spectral_pass(s, B=20, show_progress=False)
    rk = recipe_K(sp, delta=0.10, k_cv=k_cv, p_floor=0.5)
    est = estimate_rank(s, recovery_tolerance=0.10, n_bootstrap=20, random_state=0)

    rows = [
        ("k_cut",              float(est.rank),                float(rk["k_cut"]),       SCALAR_TOL),
        ("p_star",             float(est.sampling_fraction),   float(rk["p_star"]),      SCALAR_TOL),
        ("p_floor",            float(est.detectability_floor), float(rk["p_floor"]),     SCALAR_TOL),
    ]
    arr_rows = [
        ("eigenvalues_top",   est.eigenvalues,                  np.asarray(sp["evals_ref"])[:len(est.eigenvalues)]),
        ("leakage/kappa",     est.leakage,                      np.asarray(sp["kappa_hat"])[:len(est.leakage)]),
        ("p_grid",            est.sampling_grid,                np.asarray(sp["p_grid"])[:len(est.sampling_grid)]),
        ("delta_raw",         est.recovery_raw,                 np.asarray(rk["delta_emp_raw"])[:len(est.recovery_raw)]),
        ("delta_monotone",    est.recovery_monotone,            np.asarray(rk["delta_emp_iso"])[:len(est.recovery_monotone)]),
    ]

    log.info("=" * 72)
    log.info("[%s]", label)
    all_ok = True
    for name, ours, theirs, tol in rows:
        diff = abs(ours - theirs)
        ok = diff <= tol
        all_ok &= ok
        log.info("  %-16s  ours=%-14g  recipe_K=%-14g  |diff|=%.2e  %s",
                 name, ours, theirs, diff, "OK" if ok else "DIFF")
    for name, ours, theirs in arr_rows:
        diff = _max_abs_diff(ours, theirs)
        ok = diff <= CURVE_TOL
        all_ok &= ok
        log.info("  %-16s  max|diff|=%.3e  %s", name, diff, "OK" if ok else "DIFF")
    return {"label": label, "ok": all_ok}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    datasets = [
        ("SBM (n=200, K=10)", clean_weighted_sbm(n=200, K=10, mu_in=1.0, mu_out=0.0,
                                                   sigma=0.3, seed=42)["S"]),
        ("Powerlaw (n=400)",  _powerlaw_dataset()),
    ]

    summary = ["# audit: pysrf.estimate_rank vs update_pysrf.recipe_K\n"]
    overall_ok = True
    for label, s in datasets:
        result = _audit_one(label, s)
        overall_ok &= result["ok"]
        summary.append(f"  {label}:  {'PASS' if result['ok'] else 'FAIL'}")
    summary.append(f"\nOVERALL: {'PASS' if overall_ok else 'FAIL'}")
    (OUTPUT_DIR / "audit_summary.txt").write_text("\n".join(summary) + "\n")
    log.info("\n%s", "\n".join(summary))


def _powerlaw_dataset() -> np.ndarray:
    s_raw, _ = make_power_law(n=400, k_signal=10, alpha=1.0, snr=5.0, seed=42)
    return s_raw - s_raw.min() + 0.01


if __name__ == "__main__":
    main()
