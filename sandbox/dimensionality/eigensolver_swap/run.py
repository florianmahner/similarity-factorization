"""Prototype: swap the bootstrap eigensolver to dense eigh.

The diagnostic in sandbox/dimensionality/things_diagnostic showed pysrf
and update_pysrf give the same leakage profile up to dim 23, then diverge
at dim 24 (pysrf 7.66 spike vs reference 4.39 smooth) on the same matrix.
The spike pulls pysrf's F-stat changepoint to 23; reference picks 26.

ARPACK with a random v0 (our pysrf) can produce slightly less accurate
eigenvectors at tol=1e-6 than dense `eigh` (the reference's LOBPCG
fallback). In close-eigenvalue subspaces, ARPACK's basis can rotate
between replicates while dense eigh's stays canonical.

Hypothesis: simply switching to dense eigh in the bootstrap should give
rank=26 for THINGS. If yes, the fix is one line in pysrf. No LOBPCG
warm-start chain needed.

Run:
    ./scripts/submit sandbox/dimensionality/eigensolver_swap/run.py --bg
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from pysrf import estimate_rank as pysrf_estimate_rank
from pysrf._common import as_seed_sequence
from pysrf.coherence._rank_selection import _select_rank
from pysrf.coherence._sampling_fraction import _calibrate_sampling_fraction
from src.utils import get_output_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = get_output_dir()


def _top_eigenpairs_eigh(a: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense top-k eigenpairs (descending). Same algorithm reference falls back to."""
    a = 0.5 * (a + a.T)
    vals, vecs = eigh(a)
    idx = np.argsort(vals)[::-1][:k]
    return vals[idx], vecs[:, idx]


def _subsampled_replicate(s, mask, p, rng, triu):
    n = s.shape[0]
    triu_i, triu_j = triu
    sampled = rng.random(triu_i.size) < p
    sampled &= mask[triu_i, triu_j]
    rep = np.zeros((n, n), dtype=np.float64)
    v = s[triu_i[sampled], triu_j[sampled]] / p
    rep[triu_i[sampled], triu_j[sampled]] = v
    rep[triu_j[sampled], triu_i[sampled]] = v
    np.fill_diagonal(rep, np.diag(s))
    return rep


def _cumulative_subspace_overlap(v_boot, v_ref):
    sq = (v_boot.T @ v_ref) ** 2
    r = np.arange(sq.shape[0])
    return np.clip(np.cumsum(sq, axis=0)[r, r], 0.0, 1.0)


def _cumulative_recovered_mass(s, v_boot):
    return np.cumsum(np.einsum("ij,ij->j", v_boot, s @ v_boot))


def _one_replicate(seed, p, s, mask, v_ref, max_rank, triu):
    rng = np.random.default_rng(seed)
    rep = _subsampled_replicate(s, mask, p, rng, triu)
    _, v_boot = _top_eigenpairs_eigh(rep, max_rank)
    return (_cumulative_subspace_overlap(v_boot, v_ref),
            _cumulative_recovered_mass(s, v_boot))


def estimate_rank_eigh(s, n_bootstrap=20, random_state=0, n_jobs=None):
    """pysrf.estimate_rank but with dense eigh in the bootstrap."""
    s = np.asarray(s, dtype=np.float64)
    s = 0.5 * (s + s.T)
    s = np.nan_to_num(s, nan=0.0)
    n = s.shape[0]
    max_rank = max(min(n // 4, 100), 2)
    sampling_grid = np.linspace(0.05, 0.95, 20)
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    seed_eigen, seed_boot = as_seed_sequence(random_state).spawn(2)
    rng = np.random.default_rng(seed_eigen)
    top_vals, top_vecs = _top_eigenpairs_eigh(s, max_rank)

    n_p = len(sampling_grid)
    seeds = as_seed_sequence(seed_boot).spawn(n_p * n_bootstrap)
    replicate_seeds = np.array(seeds, dtype=object).reshape(n_p, n_bootstrap)
    mask = np.ones((n, n), dtype=bool)
    triu = np.triu_indices(n, k=1)

    # Flatten (p, b) -> one job each. 400 jobs over ~127 cores: good utilization.
    jobs = [(replicate_seeds[i, b], float(p), s, mask, top_vecs, max_rank, triu)
            for i, p in enumerate(sampling_grid) for b in range(n_bootstrap)]
    with threadpool_limits(limits=1):
        out = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_one_replicate)(*j) for j in jobs
        )

    coherence = np.empty((max_rank, n_p, n_bootstrap), dtype=np.float64)
    recovered = np.empty_like(coherence)
    for idx, (coh, rec) in enumerate(out):
        i = idx // n_bootstrap
        b = idx % n_bootstrap
        coherence[:, i, b] = coh
        recovered[:, i, b] = rec

    rank, leakage = _select_rank(coherence, sampling_grid, 0.85)
    p_star, _ = _calibrate_sampling_fraction(
        recovered, top_vals, sampling_grid, rank, recovery_tolerance=0.10,
    )
    return {
        "rank": int(rank),
        "sampling_fraction": float(p_star),
        "leakage_top30": [float(v) for v in leakage[:30]],
    }


def load_reference_things() -> np.ndarray:
    path = PROJECT_ROOT / "update_pysrf" / "data" / "things_behavior_similarity.npy"
    a = np.load(path)
    a = (a + a.T) / 2.0
    diag = np.eye(a.shape[0], dtype=bool)
    med = float(np.nanmedian(a[~diag]))
    return np.where(np.isnan(a), med, a)


def main():
    log.info("=" * 70)
    log.info("Dense-eigh bootstrap prototype on THINGS")
    log.info("=" * 70)

    s = load_reference_things()
    log.info(f"shape={s.shape}")

    log.info("\n[1/2] Current pysrf (ARPACK + random v0)...")
    t0 = time.time()
    pysrf_est = pysrf_estimate_rank(s, n_bootstrap=20, random_state=0)
    log.info(f"  rank={pysrf_est.rank}  p*={pysrf_est.sampling_fraction:.3f}  "
             f"({time.time() - t0:.1f}s)")

    log.info("\n[2/2] Prototype (dense eigh in bootstrap)...")
    t0 = time.time()
    eigh_est = estimate_rank_eigh(s, n_bootstrap=20, random_state=0)
    log.info(f"  rank={eigh_est['rank']}  p*={eigh_est['sampling_fraction']:.3f}  "
             f"({time.time() - t0:.1f}s)")

    log.info("\nLeakage profile around the changepoint:")
    log.info(f"{'r':>3} | {'pysrf':>10} | {'eigh':>10}")
    pysrf_leak = pysrf_est.leakage[:30]
    eigh_leak = eigh_est["leakage_top30"]
    for i in range(30):
        marker = "  <-- divergence" if i == 23 else ""
        log.info(f"{i+1:>3} | {pysrf_leak[i]:10.4f} | {eigh_leak[i]:10.4f}{marker}")

    summary = {
        "n": int(s.shape[0]),
        "pysrf": {"rank": int(pysrf_est.rank),
                  "sampling_fraction": float(pysrf_est.sampling_fraction),
                  "leakage_top30": [float(v) for v in pysrf_leak]},
        "eigh": eigh_est,
        "target_rank": 26,
        "eigh_matches_target": eigh_est["rank"] == 26,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"\nSaved {OUTPUT_DIR / 'summary.json'}")
    log.info(f"\nFINAL: pysrf={pysrf_est.rank}, dense_eigh={eigh_est['rank']}, target=26")


if __name__ == "__main__":
    main()
