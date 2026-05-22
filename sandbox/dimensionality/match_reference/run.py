"""Sandbox: identify what single change makes our pysrf match update_pysrf rank.

Background:
  Reference (update_pysrf.spectral_pass) gives k_cut=26 on THINGS.
  Our pysrf.estimate_rank gives rank=23 on the same matrix.
  Same coherence/leakage/F-stat math; matrices identical.

Two differences exist between the two bootstrap kernels:
  (A) Seed generation per (p_index, replicate)
        reference: (seed_base + 1_000_003*i + 9176*b) & 0xFFFFFFFF
        ours:      SeedSequence(random_state).spawn(P*B).reshape(P, B)
  (B) eigsh `v0` argument
        reference: not passed (scipy default; numpy global state)
        ours:      v0 = rng.standard_normal(n)

Goal: ablate seed style x v0 x backend and find the minimum change that flips
rank from 23 -> 26.

Variants (seed style, eigsh v0, joblib backend):
  V0_REF        : reference spectral_pass (calls update_pysrf code)
  V1_BASELINE   : SeedSequence + explicit v0  + threading    (current pysrf)
  V2_NO_V0      : SeedSequence + no v0        + threading
  V3_NO_V0_LOKY : SeedSequence + no v0        + loky processes
  V4_HASH_NO_V0 : magic-hash   + no v0        + loky          (reference mirror)
  V5_HASH_V0    : magic-hash   + explicit v0  + threading      (isolates seed-only)

Each records rank, p*, runtime, leakage_top30. Summary printed and saved.

Run:
    ./scripts/submit sandbox/dimensionality/match_reference/run.py --bg
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
from joblib import Parallel, delayed
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
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


# -------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------
def load_things() -> np.ndarray:
    path = PROJECT_ROOT / "update_pysrf" / "data" / "things_behavior_similarity.npy"
    a = np.load(path)
    a = (a + a.T) / 2.0
    diag = np.eye(a.shape[0], dtype=bool)
    med = float(np.nanmedian(a[~diag]))
    return np.where(np.isnan(a), med, a)


# -------------------------------------------------------------------------
# Reference (update_pysrf code, ground truth)
# -------------------------------------------------------------------------
def run_reference(s: np.ndarray) -> dict:
    sys.path.insert(0, str(PROJECT_ROOT / "update_pysrf" / "src"))
    from _common import spectral_pass  # noqa: E402

    t0 = time.time()
    spec = spectral_pass(s, B=20, smooth_window=10, show_progress=False)
    elapsed = time.time() - t0
    return {
        "label": "V0_REF",
        "rank": int(spec["k_cut"]),
        "sampling_fraction": None,
        "kappa_hat_top30": [float(v) for v in np.asarray(spec["kappa_hat"])[:30]],
        "runtime_sec": round(elapsed, 1),
    }


# -------------------------------------------------------------------------
# Baseline (current pysrf)
# -------------------------------------------------------------------------
def run_pysrf_baseline(s: np.ndarray) -> dict:
    t0 = time.time()
    est = pysrf_estimate_rank(s, n_bootstrap=20, random_state=0)
    elapsed = time.time() - t0
    return {
        "label": "V1_BASELINE",
        "rank": int(est.rank),
        "sampling_fraction": float(est.sampling_fraction),
        "kappa_hat_top30": [float(v) for v in est.leakage[:30]],
        "runtime_sec": round(elapsed, 1),
    }


# -------------------------------------------------------------------------
# Prototype bootstrap (parameterized)
# -------------------------------------------------------------------------
def _subsampled_replicate(s, mask, p, rng, triu):
    n = s.shape[0]
    ti, tj = triu
    sampled = rng.random(ti.size) < p
    sampled &= mask[ti, tj]
    rep = np.zeros((n, n), dtype=np.float64)
    v = s[ti[sampled], tj[sampled]] / p
    rep[ti[sampled], tj[sampled]] = v
    rep[tj[sampled], ti[sampled]] = v
    np.fill_diagonal(rep, np.diag(s))
    return rep


def _topk_eigenvectors(a, k, v0=None):
    """Mirror update_pysrf._topk_eigenvectors: eigsh -> dense eigh fallback."""
    n = a.shape[0]
    if k < n:
        try:
            kwargs = {"which": "LA", "tol": 1e-6}
            if v0 is not None:
                kwargs["v0"] = v0
            vals, vecs = eigsh(a, k=k, **kwargs)
            idx = np.argsort(vals)[::-1]
            return vals[idx], vecs[:, idx]
        except Exception:
            pass
    vals, vecs = eigh(a)
    idx = np.argsort(vals)[::-1][:k]
    return vals[idx], vecs[:, idx]


def _cumulative_subspace_overlap(v_boot, v_ref):
    sq = (v_boot.T @ v_ref) ** 2
    r = np.arange(sq.shape[0])
    return np.clip(np.cumsum(sq, axis=0)[r, r], 0.0, 1.0)


def _cumulative_recovered_mass(s, v_boot):
    return np.cumsum(np.einsum("ij,ij->j", v_boot, s @ v_boot))


def _one_p_job(i, p, s, mask, v_ref, max_rank, n_bootstrap,
                seed_mode, use_v0, replicate_seeds, hash_base):
    """One p value, all B reps.

    seed_mode:
      "spawn" -> use replicate_seeds[b] (a SeedSequence object)
      "hash"  -> use (hash_base + 1_000_003*i + 9176*b) & 0xFFFFFFFF
    """
    n = s.shape[0]
    triu = np.triu_indices(n, k=1)
    coh = np.empty((max_rank, n_bootstrap), dtype=np.float64)
    rec = np.empty_like(coh)
    for b in range(n_bootstrap):
        if seed_mode == "spawn":
            rng = np.random.default_rng(replicate_seeds[b])
        else:
            seed = (int(hash_base) + 1_000_003 * i + 9176 * b) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
        replicate = _subsampled_replicate(s, mask, p, rng, triu)
        v0 = rng.standard_normal(n) if use_v0 else None
        _, v_boot = _topk_eigenvectors(replicate, max_rank, v0=v0)
        coh[:, b] = _cumulative_subspace_overlap(v_boot, v_ref)
        rec[:, b] = _cumulative_recovered_mass(s, v_boot)
    return i, coh, rec


def run_prototype(
    s: np.ndarray, *, label: str, seed_mode: str, use_v0: bool,
    backend: str, n_jobs: int | None = None, n_bootstrap: int = 20,
    random_state: int = 0,
) -> dict:
    n = s.shape[0]
    max_rank = max(min(n // 4, 100), 2)
    sampling_grid = np.linspace(0.05, 0.95, 20)
    n_p = len(sampling_grid)
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    t0 = time.time()
    # Reference eigenpairs: dense eigh (one-shot, n=1854 -> a few seconds)
    s_sym = 0.5 * (s + s.T)
    vals_all, vecs_all = eigh(s_sym)
    idx = np.argsort(vals_all)[::-1][:max_rank]
    top_vals = vals_all[idx]
    top_vecs = vecs_all[:, idx]
    mask = np.ones((n, n), dtype=bool)

    # Pre-compute seeds (matches pysrf's structure exactly)
    if seed_mode == "spawn":
        all_seeds = np.array(
            as_seed_sequence(random_state).spawn(n_p * n_bootstrap),
            dtype=object,
        ).reshape(n_p, n_bootstrap)
        args = [
            (i, float(p), s, mask, top_vecs, max_rank, n_bootstrap,
             "spawn", use_v0, list(all_seeds[i]), None)
            for i, p in enumerate(sampling_grid)
        ]
    else:  # hash
        args = [
            (i, float(p), s, mask, top_vecs, max_rank, n_bootstrap,
             "hash", use_v0, None, random_state)
            for i, p in enumerate(sampling_grid)
        ]

    if backend == "threading":
        with threadpool_limits(limits=1):
            out = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(_one_p_job)(*a) for a in args
            )
    elif backend == "loky":
        out = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_one_p_job)(*a) for a in args
        )
    else:
        out = [_one_p_job(*a) for a in args]

    coherence = np.empty((max_rank, n_p, n_bootstrap), dtype=np.float64)
    recovered = np.empty_like(coherence)
    for i, c, r in out:
        coherence[:, i, :] = c
        recovered[:, i, :] = r

    rank, leakage = _select_rank(coherence, sampling_grid, 0.85)
    p_star, _ = _calibrate_sampling_fraction(
        recovered, top_vals, sampling_grid, rank, recovery_tolerance=0.10,
    )
    return {
        "label": label,
        "rank": int(rank),
        "sampling_fraction": float(p_star),
        "kappa_hat_top30": [float(v) for v in leakage[:30]],
        "runtime_sec": round(time.time() - t0, 1),
        "config": {"seed_mode": seed_mode, "use_v0": use_v0, "backend": backend},
    }


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main() -> None:
    log.info("=" * 70)
    log.info("THINGS rank: ablate seed-style x v0 x backend vs update_pysrf")
    log.info("=" * 70)

    s = load_things()
    log.info(f"loaded THINGS: shape={s.shape}")

    results = []

    log.info("\n[V0_REF] update_pysrf.spectral_pass...")
    results.append(run_reference(s))
    log.info(f"  rank={results[-1]['rank']}  runtime={results[-1]['runtime_sec']}s")

    log.info("\n[V1_BASELINE] current pysrf.estimate_rank...")
    results.append(run_pysrf_baseline(s))
    log.info(f"  rank={results[-1]['rank']}  runtime={results[-1]['runtime_sec']}s")

    variants = [
        ("V2_NO_V0",      "spawn", False, "threading"),
        ("V3_NO_V0_LOKY", "spawn", False, "loky"),
        ("V4_HASH_NO_V0", "hash",  False, "loky"),
        ("V5_HASH_V0",    "hash",  True,  "threading"),
    ]
    for label, seed_mode, use_v0, backend in variants:
        log.info(f"\n[{label}] seed={seed_mode}, v0={use_v0}, backend={backend}...")
        results.append(run_prototype(s, label=label, seed_mode=seed_mode,
                                       use_v0=use_v0, backend=backend))
        log.info(f"  rank={results[-1]['rank']}  runtime={results[-1]['runtime_sec']}s")

    log.info("\n" + "=" * 70)
    log.info("Summary")
    log.info("=" * 70)
    log.info(f"{'variant':<18}  {'rank':>4}  {'p*':>6}  {'runtime':>7}")
    for r in results:
        ps = "N/A" if r.get("sampling_fraction") is None else f"{r['sampling_fraction']:.3f}"
        log.info(f"{r['label']:<18}  {r['rank']:>4}  {ps:>6}  {r['runtime_sec']:>5}s")

    log.info("\nLeakage profile at the divergence point:")
    log.info(f"{'variant':<18}  {'dim 23':>10}  {'dim 24':>10}  {'dim 25':>10}  {'dim 26':>10}")
    for r in results:
        k = r["kappa_hat_top30"]
        log.info(f"{r['label']:<18}  {k[22]:10.4f}  {k[23]:10.4f}  {k[24]:10.4f}  {k[25]:10.4f}")

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(results, indent=2))
    log.info(f"\nSaved {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
