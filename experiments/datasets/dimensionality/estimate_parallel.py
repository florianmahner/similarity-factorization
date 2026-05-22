"""Parallel-over-(p, bootstrap) coherence estimator.

Pysrf's `estimate_rank` parallelizes ONLY over the `n_p` sampling fractions
(20 tasks for n_p=20), with `n_bootstrap` iters sequential inside each task.
With n_jobs > n_p the extra cores sit idle.

This wrapper imports pysrf's internal helpers (_top_eigenpairs,
_bootstrap_at_fraction's inner logic, _select_rank, _calibrate_sampling_fraction)
and re-parallelizes across all (p_idx, bootstrap_idx) pairs:

    n_p * n_bootstrap = 400 independent tasks at n_p=20, n_boot=20
    -> any n_jobs up to 400 actually uses cores

Per-chunk progress logging (chunk = n_jobs tasks) so we get per-wave updates
instead of one silent 5-hour batch.

Output JSON exactly matches the format that `_estimate.py` produces so downstream
plotting / validation flows are unchanged.

Usage:
    python -m experiments.datasets.dimensionality.estimate_parallel \
        --dataset things_macaque22k_tight \
        --n-jobs 64

Reads similarity from `outputs/cache/<dataset>.npy` (will build it first if
missing, using the dataset config in configs/dataset/<config>.yaml).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

# Internal pysrf helpers — these are stable utilities used by estimate_rank;
# we import (not modify) so we can drive the bootstrap loop ourselves and
# get per-chunk progress.
from pysrf.coherence._estimate import (
    _calibrate_sampling_fraction,
    _prepare_input,
    _select_rank,
)
from pysrf.coherence._bootstrap import (
    _bootstrap_parent_seed,
    _cumulative_recovered_mass,
    _cumulative_subspace_overlap,
    _replicate_seed,
    _subsampled_replicate,
    _top_eigenpairs,
)

from . import io as _io

log = logging.getLogger("estimate_parallel")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


def _single_bootstrap_iter(
    p_idx: int,
    sampling_fraction: float,
    b_idx: int,
    similarity: np.ndarray,
    observation_mask: np.ndarray,
    top_eigenvectors: np.ndarray,
    parent_seed: int,
    triu_i: np.ndarray,
    triu_j: np.ndarray,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """One bootstrap replicate at one sampling fraction. Returns (p_idx, b_idx, coherence, recovered_mass)."""
    max_rank = top_eigenvectors.shape[1]
    seed = _replicate_seed(parent_seed, p_idx, b_idx)
    rng = np.random.default_rng(seed)
    with threadpool_limits(limits=1):
        replicate = _subsampled_replicate(
            similarity, observation_mask, sampling_fraction, rng, (triu_i, triu_j),
        )
        _, replicate_eigenvectors = _top_eigenpairs(replicate, max_rank)
        coh = _cumulative_subspace_overlap(replicate_eigenvectors, top_eigenvectors)
        mass = _cumulative_recovered_mass(similarity, replicate_eigenvectors)
    return p_idx, b_idx, coh, mass


def parallel_estimate(
    similarity: np.ndarray,
    *,
    max_rank: int,
    sampling_grid: np.ndarray,
    n_bootstrap: int,
    recovery_tolerance: float,
    high_band_quantile: float,
    random_state: int,
    n_jobs: int,
) -> dict:
    """Run the full bootstrap loop parallelized over (p, b) with chunked progress logging."""
    s, observed_mask = _prepare_input(similarity)
    n = s.shape[0]

    log.info(f"computing top {max_rank} eigenpairs of n={n} similarity...")
    t0 = time.time()
    top_eigenvalues, top_eigenvectors = _top_eigenpairs(s, max_rank)
    log.info(f"  eigendecomp done in {time.time() - t0:.1f}s")

    triu_i, triu_j = np.triu_indices(n, k=1)
    parent_seed = _bootstrap_parent_seed(random_state)

    # Build the (p, b) task list
    tasks = [
        (p_idx, float(frac), b_idx)
        for p_idx, frac in enumerate(sampling_grid)
        for b_idx in range(n_bootstrap)
    ]
    n_total = len(tasks)
    log.info(f"dispatching {n_total} bootstrap fits "
             f"({len(sampling_grid)} p-values x {n_bootstrap} bootstrap iters) "
             f"on n_jobs={n_jobs}")

    # Chunk so we get progress per wave-ish; 2x n_jobs per chunk gives ~2 waves per log line.
    chunk_size = max(n_jobs, 2 * n_jobs)
    chunks = [tasks[i:i + chunk_size] for i in range(0, n_total, chunk_size)]

    all_results: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    overall_t0 = time.time()
    for chunk_idx, chunk in enumerate(chunks, 1):
        t0 = time.time()
        log.info(f"chunk {chunk_idx}/{len(chunks)}: {len(chunk)} fits starting "
                 f"(p_idx range {chunk[0][0]}..{chunk[-1][0]})")
        chunk_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_single_bootstrap_iter)(
                p_idx, frac, b_idx, s, observed_mask, top_eigenvectors,
                parent_seed, triu_i, triu_j,
            )
            for (p_idx, frac, b_idx) in chunk
        )
        elapsed = time.time() - t0
        log.info(f"  chunk {chunk_idx}/{len(chunks)} done in {elapsed:.1f}s  "
                 f"(total elapsed {time.time() - overall_t0:.1f}s, "
                 f"{(chunk_idx / len(chunks)) * 100:.0f}% complete)")
        all_results.extend(chunk_results)

    total_elapsed = time.time() - overall_t0
    log.info(f"all {n_total} fits done in {total_elapsed:.1f}s")

    # Aggregate into (max_rank, n_p, n_bootstrap) -> match pysrf's per-p arrays
    coherence_arr = np.zeros((max_rank, len(sampling_grid), n_bootstrap), dtype=np.float64)
    mass_arr = np.zeros_like(coherence_arr)
    for p_idx, b_idx, coh, mass in all_results:
        coherence_arr[:, p_idx, b_idx] = coh
        mass_arr[:, p_idx, b_idx] = mass

    # Stack to (max_rank, n_p) collapsed-over-b means / matrices, matching pysrf format
    # Pysrf stores per-rank-per-p as (max_rank, n_p) with stacked bootstraps along axis 1.
    # We want (max_rank, n_p) for downstream _select_rank.
    coherence_matrix = coherence_arr  # (max_rank, n_p, n_boot)
    mass_matrix = mass_arr  # same shape
    # _select_rank and _calibrate_sampling_fraction expect (max_rank, n_p) — they
    # internally aggregate over bootstrap iters within a (max_rank, n_p) "slice"
    # of stacked replicate arrays. Reshape to (max_rank, n_p * n_boot) is what
    # pysrf's _bootstrap_subspace_stability produces:
    coherence_stacked = coherence_matrix.reshape(max_rank, len(sampling_grid) * n_bootstrap)
    # but actually pysrf returns (max_rank, n_p) after averaging bootstrap-internal
    # operations into a single per-p slice. Let me check by reading the code again...
    # Looking at _bootstrap_subspace_stability:
    #   slices = [_bootstrap_at_fraction(...) for each p]  -> list of (max_rank, n_bootstrap) arrays
    #   coherence = np.stack([s[0] for s in slices], axis=1)  -> (max_rank, n_p)? No: axis=1 stacks p
    #   But each s[0] is shape (max_rank, n_bootstrap), so stacking on axis=1 gives
    #   (max_rank, n_p * n_bootstrap) — actually that's stack semantics on a list of 2D arrays.
    # Actually np.stack(list_of_2D, axis=1) yields a 3D array with new axis inserted.
    # Re-check: np.stack(arrs, axis=1) where each arr is (M, K) -> (M, n, K)
    # So pysrf returns (max_rank, n_p, n_bootstrap) -- same as my coherence_matrix. Good.
    coherence_per_p = coherence_matrix  # (max_rank, n_p, n_bootstrap)
    recovered_per_p = mass_matrix

    log.info("aggregating + selecting rank...")
    rank, leakage = _select_rank(coherence_per_p, sampling_grid, high_band_quantile)
    sampling_fraction, loss_curve = _calibrate_sampling_fraction(
        recovered_per_p, top_eigenvalues, sampling_grid, rank, recovery_tolerance,
    )

    return {
        "rank": int(rank),
        "sampling_fraction": float(sampling_fraction),
        "detectability_floor": float(loss_curve.floor),
        "eigenvalues": top_eigenvalues.tolist(),
        "leakage": leakage.tolist(),
        "sampling_grid": loss_curve.sampling_grid.tolist(),
        "recovery_loss_raw": loss_curve.raw.tolist(),
        "recovery_loss_monotone": loss_curve.monotone.tolist(),
        "runtime_sec": round(total_elapsed, 2),
        "n_features_in": int(n),
    }


def main() -> None:
    args = _parse_args()
    _limit_thread_env()

    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    cache_path = output_dir / "cache" / f"{args.dataset}.npy"

    if not cache_path.exists() and not args.build_sim:
        raise FileNotFoundError(
            f"No cached similarity at {cache_path}. "
            f"Build it first with: --build-sim --config-name <dataset_config>"
        )

    if args.build_sim:
        log.info(f"building similarity for {args.dataset} (config={args.config_name})")
        from similarity import build_similarity
        from hydra import compose, initialize_config_dir
        # Compose with the dataset config AND the paths config so ${paths.ssd} resolves
        cfg_root = str(project_root / "configs")
        with initialize_config_dir(version_base=None, config_dir=cfg_root):
            cfg = compose(
                config_name="base",
                overrides=[
                    f"+dataset={args.config_name}",
                    "experiment_name=adhoc",
                    f"task={args.dataset}",
                    f"project_root={project_root}",
                ],
            )
        ds_cfg = cfg.dataset
        sim = build_similarity(ds_cfg)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, sim)
        log.info(f"  wrote {cache_path}  shape={sim.shape}")

    log.info(f"loading similarity from {cache_path}")
    similarity = np.load(cache_path)
    log.info(f"  loaded n={similarity.shape[0]}")

    sampling_grid = np.linspace(args.p_min, args.p_max, args.n_p)
    estimate = parallel_estimate(
        similarity,
        max_rank=args.max_rank,
        sampling_grid=sampling_grid,
        n_bootstrap=args.n_bootstrap,
        recovery_tolerance=args.recovery_tolerance,
        high_band_quantile=args.high_band_quantile,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    estimate["params"] = {
        "n_bootstrap": int(args.n_bootstrap),
        "n_p": int(args.n_p),
        "p_min": float(args.p_min),
        "p_max": float(args.p_max),
        "recovery_tolerance": float(args.recovery_tolerance),
        "high_band_quantile": float(args.high_band_quantile),
        "random_state": int(args.random_state),
        "max_rank": int(args.max_rank),
        "n_jobs": int(args.n_jobs),
    }

    json_path = _io.coherence_path(args.dataset, output_dir)
    payload = {
        "dataset": args.dataset,
        "n": int(similarity.shape[0]),
        "estimate": estimate,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(json_path)
    log.info(f"wrote {json_path}  rank={estimate['rank']}  p*={estimate['sampling_fraction']:.3f}  "
             f"floor={estimate['detectability_floor']:.3f}")


def _limit_thread_env() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(key, "1")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Output dataset name (e.g. things_macaque22k_tight)")
    parser.add_argument("--config-name", default=None,
                        help="Dataset YAML name in configs/dataset/ (only needed if --build-sim).")
    parser.add_argument("--build-sim", action="store_true",
                        help="Build similarity matrix first (otherwise expects cached .npy).")
    parser.add_argument("--n-p", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=20)
    parser.add_argument("--p-min", type=float, default=0.05)
    parser.add_argument("--p-max", type=float, default=0.95)
    parser.add_argument("--max-rank", type=int, default=200)
    parser.add_argument("--recovery-tolerance", type=float, default=0.10)
    parser.add_argument("--high-band-quantile", type=float, default=0.85)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=64)
    parser.add_argument("--project-root", default="/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization")
    parser.add_argument("--output-dir", default="/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/datasets/dimensionality/outputs")
    return parser.parse_args()


if __name__ == "__main__":
    main()
