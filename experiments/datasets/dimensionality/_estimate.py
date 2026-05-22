"""Estimate stage: run pysrf.estimate_rank for each dataset, write JSON."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from pysrf import estimate_rank

from . import io as _io
from ._loader import load_similarity


log = logging.getLogger(__name__)


def run_estimate(cfg: DictConfig, output_dir: Path, n_jobs: int) -> list[dict]:
    project_root = Path(cfg.project_root)
    datasets = _select_datasets(cfg)
    force = bool(cfg.get("force", False))
    summary: list[dict] = []

    for i, ds_entry in enumerate(datasets, 1):
        ds = OmegaConf.create(ds_entry)
        params = _dataset_estimate_params(ds, cfg.estimate)
        ds_n_jobs = _resolve_dataset_n_jobs(ds, params, n_jobs)
        log.info(f"\n[{i}/{len(datasets)}] estimate :: {ds.name}")
        payload, did_compute = _process(ds, cfg, params, project_root, output_dir, ds_n_jobs, force)
        if did_compute and _io.cross_validation_path(ds.name, output_dir).exists():
            log.info(
                "  kept existing cross_validation.json; validate will overwrite "
                "variants whose p*/protocol/settings no longer match"
            )
        summary.append({
            "dataset": payload["dataset"],
            "n": payload["n"],
            "rank": payload["estimate"]["rank"],
            "sampling_fraction": payload["estimate"]["sampling_fraction"],
            "detectability_floor": payload["estimate"]["detectability_floor"],
            "runtime_sec": payload["estimate"]["runtime_sec"],
        })

    summary_path = _io.summary_path("coherence_estimate", output_dir, cfg.get("only", None))
    _write_summary(summary_path, summary)
    log.info(f"\nestimate summary written to {summary_path}")
    return summary


def _process(
    ds: DictConfig,
    cfg: DictConfig,
    params: DictConfig,
    project_root: Path,
    output_dir: Path,
    n_jobs: int,
    force: bool,
) -> tuple[dict, bool]:
    json_path = _io.coherence_path(ds.name, output_dir)
    payload = _read_existing(ds.name, output_dir, n=None)

    if not force and "estimate" in payload and _estimate_matches(payload, params, ds):
        log.info(f"  skip (estimate present); rank={payload['estimate']['rank']}, "
                 f"p*={payload['estimate']['sampling_fraction']:.3f}")
        _write_json(json_path, _coherence_payload(payload))
        return payload, False

    t0 = time.time()
    similarity = load_similarity(
        ds,
        project_root,
        cache_dir=output_dir / "cache",
        use_cache=bool(cfg.get("cache_similarity", True)),
        force_cache=bool(cfg.get("force_cache", False)),
    )
    n = similarity.shape[0]
    payload["n"] = int(n)
    log.info(f"  loaded n={n} in {time.time() - t0:.1f}s")

    sampling_grid = np.linspace(params.p_min, params.p_max, params.n_p)
    max_rank = ds.get("max_rank", None)  # None -> pysrf default max(min(n//4, 100), 2)

    t1 = time.time()
    est = estimate_rank(
        similarity,
        recovery_tolerance=params.recovery_tolerance,
        max_rank=max_rank,
        sampling_grid=sampling_grid,
        n_bootstrap=params.n_bootstrap,
        high_band_quantile=params.high_band_quantile,
        random_state=params.random_state,
        n_jobs=n_jobs,
    )
    elapsed = time.time() - t1

    log.info(f"  rank={est.rank}  p*={est.sampling_fraction:.3f}  "
             f"floor={est.detectability_floor:.3f}  n_jobs={n_jobs}  ({elapsed:.1f}s)")

    payload["estimate"] = {
        "rank": int(est.rank),
        "sampling_fraction": float(est.sampling_fraction),
        "detectability_floor": float(est.detectability_floor),
        "eigenvalues": est.eigenvalues.tolist(),
        "leakage": est.leakage.tolist(),
        "sampling_grid": est.sampling_grid.tolist(),
        "recovery_loss_raw": est.recovery_loss_raw.tolist(),
        "recovery_loss_monotone": est.recovery_loss_monotone.tolist(),
        "params": {
            "n_bootstrap": int(params.n_bootstrap),
            "n_p": int(params.n_p),
            "p_min": float(params.p_min),
            "p_max": float(params.p_max),
            "recovery_tolerance": float(params.recovery_tolerance),
            "high_band_quantile": float(params.high_band_quantile),
            "random_state": int(params.random_state),
            "max_rank": int(est.eigenvalues.shape[0]),
            "n_jobs": int(n_jobs),
        },
        "runtime_sec": round(elapsed, 2),
    }
    _write_json(json_path, _coherence_payload(payload))
    return payload, True


def _select_datasets(cfg: DictConfig) -> list[dict]:
    datasets = OmegaConf.to_container(cfg.datasets, resolve=True)
    only = cfg.get("only", None)
    if only is None:
        return datasets
    if isinstance(only, str):
        only_set = {s.strip() for s in only.split(",") if s.strip()}
    else:
        only_set = set(only)
    kept = [d for d in datasets if d["name"] in only_set]
    missing = only_set - {d["name"] for d in kept}
    if missing:
        raise ValueError(f"only references unknown datasets: {sorted(missing)}")
    log.info(f"Filtering to datasets: {sorted(only_set)}")
    return kept


def _read_existing(name: str, output_dir: Path, n: int | None) -> dict:
    for json_path in [_io.coherence_path(name, output_dir), *_io.legacy_result_paths(name, output_dir)]:
        if json_path.exists():
            return json.loads(json_path.read_text())
    return {"dataset": name, "n": n}


def _coherence_payload(payload: dict) -> dict:
    return {
        "dataset": payload.get("dataset"),
        "n": payload.get("n"),
        "estimate": payload.get("estimate"),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.write_text(json.dumps(rows, indent=2))


def _resolve_dataset_n_jobs(ds: DictConfig, params: DictConfig, fallback: int) -> int:
    value = ds.get("estimate_n_jobs", ds.get("n_jobs", params.get("n_jobs", fallback)))
    value = int(value)
    return fallback if value <= 0 else value


def _dataset_estimate_params(ds: DictConfig, params: DictConfig) -> DictConfig:
    overrides = ds.get("estimate", {})
    return OmegaConf.merge(params, overrides)


def _estimate_matches(payload: dict, params: DictConfig, ds: DictConfig) -> bool:
    existing = payload.get("estimate", {}).get("params", {})
    checks = {
        "n_bootstrap": int(params.n_bootstrap),
        "n_p": int(params.n_p),
        "p_min": float(params.p_min),
        "p_max": float(params.p_max),
        "recovery_tolerance": float(params.recovery_tolerance),
        "high_band_quantile": float(params.high_band_quantile),
        "random_state": int(params.random_state),
    }
    for key, expected in checks.items():
        if key not in existing:
            return False
        if isinstance(expected, float):
            if abs(float(existing[key]) - expected) > 1e-12:
                return False
        elif int(existing[key]) != expected:
            return False
    if ds.get("max_rank", None) is not None:
        return int(existing.get("max_rank", -1)) == int(ds.max_rank)
    return True
