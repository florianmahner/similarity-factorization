"""Validate stage: adaptive cross_val_score at p*, stored in cross_validation.json."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from pysrf import cross_val_score

from . import io as _io
from ._loader import load_similarity

log = logging.getLogger(__name__)
CV_PROTOCOL_VERSION = "entry_prekfold_diag_v2"


def run_validate(cfg: DictConfig, output_dir: Path, n_jobs: int) -> list[dict]:
    project_root = Path(cfg.project_root)
    datasets = _select_datasets(cfg)
    force = bool(cfg.get("force", False))
    summary: list[dict] = []

    for i, ds_entry in enumerate(datasets, 1):
        ds = OmegaConf.create(ds_entry)
        log.info(f"\n[{i}/{len(datasets)}] validate :: {ds.name}")
        summary.extend(_process(ds, cfg, cfg.validate, project_root, output_dir, n_jobs, force))

    summary_path = _io.summary_path("cross_validation", output_dir, cfg.get("only", None))
    _write_summary(summary_path, summary)
    log.info(f"\nvalidate summary written to {summary_path}")
    return summary


def _process(
    ds: DictConfig,
    cfg: DictConfig,
    params: DictConfig,
    project_root: Path,
    output_dir: Path,
    fallback_n_jobs: int,
    force: bool,
) -> list[dict]:
    coherence_path = _io.coherence_path(ds.name, output_dir)
    if not coherence_path.exists() and not any(path.exists() for path in _io.legacy_result_paths(ds.name, output_dir)):
        log.warning(f"  no coherence_estimate.json at {coherence_path}; skipping.")
        return []

    coherence = _io.read_coherence(ds.name, output_dir)
    if "estimate" not in coherence:
        log.warning(f"  no 'estimate' block in {coherence_path}; skipping.")
        return []

    ds_params = _dataset_validate_params(ds, params)
    variants = _validation_variants(ds_params)
    primary_variant = variants[0]["name"]
    cv_path = _io.cross_validation_path(ds.name, output_dir)
    cv_payload = _read_cv_existing(cv_path, coherence)
    validations = cv_payload.setdefault("validations", {})

    k_cut = int(coherence["estimate"]["rank"])
    sampling_fraction = float(coherence["estimate"]["sampling_fraction"])
    n = int(coherence["n"])
    seed_ranks = _compose_seed_ranks(ds, ds_params, k_cut, n)
    ds_n_jobs = _resolve_dataset_n_jobs(ds, ds_params, fallback_n_jobs)

    log.info(
        f"  k_cut={k_cut}  p*={sampling_fraction:.3f}  "
        f"seed_ranks={seed_ranks}  n_jobs={ds_n_jobs}"
    )

    similarity = None
    records: list[dict] = []
    for variant in variants:
        name = variant["name"]
        block = validations.get(name)
        can_resume = (
            not force
            and isinstance(block, dict)
            and _matches_existing(block, seed_ranks, sampling_fraction, variant, ds_params)
        )
        if (
            can_resume
            and _block_complete(block)
        ):
            log.info(f"  skip {name} (present); argmin_rank={block.get('argmin_rank')}")
            records.append(_summary_row(coherence, name, block))
            continue

        if similarity is None:
            t0 = time.time()
            similarity = load_similarity(
                ds,
                project_root,
                cache_dir=output_dir / "cache",
                use_cache=bool(cfg.get("cache_similarity", False)),
                force_cache=bool(cfg.get("force_cache", False)),
            )
            log.info(f"  loaded n={similarity.shape[0]} in {time.time() - t0:.1f}s")

        def checkpoint(record: dict) -> None:
            validations[name] = record
            if name == primary_variant:
                cv_payload["primary_variant"] = name
            cv_payload["updated_at_unix"] = time.time()
            _write_json_atomic(cv_path, cv_payload)

        record = _run_variant(
            dataset=ds.name,
            similarity=similarity,
            seed_ranks=seed_ranks,
            k_cut=k_cut,
            n=n,
            sampling_fraction=sampling_fraction,
            variant=variant,
            params=ds_params,
            n_jobs=ds_n_jobs,
            existing_block=block if can_resume else None,
            checkpoint=checkpoint,
        )
        checkpoint(record)
        records.append(_summary_row(coherence, name, record))

    if primary_variant in validations:
        cv_payload["primary_variant"] = primary_variant
        _write_json_atomic(cv_path, cv_payload)
    return records


def _run_variant(
    dataset: str,
    similarity: np.ndarray,
    seed_ranks: list[int],
    k_cut: int,
    n: int,
    sampling_fraction: float,
    variant: dict,
    params: DictConfig,
    n_jobs: int,
    existing_block: dict | None,
    checkpoint: Callable[[dict], None],
) -> dict:
    n_folds = int(variant["n_folds"])
    n_repeats = int(variant["n_repeats"])
    expected_per_rank = n_folds * n_repeats
    strategy = str(params.get("strategy", "fixed")).lower()
    curve = _curve_from_block(existing_block, expected_per_rank)
    completed = _completed_ranks(curve, seed_ranks, expected_per_rank)

    log.info(
        f"  {variant['name']}: strategy={strategy} folds={n_folds} repeats={n_repeats} "
        f"completed={sorted(completed)}"
    )
    started = time.time()

    if strategy == "adaptive":
        curve, final_ranks = _run_adaptive_curve(
            curve=curve,
            dataset=dataset,
            similarity=similarity,
            seed_ranks=seed_ranks,
            k_cut=k_cut,
            n=n,
            sampling_fraction=sampling_fraction,
            variant=variant,
            params=params,
            n_jobs=n_jobs,
            started=started,
            checkpoint=checkpoint,
        )
    else:
        final_ranks = seed_ranks
        curve = _run_rank_list(
            curve=curve,
            dataset=dataset,
            similarity=similarity,
            ranks_to_run=_missing_ranks(curve, final_ranks, expected_per_rank),
            target_ranks=final_ranks,
            sampling_fraction=sampling_fraction,
            variant=variant,
            params=params,
            n_jobs=n_jobs,
            started=started,
            checkpoint=checkpoint,
        )

    return _record_from_curve(
        curve=curve,
        target_ranks=final_ranks,
        sampling_fraction=sampling_fraction,
        variant=variant,
        params=params,
        n_jobs=n_jobs,
        started=started,
        resumed_from_json=bool(completed),
    )


def _run_adaptive_curve(
    curve: pd.DataFrame,
    dataset: str,
    similarity: np.ndarray,
    seed_ranks: list[int],
    k_cut: int,
    n: int,
    sampling_fraction: float,
    variant: dict,
    params: DictConfig,
    n_jobs: int,
    started: float,
    checkpoint: Callable[[dict], None],
) -> tuple[pd.DataFrame, list[int]]:
    settings = _adaptive_settings(params, n)
    max_rank = min(int(settings.max_rank), n - 1)
    expected_per_rank = int(variant["n_folds"]) * int(variant["n_repeats"])
    ranks = _adaptive_initial_ranks(seed_ranks, k_cut, max_rank, settings)

    # Outer-in bracketing: probe the extremes first, then converge inward.
    # Gives early signal on curve shape and degrades gracefully if interrupted.
    coarse_order = _outer_in_order(_missing_ranks(curve, ranks, expected_per_rank))
    log.info(f"  {variant['name']}: adaptive initial (outer-in)={coarse_order}")
    curve = _run_rank_list(
        curve, dataset, similarity,
        coarse_order,
        ranks, sampling_fraction, variant, params, n_jobs, started, checkpoint,
    )

    for _ in range(int(settings.expansion_rounds)):
        stats = _rank_stats(curve)
        if len(stats) < 2:
            break
        high = int(stats.index.max())
        if high >= max_rank:
            break
        best = int(stats["mean"].idxmin())
        prev_high = int(stats.index[stats.index < high].max())
        high_mean = float(stats.loc[high, "mean"])
        prev_mean = float(stats.loc[prev_high, "mean"])
        rel_drop = (prev_mean - high_mean) / max(abs(prev_mean), 1e-12)
        if best != high and rel_drop <= float(settings.expansion_min_rel_drop):
            break

        new_high = min(max_rank, max(high + 1, int(math.ceil(high * float(settings.expansion_factor)))))
        additions = sorted({int(round((high + new_high) / 2)), int(new_high)})
        additions = [r for r in additions if 2 <= r <= max_rank and r not in stats.index]
        if not additions:
            break
        log.info(
            f"  {variant['name']}: expanding {high}->{new_high} "
            f"(best={best}, rel_drop={rel_drop:.4f})"
        )
        ranks = sorted(set(ranks) | set(additions))
        curve = _run_rank_list(
            curve, dataset, similarity, additions, ranks, sampling_fraction,
            variant, params, n_jobs, started, checkpoint,
        )

    for _ in range(int(settings.refine_rounds)):
        stats = _rank_stats(curve)
        additions = [r for r in _adaptive_refinement_ranks(stats, k_cut, max_rank, settings) if r not in stats.index]
        if not additions:
            break
        log.info(f"  {variant['name']}: refining ranks={additions}")
        ranks = sorted(set(ranks) | set(additions))
        curve = _run_rank_list(
            curve, dataset, similarity, additions, ranks, sampling_fraction,
            variant, params, n_jobs, started, checkpoint,
        )

    return curve, sorted(int(r) for r in _rank_stats(curve).index)


def _run_rank_list(
    curve: pd.DataFrame,
    dataset: str,
    similarity: np.ndarray,
    ranks_to_run: list[int],
    target_ranks: list[int],
    sampling_fraction: float,
    variant: dict,
    params: DictConfig,
    n_jobs: int,
    started: float,
    checkpoint: Callable[[dict], None],
) -> pd.DataFrame:
    for rank in ranks_to_run:
        log.info(f"  {variant['name']}: rank={rank}")
        rank_curve = cross_val_score(
            similarity,
            ranks=[rank],
            sampling_fraction=sampling_fraction,
            n_folds=int(variant["n_folds"]),
            n_repeats=int(variant["n_repeats"]),
            random_state=int(variant["random_state"]),
            n_jobs=n_jobs,
            srf_kwargs=_plain(params.srf_kwargs),
        )
        rank_curve = rank_curve.assign(dataset=dataset, cv_variant=variant["name"])
        curve = _replace_rank(curve, rank_curve, rank)
        checkpoint(_record_from_curve(
            curve=curve,
            target_ranks=target_ranks,
            sampling_fraction=sampling_fraction,
            variant=variant,
            params=params,
            n_jobs=n_jobs,
            started=started,
            resumed_from_json=True,
        ))
    return curve


def _record_from_curve(
    curve: pd.DataFrame,
    target_ranks: list[int],
    sampling_fraction: float,
    variant: dict,
    params: DictConfig,
    n_jobs: int,
    started: float,
    resumed_from_json: bool,
) -> dict:
    n_folds = int(variant["n_folds"])
    n_repeats = int(variant["n_repeats"])
    expected_per_rank = n_folds * n_repeats
    completed = sorted(_completed_ranks(curve, target_ranks, expected_per_rank))
    stats = curve[curve["rank"].astype(int).isin(completed)].groupby("rank")["val_mse"].agg(["mean", "std", "count"]).reindex(completed)

    mean = stats["mean"].to_numpy(dtype=float) if completed else np.array([])
    count = stats["count"].fillna(0).to_numpy(dtype=int) if completed else np.array([], dtype=int)
    sem = stats["std"].to_numpy(dtype=float) / np.sqrt(np.maximum(count, 1)) if completed else np.array([])
    if completed:
        argmin_idx = int(np.nanargmin(mean))
        argmin_rank = int(completed[argmin_idx])
        one_se_rank = _one_se_rank(completed, mean, sem, argmin_idx)
        edge_status = _edge_status(completed, mean, argmin_idx)
    else:
        argmin_rank = None
        one_se_rank = None
        edge_status = "none"

    return {
        "variant": variant["name"],
        "status": "complete" if set(completed) == set(target_ranks) else "partial",
        "target_ranks": [int(r) for r in target_ranks],
        "completed_ranks": [int(r) for r in completed],
        "ranks": [int(r) for r in completed],
        "argmin_rank": argmin_rank,
        "one_se_rank": one_se_rank,
        "edge_status": edge_status,
        "val_mse_mean": mean.tolist(),
        "val_mse_sem": sem.tolist(),
        "val_mse_count": [int(c) for c in count],
        "scores": _scores_to_json(curve, completed, n_repeats, n_folds),
        "params": {
            "cv_protocol": CV_PROTOCOL_VERSION,
            "n_folds": n_folds,
            "n_repeats": n_repeats,
            "sampling_fraction": float(sampling_fraction),
            "srf_kwargs": _plain(params.srf_kwargs),
            "random_state": int(variant["random_state"]),
            "rank_strategy": str(params.get("strategy", "fixed")).lower(),
            "rank_spec": _rank_spec_for_json(params),
            "adaptive": _plain(params.get("adaptive", {})),
            "n_jobs": int(n_jobs),
        },
        "runtime_sec": round(time.time() - started, 2),
        "resumed_from_json": bool(resumed_from_json),
    }


def _scores_to_json(curve: pd.DataFrame, ranks: list[int], n_repeats: int, n_folds: int) -> dict[str, list[list[float]]]:
    scores: dict[str, list[list[float]]] = {}
    for rank in ranks:
        matrix = np.full((n_repeats, n_folds), np.nan, dtype=float)
        subset = curve[curve["rank"].astype(int) == int(rank)]
        for row in subset.itertuples(index=False):
            matrix[int(row.rep), int(row.fold)] = float(row.val_mse)
        scores[str(int(rank))] = matrix.tolist()
    return scores


def _curve_from_block(block: dict | None, expected_per_rank: int) -> pd.DataFrame:
    columns = ["rep", "fold", "rank", "val_mse", "dataset", "cv_variant"]
    if not block or "scores" not in block:
        return pd.DataFrame(columns=columns)
    rows = []
    dataset = block.get("dataset", "")
    variant = block.get("variant", "")
    for rank_text, matrix in block.get("scores", {}).items():
        rank = int(rank_text)
        for rep, fold_values in enumerate(matrix):
            for fold, value in enumerate(fold_values):
                if value is not None and np.isfinite(value):
                    rows.append((rep, fold, rank, float(value), dataset, variant))
    curve = pd.DataFrame(rows, columns=columns)
    completed = _completed_ranks(curve, [int(r) for r in block.get("completed_ranks", [])], expected_per_rank)
    return curve[curve["rank"].astype(int).isin(completed)]


def _adaptive_settings(params: DictConfig, n: int) -> DictConfig:
    defaults = OmegaConf.create({
        "max_rank": min(300, n - 1),
        "initial_multipliers": [0.5, 1.0, 1.5, 2.0, 3.0],
        "include_upper_probe": True,
        "expansion_factor": 1.6,
        "expansion_rounds": 4,
        "expansion_min_rel_drop": 0.005,
        "refine_rounds": 2,
        "refine_step": 5,
        "refine_radius": 25,
        "max_refine_points": 25,
        "plateau_rel_tol": 0.01,
    })
    return OmegaConf.merge(defaults, params.get("adaptive", {}))


def _adaptive_initial_ranks(seed_ranks: list[int], k_cut: int, max_rank: int, settings: DictConfig) -> list[int]:
    ranks = {2, *seed_ranks}
    for multiplier in settings.initial_multipliers:
        ranks.add(int(round(k_cut * float(multiplier))))
    if bool(settings.include_upper_probe):
        ranks.add(max_rank)
    return sorted({r for r in ranks if 2 <= r <= max_rank})


def _outer_in_order(ranks: list[int]) -> list[int]:
    """Sort ranks low/high/low/high so the extremes are evaluated first.

    Example: [10, 20, 30, 100, 180, 200] -> [10, 200, 20, 180, 30, 100].
    Used for the adaptive coarse sweep to bracket the minimum and produce
    informative coverage even if the run is interrupted before completion.
    """
    if not ranks:
        return []
    sorted_ranks = sorted({int(r) for r in ranks})
    out: list[int] = []
    lo, hi = 0, len(sorted_ranks) - 1
    while lo <= hi:
        if lo == hi:
            out.append(sorted_ranks[lo])
        else:
            out.append(sorted_ranks[lo])
            out.append(sorted_ranks[hi])
        lo += 1
        hi -= 1
    return out


def _adaptive_refinement_ranks(stats: pd.DataFrame, k_cut: int, max_rank: int, settings: DictConfig) -> list[int]:
    if stats.empty:
        return []
    means = stats["mean"]
    threshold = float(means.min()) * (1.0 + float(settings.plateau_rel_tol))
    near_best = [int(r) for r in stats.index[means <= threshold]]
    center = min(near_best) if near_best else int(means.idxmin())
    ranks = [int(r) for r in stats.index]
    lower = max([r for r in ranks if r < center], default=max(2, center - int(settings.refine_radius)))
    upper = min([r for r in ranks if r > center], default=min(max_rank, center + int(settings.refine_radius)))
    step = max(1, int(settings.refine_step))
    refined = set(range(max(2, lower), min(max_rank, upper) + 1, step))
    max_points = max(3, int(settings.max_refine_points))
    if len(refined) > max_points:
        step = max(step, int(math.ceil((upper - lower + 1) / max_points)))
        refined = set(range(max(2, lower), min(max_rank, upper) + 1, step))
    refined.update({center, int(means.idxmin()), int(k_cut)})
    return sorted({r for r in refined if 2 <= r <= max_rank})


def _compose_seed_ranks(ds: DictConfig, params: DictConfig, k_cut: int, n: int) -> list[int]:
    rank_set: set[int] = set()
    if str(params.get("strategy", "fixed")).lower() == "adaptive":
        rank_set.update(int(r) for r in ds.get("cv_seed_ranks", params.get("seed_ranks", [])))
    elif "cv_ranks" in ds:
        rank_set.update(int(r) for r in ds.cv_ranks)
    else:
        rank_range = ds.get("cv_rank_range", params.get("rank_range", None))
        if rank_range is not None:
            start, stop, step = [int(v) for v in rank_range]
            rank_set.update(range(start, stop + 1, step))
        else:
            window = int(params.get("rank_window", 5))
            high = max(k_cut + window, int(math.ceil(k_cut * float(params.get("rank_high_multiplier", 2.0)))))
            rank_set.update(range(max(2, k_cut - window), high + 1, int(params.get("rank_step", 1))))
    rank_set.update(int(r) for r in ds.get("cv_extra_ranks", []))
    rank_set.update(int(r) for r in params.get("extra_ranks", []))
    if bool(params.get("include_k_cut", True)):
        rank_set.add(int(k_cut))
    return sorted({r for r in rank_set if 2 <= r <= n - 1})


def _completed_ranks(curve: pd.DataFrame, ranks: list[int], expected_per_rank: int) -> set[int]:
    if curve.empty:
        return set()
    counts = curve.groupby("rank").size()
    return {int(rank) for rank in ranks if int(counts.get(rank, 0)) >= expected_per_rank}


def _missing_ranks(curve: pd.DataFrame, ranks: list[int], expected_per_rank: int) -> list[int]:
    completed = _completed_ranks(curve, ranks, expected_per_rank)
    return [int(rank) for rank in ranks if int(rank) not in completed]


def _rank_stats(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame(columns=["mean", "std", "count"])
    return curve.groupby("rank")["val_mse"].agg(["mean", "std", "count"]).sort_index()


def _replace_rank(curve: pd.DataFrame, rank_curve: pd.DataFrame, rank: int) -> pd.DataFrame:
    kept = curve[curve["rank"].astype(int) != int(rank)] if not curve.empty else curve
    return pd.concat([kept, rank_curve], ignore_index=True)


def _one_se_rank(ranks: list[int], mean: np.ndarray, sem: np.ndarray, argmin_idx: int) -> int | None:
    sem_min = sem[argmin_idx]
    if not np.isfinite(sem_min):
        return None
    threshold = mean[argmin_idx] + sem_min
    eligible = [int(r) for r, m in zip(ranks, mean) if np.isfinite(m) and m <= threshold]
    return min(eligible) if eligible else None


def _edge_status(ranks: list[int], mean: np.ndarray, argmin_idx: int) -> str:
    if argmin_idx == 0:
        return "lower_edge"
    if argmin_idx == len(ranks) - 1:
        return "upper_edge"
    return "interior"


def _dataset_validate_params(ds: DictConfig, params: DictConfig) -> DictConfig:
    return OmegaConf.merge(params, ds.get("validate", {}))


def _validation_variants(params: DictConfig) -> list[dict]:
    raw_variants = params.get("variants", None)
    if raw_variants is None:
        raw_variants = [{"name": f"{int(params.n_folds)}fold", "n_folds": int(params.n_folds)}]
    variants = []
    for raw in _plain(raw_variants):
        n_folds = int(raw.get("n_folds", params.get("n_folds", 5)))
        variants.append({
            "name": str(raw.get("name", f"{n_folds}fold")),
            "n_folds": n_folds,
            "n_repeats": int(raw.get("n_repeats", params.get("n_repeats", 1))),
            "random_state": int(raw.get("random_state", params.get("random_state", 0))),
        })
    return variants


def _matches_existing(block: dict, seed_ranks: list[int], sampling_fraction: float, variant: dict, params: DictConfig) -> bool:
    block_params = block.get("params", {})
    strategy = str(params.get("strategy", "fixed")).lower()
    ranks_match = True if strategy == "adaptive" else block.get("target_ranks") == [int(r) for r in seed_ranks]
    adaptive_match = (
        strategy != "adaptive"
        or block_params.get("adaptive", {}) == _plain(params.get("adaptive", {}))
    )
    return (
        ranks_match
        and block.get("variant") == variant["name"]
        and block_params.get("cv_protocol") == CV_PROTOCOL_VERSION
        and int(block_params.get("n_folds", -1)) == int(variant["n_folds"])
        and int(block_params.get("n_repeats", -1)) == int(variant["n_repeats"])
        and int(block_params.get("random_state", -1)) == int(variant["random_state"])
        and abs(float(block_params.get("sampling_fraction", -1.0)) - sampling_fraction) < 1e-12
        and block_params.get("srf_kwargs") == _plain(params.srf_kwargs)
        and block_params.get("rank_strategy", "fixed") == strategy
        and adaptive_match
    )


def _block_complete(block: dict) -> bool:
    return block.get("status") == "complete" and set(block.get("target_ranks", [])) == set(block.get("completed_ranks", []))


def _rank_spec_for_json(params: DictConfig) -> dict:
    keys = ["ranks", "rank_range", "rank_window", "rank_high_multiplier", "rank_step", "seed_ranks"]
    return {key: _plain(params.get(key)) for key in keys if key in params}


def _summary_row(payload: dict, variant: str, block: dict) -> dict:
    return {
        "dataset": payload["dataset"],
        "variant": variant,
        "n": payload["n"],
        "rank_estimate": payload["estimate"]["rank"],
        "rank_argmin": block.get("argmin_rank"),
        "rank_one_se": block.get("one_se_rank"),
        "edge_status": block.get("edge_status"),
        "sampling_fraction": block["params"]["sampling_fraction"],
        "n_folds": block["params"]["n_folds"],
        "n_repeats": block["params"]["n_repeats"],
        "n_ranks": len(block.get("completed_ranks", [])),
        "runtime_sec": block.get("runtime_sec"),
    }


def _read_cv_existing(path: Path, coherence: dict) -> dict:
    payload = json.loads(path.read_text()) if path.exists() else {}
    return {
        "dataset": coherence.get("dataset"),
        "n": coherence.get("n"),
        "rank_estimate": coherence.get("estimate", {}).get("rank"),
        "sampling_fraction": coherence.get("estimate", {}).get("sampling_fraction"),
        "validations": payload.get("validations", {}),
        "primary_variant": payload.get("primary_variant"),
    }


def _select_datasets(cfg: DictConfig) -> list[dict]:
    datasets = OmegaConf.to_container(cfg.datasets, resolve=True)
    only = cfg.get("only", None)
    if only is None:
        return datasets
    only_set = {s.strip() for s in only.split(",") if s.strip()} if isinstance(only, str) else set(only)
    kept = [d for d in datasets if d["name"] in only_set]
    missing = only_set - {d["name"] for d in kept}
    if missing:
        raise ValueError(f"only references unknown datasets: {sorted(missing)}")
    log.info(f"Filtering to datasets: {sorted(only_set)}")
    return kept


def _resolve_dataset_n_jobs(ds: DictConfig, params: DictConfig, fallback: int) -> int:
    value = int(ds.get("validate_n_jobs", ds.get("n_jobs", params.get("n_jobs", fallback))))
    return fallback if value <= 0 else value


def _write_summary(path: Path, rows: list[dict]) -> None:
    if rows:
        path.write_text(json.dumps(rows, indent=2))


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
