"""Cross-validation rank estimation centered on kappa k*.

Reads k* from ../kappa/outputs/<kappa_name>.json, builds a rank grid
centered on k*, estimates bounds inline, runs pysrf.cross_val_score.

Usage:
    ./scripts/submit experiments/datasets/ranks/cv/run.py dataset=mur92
    ./scripts/submit experiments/datasets/ranks/cv/run.py dataset=nsd subject_id=1
    ./scripts/submit experiments/datasets/ranks/cv/run.py dataset=things_behavior kappa_name=things_100pct
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from pysrf import cross_val_score
from pysrf.bounds import estimate_sampling_bounds_fast

from similarity import build_similarity
from src.utils.figure_theme import CMAP, GRAY, create_figure, despine, save_figure

log = logging.getLogger(__name__)

KAPPA_DIR = Path(__file__).resolve().parent.parent / "kappa" / "outputs"


def _resolve_kappa_name(cfg: DictConfig) -> str:
    """Map Hydra dataset config to kappa output filename.

    Kappa outputs use flat names: mur92, peterson_animals, nsd_subj01,
    things_100pct, etc. The mapping from (dataset.name, subject_id) is:
      - kappa_name override (if set): use directly
      - multi-subject datasets: {name}_subj{id:02d}
      - otherwise: dataset.name
    """
    if cfg.get("kappa_name"):
        return cfg.kappa_name
    ds_name = cfg.dataset.name
    subject_id = cfg.get("subject_id")
    if subject_id is not None:
        return f"{ds_name}_subj{subject_id:02d}"
    return ds_name


def _load_kappa_kstar(kappa_name: str) -> int | None:
    json_path = KAPPA_DIR / f"{kappa_name}.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)["k_star_kappa"]


def _build_rank_grid(
    ds_cfg: DictConfig, k_star: int | None, n_around: int,
) -> list[int]:
    """Build rank grid centered on kappa k*.

    Uses step from dataset rank_range config. Falls back to full rank_range
    if k_star is None.
    """
    start, stop, step = ds_cfg.rank_range
    if k_star is None:
        return list(range(start, stop + 1, step))

    lo = max(step, k_star - n_around * step)
    hi = k_star + n_around * step
    lo = max(lo, start)
    hi = min(hi, stop)
    grid = list(range(lo, hi + 1, step))
    if k_star not in grid and start <= k_star <= stop:
        grid.append(k_star)
        grid.sort()
    return grid


def run(cfg: DictConfig) -> None:
    subject_id = cfg.get("subject_id")
    ds_name = cfg.dataset.name

    output_dir = Path.cwd() / ds_name
    if subject_id is not None:
        output_dir = output_dir / f"subj{subject_id:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    kappa_name = _resolve_kappa_name(cfg)
    k_star = _load_kappa_kstar(kappa_name)
    if k_star is not None:
        log.info(f"Kappa k* = {k_star} (from {kappa_name}.json)")
    else:
        log.warning(f"No kappa output for {kappa_name}, using full rank_range")

    n_around = cfg.cv.n_ranks_around_kstar
    rank_grid = _build_rank_grid(cfg.dataset, k_star, n_around)
    log.info(f"Rank grid: {rank_grid} ({len(rank_grid)} values)")

    log.info(f"Building similarity for {ds_name}...")
    similarity = build_similarity(cfg.dataset, subject_id=subject_id)
    log.info(f"Shape: {similarity.shape}")

    log.info("Estimating sampling bounds...")
    pmin, pmax, _ = estimate_sampling_bounds_fast(
        similarity, random_state=cfg.common.random_state, n_jobs=cfg.common.n_jobs,
    )
    sel = cfg.cv.sampling_selection
    p_star = {"min": pmin, "mean": 0.5 * (pmin + pmax), "max": pmax}[sel]
    log.info(f"Bounds: pmin={pmin:.4f}, pmax={pmax:.4f}, p*={p_star:.4f} ({sel})")

    n_repeats = cfg.cv.n_repeats
    log.info(f"Running CV: {len(rank_grid)} ranks x {n_repeats} repeats...")
    cv_result = cross_val_score(
        similarity,
        param_grid={"rank": rank_grid},
        n_repeats=n_repeats,
        sampling_fraction=p_star,
        estimate_sampling_fraction=False,
        fit_final_estimator=False,
        random_state=cfg.common.random_state,
        n_jobs=cfg.common.n_jobs,
        verbose=1,
    )

    optimal_rank = cv_result.best_params_["rank"]
    log.info(f"Optimal rank: {optimal_rank} (CV score: {cv_result.best_score_:.6f})")

    df = cv_result.cv_results_
    cv_scores = {
        int(r): {"mean": float(g["score"].mean()), "std": float(g["score"].std())}
        for r, g in df.groupby("rank")
    }

    summary = {
        "dataset": ds_name,
        "subject_id": int(subject_id) if subject_id else None,
        "n_samples": int(similarity.shape[0]),
        "optimal_rank": int(optimal_rank),
        "best_cv_score": float(cv_result.best_score_),
        "kappa_k_star": k_star,
        "kappa_name": kappa_name,
        "rank_grid": [int(r) for r in rank_grid],
        "n_repeats": n_repeats,
        "pmin": float(pmin),
        "pmax": float(pmax),
        "p_star": float(p_star),
        "cv_scores": cv_scores,
    }
    (output_dir / "rank_estimation.json").write_text(json.dumps(summary, indent=2))
    df.to_csv(output_dir / "cv_results.csv", index=False)

    fig, ax = create_figure("single")
    ranks = sorted(cv_scores.keys())
    means = [cv_scores[r]["mean"] for r in ranks]
    stds = [cv_scores[r]["std"] for r in ranks]
    ax.errorbar(ranks, means, yerr=stds, fmt="o-", color=CMAP[1], capsize=3, markersize=4)
    ax.axvline(optimal_rank, color=CMAP[0], linestyle="--", linewidth=1.5,
               label=f"CV optimal: {optimal_rank}")
    if k_star is not None:
        ax.axvline(k_star, color=GRAY["medium"], linestyle=":", linewidth=1,
                   label=f"Kappa k*: {k_star}")
    ax.set_xlabel("Rank")
    ax.set_ylabel("CV Score (MSE)")
    ax.legend(fontsize=8)
    despine(ax)
    save_figure(fig, output_dir / "cv_curve.pdf")
    plt.close(fig)

    log.info(f"Saved to {output_dir}")
