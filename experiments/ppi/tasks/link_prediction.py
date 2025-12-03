from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig

# Ensure these imports point to your actual file structure
from ..lib.utils import (
    evaluate_open_world,
    prepare_splits,
    load_fold_data,
    FILENAME_MAP,
)
from ..lib.models import get_predictor


def _run_single_task(fold_idx, method, dataset, split_dir, seed, cfg):
    # Construct output path first to skip if needed (optional, but good for resume)
    # But we overwrite by default as per standard behavior unless checked
    rank = cfg.get("rank", 0)

    output_dir = Path.cwd() / "link_prediction" / dataset / method
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"rank{rank}_fold{fold_idx}.json"

    print(f"[{method}] Fold {fold_idx} (Rank {rank}) -> {output_file}")

    data = load_fold_data(split_dir, dataset, fold_idx)
    train_edges = data["train_edges"]
    test_pos = data["test_pos_edges"]
    n_nodes = data["n_nodes"]

    model = get_predictor(method, seed + fold_idx, cfg)
    model.fit(train_edges, n_nodes)
    score_matrix = model.predict_all()
    metrics = evaluate_open_world(score_matrix, train_edges, test_pos)

    result = {
        "fold": fold_idx,
        "method": method,
        "dataset": dataset,
        "rank": rank,
        "seed": seed,
        **metrics,
    }

    # Atomic write
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    return str(output_file)


def run(cfg: DictConfig) -> None:

    dataset = FILENAME_MAP[cfg.dataset]
    split_dir = Path(cfg.data_dir) / "splits" / dataset
    if not (split_dir / "metadata.json").exists():
        print(f"Preparing splits for {cfg.dataset}...")
        prepare_splits(
            cfg.dataset,
            Path(cfg.data_dir),
            split_dir,
            cfg.n_folds,
            cfg.seed,
            n_jobs=cfg.n_jobs,
        )

    methods = list(cfg.methods) if cfg.methods else ["srf", "cn", "aa"]

    tasks = [
        (f, m, cfg.dataset, split_dir, cfg.seed, cfg)
        for f in range(cfg.n_folds)
        for m in methods
    ]

    print(f"Running {len(tasks)} tasks on {cfg.dataset} with {cfg.n_jobs} jobs...")

    # Run in parallel
    saved_files = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(_run_single_task)(*t) for t in tasks
    )

    print(
        f"Completed. Saved {len(saved_files)} result files to results/link_prediction/{dataset}/"
    )
