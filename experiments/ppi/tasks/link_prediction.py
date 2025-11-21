from __future__ import annotations

from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig

from ..lib.utils import prepare_splits, load_fold_npz, compute_all_vs_all_metrics
from ..lib.models import get_predictor


def _evaluate_fold_method(fold_idx, method, dataset, splits_dir, seed, model_params):
    data = load_fold_npz(Path(splits_dir) / dataset, fold_idx)
    nodes = data["nodes"]
    train_edges = data["train_edges"]
    test_pos_edges = data["test_pos_edges"]

    model = get_predictor(method, seed + fold_idx, model_params)
    model.fit(train_edges, len(nodes), mask_edges=test_pos_edges)
    score_matrix = model.predict_all()
    metrics = compute_all_vs_all_metrics(score_matrix, test_pos_edges, train_edges)

    return [
        {"fold": fold_idx, "method": method, "metric": k, "value": v}
        for k, v in metrics.items()
    ]


def run(cfg: DictConfig) -> None:
    if not cfg.dataset:
        raise ValueError("dataset required")

    splits_dir = Path(cfg.splits_dir)
    meta_path = splits_dir / cfg.dataset / "metadata.json"
    if not meta_path.exists():
        prepare_splits(
            cfg.dataset,
            Path(cfg.base_dir) / "data/ppi",
            splits_dir,
            cfg.n_folds,
            cfg.seed,
            cfg.n_jobs,
        )

    model_params = {
        "rank": cfg.rank,
        "embedding_params": {
            "dim": cfg.embedding_dim,
            "epochs": cfg.embedding_epochs,
        },
    }

    methods = list(cfg.methods) if cfg.methods else ["srf", "cn", "aa", "ra", "jc"]
    tasks = [
        (f, m, cfg.dataset, splits_dir, cfg.seed, model_params)
        for f in range(cfg.n_folds)
        for m in methods
    ]

    outputs = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(_evaluate_fold_method)(*t) for t in tasks
    )

    results = pd.DataFrame([row for batch in outputs for row in batch])
    if not results.empty:
        pivot = results.pivot_table(
            index=["fold", "method"], columns="metric", values="value"
        )
        pivot.reset_index().to_csv(Path.cwd() / "results.csv", index=False)
