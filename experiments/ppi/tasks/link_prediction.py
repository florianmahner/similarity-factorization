from __future__ import annotations
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
    data = load_fold_data(split_dir, dataset, fold_idx)
    train_edges = data["train_edges"]
    test_pos = data["test_pos_edges"]
    n_nodes = data["n_nodes"]

    model = get_predictor(method, seed + fold_idx, cfg)
    model.fit(train_edges, n_nodes)
    score_matrix = model.predict_all()
    metrics = evaluate_open_world(score_matrix, train_edges, test_pos)

    return {
        "fold": fold_idx,
        "method": method,
        **metrics,
    }


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
    results = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(_run_single_task)(*t) for t in tasks
    )

    df = pd.DataFrame(results)
    summary = df.groupby("method")[["auroc", "auprc", "p500", "ndcg"]].agg(
        ["mean", "std"]
    )
    print(summary)

    # Save to CSV
    out_path = Path.cwd() / "outputs" / dataset / "results.csv"
    df.to_csv(out_path, index=False)
    # make a quick plot of the results
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 5))
    sns.barplot(x="method", y="auroc", data=df)
    plt.savefig(out_path / "results_plot.pdf")
