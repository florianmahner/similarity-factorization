from __future__ import annotations
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig

# Ensure these imports point to your actual file structure
from ..lib.utils import evaluate_open_world, prepare_splits, load_fold_data
from ..lib.models import get_predictor


def _run_single_task(fold_idx, method, dataset, splits_dir, seed, cfg):
    """
    Worker function for a single fold/method combination.
    """
    # 1. Load Data
    # (Using the lightweight numpy loader from previous answers)
    data = load_fold_data(splits_dir, dataset, fold_idx)
    train_edges = data["train_edges"]
    test_pos = data["test_pos_edges"]
    n_nodes = data["n_nodes"]

    # 2. Init Model
    # We pass the whole 'cfg' object. The factory extracts what it needs.
    model = get_predictor(method, seed + fold_idx, cfg)

    # 3. Fit
    # We only pass Positive Train Edges.
    # The Model handles strict/balanced sampling internally.
    model.fit(train_edges, n_nodes)

    # 4. Predict Global Matrix
    score_matrix = model.predict_all()

    # 5. Evaluate
    # Uses Upper Triangle logic to strictly separate Train/Test/Background
    metrics = evaluate_open_world(score_matrix, train_edges, test_pos)

    return {
        "fold": fold_idx,
        "method": method,
        **metrics,  # Unpacks 'auroc', 'auprc', 'p500', 'ndcg500'
    }


def run(cfg: DictConfig) -> None:
    splits_dir = Path(cfg.splits_dir)

    # 1. Prepare Data (Only if metadata missing)
    if not (splits_dir / cfg.dataset / "metadata.json").exists():
        print(f"Preparing splits for {cfg.dataset}...")
        prepare_splits(
            cfg.dataset,
            Path(cfg.data_dir),
            splits_dir,
            cfg.n_folds,
            cfg.seed,
            n_jobs=cfg.n_jobs,
        )

    # 2. Define Tasks
    methods = list(cfg.methods) if cfg.methods else ["srf", "cn", "aa"]

    # Create a list of tasks. Note we pass 'cfg' as the last argument
    tasks = [
        (f, m, cfg.dataset, splits_dir, cfg.seed, cfg)
        for f in range(cfg.n_folds)
        for m in methods
    ]

    # 3. Execute Parallel
    print(f"Running {len(tasks)} tasks on {cfg.dataset} with {cfg.n_jobs} jobs...")
    results_list = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(_run_single_task)(*t) for t in tasks
    )

    # 4. Save Results
    df = pd.DataFrame(results_list)

    # Print Summary
    summary = df.groupby("method")[["auroc", "auprc", "p500", "ndcg"]].agg(
        ["mean", "std"]
    )
    print(summary)

    # Save to CSV
    out_path = Path.cwd()
    df.to_csv(out_path / "results_raw.csv", index=False)
    summary.to_csv(out_path / "results_summary.csv")

    # make a quick plot of the results
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 5))
    sns.barplot(x="method", y="auroc", data=df)
    plt.savefig(out_path / "results_plot.pdf")
