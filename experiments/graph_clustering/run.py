import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from experiments.clustering.graph import (
    load_all_datasets,
    create_models,
    evaluate_model_on_dataset,
    DATASET_REGISTRY,
    MODEL_REGISTRY,
)


# Constants
MAX_OUTER = 10
MAX_INNER = 100
SEEDS = range(100)
VERBOSE = False


def run_clustering_benchmark(
    seeds=None,
    datasets=None,
    max_outer=100,
    max_inner=100,
    verbose=False,
    n_jobs=-1,
    output_dir="./results/benchmarks",
):
    if seeds is None:
        seeds = range(100)

    # Set module-level constants
    global MAX_OUTER, MAX_INNER, VERBOSE
    MAX_OUTER = max_outer
    MAX_INNER = max_inner
    VERBOSE = verbose

    load_all_datasets()
    tasks = []

    for seed in seeds:
        for dname, ds_info in DATASET_REGISTRY.items():
            if datasets is None or dname in datasets:
                rank = len(np.unique(ds_info["y"]))
                create_models(rank, seed, max_outer, max_inner, verbose)

                tasks.extend(
                    (
                        model_info["model"],
                        dname,
                        model_name,
                        seed,
                    )
                    for model_name, model_info in MODEL_REGISTRY.items()
                )

    results = joblib.Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        joblib.delayed(evaluate_model_on_dataset)(*task) for task in tasks
    )

    results_df = pd.DataFrame(results)

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / "clustering_benchmarks.csv"
    results_df.to_csv(fname, index=False)

    if verbose:
        print(f"Clustering benchmark complete. Results saved to '{fname}'.")

    return results_df


if __name__ == "__main__":
    run_clustering_benchmark()
