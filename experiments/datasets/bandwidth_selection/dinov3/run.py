"""Bandwidth (sigma_scale alpha) selection for DINOv3 ViT-L."""
from __future__ import annotations

import time

from datasets import load_dataset
from experiments.datasets.bandwidth_selection._lib import run_bandwidth_selection
from src.utils import get_output_dir

RANK = 15
ALPHAS = [0.2, 0.4, 0.6, 0.8, 1.0]
N_RUNS = 5
N_JOBS = 16

if __name__ == "__main__":
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading DINOv3 features...", flush=True)
    ds = load_dataset("dinov3")
    features = ds.data
    print(f"[{time.strftime('%H:%M:%S')}]   features shape={features.shape}  load={time.time()-t0:.1f}s", flush=True)
    run_bandwidth_selection(
        dataset_name="dinov3", features=features,
        rank=RANK, alphas=ALPHAS, n_runs=N_RUNS, n_jobs=N_JOBS,
        output_dir=get_output_dir(),
    )
