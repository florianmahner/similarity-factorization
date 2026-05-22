"""Bandwidth (sigma_scale alpha) selection for things_macaque22k (monkey IT)."""
from __future__ import annotations

import time

from datasets.monkey import load_macaque
from experiments.datasets.bandwidth_selection._lib import run_bandwidth_selection
from src.utils import get_output_dir

RANK = 89  # from coherence estimate at alpha=0.4, n_p=20
ALPHAS = [0.2, 0.4, 0.6, 0.8, 1.0]
N_RUNS = 5
N_JOBS = 32
SRF_MAX_OUTER = 20  # n=22k is heavy, use shorter training for selection only

if __name__ == "__main__":
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading macaque features...", flush=True)
    features, _, _ = load_macaque(
        "22k", "F",
        root="/data/labshare/_stachelschwein/SSD/datasets/things/macaque",
        roi="it", min_reliab=0.3, average_exemplars=False,
    )
    print(f"[{time.strftime('%H:%M:%S')}]   features shape={features.shape}  load={time.time()-t0:.1f}s", flush=True)
    run_bandwidth_selection(
        dataset_name="things_macaque22k", features=features,
        rank=RANK, alphas=ALPHAS, n_runs=N_RUNS, n_jobs=N_JOBS,
        srf_max_outer=SRF_MAX_OUTER,
        output_dir=get_output_dir(),
    )
