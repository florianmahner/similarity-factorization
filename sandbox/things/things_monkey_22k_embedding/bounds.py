"""Estimate sampling bounds for THINGS macaque IT neural data."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from datasets.loaders import load_things_monkey_2k
from pysrf.bounds import estimate_sampling_bounds_ultra
from tools.metrics import gaussian_kernel_similarity

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main(
    monkey_type: str = "F",
    roi: str = "it",
    random_state: int = 42,
    output_dir: Path | None = None,
):
    if output_dir is None:
        output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(f"Sampling Bounds: THINGS Macaque ({monkey_type}, {roi})")
    log.info("=" * 60)

    log.info("\n[1] Loading neural data...")
    ds = load_things_monkey_2k(monkey_type=monkey_type, roi=roi)
    log.info(f"  Shape: {ds.data.shape}")
    log.info(f"  N concepts: {len(ds.metadata['filenames'])}")

    log.info("\n[2] Computing RSM (Gaussian kernel, median heuristic)...")
    rsm = gaussian_kernel_similarity(ds.data, ds.data, sigma=None)
    np.fill_diagonal(rsm, 1.0)
    log.info(f"  RSM shape: {rsm.shape}")
    log.info(f"  RSM range: [{rsm.min():.4f}, {rsm.max():.4f}]")
    log.info(f"  RSM symmetric: {np.allclose(rsm, rsm.T)}")

    log.info("\n[3] Estimating sampling bounds (ultra)...")
    start_time = time.time()
    pmin, pmax, _ = estimate_sampling_bounds_ultra(
        rsm,
        random_state=random_state,
        verbose=True,
    )
    elapsed = time.time() - start_time

    log.info(f"\n[4] Results:")
    log.info(f"  pmin = {pmin:.4f}")
    log.info(f"  pmax = {pmax:.4f}")
    log.info(f"  mean sampling fraction = {0.5 * (pmin + pmax):.4f}")
    log.info(f"  computation time = {elapsed:.1f}s")

    bounds = {
        "dataset": "things-macaque-2k",
        "monkey_type": monkey_type,
        "roi": roi,
        "pmin": float(pmin),
        "pmax": float(pmax),
        "mean_sampling_fraction": float(0.5 * (pmin + pmax)),
        "matrix_shape": list(rsm.shape),
        "n_concepts": int(ds.data.shape[0]),
        "n_channels": int(ds.data.shape[1]),
        "random_state": random_state,
        "computation_time_seconds": float(elapsed),
    }

    out_file = output_dir / "bounds.json"
    with open(out_file, "w") as f:
        json.dump(bounds, f, indent=2)

    log.info(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monkey", type=str, default="F", choices=["F", "N"])
    parser.add_argument("--roi", type=str, default="it", choices=["v1", "v4", "it"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(monkey_type=args.monkey, roi=args.roi, random_state=args.seed)
