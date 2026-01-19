"""Multi-GPU job runner for VICE low-data experiment.

Runs all (percentage, partition, seed) combinations across 4 GPUs.

Usage:
    poetry run python experiments/things_behavior/vice_lowdata_runner.py
"""
from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PARTITIONS = {
    5: 20,
    10: 10,
    20: 5,
    50: 2,
}
N_GPUS = 4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "things_behavior" / "vice_lowdata.py"


def run_single(pct: int, part: int, gpu: int) -> tuple[str, bool, str]:
    """Run a single training job. Returns (identifier, success, message)."""
    identifier = f"vice_{pct}pct_part{part}"
    cmd = [
        sys.executable, str(SCRIPT_PATH),
        "--train",
        "--percentage", str(pct),
        "--partition", str(part),
        "--gpu", str(gpu),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return identifier, True, "completed"
        return identifier, False, result.stderr[:500]
    except Exception as e:
        return identifier, False, str(e)


def run_aggregation():
    """Run aggregation after training."""
    cmd = [sys.executable, str(SCRIPT_PATH), "--aggregate"]
    print("\n" + "="*50)
    print("Running aggregation...")
    print("="*50)
    subprocess.run(cmd)


def run_plotting():
    """Run plotting after aggregation."""
    plot_script = PROJECT_ROOT / "experiments" / "things_behavior" / "vice_lowdata_plot.py"
    cmd = [sys.executable, str(plot_script)]
    print("\n" + "="*50)
    print("Creating plots...")
    print("="*50)
    subprocess.run(cmd)


def main():
    jobs = []
    for pct, n_parts in PARTITIONS.items():
        for part in range(n_parts):
            jobs.append((pct, part))

    print(f"Total jobs: {len(jobs)}")
    print(f"Running on {N_GPUS} GPUs")

    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=N_GPUS) as executor:
        futures = {}
        for i, (pct, part) in enumerate(jobs):
            gpu = i % N_GPUS
            future = executor.submit(run_single, pct, part, gpu)
            futures[future] = (pct, part)

        for future in as_completed(futures):
            pct, part = futures[future]
            identifier, success, message = future.result()
            if success:
                completed += 1
                print(f"[{completed}/{len(jobs)}] {identifier}: {message}")
            else:
                failed += 1
                print(f"[FAILED] {identifier}: {message}")

    print(f"\nTraining complete: {completed} succeeded, {failed} failed")

    # Auto-run aggregation and plotting
    run_aggregation()
    run_plotting()

    print("\n" + "="*50)
    print("ALL DONE!")
    print("="*50)


if __name__ == "__main__":
    main()
