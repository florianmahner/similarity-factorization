"""Run VICE low-data experiment with multiple seeds per partition.

We already have seed 0 for all partitions. This runs seeds 1-9.

Usage:
    # Run all seeds on all GPUs (parallel within each GPU)
    poetry run python experiments/things_behavior/vice_lowdata_seeds.py --run-all

    # Run specific GPU's allocation
    poetry run python experiments/things_behavior/vice_lowdata_seeds.py --gpu 0

    # Show distribution
    poetry run python experiments/things_behavior/vice_lowdata_seeds.py --show-distribution
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARTITION_DIR = PROJECT_ROOT / "data" / "things" / "partitions"
MODEL_DIR = PROJECT_ROOT / "outputs" / "experiments" / "things_behavior" / "vice_lowdata" / "models"
OBJDIM_ROOT = Path("/LOCAL/fmahner/object-dimensions")
OBJDIM_PYTHON = OBJDIM_ROOT / ".venv" / "bin" / "python"
OBJDIM_SCRIPT = OBJDIM_ROOT / "run_optimization.py"

PARTITIONS = {
    5: 20,
    10: 10,
    20: 5,
    50: 2,
}

N_GPUS = 4
JOBS_PER_GPU = 20
# Original runs use seed 42, multi-seed runs use 0-9
SEEDS = [42] + list(range(10))  # Seeds 42, 0-9 (will skip completed ones)


def get_all_tasks() -> list[tuple[int, int, int]]:
    """Get all (pct, part, seed) tasks."""
    tasks = []
    for pct, n_parts in PARTITIONS.items():
        for part in range(n_parts):
            for seed in SEEDS:
                tasks.append((pct, part, seed))
    return tasks


def get_gpu_tasks(gpu_id: int) -> list[tuple[int, int, int]]:
    """Get tasks assigned to a specific GPU."""
    all_tasks = get_all_tasks()
    return [t for i, t in enumerate(all_tasks) if i % N_GPUS == gpu_id]


def is_completed(pct: int, part: int, seed: int) -> bool:
    """Check if a run is already completed (has parameters.npz)."""
    identifier = f"vice_{pct}pct_part{part}_seed{seed}"
    result_dir = MODEL_DIR / identifier
    if not result_dir.exists():
        return False
    # Search for parameters.npz anywhere in the result dir
    for path in result_dir.rglob("parameters.npz"):
        return True
    return False


def train_single(pct: int, part: int, seed: int, gpu: int) -> str:
    """Train VICE on a single partition with specific seed."""
    partition_dir = PARTITION_DIR / f"{pct}pct_part{part}"
    identifier = f"vice_{pct}pct_part{part}_seed{seed}"

    if is_completed(pct, part, seed):
        return f"Skipped {identifier} - already completed"

    if not partition_dir.exists():
        return f"Error {identifier}: partition dir not found"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(OBJDIM_PYTHON), str(OBJDIM_SCRIPT),
        "--triplet_path", str(partition_dir),
        "--method", "variational",
        "--prior", "sslab",
        "--init_dim", "90",
        "--n_epochs", "1000",
        "--batch_size", "256",
        "--beta", "1.0",
        "--lr", "0.001",
        "--stability_time", "300",
        "--mc_samples", "50",
        "--non_zero_weights", "5",
        "--params_interval", "100",
        "--checkpoint_interval", "100",
        "--device_id", "0",
        "--log_path", str(MODEL_DIR),
        "--identifier", identifier,
        "--seed", str(seed),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    try:
        subprocess.run(cmd, check=True, env=env, capture_output=True)
        return f"Completed {identifier}"
    except subprocess.CalledProcessError as e:
        return f"Error {identifier}: {e}"


def run_gpu_parallel(gpu_id: int, jobs_per_gpu: int = JOBS_PER_GPU) -> None:
    """Run all tasks assigned to a GPU with parallel execution."""
    tasks = get_gpu_tasks(gpu_id)
    pending = [(pct, part, seed) for pct, part, seed in tasks if not is_completed(pct, part, seed)]

    print(f"GPU {gpu_id}: {len(pending)} pending tasks ({len(tasks) - len(pending)} already done)")

    if not pending:
        print(f"GPU {gpu_id}: All tasks completed!")
        return

    completed = 0
    with ProcessPoolExecutor(max_workers=jobs_per_gpu) as executor:
        futures = {
            executor.submit(train_single, pct, part, seed, gpu_id): (pct, part, seed)
            for pct, part, seed in pending
        }

        for future in as_completed(futures):
            completed += 1
            result = future.result()
            print(f"GPU {gpu_id} [{completed}/{len(pending)}]: {result}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, help="Run tasks for specific GPU")
    parser.add_argument("--run-all", action="store_true", help="Launch all GPUs in background")
    parser.add_argument("--show-distribution", action="store_true", help="Show task distribution")
    parser.add_argument("--jobs-per-gpu", type=int, default=JOBS_PER_GPU, help="Parallel jobs per GPU")
    args = parser.parse_args()

    jobs_per_gpu = args.jobs_per_gpu

    if args.show_distribution:
        all_tasks = get_all_tasks()
        pending = [t for t in all_tasks if not is_completed(*t)]
        print(f"Total tasks: {len(all_tasks)} ({len(pending)} pending)")
        for gpu in range(N_GPUS):
            tasks = get_gpu_tasks(gpu)
            gpu_pending = [t for t in tasks if not is_completed(*t)]
            print(f"  GPU {gpu}: {len(tasks)} tasks ({len(gpu_pending)} pending)")
        return

    if args.run_all:
        import sys
        for gpu in range(N_GPUS):
            cmd = [
                sys.executable, __file__, "--gpu", str(gpu),
                "--jobs-per-gpu", str(jobs_per_gpu)
            ]
            log_file = MODEL_DIR / f"gpu{gpu}_seeds.log"
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Launching GPU {gpu} with {jobs_per_gpu} parallel jobs, log: {log_file}")
            with open(log_file, "w") as f:
                subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        print(f"\nAll GPUs launched. Monitor with: tail -f {MODEL_DIR}/gpu*.log")
        return

    if args.gpu is not None:
        run_gpu_parallel(args.gpu, jobs_per_gpu)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
