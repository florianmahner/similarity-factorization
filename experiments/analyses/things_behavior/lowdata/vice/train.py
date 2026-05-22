"""VICE low-data experiment: Train VICE on subsampled THINGS triplets.

Usage:
    # Prepare partitions (run once)
    poetry run python experiments/things_behavior/vice_lowdata.py --prepare

    # Train single configuration
    poetry run python experiments/things_behavior/vice_lowdata.py \
        --train --percentage 5 --partition 0 --seed 0 --gpu 0

    # Aggregate results
    poetry run python experiments/things_behavior/vice_lowdata.py --aggregate
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DATA_DIR = PROJECT_ROOT / "data" / "things"
TRIPLET_DIR = DATA_DIR / "triplets_47"
PARTITION_DIR = DATA_DIR / "partitions"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
OBJDIM_ROOT = Path("/LOCAL/fmahner/object-dimensions")
OBJDIM_PYTHON = OBJDIM_ROOT / ".venv" / "bin" / "python"
OBJDIM_SCRIPT = OBJDIM_ROOT / "run_optimization.py"

PARTITIONS = {
    5: 20,
    10: 10,
    20: 5,
    50: 2,
}


def prepare_partitions() -> None:
    """Create non-overlapping partitions for all percentages."""
    trainset_path = TRIPLET_DIR / "trainset.txt"
    valset_path = TRIPLET_DIR / "test_10.txt"

    print(f"Loading triplets from {trainset_path}")
    triplets = np.loadtxt(trainset_path, dtype=int)
    print(f"Loaded {len(triplets):,} triplets")

    rng = np.random.default_rng(42)
    rng.shuffle(triplets)

    for pct, n_parts in PARTITIONS.items():
        part_size = len(triplets) // n_parts
        print(f"\nCreating {n_parts} partitions for {pct}% ({part_size:,} triplets each)")

        for i in range(n_parts):
            part_dir = PARTITION_DIR / f"{pct}pct_part{i}"
            part_dir.mkdir(parents=True, exist_ok=True)

            start = i * part_size
            end = start + part_size
            part = triplets[start:end]

            train_path = part_dir / "train_90.txt"
            np.savetxt(train_path, part, fmt="%d")

            test_path = part_dir / "test_10.txt"
            if not test_path.exists():
                shutil.copy(valset_path, test_path)

            print(f"  {part_dir.name}: {len(part):,} triplets")

    print(f"\nPartitions saved to {PARTITION_DIR}")


def train_single(percentage: int, partition: int, gpu: int) -> None:
    """Train VICE on a single partition."""
    partition_dir = PARTITION_DIR / f"{percentage}pct_part{partition}"
    identifier = f"vice_{percentage}pct_part{partition}"

    if not partition_dir.exists():
        raise FileNotFoundError(f"Partition directory not found: {partition_dir}")

    result_dir = MODEL_DIR / identifier
    params_file = result_dir / "params" / "parameters.npz"
    if params_file.exists():
        print(f"Skipping {identifier} - already completed")
        return

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
        "--device_id", "0",  # Always 0 because CUDA_VISIBLE_DEVICES restricts to one GPU
        "--log_path", str(MODEL_DIR),
        "--identifier", identifier,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"Training {identifier} on GPU {gpu}")
    subprocess.run(cmd, check=True, env=env)


def extract_result(identifier: str) -> dict | None:
    """Extract results from a single trained model with validation."""
    result_dir = MODEL_DIR / identifier
    params_dir = result_dir / "params"

    if not params_dir.exists():
        return None

    # Find the final params file (parameters.npz or last epoch)
    final_params = params_dir / "parameters.npz"
    if not final_params.exists():
        param_files = sorted(params_dir.glob("params_epoch_*.npz"))
        if not param_files:
            return None
        final_params = param_files[-1]

    params = np.load(final_params, allow_pickle=True)

    # Extract embedding - try multiple keys for robustness
    embedding = None
    for key in ["pruned_q_mu", "embedding", "pruned_weights"]:
        if key in params:
            embedding = params[key]
            break

    if embedding is None:
        print(f"WARNING: No embedding found in {identifier}")
        return None

    n_dims = embedding.shape[1]

    # Extract accuracy - handle both scalar and array
    val_acc = params.get("val_acc", np.nan)
    if hasattr(val_acc, "__len__"):
        val_acc = float(val_acc[-1])  # Last epoch
    else:
        val_acc = float(val_acc)

    # Extract epoch info
    epoch = int(params.get("epoch", -1))

    return {
        "n_dimensions": n_dims,
        "accuracy": val_acc,
        "final_epoch": epoch,
        "embedding_shape": embedding.shape,
    }


def aggregate_results() -> None:
    """Aggregate all training results into a single CSV with validation."""
    rows = []
    missing = []
    failed = []

    total_expected = sum(PARTITIONS.values())
    print(f"Expecting {total_expected} results...")

    for pct, n_parts in PARTITIONS.items():
        for part in range(n_parts):
            identifier = f"vice_{pct}pct_part{part}"
            result = extract_result(identifier)

            if result is None:
                if not (MODEL_DIR / identifier).exists():
                    missing.append(identifier)
                else:
                    failed.append(identifier)
                continue

            rows.append({
                "percentage": pct,
                "partition": part,
                "accuracy": result["accuracy"],
                "n_dimensions": result["n_dimensions"],
                "final_epoch": result["final_epoch"],
            })

    # Report status
    print(f"\nResults: {len(rows)}/{total_expected} successful")
    if missing:
        print(f"Missing ({len(missing)}): {missing[:5]}..." if len(missing) > 5 else f"Missing: {missing}")
    if failed:
        print(f"Failed ({len(failed)}): {failed[:5]}..." if len(failed) > 5 else f"Failed: {failed}")

    if not rows:
        print("ERROR: No results to aggregate!")
        return

    # Save results
    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} results to {output_path}")

    # Validation checks
    print("\n=== Validation ===")
    print(f"Accuracy range: [{df['accuracy'].min():.3f}, {df['accuracy'].max():.3f}]")
    print(f"Dimensions range: [{df['n_dimensions'].min()}, {df['n_dimensions'].max()}]")
    if df['accuracy'].min() < 0.33:
        print("WARNING: Some accuracies below chance (0.33)!")
    if df['n_dimensions'].max() > 90:
        print("WARNING: Some models have more dims than init_dim!")

    print("\n=== Summary by percentage ===")
    summary = df.groupby("percentage").agg({
        "accuracy": ["mean", "std", "count"],
        "n_dimensions": ["mean", "std"],
    }).round(3)
    print(summary)

    # Save summary too
    summary_path = OUTPUT_DIR / "summary.csv"
    summary.to_csv(summary_path)
    print(f"\nSaved summary to {summary_path}")


def verify_progress() -> None:
    """Check progress of training without aggregating."""
    print("=== VICE Low-Data Experiment Progress ===\n")

    total_expected = sum(PARTITIONS.values())
    completed = 0
    in_progress = 0

    for pct, n_parts in PARTITIONS.items():
        pct_completed = 0
        pct_in_progress = 0
        for part in range(n_parts):
            identifier = f"vice_{pct}pct_part{part}"
            model_dir = MODEL_DIR / identifier

            if not model_dir.exists():
                continue

            params_dir = model_dir / "params"
            if (params_dir / "parameters.npz").exists():
                pct_completed += 1
                completed += 1
            elif list(params_dir.glob("params_epoch_*.npz")):
                pct_in_progress += 1
                in_progress += 1

        print(f"{pct}%: {pct_completed}/{n_parts} complete, {pct_in_progress} in progress")

    print(f"\nTotal: {completed}/{total_expected} complete ({100*completed/total_expected:.1f}%)")
    if in_progress:
        print(f"In progress: {in_progress}")

    # Check disk usage
    if MODEL_DIR.exists():
        import shutil
        total, used, free = shutil.disk_usage(MODEL_DIR)
        model_size = sum(f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file())
        print(f"\nDisk: {model_size/1e9:.2f} GB used by models, {free/1e9:.1f} GB free")


def main():
    parser = argparse.ArgumentParser(description="VICE low-data experiment")
    parser.add_argument("--prepare", action="store_true", help="Create partition directories")
    parser.add_argument("--train", action="store_true", help="Train single configuration")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate results to CSV")
    parser.add_argument("--verify", action="store_true", help="Check training progress")
    parser.add_argument("--percentage", type=int, help="Data percentage (5, 10, 20, 50)")
    parser.add_argument("--partition", type=int, help="Partition index")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")

    args = parser.parse_args()

    if args.prepare:
        prepare_partitions()
    elif args.train:
        if args.percentage is None or args.partition is None:
            parser.error("--train requires --percentage and --partition")
        train_single(args.percentage, args.partition, args.gpu)
    elif args.aggregate:
        aggregate_results()
    elif args.verify:
        verify_progress()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
