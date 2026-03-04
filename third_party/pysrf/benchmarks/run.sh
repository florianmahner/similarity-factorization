#!/bin/bash
#SBATCH --job-name=bsum-bench
#SBATCH --partition=p.large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=benchmarks_%j.log

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

PYTHON=${PYTHON:-python3}

echo "=== Correctness Test ==="
$PYTHON -u benchmarks/correctness.py

echo ""
echo "=== Performance Benchmark ==="
$PYTHON -u benchmarks/benchmark.py
