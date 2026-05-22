#!/usr/bin/env bash
set -euo pipefail

PROJ=/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization
OUT="$PROJ/experiments/datasets/dimensionality/outputs/things_macaque22_sigma0.4"
LOG_DIR="$OUT/cv_logs"
mkdir -p "$LOG_DIR"

cd "$PROJ"

# Run the 30 rank x fold fits in two waves. The all-at-once 30 x 2 layout
# exceeded this machine's memory footprint before any fit could finish.
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6

.venv/bin/python -m experiments.datasets.dimensionality.cv_resumable \
  --dataset things_macaque22_sigma0.4 \
  --cache-dataset things_macaque22k \
  --variant 5fold_outer200_r50_100_125_150_175_200 \
  --ranks 50,100,125,150,175,200 \
  --n-folds 5 --n-repeats 1 \
  --max-outer 200 --max-inner 100 --tol 1e-4 --rho 3.0 \
  --n-jobs 15 --blas-threads 6 \
  --per-fit-dir "$OUT/cv_per_fit_outer200_r50_100_125_150_175_200" \
  --save-estimators \
  --estimator-dir "$OUT/cv_estimators_outer200_r50_100_125_150_175_200"
