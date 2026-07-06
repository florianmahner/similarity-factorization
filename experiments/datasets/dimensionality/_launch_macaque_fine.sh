#!/usr/bin/env bash
cd /data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization
nohup .venv/bin/python -m experiments.datasets.dimensionality.cv_resumable \
  --dataset things_macaque2k_F_sigma0.4 \
  --cache-dataset things_macaque2k_F_sigma0.4 \
  --ranks 70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90 \
  --n-folds 5 \
  --n-repeats 5 \
  --variant 5fold \
  --max-outer 200 \
  --max-inner 30 \
  --tol 1e-4 \
  --rho 3.0 \
  --n-jobs 120 \
  --blas-threads 1 \
  --output-dir /data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/datasets/dimensionality/outputs \
  --per-fit-dir /data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/datasets/dimensionality/outputs/things_macaque2k_F_sigma0.4/cv_per_fit \
  > experiments/datasets/dimensionality/logs/macaque_fine70_90_20260523_151500.log 2>&1 < /dev/null &
echo "launched pid=$!"
