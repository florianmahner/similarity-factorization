#!/usr/bin/env bash
cd /data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization
nohup .venv/bin/python -m experiments.datasets.dimensionality.cv_resumable \
  --dataset things_behavior \
  --cache-dataset things_behavior \
  --ranks 5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,150 \
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
  --per-fit-dir /data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/datasets/dimensionality/outputs/things_behavior/cv_per_fit \
  > experiments/datasets/dimensionality/logs/things_behavior_cv_20260523_130000.log 2>&1 < /dev/null &
echo "launched pid=$!"
