#!/usr/bin/env bash
cd /data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization
nohup .venv/bin/python -m experiments.datasets.dimensionality.consensus_batched \
  --dataset things_macaque2k_F_sigma0.4 \
  --rank 78 \
  --n-runs 30 \
  --n-jobs 30 \
  > experiments/datasets/dimensionality/logs/macaque_consensus_r78_20260523_163000.log 2>&1 < /dev/null &
echo "launched pid=$!"
