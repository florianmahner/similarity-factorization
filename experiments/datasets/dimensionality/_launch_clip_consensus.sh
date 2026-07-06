#!/usr/bin/env bash
cd /data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization
nohup .venv/bin/python -m experiments.datasets.dimensionality.consensus_batched \
  --dataset clip_vit_l14_sigma0.4 --n-runs 30 --n-jobs 30 \
  > experiments/datasets/dimensionality/logs/consensus_clip_20260523_110500.log 2>&1 < /dev/null &
echo "launched pid=$!"
