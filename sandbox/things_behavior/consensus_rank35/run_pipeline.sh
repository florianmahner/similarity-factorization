#!/usr/bin/env bash
# things_behavior consensus at k*=35 (post-longer-training migration).
#
# Reads rank from cross_validation.json (= 35 after the May 21 migration),
# runs 50-run ensemble + AlignedConsensus + dimension reliability, writes
# to experiments/datasets/consensus/outputs/things_behavior/.

set -euo pipefail

PROJ=/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization
LOG_DIR="$PROJ/sandbox/things_behavior/consensus_rank35/outputs"
mkdir -p "$LOG_DIR"

cd "$PROJ"
ts() { date +'%H:%M:%S'; }
log_to() { tee -a "$LOG_DIR/run.log" ; }

echo "[$(ts)] === things_behavior consensus at k*=35 start (host=$(hostname)) ===" | log_to

.venv/bin/python -m experiments.datasets.dimensionality.consensus_batched \
  --dataset things_behavior \
  --n-runs 50 --n-jobs 50 2>&1 | log_to

echo "[$(ts)] === pipeline complete ===" | log_to
