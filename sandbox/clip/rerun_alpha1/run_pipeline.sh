#!/usr/bin/env bash
# CLIP ViT-L/14 pipeline at α=1.0 (plain median heuristic).
#
# Runs estimate → validate → consensus → plot for the canonical dimensionality
# outputs at experiments/datasets/dimensionality/outputs/clip_vit_l14/.
#
# Re-runs from scratch — the α=0.4 outputs have been archived to
# outputs/clip_vit_l14_sigma0.4/ and outputs/cache/clip_vit_l14_sigma0.4.npy.

set -euo pipefail

PROJ=/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization
OUTPUTS="$PROJ/experiments/datasets/dimensionality/outputs"
LOG_DIR="$PROJ/sandbox/clip/rerun_alpha1/outputs"
mkdir -p "$LOG_DIR"

cd "$PROJ"
ts() { date +'%H:%M:%S'; }
log_to() { tee -a "$LOG_DIR/run.log" ; }

echo "[$(ts)] === CLIP α=1.0 pipeline start (host=$(hostname)) ===" | log_to

# 1. Estimate (regenerates cache at σ=median, writes coherence_estimate.json)
echo "[$(ts)] === [1/4] estimate ===" | log_to
.venv/bin/python -m experiments.datasets.dimensionality.run \
  --config-dir=configs \
  project_root="$PROJ" \
  mode=estimate \
  only=[clip_vit_l14] \
  cache_similarity=true \
  force=true \
  hydra.run.dir="$OUTPUTS" \
  hydra.job.chdir=true 2>&1 | log_to

# 2. CV at max_outer=200 (batched_validate uses the cached similarity)
echo "[$(ts)] === [2/4] validate (max_outer=200) ===" | log_to
.venv/bin/python -m experiments.datasets.dimensionality.batched_validate \
  --dataset clip_vit_l14 \
  --ranks 5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100 \
  --n-folds 5 --n-repeats 1 \
  --n-jobs 100 \
  --output-dir "$OUTPUTS" 2>&1 | log_to

# 3. Consensus + dimension reliability (writes to experiments/datasets/consensus/outputs/clip_vit_l14/)
echo "[$(ts)] === [3/4] consensus (n_runs=50) ===" | log_to
.venv/bin/python -m experiments.datasets.dimensionality.consensus_batched \
  --dataset clip_vit_l14 \
  --n-runs 50 --n-jobs 50 2>&1 | log_to

# 4. Plot CV
echo "[$(ts)] === [4/4] plot_cv ===" | log_to
.venv/bin/python -m experiments.datasets.dimensionality.plot_cv \
  --datasets clip_vit_l14 \
  --output-dir "$OUTPUTS" \
  --output "$OUTPUTS/clip_vit_l14/cv_curve.pdf" 2>&1 | log_to

echo "[$(ts)] === pipeline complete ===" | log_to
