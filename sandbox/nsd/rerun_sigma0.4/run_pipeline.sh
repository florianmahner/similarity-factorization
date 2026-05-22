#!/usr/bin/env bash
# NSD subj01 pipeline at α=0.4 — robust version using cv_resumable.py.
# Estimate already done yesterday; this resumes from CV.
# No `set -e` — each step is run with its own log so failures are visible
# but don't silently kill the rest of the pipeline.

PROJ=/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization
OUTPUTS="$PROJ/experiments/datasets/dimensionality/outputs"
LOG_DIR="$PROJ/sandbox/nsd/rerun_sigma0.4/outputs"
mkdir -p "$LOG_DIR"

cd "$PROJ"
ts() { date +'%H:%M:%S'; }
log_to() { tee -a "$LOG_DIR/run.log" ; }

# Match BLAS configuration to macaque pipeline.
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6

echo "[$(ts)] === NSD subj01 α=0.4 pipeline (RESUMABLE) start (host=$(hostname)) ===" | log_to

# Sanity check: estimate must have completed previously.
if [ ! -f "$OUTPUTS/nsd_subj01/coherence_estimate.json" ]; then
  echo "[$(ts)] ERROR: no coherence_estimate.json — need to run estimate first" | log_to
  exit 1
fi
if [ ! -f "$OUTPUTS/cache/nsd_subj01.npy" ]; then
  echo "[$(ts)] ERROR: no cached similarity" | log_to
  exit 1
fi
echo "[$(ts)] estimate state OK; cache present" | log_to

# 1. CV via cv_resumable (same robust pattern as macaque).
echo "[$(ts)] === [1/3] cv_resumable (max_outer=50 max_inner=100 tol=1e-4) ===" | log_to
.venv/bin/python -m experiments.datasets.dimensionality.cv_resumable \
  --dataset nsd_subj01 \
  --ranks 5,10,15,20,30,40,50,60,80,100,120,150,200 \
  --n-folds 5 --n-repeats 1 \
  --max-outer 50 --max-inner 100 --tol 1e-4 --rho 3.0 \
  --n-jobs 30 --blas-threads 4 \
  --output-dir "$OUTPUTS" 2>&1 | log_to
cv_exit=${PIPESTATUS[0]}
echo "[$(ts)] cv_resumable exit=$cv_exit" | log_to

# 2. Consensus (n_runs=30 to match other datasets).
if [ "$cv_exit" -eq 0 ]; then
  echo "[$(ts)] === [2/3] consensus (n_runs=30) ===" | log_to
  .venv/bin/python -m experiments.datasets.dimensionality.consensus_batched \
    --dataset nsd_subj01 \
    --n-runs 30 --n-jobs 30 2>&1 | log_to
  consensus_exit=${PIPESTATUS[0]}
  echo "[$(ts)] consensus exit=$consensus_exit" | log_to
else
  echo "[$(ts)] === [2/3] SKIPPED (CV failed) ===" | log_to
  consensus_exit=1
fi

# 3. Plot.
if [ "$consensus_exit" -eq 0 ]; then
  echo "[$(ts)] === [3/3] plot_cv ===" | log_to
  .venv/bin/python -m experiments.datasets.dimensionality.plot_cv \
    --datasets nsd_subj01 \
    --output-dir "$OUTPUTS" \
    --output "$OUTPUTS/nsd_subj01/cv_curve.pdf" 2>&1 | log_to
fi

echo "[$(ts)] === pipeline complete (cv=$cv_exit consensus=$consensus_exit) ===" | log_to
