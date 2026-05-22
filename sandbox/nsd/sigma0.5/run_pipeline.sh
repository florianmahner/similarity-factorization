#!/usr/bin/env bash
# NSD subj01 pipeline at α=0.5 — bandwidth multiplier sweep variant.
# Mirrors sandbox/nsd/rerun_sigma0.4/run_pipeline.sh but writes everything
# to nsd_subj01_sigma0.5 so the canonical α=0.4 outputs are untouched.
#
# Solver: max_outer=200, max_inner=50, tol=1e-4 — fewer inner iters per outer
# step but many more outer steps so ADMM actually converges (per user
# instruction).
#
# No `set -e` — each step gets its own log line so failures are visible
# but don't kill the rest of the pipeline.

set -uo pipefail

PROJ=/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization
OUTPUTS="$PROJ/experiments/datasets/dimensionality/outputs"
DATASET=nsd_subj01_sigma0.5
LOG_DIR="$OUTPUTS/$DATASET"
mkdir -p "$LOG_DIR"

cd "$PROJ"
ts() { date +'%H:%M:%S'; }
log_to() { tee -a "$LOG_DIR/run.log" ; }

# All 144 cores on insula, single BLAS thread (per user instruction).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[$(ts)] === NSD subj01 α=0.5 pipeline start (host=$(hostname)) ===" | log_to

# 1. Estimate: builds similarity at σ_scale=0.5, caches as
#    outputs/cache/nsd_subj01_sigma0.5.npy, writes coherence_estimate.json
echo "[$(ts)] === [1/4] estimate ===" | log_to
.venv/bin/python -m experiments.datasets.dimensionality.run \
  --config-dir=configs \
  project_root="$PROJ" \
  mode=estimate \
  only=[${DATASET}] \
  cache_similarity=true \
  force=true \
  hydra.run.dir="$OUTPUTS" \
  hydra.job.chdir=true 2>&1 | log_to
estimate_exit=${PIPESTATUS[0]}
echo "[$(ts)] estimate exit=${estimate_exit}" | log_to

# 2. CV — resumable, max_outer=200 max_inner=50 tol=1e-4
if [ "${estimate_exit}" -eq 0 ]; then
  echo "[$(ts)] === [2/4] cv_resumable (max_outer=200 max_inner=50 tol=1e-4 n_repeats=5 n_jobs=144) ===" | log_to
  .venv/bin/python -m experiments.datasets.dimensionality.cv_resumable \
    --dataset ${DATASET} \
    --ranks 5,10,15,20,30,40,50,60,80,100,120,150,200 \
    --n-folds 5 --n-repeats 5 \
    --max-outer 200 --max-inner 50 --tol 1e-4 --rho 3.0 \
    --n-jobs 144 --blas-threads 1 \
    --output-dir "$OUTPUTS" 2>&1 | log_to
  cv_exit=${PIPESTATUS[0]}
else
  echo "[$(ts)] === [2/4] SKIPPED (estimate failed) ===" | log_to
  cv_exit=1
fi
echo "[$(ts)] cv exit=${cv_exit}" | log_to

# 3. Consensus — match CV kwargs exactly so reconstruction is comparable.
if [ "${cv_exit}" -eq 0 ]; then
  echo "[$(ts)] === [3/4] consensus (n_runs=30, matching CV kwargs) ===" | log_to
  .venv/bin/python -c "
import sys
sys.path.insert(0, '.')
from experiments.datasets.dimensionality import consensus_batched
consensus_batched.SRF_KWARGS = {
    'rho': 3.0, 'max_inner': 50, 'tol': 1e-4,
    'max_outer': 200, 'check_input': False,
}
sys.argv = ['consensus_batched',
    '--dataset', '${DATASET}',
    '--n-runs', '30', '--n-jobs', '30',
    # consensus only needs 30 workers since n_runs=30; CV is what needs 144.
]
consensus_batched.main()
" 2>&1 | log_to
  consensus_exit=${PIPESTATUS[0]}
else
  echo "[$(ts)] === [3/4] SKIPPED (CV failed) ===" | log_to
  consensus_exit=1
fi
echo "[$(ts)] consensus exit=${consensus_exit}" | log_to

# 4. Plot CV curve
if [ "${cv_exit}" -eq 0 ]; then
  echo "[$(ts)] === [4/4] plot_cv ===" | log_to
  .venv/bin/python -m experiments.datasets.dimensionality.plot_cv \
    --datasets ${DATASET} \
    --output-dir "$OUTPUTS" \
    --output "$OUTPUTS/${DATASET}/cv_curve.pdf" 2>&1 | log_to
else
  echo "[$(ts)] === [4/4] SKIPPED (CV failed) ===" | log_to
fi

echo "[$(ts)] === pipeline complete (estimate=${estimate_exit} cv=${cv_exit} consensus=${consensus_exit}) ===" | log_to
