#!/usr/bin/env bash
# Macaque (things_macaque22k) pipeline at α=0.4 — full end-to-end on prefrontal.
#
# Uses the new resumable CV (cv_resumable.py) so ranks can be added/refined
# later without redoing existing fits. SRF kwargs: max_outer=50, max_inner=100,
# tol=1e-4 — same fast-but-converged regime as deep-similarity-neurips.

set -euo pipefail

PROJ=/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization
OUTPUTS="$PROJ/experiments/datasets/dimensionality/outputs"
LOG_DIR="$PROJ/sandbox/macaque/rerun_sigma0.4/outputs"
mkdir -p "$LOG_DIR"

cd "$PROJ"
ts() { date +'%H:%M:%S'; }
log_to() { tee -a "$LOG_DIR/run.log" ; }

# CRITICAL for consensus: EnsembleFit workers inherit BLAS thread count from env.
# n_jobs=20 × 6 BLAS threads = 120 cores (well within 128 on prefrontal).
# Memory per worker: ~30 GB (n=22k, n²·float64 ADMM aux + train matrix).
# 20 workers × 30 GB = 600 GB peak — safely under 1 TB.
# More parallel risks OOM; more BLAS threads gives diminishing returns past 6.
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6

echo "[$(ts)] === macaque α=0.4 pipeline start (host=$(hostname)) ===" | log_to

# 1. Estimate (regenerate cache at σ=0.4·median, run coherence)
echo "[$(ts)] === [1/4] estimate ===" | log_to
.venv/bin/python -m experiments.datasets.dimensionality.run \
  --config-dir=configs \
  project_root="$PROJ" \
  mode=estimate \
  only=[things_macaque22k] \
  cache_similarity=true \
  force=true \
  hydra.run.dir="$OUTPUTS" \
  hydra.job.chdir=true 2>&1 | log_to

# 2. CV — resumable, max_outer=50, max_inner=100, tol=1e-4
#    ranks 10..100. Extend later by re-running with a longer --ranks list.
echo "[$(ts)] === [2/4] cv_resumable (max_outer=50 max_inner=100 tol=1e-4) ===" | log_to
.venv/bin/python -m experiments.datasets.dimensionality.cv_resumable \
  --dataset things_macaque22k \
  --ranks 10,20,30,40,50,60,80,100 \
  --n-folds 5 --n-repeats 1 \
  --max-outer 50 --max-inner 100 --tol 1e-4 --rho 3.0 \
  --n-jobs 20 --blas-threads 6 \
  --output-dir "$OUTPUTS" 2>&1 | log_to

# 3. Consensus — same SRF kwargs as CV (matching is important for fair recon comparison)
echo "[$(ts)] === [3/4] consensus (n_runs=50) ===" | log_to
.venv/bin/python -c "
import sys
sys.path.insert(0, '.')
from experiments.datasets.dimensionality import consensus_batched
consensus_batched.SRF_KWARGS = {
    'rho': 3.0, 'max_inner': 100, 'tol': 1e-4,
    'max_outer': 50, 'check_input': False,
}
sys.argv = ['consensus_batched',
    '--dataset', 'things_macaque22k',
    '--n-runs', '30', '--n-jobs', '20',
]
consensus_batched.main()
" 2>&1 | log_to

# 4. Plot CV curve
echo "[$(ts)] === [4/4] plot_cv ===" | log_to
.venv/bin/python -m experiments.datasets.dimensionality.plot_cv \
  --datasets things_macaque22k \
  --output-dir "$OUTPUTS" \
  --output "$OUTPUTS/things_macaque22k/cv_curve.pdf" 2>&1 | log_to

echo "[$(ts)] === pipeline complete ===" | log_to
