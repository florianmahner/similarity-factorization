#!/usr/bin/env bash
# CLIP ViT-L/14 pipeline at α=0.5 — bandwidth multiplier sweep variant.
#
# Mirrors sandbox/clip/rerun_alpha1/run_pipeline.sh but writes to a separate
# dataset alias so the canonical clip_vit_l14_sigma0.4 / clip_vit_l14
# outputs are not touched.
#
# Steps:
#   1. estimate  -> outputs/clip_vit_l14_sigma0.5/coherence_estimate.json
#                  outputs/cache/clip_vit_l14_sigma0.5.npy
#   2. cv_resumable (max_outer=200, max_inner=100, tol=1e-4, n_repeats=10)
#   3. consensus_batched at CV argmin rank
#   4. plot_cv

set -uo pipefail
# no `set -e` — each step has its own log line; one failing should not silence the rest

PROJ=/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization
OUTPUTS="$PROJ/experiments/datasets/dimensionality/outputs"
DATASET=clip_vit_l14_sigma0.5
LOG_DIR="$OUTPUTS/$DATASET"
mkdir -p "$LOG_DIR"

cd "$PROJ"
ts() { date +'%H:%M:%S'; }
log_to() { tee -a "$LOG_DIR/run.log" ; }

# 128 cores available, n=1854 → workers are light. All cores, single BLAS thread.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[$(ts)] === CLIP α=0.5 pipeline start (host=$(hostname)) ===" | log_to

# 1. Estimate: builds similarity at σ_scale=0.5, caches as
#    outputs/cache/clip_vit_l14_sigma0.5.npy, writes coherence_estimate.json
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

# 2. CV — resumable, rigorous kwargs (matches latest nsd_subj01 / macaque reruns)
if [ "${estimate_exit}" -eq 0 ]; then
  echo "[$(ts)] === [2/4] cv_resumable (max_outer=200 max_inner=50 tol=1e-4 n_repeats=5 n_jobs=128) ===" | log_to
  .venv/bin/python -m experiments.datasets.dimensionality.cv_resumable \
    --dataset ${DATASET} \
    --ranks 5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100 \
    --n-folds 5 --n-repeats 5 \
    --max-outer 200 --max-inner 50 --tol 1e-4 --rho 3.0 \
    --n-jobs 128 --blas-threads 1 \
    --output-dir "$OUTPUTS" 2>&1 | log_to
  cv_exit=${PIPESTATUS[0]}
else
  echo "[$(ts)] === [2/4] SKIPPED (estimate failed) ===" | log_to
  cv_exit=1
fi
echo "[$(ts)] cv exit=${cv_exit}" | log_to

# 3. Consensus at CV-selected rank — match the SRF kwargs used in CV.
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
    # consensus only needs 30 workers since n_runs=30; CV is what needs 128.
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
