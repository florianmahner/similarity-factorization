#!/usr/bin/env bash
set -euo pipefail

# Final paper CV protocol for alpha=0.4 RBF datasets.
# Each command is resumable and writes per-fit JSONs into outputs/<dataset>/cv_per_fit/.
#
# Protocol:
#   - 5-fold entry CV
#   - 5 repeats
#   - SRF rho=3, max_outer=200, max_inner=30, tol=1e-4
#   - explicit rank grids per dataset
#
# Note: existing per-fit JSONs were fit at max_inner=50 (more iterations than
# the current protocol requires). They satisfy the convergence target and are
# reused as-is; only new (rank, fold, repeat) combos are fit at max_inner=30.
# Aggregator emits a "mixed kwargs" warning — expected, not an error.

ROOT="/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization"
OUT="$ROOT/experiments/datasets/dimensionality/outputs"
PY="$ROOT/.venv/bin/python"

run_cv() {
  local dataset="$1"
  local cache_dataset="$2"
  local ranks="$3"
  local n_jobs="$4"

  "$PY" -m experiments.datasets.dimensionality.cv_resumable \
    --dataset "$dataset" \
    --cache-dataset "$cache_dataset" \
    --ranks "$ranks" \
    --n-folds 5 \
    --n-repeats 5 \
    --variant 5fold \
    --max-outer 200 \
    --max-inner 30 \
    --tol 1e-4 \
    --rho 3.0 \
    --n-jobs "$n_jobs" \
    --blas-threads 1 \
    --output-dir "$OUT" \
    --per-fit-dir "$OUT/$dataset/cv_per_fit"
}

case "${1:-}" in
  clip)
    run_cv \
      "clip_vit_l14_sigma0.4" \
      "clip_vit_l14" \
      "${3:-5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100}" \
      "${2:-64}"
    ;;
  nsd)
    run_cv \
      "nsd_subj01_sigma0.4" \
      "nsd_subj01" \
      "${3:-10,20,30,40,50,60,70,80,90,100,120,160,200}" \
      "${2:-120}"
    ;;
  macaque2k)
    run_cv \
      "things_macaque2k_F_sigma0.4" \
      "things_macaque2k_F_sigma0.4" \
      "${3:-5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,120}" \
      "${2:-64}"
    ;;
  *)
    echo "Usage: $0 {clip|nsd|macaque2k} [n_jobs] [comma_separated_ranks]" >&2
    exit 2
    ;;
esac
