#!/bin/bash

DATASET="${1:-peterson-animals}"
REPO_ROOT="/u/fmahner/similarity-factorization"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${REPO_ROOT}/experiments/embedding_generation/logs"
mkdir -p "${LOG_DIR}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=debug_${DATASET}
#SBATCH --output=${LOG_DIR}/debug_${DATASET}_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/debug_${DATASET}_${TIMESTAMP}_%j.err
#SBATCH --partition=interactive
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

cd ${REPO_ROOT}
source setup_env.sh
poetry install --only main

poetry run python experiments/embedding_generation/run.py \
    --dataset ${DATASET} \
    --n_jobs 4 \
    --random_state 0 \
    --n_cv_repeats 2 \
    --n_stable_runs 5
EOF

echo "submitted debug job for ${DATASET}"
echo "logs: ${LOG_DIR}/debug_${DATASET}_${TIMESTAMP}_*.{out,err}"
