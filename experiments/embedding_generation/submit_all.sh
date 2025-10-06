#!/bin/bash

REPO_ROOT="/u/fmahner/similarity-factorization"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${REPO_ROOT}/experiments/embedding_generation/logs"
mkdir -p "${LOG_DIR}"

submit_single() {
    local DATASET=$1
    local CPUS=$2
    local MEM=$3
    local TIME=$4
    local EXTRA_ARGS=$5
    
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${DATASET}
#SBATCH --output=${LOG_DIR}/${DATASET}_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/${DATASET}_${TIMESTAMP}_%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}

cd ${REPO_ROOT}
source setup_env.sh
poetry install --only main

poetry run python experiments/embedding_generation/run.py \
    --dataset ${DATASET} \
    --n_jobs ${CPUS} \
    --random_state 0 \
    --n_cv_repeats 5 \
    --n_stable_runs 50 \
    ${EXTRA_ARGS}
EOF
    
    echo "submitted ${DATASET}"
}

submit_chain() {
    local DATASET=$1
    local CPUS=$2
    local MEM=$3
    local TIME=$4
    local N_CHAINS=$5
    local EXTRA_ARGS=$6
    
    local CHAIN_LOG="${LOG_DIR}/${DATASET}_${TIMESTAMP}_chain.log"
    echo "[ $(date -Is) ] Submitting ${N_CHAINS} chained jobs for ${DATASET}" | tee -a "${CHAIN_LOG}"
    
    local JOBID=""
    local DEPENDENCY=""
    
    for CHAIN_IDX in $(seq 1 ${N_CHAINS}); do
        if [ -n "${JOBID}" ]; then
            DEPENDENCY="#SBATCH --dependency=afterany:${JOBID}"
        fi
        
        JOBID=$(sbatch ${DEPENDENCY:+--dependency=afterany:${JOBID}} <<EOF | awk '{print \$NF}'
#!/bin/bash
#SBATCH --job-name=${DATASET}_chain${CHAIN_IDX}
#SBATCH --output=${LOG_DIR}/${DATASET}_${TIMESTAMP}_chain${CHAIN_IDX}_%j.out
#SBATCH --error=${LOG_DIR}/${DATASET}_${TIMESTAMP}_chain${CHAIN_IDX}_%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}

cd ${REPO_ROOT}
source setup_env.sh
poetry install --only main

poetry run python experiments/embedding_generation/run.py \
    --dataset ${DATASET} \
    --n_jobs ${CPUS} \
    --random_state 0 \
    --n_cv_repeats 5 \
    --n_stable_runs 50 \
    ${EXTRA_ARGS}
EOF
)
        echo "[ $(date -Is) ]   chain ${CHAIN_IDX} -> job ${JOBID}" | tee -a "${CHAIN_LOG}"
    done
    
    echo "submitted ${DATASET} (${N_CHAINS} chained jobs)"
}

# Regular datasets
submit_single "mur92" 16 32G 06:00:00 ""
submit_single "cichy118" 16 32G 06:00:00 ""
submit_single "peterson-animals" 16 32G 06:00:00 ""
submit_single "peterson-various" 16 32G 06:00:00 ""

# Large datasets with chaining (3 attempts each)
submit_chain "things-monkey-22k" 64 200G 24:00:00 3 ""
submit_chain "vit" 64 480G 24:00:00 3 ""

# NSD subjects with chaining
for SUBJ in {1..8}; do
    submit_chain "nsd" 32 400G 24:00:00 3 "--subject_id ${SUBJ}"
done

echo ""
echo "all jobs submitted with timestamp ${TIMESTAMP}"
echo "logs: ${LOG_DIR}/*_${TIMESTAMP}_*.{out,err}"
echo "chain logs: ${LOG_DIR}/*_${TIMESTAMP}_chain.log"
