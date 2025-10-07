#!/bin/bash
set -euo pipefail

REPO_ROOT="/u/fmahner/similarity-factorization"
RUN_CMD="${REPO_ROOT}/run"

submit_single() {
    local dataset=$1
    local cpus=$2
    local mem=$3
    local time=$4
    shift 4
    local extra_args=("$@")
    
    "${RUN_CMD}" embedding_generation \
        --partition normal \
        --cpus "${cpus}" \
        --mem "${mem}" \
        --time "${time}" \
        --output-name "${dataset}" \
        -- --dataset "${dataset}" \
        --n_jobs "${cpus}" \
        --random_state 0 \
        --n_cv_repeats 20 \
        --n_stable_runs 50 \
        "${extra_args[@]}"
}

submit_chain() {
    local dataset=$1
    local cpus=$2
    local mem=$3
    local time=$4
    local n_chains=$5
    
    echo "submitting ${n_chains} chained jobs for ${dataset}..."
    
    local prev_job=""
    for i in $(seq 1 "${n_chains}"); do
        local name="${dataset}_chain$(printf '%02d' "${i}")"
        local dep_arg=""
        if [ -n "${prev_job}" ]; then
            dep_arg="--dependency afterany:${prev_job}"
        fi
        
        local output
        output=$("${RUN_CMD}" embedding_generation \
            --partition normal \
            --cpus "${cpus}" \
            --mem "${mem}" \
            --time "${time}" \
            --name "${name}" \
            --output-name "${dataset}" \
            ${dep_arg} \
            -- --dataset "${dataset}" \
            --n_jobs "${cpus}" \
            --random_state 0 \
            --n_cv_repeats 10 \
            --n_stable_runs 50)
        
        prev_job=$(echo "${output}" | awk '{print $NF}')
        echo "  chain ${i} -> job ${prev_job}"
    done
}

echo "submitting small datasets..."
submit_single "mur92" 4 16G 01:00:00
# submit_single "cichy118" 4 16G 01:00:00
# submit_single "peterson-animals" 4 16G 01:00:00
# submit_single "peterson-various" 4 16G 01:00:00

# echo ""
# echo "submitting large datasets with chaining..."
# submit_chain "things-monkey-22k" 72 400G 24:00:00 4
# submit_chain "vit" 72 400G 24:00:00 4

# echo ""
# echo "submitting nsd subjects..."
# for subj in $(seq 1 8); do
#     "${RUN_CMD}" embedding_generation \
#         --partition normal \
#         --cpus 64 \
#         --mem 240G \
#         --time 24:00:00 \
#         --name "nsd_subj${subj}" \
#         --output-name "nsd_subj${subj}" \
#         -- --dataset nsd \
#         --n_jobs 64 \
#         --random_state 0 \
#         --n_cv_repeats 10 \
#         --n_stable_runs 50 \
#         --subject_id "${subj}"
# done

# echo ""
# echo "all jobs submitted"