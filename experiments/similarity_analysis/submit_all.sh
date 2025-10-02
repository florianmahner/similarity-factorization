#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$LOG_DIR"

TEMPLATE="$REPO_ROOT/scripts/slurm/production.template.sh"
WORK_DIR="$REPO_ROOT"

submit_job() {
    local dataset=$1
    local subject_id=$2
    local cpus=$3
    local mem=$4
    local time=$5
    
    if [ -z "$subject_id" ]; then
        JOB_NAME="${dataset}"
        COMMAND="$REPO_ROOT/run similarity_analysis --dataset $dataset --n_jobs $cpus --random_state 0 --n_cv_repeats 10 --n_stable_runs 50"
    else
        JOB_NAME="${dataset}_subj$(printf "%02d" $subject_id)"
        COMMAND="$REPO_ROOT/run similarity_analysis --dataset $dataset --subject_id $subject_id --n_jobs $cpus --random_state 0 --n_cv_repeats 10 --n_stable_runs 50"
    fi
    
    TEMP_SCRIPT="$LOG_DIR/${JOB_NAME}.sbatch"
    
    sed -e "s|{log_dir}|$LOG_DIR|g" \
        -e "s|{job_name}|$JOB_NAME|g" \
        -e "s|{work_dir}|$WORK_DIR|g" \
        -e "s|{cpus}|$cpus|g" \
        -e "s|{mem}|$mem|g" \
        -e "s|{time}|$time|g" \
        -e "s|{command}|$COMMAND|g" \
        "$TEMPLATE" > "$TEMP_SCRIPT"
    
    echo "submitting job: $JOB_NAME (cpus=$cpus, mem=${mem}MB, time=$time)"
    sbatch "$TEMP_SCRIPT"
}

echo "submitting similarity analysis jobs"
echo ""

submit_job "mur92" "" 16 32000 "1:00:00"
submit_job "cichy118" "" 16 32000 "1:00:00"
submit_job "peterson-animals" "" 16 32000 "1:00:00"
submit_job "peterson-various" "" 16 32000 "1:00:00"
submit_job "things-monkey-22k" "" 96 480000 "24:00:00"
submit_job "vit" "" 96 480000 "24:00:00"

for subj_id in 1 2 3 4 5 6 7 8; do
    submit_job "nsd" $subj_id 96 480000 "24:00:00"
done

echo ""
echo "all jobs submitted"
echo "check status: squeue -u $USER"
echo "logs: $LOG_DIR/"
echo "results: $SCRIPT_DIR/outputs/"
