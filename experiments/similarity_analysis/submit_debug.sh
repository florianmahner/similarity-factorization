#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$LOG_DIR"

DATASET="${1:-peterson-animals}"

JOB_NAME="debug_${DATASET}"
WORK_DIR="$REPO_ROOT"
CPUS=4
MEM="8000"

COMMAND="$REPO_ROOT/run similarity_analysis --dataset $DATASET --n_jobs $CPUS --random_state 0 --n_cv_repeats 10 --n_stable_runs 50"

TEMPLATE="$REPO_ROOT/scripts/slurm/debug.template.sh"

TEMP_SCRIPT="$LOG_DIR/${JOB_NAME}.sbatch"

sed -e "s|{log_dir}|$LOG_DIR|g" \
    -e "s|{job_name}|$JOB_NAME|g" \
    -e "s|{work_dir}|$WORK_DIR|g" \
    -e "s|{cpus}|$CPUS|g" \
    -e "s|{mem}|$MEM|g" \
    -e "s|{command}|$COMMAND|g" \
    "$TEMPLATE" > "$TEMP_SCRIPT"

echo "submitting debug job for $DATASET"
echo "logs: $LOG_DIR/${JOB_NAME}_*.{out,err}"
echo ""

sbatch "$TEMP_SCRIPT"
