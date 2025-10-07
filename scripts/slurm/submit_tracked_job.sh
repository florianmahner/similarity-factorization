#!/bin/bash
set -e

if [ $# -lt 8 ]; then
    echo "Usage: $0 <run_id> <experiment> <log_dir> <output_dir> <repo_root> <cpus> <mem> <time> [partition] [extra_args...]"
    exit 1
fi

RUN_ID="$1"
EXPERIMENT="$2"
LOG_DIR="$3"
OUTPUT_DIR="$4"
REPO_ROOT="$5"
CPUS="$6"
MEM="$7"
TIME="$8"
if [ $# -ge 9 ]; then
    PARTITION="$9"
    shift 9
else
    PARTITION="normal"
    shift 8
fi
EXTRA_ARGS="$@"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${RUN_ID}
#SBATCH --output=${LOG_DIR}/slurm.out
#SBATCH --error=${LOG_DIR}/slurm.err
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}

cd ${REPO_ROOT}
source ${REPO_ROOT}/scripts/slurm/slurm_modules.sh

export PYTHONPATH="${REPO_ROOT}:\$PYTHONPATH"

poetry run python experiments/${EXPERIMENT}/run.py ${EXTRA_ARGS} \\
    --output_dir ${OUTPUT_DIR} \\
    2>&1 | tee ${LOG_DIR}/main.log

exit_code=\${PIPESTATUS[0]}

if [ \$exit_code -eq 0 ]; then
    poetry run python -c "import sys; sys.path.insert(0, '${REPO_ROOT}/experiments'); from tracker import Tracker; Tracker().update_status('${RUN_ID}', 'completed')"
else
    poetry run python -c "import sys; sys.path.insert(0, '${REPO_ROOT}/experiments'); from tracker import Tracker; Tracker().update_status('${RUN_ID}', 'failed')"
fi

exit \$exit_code
EOF

