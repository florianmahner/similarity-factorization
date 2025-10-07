#!/bin/bash
set -e

job_name=$1
partition=$2
cpus=$3
mem=$4
time=$5
log_dir=$6
repo_root=$7
run_script=$8
output_dir=$9
dependency=${10}
shift 10
script_args="$@"

dependency_directive=""
if [ -n "${dependency}" ]; then
    dependency_directive="#SBATCH --dependency=${dependency}"
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${partition}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH --time=${time}
#SBATCH --output=${log_dir}/slurm.out
#SBATCH --error=${log_dir}/slurm.err
${dependency_directive}

cd ${repo_root}
source /u/fmahner/.cache/pypoetry/virtualenvs/similarity-factorization-8QbpfJl5-py3.12/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}

python ${run_script} --output_dir ${output_dir} ${script_args}
EOF

