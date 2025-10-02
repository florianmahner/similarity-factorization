#!/bin/bash -l

#SBATCH -e {log_dir}/{job_name}_%j.err
#SBATCH -o {log_dir}/{job_name}_%j.out
#SBATCH -D {work_dir}
#SBATCH -J {job_name}

#SBATCH --partition=interactive
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}

module purge
module load intel/21.2.0
module load python-waterboa/2024.06

if [ ! -d ".venv" ]; then
    poetry install --no-interaction
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun {command}

