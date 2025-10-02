#!/bin/bash -l

#SBATCH -e /u/fmahner/similarity-factorization/experiments/similarity_analysis/logs/debug_peterson-animals_%j.err
#SBATCH -o /u/fmahner/similarity-factorization/experiments/similarity_analysis/logs/debug_peterson-animals_%j.out
#SBATCH -D /u/fmahner/similarity-factorization
#SBATCH -J debug_peterson-animals

#SBATCH --partition=interactive
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000

module purge
module load intel/21.2.0
module load python-waterboa/2024.06

if [ ! -d ".venv" ]; then
    poetry install --no-interaction
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun /u/fmahner/similarity-factorization/run similarity_analysis --dataset peterson-animals --n_jobs 4 --random_state 0 --n_cv_repeats 10 --n_stable_runs 50

