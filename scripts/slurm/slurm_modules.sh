#!/bin/bash

module purge
module load intel/21.2.0
module load python-waterboa/2024.06

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

