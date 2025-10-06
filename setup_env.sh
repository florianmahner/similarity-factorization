#!/bin/bash
module purge
module load intel/21.2.0
module load python-waterboa/2024.06
echo "Environment loaded: Python $(python3 --version)"

