#!/bin/bash
# Submit a chain of batch jobs with dependencies
#
# Number of jobs to submit:
NR_OF_JOBS=24

# Batch job script:
JOB_SCRIPT=./scripts/hyperparameter_analysis.sh

echo "Submitting job chain of ${NR_OF_JOBS} jobs for batch script ${JOB_SCRIPT}:" >> job_chain.out

JOBID=$(sbatch ${JOB_SCRIPT} 2>&1 | awk '{print $(NF)}')
echo "  " ${JOBID} >> job_chain.out

job_idx=1
while [[ ${job_idx} -lt ${NR_OF_JOBS} ]]; do
    JOBID=$(sbatch --dependency=afterany:${JOBID} ${JOB_SCRIPT} 2>&1 | awk '{print $(NF)}')
    echo "  " ${JOBID} >> job_chain.out
    let job_idx=${job_idx}+1
done

