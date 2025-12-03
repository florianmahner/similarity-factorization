import time
import random
import sys
import os

# Add current directory to path to import job_logger
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from job_logger import JobLogger, JobType, JobStatus

def run():
    logger = JobLogger()
    
    # Start Job
    job_id = logger.start_job(
        name="Data Processing (Fail)",
        job_type=JobType.LOCAL,
        command="python process_data.py",
    )
    print(f"Started job {job_id}")
    
    logger.log(job_id, stdout="Loading large dataset...")
    time.sleep(3)
    
    logger.log(job_id, stdout="Dataset loaded. Starting preprocessing...")
    time.sleep(2)
    
    # Simulate failure
    error_msg = "RuntimeError: Out of memory. Tried to allocate 50GB but only 10GB available."
    print(error_msg)
    logger.log(job_id, stderr=error_msg)
    
    logger.update_status(job_id, JobStatus.FAILED, duration="5.2s")
    print("Job failed")

if __name__ == "__main__":
    run()

