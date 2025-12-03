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
        name="Training Experiment A (Success)",
        job_type=JobType.LOCAL,
        command="python train.py --epochs 100",
        hydra_config={"configName": "train", "overrides": ["lr=0.01", "batch_size=32"], "outputDir": "/tmp/exp_a"}
    )
    print(f"Started job {job_id}")
    
    logger.log(job_id, stdout="Initializing training parameters...")
    time.sleep(2)
    
    total_epochs = 10
    start_time = time.time()
    
    for epoch in range(total_epochs):
        # Simulate work
        time.sleep(1.5) 
        
        # Log output
        loss = 0.5 - (0.4 * epoch / total_epochs) + (random.random() * 0.05)
        acc = 0.5 + (0.45 * epoch / total_epochs)
        log_msg = f"Epoch {epoch+1}/{total_epochs}: loss={loss:.4f}, acc={acc:.4f}"
        print(log_msg)
        logger.log(job_id, stdout=log_msg)
        
        # Update metrics
        cpu = 40 + random.random() * 30
        mem = 2048 + random.random() * 512
        logger.update_metrics(job_id, cpu_usage=cpu, memory_usage=mem)

    duration = f"{time.time() - start_time:.2f}s"
    logger.log(job_id, stdout="Training completed successfully.")
    logger.update_status(job_id, JobStatus.COMPLETED, duration=duration)
    print("Job completed")

if __name__ == "__main__":
    run()

