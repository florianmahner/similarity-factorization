from __future__ import annotations
import sys
import time
import hydra
from pathlib import Path
from omegaconf import DictConfig

def run(cfg: DictConfig) -> None:
    # Add dashboard to path
    # We are in development/demo
    # root is up 2 levels
    root_dir = Path(hydra.utils.get_original_cwd()).parent.parent
    sys.path.append(str(root_dir / "dashboard"))
    
    try:
        from job_logger import JobLogger, JobType, JobStatus
    except ImportError:
        print(f"Could not import JobLogger. Root dir: {root_dir}. Dashboard logging disabled.")
        return

    logger = JobLogger()
    
    # Determine job name
    task_type = cfg.get("type", "unknown")
    job_name = f"Demo: {task_type.capitalize()}"
    
    job_id = logger.start_job(
        name=job_name,
        job_type=JobType.LOCAL,
        command=f"python run.py type={task_type}",
        hydra_config={
            "configName": "demo",
            "overrides": [f"type={task_type}"],
            "outputDir": str(Path.cwd())
        }
    )
    
    logger.log(job_id, stdout=f"Started Hydra task in {Path.cwd()}")
    
    try:
        if task_type == "long_running":
            logger.log(job_id, stdout="Starting long running process (waiting for cancellation)...")
            i = 0
            while True:
                time.sleep(2)
                i += 1
                logger.log(job_id, stdout=f"Working... step {i}")
                # Updates metrics
                import random
                logger.update_metrics(job_id, cpu_usage=random.uniform(10, 20), memory_usage=random.uniform(200, 400))
                
        elif task_type == "failure":
            logger.log(job_id, stdout="Initializing...")
            time.sleep(3)
            logger.log(job_id, stdout="Critical error encountered.")
            error_msg = "ValueError: Invalid input data shape (3, 256, 256) expected (1, 256, 256)"
            logger.log(job_id, stderr=error_msg)
            raise ValueError(error_msg)
            
        elif task_type == "success":
            logger.log(job_id, stdout="Processing data...")
            for i in range(5):
                time.sleep(1)
                logger.log(job_id, stdout=f"Batch {i+1}/5 processed.")
            logger.log(job_id, stdout="Done.")
            
        else:
            logger.log(job_id, stderr=f"Unknown task type: {task_type}")
            return

        logger.update_status(job_id, JobStatus.COMPLETED, duration="Done")
        
    except Exception as e:
        logger.update_status(job_id, JobStatus.FAILED, duration="Failed")
        logger.log(job_id, stderr=str(e))
        raise e

