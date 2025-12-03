import json
import os
import time
import uuid
import threading
from datetime import datetime
from enum import Enum

# Ensure dashboard directory exists
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_FILE = os.path.join(DASHBOARD_DIR, "jobs.json")

class JobStatus(str, Enum):
    RUNNING = 'RUNNING'
    PENDING = 'PENDING'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'
    CANCELLED = 'CANCELLED'

class JobType(str, Enum):
    LOCAL = 'LOCAL'
    SLURM = 'SLURM'

class JobLogger:
    _lock = threading.Lock()

    def __init__(self):
        self._ensure_jobs_file()

    def _ensure_jobs_file(self):
        if not os.path.exists(JOBS_FILE):
            with open(JOBS_FILE, 'w') as f:
                json.dump([], f)

    def _read_jobs(self):
        with self._lock:
            try:
                with open(JOBS_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return []

    def _write_jobs(self, jobs):
        with self._lock:
            with open(JOBS_FILE, 'w') as f:
                json.dump(jobs, f, indent=2)

    def start_job(self, name, job_type=JobType.LOCAL, command=None, hydra_config=None, slurm_info=None):
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "name": name,
            "type": job_type,
            "status": JobStatus.RUNNING,
            "startTime": datetime.now().isoformat(),
            "command": command,
            "hydra": hydra_config,
            "slurm": slurm_info,
            "logs": {
                "stdout": "",
                "stderr": ""
            },
            "cpuUsage": 0,
            "memoryUsage": 0
        }
        
        jobs = self._read_jobs()
        # Prepend to list so newest is first
        jobs.insert(0, job)
        self._write_jobs(jobs)
        return job_id

    def update_status(self, job_id, status, duration=None):
        jobs = self._read_jobs()
        for job in jobs:
            if job["id"] == job_id:
                job["status"] = status
                if duration:
                    job["duration"] = duration
                self._write_jobs(jobs)
                break

    def log(self, job_id, stdout=None, stderr=None):
        jobs = self._read_jobs()
        for job in jobs:
            if job["id"] == job_id:
                if stdout:
                    job["logs"]["stdout"] += stdout + "\n"
                if stderr:
                    job["logs"]["stderr"] += stderr + "\n"
                self._write_jobs(jobs)
                break

    def update_metrics(self, job_id, cpu_usage=None, memory_usage=None):
        jobs = self._read_jobs()
        for job in jobs:
            if job["id"] == job_id:
                if cpu_usage is not None:
                    job["cpuUsage"] = cpu_usage
                if memory_usage is not None:
                    job["memoryUsage"] = memory_usage
                self._write_jobs(jobs)
                break

    def get_jobs(self):
        return self._read_jobs()

    def delete_job(self, job_id):
        jobs = self._read_jobs()
        jobs = [j for j in jobs if j["id"] != job_id]
        self._write_jobs(jobs)
        return True

    def get_job(self, job_id):
        jobs = self._read_jobs()
        for job in jobs:
            if job["id"] == job_id:
                return job
        return None

    def clear_all_jobs(self):
         with self._lock:
            with open(JOBS_FILE, 'w') as f:
                json.dump([], f)
