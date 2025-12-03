from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from job_logger import JobLogger, JobStatus, JobType


class SlurmMonitor:
    SLURM_TO_DASHBOARD_STATUS = {
        "PENDING": JobStatus.PENDING,
        "RUNNING": JobStatus.RUNNING,
        "COMPLETED": JobStatus.COMPLETED,
        "FAILED": JobStatus.FAILED,
        "CANCELLED": JobStatus.CANCELLED,
        "TIMEOUT": JobStatus.FAILED,
        "OUT_OF_MEMORY": JobStatus.FAILED,
        "NODE_FAIL": JobStatus.FAILED,
    }

    def __init__(self, logger: JobLogger):
        self.logger = logger
        self._check_slurm_available()

    def _check_slurm_available(self) -> bool:
        try:
            result = subprocess.run(
                ["which", "squeue"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.slurm_available = result.returncode == 0
        except Exception:
            self.slurm_available = False

        return self.slurm_available

    def update_slurm_jobs(self) -> dict[str, Any]:
        if not self.slurm_available:
            return {"error": "SLURM not available", "updated": 0}

        jobs = self.logger.get_jobs()
        slurm_jobs = [j for j in jobs if j.get("type") == JobType.SLURM]

        stats = {"updated": 0, "unchanged": 0, "errors": 0}

        for job in slurm_jobs:
            try:
                slurm_job_id = self._extract_slurm_job_id(job)
                if not slurm_job_id:
                    continue

                slurm_status = self._query_slurm_status(slurm_job_id)
                if slurm_status:
                    dashboard_status = self.SLURM_TO_DASHBOARD_STATUS.get(
                        slurm_status, JobStatus.RUNNING
                    )

                    if job.get("status") != dashboard_status:
                        self.logger.update_status(job["id"], dashboard_status)
                        stats["updated"] += 1
                    else:
                        stats["unchanged"] += 1
            except Exception:
                stats["errors"] += 1

        return stats

    def _extract_slurm_job_id(self, job: dict[str, Any]) -> str | None:
        output_dir = job.get("hydra", {}).get("outputDir")
        if not output_dir:
            return None

        output_path = Path(output_dir)
        submitit_dir = output_path / ".submitit"

        if not submitit_dir.exists():
            return None

        for item in submitit_dir.iterdir():
            if item.name.endswith("_submitted.pkl"):
                job_id = item.name.split("_")[0]
                if job_id.isdigit():
                    return job_id

        return None

    def _query_slurm_status(self, job_id: str) -> str | None:
        try:
            result = subprocess.run(
                [
                    "sacct",
                    "-j",
                    job_id,
                    "--format=State",
                    "--noheader",
                    "--parsable2",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return None

            lines = result.stdout.strip().split("\n")
            if lines:
                state = lines[0].strip()
                return state

        except Exception:
            pass

        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()

        except Exception:
            pass

        return None
