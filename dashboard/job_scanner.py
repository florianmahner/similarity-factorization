"""
Simplified job scanner that reads from ~/.experiment_logs/
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from job_logger import JobLogger, JobStatus, JobType

LOG_DIR = Path.home() / ".experiment_logs"


class JobScanner:
    def __init__(self, logger: JobLogger):
        self.logger = logger
        self.log_dir = LOG_DIR

    def scan_all_experiments(self) -> list[dict[str, Any]]:
        """Scan log files in both ~/.experiment_logs/ and development/*/outputs/"""
        discovered_jobs = []

        # Scan ~/.experiment_logs/ (legacy/production)
        if self.log_dir.exists():
            for date_dir in sorted(self.log_dir.iterdir(), reverse=True):
                if not date_dir.is_dir():
                    continue
                if not re.match(r"^\d{6}$", date_dir.name):
                    continue
                for log_file in sorted(date_dir.glob("*.log"), reverse=True):
                    job = self._parse_log_file(log_file, date_dir.name)
                    if job:
                        discovered_jobs.append(job)

        # Scan development/*/outputs/ (new self-contained modules)
        project_root = Path(__file__).parent.parent
        dev_dir = project_root / "development"
        if dev_dir.exists():
            for exp_dir in dev_dir.iterdir():
                if not exp_dir.is_dir() or exp_dir.name.startswith("_"):
                    continue
                outputs_dir = exp_dir / "outputs"
                if not outputs_dir.exists():
                    continue
                # Scan date folders in outputs
                for date_dir in sorted(outputs_dir.iterdir(), reverse=True):
                    if not date_dir.is_dir():
                        continue
                    # Date format: YYYY-MM-DD
                    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_dir.name):
                        continue
                    # Scan time folders
                    for time_dir in sorted(date_dir.iterdir(), reverse=True):
                        if not time_dir.is_dir():
                            continue
                        # Time format: HH-MM-SS
                        if not re.match(r"^\d{2}-\d{2}-\d{2}$", time_dir.name):
                            continue
                        # Look for run.log
                        log_file = time_dir / "run.log"
                        if log_file.exists():
                            job = self._parse_dev_log_file(log_file, exp_dir.name, date_dir.name, time_dir.name)
                            if job:
                                discovered_jobs.append(job)

        return discovered_jobs

    def _parse_dev_log_file(self, log_path: Path, exp_name: str, date_str: str, time_str: str) -> dict[str, Any] | None:
        """
        Parse development log file from outputs/YYYY-MM-DD/HH-MM-SS/run.log
        Returns job dictionary or None if invalid.
        """
        # Parse timestamp: YYYY-MM-DD_HH-MM-SS -> datetime
        try:
            timestamp_str = f"{date_str}_{time_str}".replace("-", "")
            start_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except ValueError:
            start_time = None

        # Read log file to detect status and get info
        status, end_time, error_msg = self._analyze_log_content(log_path)

        # Calculate duration
        duration = None
        if start_time and end_time:
            duration = (end_time - start_time).total_seconds()

        # Generate job ID
        job_id = f"dev_{exp_name}_{date_str}_{time_str}".replace("-", "")

        # Output directory is the parent of the log file
        output_dir = log_path.parent

        return {
            "id": job_id,
            "name": f"dev/{exp_name}",
            "type": JobType.LOCAL.value,
            "status": status.value,
            "startTime": start_time.isoformat() if start_time else None,
            "endTime": end_time.isoformat() if end_time else None,
            "duration": duration,
            "hydra": {
                "experiment": exp_name,
                "task": "development",
                "outputDir": str(output_dir),
            },
            "logs": {
                "stdout": str(log_path),
                "stderr": str(log_path),
            },
            "logPath": str(log_path),
            "errorMessage": error_msg,
        }

    def _parse_log_file(self, log_path: Path, date_str: str) -> dict[str, Any] | None:
        """
        Parse log filename: HHMMSS_experiment_task.log
        Returns job dictionary or None if invalid.
        """
        # Parse filename: HHMMSS_experiment_task.log
        filename = log_path.stem  # Remove .log
        parts = filename.split("_", 1)

        if len(parts) < 2:
            return None

        time_str = parts[0]  # HHMMSS
        rest = parts[1]  # experiment_task

        # Try to split experiment and task
        rest_parts = rest.split("_")
        if len(rest_parts) >= 2:
            experiment = rest_parts[0]
            task = "_".join(rest_parts[1:])
        else:
            experiment = rest
            task = "unknown"

        # Parse timestamp: YYMMDD_HHMMSS
        try:
            timestamp_str = f"20{date_str}_{time_str}"  # Add century
            start_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except ValueError:
            start_time = None

        # Read log file to detect status and get info
        status, end_time, error_msg = self._analyze_log_content(log_path)

        # Calculate duration
        duration = None
        if start_time and end_time:
            duration = (end_time - start_time).total_seconds()

        # Generate job ID
        job_id = f"{date_str}_{time_str}_{experiment}_{task}"

        return {
            "id": job_id,
            "name": f"{experiment}/{task}",
            "type": JobType.LOCAL.value,
            "status": status.value,
            "startTime": start_time.isoformat() if start_time else None,
            "endTime": end_time.isoformat() if end_time else None,
            "duration": duration,
            "hydra": {
                "experiment": experiment,
                "task": task,
                "outputDir": str(log_path.parent),
            },
            "logs": {
                "stdout": str(log_path),
                "stderr": str(log_path),
            },
            "logPath": str(log_path),
            "errorMessage": error_msg,
        }

    def _analyze_log_content(
        self, log_path: Path
    ) -> tuple[JobStatus, datetime | None, str | None]:
        """
        Analyze log file to determine job status.

        Returns:
            (status, end_time, error_message)
        """
        if not log_path.exists():
            return JobStatus.PENDING, None, None

        try:
            content = log_path.read_text()
        except Exception as e:
            return JobStatus.FAILED, None, f"Could not read log: {e}"

        if not content.strip():
            return JobStatus.RUNNING, None, None

        # Check for errors
        error_keywords = ["ERROR", "FAILED", "Exception", "Traceback", "error:"]
        has_error = any(keyword.lower() in content.lower() for keyword in error_keywords)

        # Try to extract end time from last log line
        end_time = None
        lines = content.strip().split("\n")
        if lines:
            # Try to parse timestamp from last line
            last_line = lines[-1]
            # Format: [YYYY-MM-DD HH:MM:SS] [name] [level] message
            match = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", last_line)
            if match:
                try:
                    end_time = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass

        # Determine status
        if has_error:
            status = JobStatus.FAILED
            # Extract error message (last few lines)
            error_msg = "\n".join(lines[-10:])
        else:
            # Check if job completed (has end marker or recent timestamp)
            if end_time and (datetime.now() - end_time).seconds > 60:
                status = JobStatus.COMPLETED
            else:
                status = JobStatus.RUNNING
            error_msg = None

        return status, end_time, error_msg

    def sync_jobs(self):
        """
        Scan logs and update job registry.
        """
        discovered_jobs = self.scan_all_experiments()

        # Update job logger
        for job in discovered_jobs:
            existing_job = self.logger.get_job(job["id"])

            if existing_job:
                # Update status if changed
                if existing_job["status"] != job["status"]:
                    self.logger.update_job_status(
                        job["id"], JobStatus(job["status"])
                    )
            else:
                # Add new job
                self.logger.add_job(job)

    def _scan_screen_sessions(self) -> list[dict[str, Any]]:
        """Scan for active screen sessions (gemini_*)."""
        sessions = []
        try:
            result = subprocess.run(
                ["screen", "-ls"], capture_output=True, text=True
            )
            output = result.stdout + result.stderr

            for line in output.split("\n"):
                if "gemini_" in line:
                    parts = line.split()
                    if len(parts) >= 1:
                        session_name = parts[0].split(".")[1] if "." in parts[0] else parts[0]
                        if session_name.startswith("gemini_"):
                            # Parse: gemini_experiment_task
                            name_parts = session_name.replace("gemini_", "").split("_", 1)
                            if len(name_parts) >= 2:
                                sessions.append({
                                    "name": session_name,
                                    "experiment": name_parts[0],
                                    "task": name_parts[1],
                                })
        except Exception:
            pass

        return sessions
