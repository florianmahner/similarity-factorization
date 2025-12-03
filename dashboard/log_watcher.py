from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator

from job_logger import JobLogger, JobStatus


class LogWatcher:
    def __init__(self, logger: JobLogger):
        self.logger = logger
        self._file_positions: dict[str, int] = {}

    async def watch_log(
        self, job_id: str, poll_interval: float = 1.0
    ) -> AsyncGenerator[str, None]:
        job = self.logger.get_job(job_id)
        if not job:
            yield "Error: Job not found\n"
            return

        # Get log path directly from job
        log_path = job.get("logPath")
        if not log_path:
            yield "Error: No log path found\n"
            return

        log_file = Path(log_path)

        if not log_file.exists():
            yield f"Waiting for log file: {log_file.name}\n"

        position = self._file_positions.get(job_id, 0)

        while True:
            if not log_file.exists():
                await asyncio.sleep(poll_interval)
                continue

            try:
                with open(log_file) as f:
                    f.seek(position)
                    new_content = f.read()
                    new_position = f.tell()

                    if new_content:
                        yield new_content
                        position = new_position
                        self._file_positions[job_id] = position

            except Exception as e:
                yield f"Error reading log: {e}\n"

            job = self.logger.get_job(job_id)
            if not job:
                break

            if job.get("status") in [
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ]:
                break

            await asyncio.sleep(poll_interval)

    def get_full_log(self, job_id: str) -> str:
        job = self.logger.get_job(job_id)
        if not job:
            return "Error: Job not found"

        output_dir = job.get("hydra", {}).get("outputDir")
        if not output_dir:
            return "Error: No output directory found"

        log_file = Path(output_dir) / "run.log"

        if not log_file.exists():
            return f"Log file not found: {log_file}"

        try:
            with open(log_file) as f:
                return f.read()
        except Exception as e:
            return f"Error reading log: {e}"

    def tail_log(self, job_id: str, n_lines: int = 100) -> str:
        full_log = self.get_full_log(job_id)
        if full_log.startswith("Error:"):
            return full_log

        lines = full_log.split("\n")
        return "\n".join(lines[-n_lines:])
