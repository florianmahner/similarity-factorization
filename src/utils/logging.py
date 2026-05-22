"""Logging utilities for job tracking."""

import logging
import sys
from pathlib import Path


class StatusFileHandler(logging.FileHandler):
    """FileHandler that creates parent directories and writes to .status/log.

    Ensures the .status/ directory exists before opening the log file.
    Used by Hydra's job_logging config.
    """

    def __init__(self, filename: str = ".status/log", mode: str = "a", encoding: str = "utf-8"):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(filename, mode=mode, encoding=encoding)


class TeeStream:
    """Stream that writes to both original stream and a file.

    Used to capture stdout/stderr to .status/log while preserving console output.
    """

    def __init__(self, original_stream, log_path: Path):
        self.original = original_stream
        self.log_path = log_path
        self._file = None

    def _ensure_file(self):
        if self._file is None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.log_path, "a", encoding="utf-8")

    def write(self, data: str):
        if data:
            self._ensure_file()
            self._file.write(data)
            self._file.flush()
        if self.original is not None:
            self.original.write(data)

    def flush(self):
        if self._file is not None:
            self._file.flush()
        if self.original is not None:
            self.original.flush()

    def fileno(self):
        if self.original is not None:
            return self.original.fileno()
        raise OSError("No file descriptor available")

    def isatty(self):
        if self.original is not None:
            return self.original.isatty()
        return False

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def setup_output_capture(log_path: Path | None = None):
    """Redirect stdout/stderr to .status/log while preserving console output.

    Call this early in run_task.py to capture all output (including progress bars,
    prints, and non-logging output) to the status log file.

    Args:
        log_path: Path to log file. Defaults to .status/log in cwd.
    """
    if log_path is None:
        log_path = Path.cwd() / ".status" / "log"

    log_path.parent.mkdir(parents=True, exist_ok=True)

    sys.stdout = TeeStream(sys.stdout, log_path)
    sys.stderr = TeeStream(sys.stderr, log_path)


def get_log_path() -> Path:
    """Get the standard log path for the current output directory."""
    return Path.cwd() / ".status" / "log"
