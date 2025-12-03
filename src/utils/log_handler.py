"""Custom logging handler that creates parent directories."""

from __future__ import annotations

import logging
from pathlib import Path


class DirectoryFileHandler(logging.FileHandler):
    """FileHandler that creates parent directories if they don't exist."""

    def __init__(self, filename, mode="a", encoding=None, delay=False):
        # Create parent directory if it doesn't exist
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(filename, mode, encoding, delay)
