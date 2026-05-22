"""Utility functions and modules."""

import os
from pathlib import Path

from src.colors import (
    ROSE, TEAL, CYAN, SAND, PURPLE,
    INDIGO, GREEN, WINE, OLIVE,
    GRAY, GRAY_LIGHT, GRAY_DARK, GRAY_PALE,
    CYCLE, PAIR, TRIPLE, QUAD, FIVE,
    PAIR_ALT1, PAIR_ALT2, PAIR_ALT3,
    CMAP_DIV, CMAP_SEQ, CMAP_IRID, CMAP_GRAY,
    setup_style,
)
from src.utils.figure_theme import (
    SIZES,
    DEFAULT_PAD,
    clean_axis,
    create_figure,
    despine,
    save_figure,
    add_reference_line,
    pad_limits,
)
from src.utils.logging import StatusFileHandler, TeeStream, get_log_path, setup_output_capture


def get_output_dir() -> Path:
    """Get output directory for sandbox scripts.

    Uses SANDBOX_OUTPUT_DIR env var (set by ./scripts/submit) or falls back
    to outputs/ relative to the calling script for direct execution.

    Usage:
        from src.utils import get_output_dir
        OUTPUT_DIR = get_output_dir()
    """
    import inspect

    if env_dir := os.environ.get("SANDBOX_OUTPUT_DIR"):
        output_dir = Path(env_dir)
    else:
        frame = inspect.currentframe()
        caller_file = frame.f_back.f_globals.get("__file__") if frame else None
        if caller_file:
            output_dir = Path(caller_file).parent / "outputs"
        else:
            output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
