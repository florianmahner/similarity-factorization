# Color Migration & Experiment Restructuring

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old Flat UI color system with the Tol Muted palette, and co-locate experiment code, configs, and outputs into a unified directory structure.

**Architecture:** Two-phase approach. Phase 1 migrates the color system (mechanical find-and-replace with semantic mapping). Phase 2 restructures directories (move files, update infrastructure scripts, update Hydra config resolution). Colors first because it doesn't change file locations.

**Tech Stack:** Python, matplotlib, Hydra, Poetry

---

## Color Mapping Reference

All tasks in Phase 1 use this mapping. **Note**: The new palette (Tol Muted) has different visual hues than the old one (Flat UI). This is intentional -- the entire palette is changing. Any inline comments referencing old color names (e.g., `# blue`, `# orange`) must be updated or removed during migration.

| Old (figure_theme.py) | New (colors.py) | Visual Change |
|------------------------|-----------------|---------------|
| `CMAP[0]` (red #e74c3c) | `ROSE` (#CC6677) | Bright red -> muted rose |
| `CMAP[1]` (blue #3498db) | `TEAL` (#44AA99) | Blue -> teal-green |
| `CMAP[2]` (green #2ecc71) | `CYAN` (#88CCEE) | Green -> light blue |
| `CMAP[3]` (purple #9b59b6) | `SAND` (#DDCC77) | Purple -> sand-yellow |
| `CMAP[4]` (orange #e67e22) | `PURPLE` (#AA4499) | Orange -> purple |
| `CMAP[5]` (turquoise #1abc9c) | `INDIGO` (#332288) | Turquoise -> indigo |
| `GRAY["dark"]` | `GRAY_DARK` | Similar |
| `GRAY["medium"]` | `GRAY` | Similar |
| `GRAY["light"]` | `GRAY_LIGHT` | Similar |
| `GRAY["faint"]` | `GRAY_PALE` | Similar |
| `HEATMAPS["diverging"]` | `CMAP_DIV` | RdYlBu_r -> Tol BuRd |
| `HEATMAPS["sequential"]` | `CMAP_SEQ` | YlOrRd -> Tol Sunset |
| `apply_theme()` | `setup_style()` | |

When replacing colors, also remove any now-incorrect inline comments like `# blue`, `# red`, etc. The named color variables are self-documenting.

Import pattern change:
```python
# OLD
from src import CMAP, GRAY, HEATMAPS, apply_theme, despine, save_figure
from src.utils import CMAP, GRAY, create_figure, despine, save_figure
from src.utils.figure_theme import CMAP, GRAY, ...

# NEW
from src.colors import ROSE, TEAL, CYAN, SAND, PURPLE, GRAY, GRAY_LIGHT, GRAY_DARK, GRAY_PALE, CMAP_DIV, CMAP_SEQ, setup_style
from src.utils.figure_theme import create_figure, save_figure, despine, clean_axis
# OR via re-exports:
from src import ROSE, TEAL, CYAN, SAND, PURPLE, GRAY, GRAY_LIGHT, setup_style, create_figure, despine, save_figure
```

---

## Phase 1: Color System Migration

### Task 1: Refactor `src/utils/figure_theme.py`

Strip colors from figure_theme.py, keep only figure utilities. Import style setup from colors.py.

**Files:**
- Modify: `src/utils/figure_theme.py`
- Reference: `src/colors.py` (read-only, already correct)

- [ ] **Step 1: Read both files**

Read `src/utils/figure_theme.py` and `src/colors.py` to confirm current state.

- [ ] **Step 2: Rewrite figure_theme.py**

Remove: `CMAP`, `GRAY`, `HEATMAPS`, `BASE_STYLE`, `apply_theme()`, `theme()` context manager.

Keep: `SIZES`, `DEFAULT_PAD`, `create_figure()`, `save_figure()`, `despine()`, `clean_axis()`, `add_reference_line()`, `pad_limits()`.

Replace `apply_theme()` call inside `create_figure()` with `setup_style()` from colors.py.

```python
"""Figure utilities for publication-quality plots."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from src.colors import setup_style

# =============================================================================
# Figure Sizes
# =============================================================================

SIZES = {
    "single": (2.7, 2.2),
    "square": (2.2, 2.2),
    "wide": (3.4, 2.2),
    "full_width": (5.6, 2.2),
}

DEFAULT_PAD = {
    "left": 0.5,
    "right": 0.2,
    "bottom": 0.5,
    "top": 0.1,
}


# =============================================================================
# Figure Helpers
# =============================================================================


def create_figure(
    size: str = "single",
    nrows: int = 1,
    ncols: int = 1,
    pad_left: float | None = None,
    pad_right: float | None = None,
    pad_bottom: float | None = None,
    pad_top: float | None = None,
    **subplot_kw,
) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    """Create figure with fixed size and consistent plotting area."""
    setup_style()

    pl = pad_left if pad_left is not None else DEFAULT_PAD["left"]
    pr = pad_right if pad_right is not None else DEFAULT_PAD["right"]
    pb = pad_bottom if pad_bottom is not None else DEFAULT_PAD["bottom"]
    pt = pad_top if pad_top is not None else DEFAULT_PAD["top"]

    plot_w, plot_h = SIZES[size]
    total_plot_w = plot_w * ncols
    total_plot_h = plot_h * nrows

    fig_w = pl + total_plot_w + pr
    fig_h = pb + total_plot_h + pt

    fig, ax = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), **subplot_kw)

    left_frac = pl / fig_w
    right_frac = (pl + total_plot_w) / fig_w
    bottom_frac = pb / fig_h
    top_frac = (pb + total_plot_h) / fig_h

    fig.subplots_adjust(
        left=left_frac,
        right=right_frac,
        bottom=bottom_frac,
        top=top_frac,
    )

    return fig, ax


def save_figure(fig: plt.Figure, path: Path | str, close: bool = True, tight: bool = True) -> None:
    """Save figure as PDF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix != ".pdf":
        path = path.with_suffix(".pdf")
    if tight:
        fig.savefig(path, format="pdf", bbox_inches="tight")
    else:
        fig.savefig(path, format="pdf")
    if close:
        plt.close(fig)


# =============================================================================
# Axis Helpers
# =============================================================================

def despine(ax: plt.Axes, left: bool = False, bottom: bool = False) -> None:
    """Remove top/right spines, optionally left/bottom."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(False)


def clean_axis(ax: plt.Axes) -> None:
    """Clean axis for heatmaps."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def add_reference_line(
    ax: plt.Axes,
    value: float,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    color: str = "#CCCCCC",  # GRAY_LIGHT from new palette
    linestyle: str = "--",
    linewidth: float = 0.8,
) -> None:
    """Add a reference line."""
    if orientation == "horizontal":
        ax.axhline(value, color=color, linestyle=linestyle, linewidth=linewidth, zorder=0)
    else:
        ax.axvline(value, color=color, linestyle=linestyle, linewidth=linewidth, zorder=0)


def pad_limits(
    data_min: float,
    data_max: float,
    pad_frac: float = 0.05,
) -> tuple[float, float]:
    """Calculate axis limits with breathing room."""
    data_range = data_max - data_min
    if data_range == 0:
        data_range = abs(data_max) * 0.1 if data_max != 0 else 1.0
    pad = data_range * pad_frac
    return data_min - pad, data_max + pad
```

- [ ] **Step 3: Verify no import errors**

Run: `poetry run python -c "from src.utils.figure_theme import create_figure, save_figure, despine"`
Expected: No error

- [ ] **Step 4: Commit**

```bash
git add src/utils/figure_theme.py
git commit -m "refactor: strip colors from figure_theme, keep only figure utilities"
```

---

### Task 2: Update re-exports in `src/__init__.py` and `src/utils/__init__.py`

**Files:**
- Modify: `src/__init__.py`
- Modify: `src/utils/__init__.py`

- [ ] **Step 1: Rewrite `src/__init__.py`**

```python
"""Core utilities."""

from src.colors import (
    ROSE, TEAL, CYAN, SAND, PURPLE,
    INDIGO, GREEN, WINE, OLIVE,
    GRAY, GRAY_LIGHT, GRAY_DARK, GRAY_PALE,
    CYCLE, PAIR, TRIPLE,
    CMAP_DIV, CMAP_SEQ, CMAP_IRID, CMAP_GRAY,
    setup_style,
)
from src.utils.figure_theme import (
    create_figure, save_figure, despine, clean_axis,
    add_reference_line, pad_limits, SIZES,
)
```

- [ ] **Step 2: Rewrite `src/utils/__init__.py`**

Replace color imports with new ones. Keep `get_output_dir`, logging exports.

```python
"""Utility functions and modules."""

import os
from pathlib import Path

from src.colors import (
    ROSE, TEAL, CYAN, SAND, PURPLE,
    INDIGO, GREEN, WINE, OLIVE,
    GRAY, GRAY_LIGHT, GRAY_DARK, GRAY_PALE,
    CYCLE, PAIR, TRIPLE,
    CMAP_DIV, CMAP_SEQ, CMAP_IRID, CMAP_GRAY,
    setup_style,
)
from src.utils.figure_theme import (
    SIZES,
    clean_axis,
    create_figure,
    despine,
    save_figure,
    add_reference_line,
    pad_limits,
)
from src.utils.logging import StatusFileHandler, TeeStream, get_log_path, setup_output_capture


def get_output_dir() -> Path:
    """Get output directory for sandbox scripts."""
    import inspect

    if env_dir := os.environ.get("SANDBOX_OUTPUT_DIR"):
        output_dir = Path(env_dir)
    else:
        frame = inspect.currentframe()
        caller_file = frame.f_back.f_globals.get("__file__") if frame else None
        if caller_file:
            output_dir = Path(caller_file).parent / "outputs" / "dev"
        else:
            output_dir = Path.cwd() / "outputs" / "dev"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
```

- [ ] **Step 3: Verify imports work**

Run: `poetry run python -c "from src import ROSE, TEAL, GRAY, setup_style, create_figure, despine, save_figure"`
Expected: No error

- [ ] **Step 4: Commit**

```bash
git add src/__init__.py src/utils/__init__.py
git commit -m "refactor: update re-exports to use new color system"
```

---

### Task 3: Migrate experiment plotting files (19 files)

Systematic find-and-replace across all experiment files. Each file needs:
1. Import line changed (replace CMAP/GRAY/HEATMAPS/apply_theme with named colors)
2. Color references changed (CMAP[0] -> ROSE, etc.)

**Files to modify** (read each before editing):
- `experiments/estimate_rank.py`
- `experiments/plot_dimensions.py`
- `experiments/plot_mds.py`
- `experiments/plot_rsm.py`
- `experiments/coherence/plot_rank.py`
- `experiments/simulation/plot.py`
- `experiments/simulation/plot_rank_detection.py`
- `experiments/simulation/plot_kappa_rank_detection.py`
- `experiments/swow/plot.py`
- `experiments/ppi/plot.py`
- `experiments/rsa_comparison/plotting.py`
- `experiments/things_behavior/plot.py`
- `experiments/things_behavior/lowdata/plot.py`
- `experiments/things_behavior/lowdata/plot_coherence_comparison.py`
- `paper/Flo-SRF/make_supplementary_figures.py`

**Pattern for each file:**

- [ ] **Step 1: Read file, identify all color references**

For each file, grep for: `CMAP`, `GRAY`, `HEATMAPS`, `apply_theme`, hardcoded hex colors from old palette.

- [ ] **Step 2: Replace imports**

Change import line. Example:
```python
# OLD
from src import CMAP, GRAY, HEATMAPS, apply_theme, despine, save_figure
# NEW
from src.colors import ROSE, TEAL, CYAN, SAND, PURPLE, GRAY, GRAY_LIGHT, GRAY_DARK, GRAY_PALE, CMAP_DIV, CMAP_SEQ, setup_style
from src.utils.figure_theme import create_figure, despine, save_figure
```

Only import the specific colors actually used in that file.

- [ ] **Step 3: Replace color references in code**

Apply the mapping table. Examples:
```python
# OLD → NEW
ax.plot(x, y, color=CMAP[0])          → ax.plot(x, y, color=ROSE)
ax.plot(x, y, color=CMAP[1])          → ax.plot(x, y, color=TEAL)
ax.axhline(0, color=GRAY["light"])    → ax.axhline(0, color=GRAY_LIGHT)
ax.bar(x, y, color=GRAY["faint"])     → ax.bar(x, y, color=GRAY_PALE)
plt.imshow(m, cmap=HEATMAPS["diverging"])  → plt.imshow(m, cmap=CMAP_DIV)
apply_theme()                          → setup_style()
```

Also replace hardcoded duplicates:
```python
BLUE = "#3498db"  → remove, use TEAL
RED = "#e74c3c"   → remove, use ROSE
```

- [ ] **Step 4: Replace CMAP[i % len(CMAP)] patterns**

Some files cycle through colors. Replace with `CYCLE`:
```python
# OLD
color = CMAP[i % len(CMAP)]
# NEW
from src.colors import CYCLE
color = CYCLE[i % len(CYCLE)]
```

- [ ] **Step 5: Verify each file imports correctly**

Run: `poetry run python -c "import experiments.simulation.plot"` (etc. for each module)

- [ ] **Step 6: Commit**

```bash
git add experiments/ paper/
git commit -m "refactor: migrate experiment files to Tol Muted color palette"
```

---

### Task 4: Migrate active sandbox files (~100 files)

Same pattern as Task 3 but for sandbox scripts. Skip `sandbox/_archive/`.

**Directories to process** (each may have multiple .py files with color imports):
- `sandbox/coherence/` (~20 files, including kachun/v2/ subdirectory)
- `sandbox/simulation/` (~45 files)
- `sandbox/things/` (~15 files)
- `sandbox/consensus/` (~4 files)
- `sandbox/swow/` (~4 files)
- `sandbox/ppi/` (~4 files)
- `sandbox/semantic/` (~3 files)
- `sandbox/nsd/` (~4 files)
- `sandbox/monkey/` (~3 files)
- `sandbox/samuel/` (~4 files)

**Approach**: Process one directory at a time. For each directory:

- [ ] **Step 1: Find all files with old color imports**

```bash
grep -rl "from src.*import.*CMAP\|from src.*import.*GRAY\|from src.*import.*HEATMAPS\|from src.*import.*apply_theme" sandbox/<dir>/ --include="*.py"
```

Skip any files under `sandbox/_archive/`.

- [ ] **Step 2: Apply same replacement pattern as Task 3**

For each file: update imports, replace CMAP[i] with named colors, replace GRAY["x"] with GRAY_X, replace apply_theme() with setup_style().

- [ ] **Step 3: Commit per directory group**

```bash
git add sandbox/<dir>/
git commit -m "refactor: migrate sandbox/<dir> to Tol Muted colors"
```

Repeat for each sandbox subdirectory.

---

### Task 5: Run tests and verify

- [ ] **Step 1: Run test suite**

```bash
poetry run pytest tests/ -v
```

Expected: All tests pass (color changes shouldn't affect test logic).

- [ ] **Step 2: Verify a sample plot script**

Pick one experiment with plotting (e.g., `experiments/simulation/plot.py`) and run it to visually confirm colors render.

- [ ] **Step 3: Commit any fixes**

---

## Phase 2: Directory Restructuring

### Task 6: Create new experiment directory structure

Move each Hydra task runner into its own subdirectory with `run.py` naming.

**Nested experiments to restructure:**

| Current | New |
|---------|-----|
| `experiments/ppi/link_prediction.py` | `experiments/ppi/link_prediction/run.py` |
| `experiments/ppi/node_classification.py` | `experiments/ppi/node_classification/run.py` |
| `experiments/ppi/validate_corum.py` | `experiments/ppi/validate_corum/run.py` |
| `experiments/simulation/rank_detection.py` | `experiments/simulation/rank_detection/run.py` |
| `experiments/simulation/imputation.py` | `experiments/simulation/imputation/run.py` |
| `experiments/simulation/interpretability.py` | `experiments/simulation/interpretability/run.py` |
| `experiments/simulation/kappa_rank_detection.py` | `experiments/simulation/kappa_rank_detection/run.py` |
| `experiments/rsa_comparison/factorial.py` | `experiments/rsa_comparison/factorial/run.py` |
| `experiments/rsa_comparison/spose.py` | `experiments/rsa_comparison/spose/run.py` |
| `experiments/things_behavior/coherence.py` | `experiments/things_behavior/coherence/run.py` |
| `experiments/things_behavior/low_data.py` | `experiments/things_behavior/low_data/run.py` |
| `experiments/things_behavior/lowdata_comparison.py` | `experiments/things_behavior/lowdata_comparison/run.py` |
| `experiments/things_behavior/pairwise.py` | `experiments/things_behavior/pairwise/run.py` |
| `experiments/things_behavior/performance48.py` | `experiments/things_behavior/performance48/run.py` |
| `experiments/things_behavior/dimension_reliability.py` | `experiments/things_behavior/dimension_reliability/run.py` |
| `experiments/things_behavior/spose_dimensionality.py` | `experiments/things_behavior/spose_dimensionality/run.py` |
| `experiments/coherence/kappa.py` | `experiments/coherence/kappa/run.py` |
| `experiments/coherence/pct.py` | `experiments/coherence/pct/run.py` |
| `experiments/coherence/plot_rank.py` | `experiments/coherence/plot_rank/run.py` |
| `experiments/bounds/estimate_bounds.py` | `experiments/bounds/estimate_bounds/run.py` |
| `experiments/swow/generate_embedding.py` | `experiments/swow/generate_embedding/run.py` |
| `experiments/swow/predict_behavioral_properties.py` | `experiments/swow/predict_behavioral_properties/run.py` |
| `experiments/graph_clustering/benchmark.py` | `experiments/graph_clustering/benchmark/run.py` |

**Flat experiments to restructure:**

| Current | New |
|---------|-----|
| `experiments/consensus.py` | `experiments/consensus/run.py` |
| `experiments/estimate_rank.py` | `experiments/estimate_rank/run.py` |
| `experiments/plot_dimensions.py` | `experiments/plot_dimensions/run.py` |
| `experiments/plot_mds.py` | `experiments/plot_mds/run.py` |
| `experiments/plot_rsm.py` | `experiments/plot_rsm/run.py` |
| `experiments/test_job.py` | `experiments/test_job/run.py` |

**Shared utility modules stay at domain level:**
- `experiments/ppi/models.py`, `utils.py`, `baselines.py`, `evaluators.py`, `embeddings.py`, `corum.py`, `seal_link_pred.py`, `node_classification_full.py` stay at `experiments/ppi/`
- `experiments/things_behavior/common.py`, `resources.py`, `utils.py`, `summary_stats.py` stay at `experiments/things_behavior/`
- `experiments/things_behavior/lowdata/` subpackage stays at `experiments/things_behavior/lowdata/`
- `experiments/swow/data.py`, `embedding.py`, `ppmi.py`, `semantic_helpers.py`, `validation.py`, `plotting.py` stay at `experiments/swow/`
- `experiments/bounds/bounds_missing.py` stays at `experiments/bounds/`
- `experiments/graph_clustering/graph.py`, `embeddings.py` stay at `experiments/graph_clustering/`

**Plotting utilities**: Each domain's `plot.py`/`plotting.py` stays at domain level (shared across tasks).

**Preprocessing stays as-is**: `experiments/preprocessing/monkey_2k/` uses argparse (not Hydra) and is not restructured. Its step scripts remain standalone.

**Task runners without existing configs** (3 files): These Hydra tasks currently have no config yaml. Create minimal configs for them during Task 7:
- `experiments/simulation/kappa_rank_detection/config.yaml` (create new)
- `experiments/ppi/validate_corum/config.yaml` (create new)
- `experiments/swow/generate_embedding/config.yaml` (create new)

**Orphan configs** (no corresponding task runner): Move to an `_unused/` subfolder or delete after confirming with user:
- `configs/experiment/rsa_comparison/gaussian_tuning.yaml`
- `configs/experiment/simulation/rank_detection_n300.yaml`

- [ ] **Step 1: Create directories and move task runners**

For each task runner:
```bash
mkdir -p experiments/<domain>/<task>/
git mv experiments/<domain>/<task>.py experiments/<domain>/<task>/run.py
```

For flat experiments:
```bash
mkdir -p experiments/<task>/
git mv experiments/<task>.py experiments/<task>/run.py
```

- [ ] **Step 2: Add `__init__.py` to new task directories and any domain directories missing them**

Each new task directory needs an empty `__init__.py` so Python module imports work (the `run_task.py` does `importlib.import_module`). Also check that domain-level directories have `__init__.py` (e.g., `experiments/bounds/`, `experiments/coherence/`).

- [ ] **Step 3: Update relative imports in moved files**

Task runners that import from sibling utility modules need path adjustment:
```python
# OLD (when link_prediction.py was at experiments/ppi/link_prediction.py)
from experiments.ppi.utils import load_data
from experiments.ppi.models import SRFPredictor

# NEW (now at experiments/ppi/link_prediction/run.py)
from experiments.ppi.utils import load_data       # Same - absolute imports still work
from experiments.ppi.models import SRFPredictor    # Same - still correct
```

**Important**: Since these use absolute imports (not relative `.` imports), the imports should still work after moving one level deeper. Verify for each moved file.

Cross-domain imports also still work:
```python
# In experiments/things_behavior/coherence/run.py
from experiments.bounds.bounds_missing import estimate_bounds  # Still correct
```

- [ ] **Step 4: Commit**

```bash
git add experiments/
git commit -m "refactor: move task runners into subdirectories with run.py pattern"
```

---

### Task 7: Move experiment configs to co-located config.yaml

Move each experiment-specific config next to its run.py.

**Nested experiment configs:**

| Current | New |
|---------|-----|
| `configs/experiment/ppi/link_prediction.yaml` | `experiments/ppi/link_prediction/config.yaml` |
| `configs/experiment/ppi/node_classification.yaml` | `experiments/ppi/node_classification/config.yaml` |
| `configs/experiment/simulation/rank_detection.yaml` | `experiments/simulation/rank_detection/config.yaml` |
| `configs/experiment/simulation/imputation.yaml` | `experiments/simulation/imputation/config.yaml` |
| `configs/experiment/simulation/interpretability.yaml` | `experiments/simulation/interpretability/config.yaml` |
| ... (all others follow same pattern) |

**Flat experiment configs:**

| Current | New |
|---------|-----|
| `configs/consensus.yaml` | `experiments/consensus/config.yaml` |
| `configs/estimate_rank.yaml` | `experiments/estimate_rank/config.yaml` |
| `configs/estimate_bounds.yaml` | `experiments/bounds/estimate_bounds/config.yaml` |
| `configs/plot_dimensions.yaml` | `experiments/plot_dimensions/config.yaml` |
| `configs/plot_mds.yaml` | `experiments/plot_mds/config.yaml` |
| `configs/plot_rsm.yaml` | `experiments/plot_rsm/config.yaml` |
| `configs/test_job.yaml` | `experiments/test_job/config.yaml` |

**Note**: If both `configs/estimate_bounds.yaml` (flat) and `configs/experiment/bounds/estimate_bounds.yaml` (nested) exist, check if they are duplicates. Keep the one that matches the task runner's config expectations and delete the other.

**Shared configs stay in `configs/`:**
- `configs/base.yaml`
- `configs/dataset/*.yaml`
- `configs/paths/local.yaml`
- `configs/hydra/*.yaml`

- [ ] **Step 1: Move config files**

```bash
# Nested
git mv configs/experiment/ppi/link_prediction.yaml experiments/ppi/link_prediction/config.yaml
# ... repeat for all

# Flat
git mv configs/consensus.yaml experiments/consensus/config.yaml
# ... repeat for all
```

- [ ] **Step 2: Rename config references inside moved files**

The configs themselves don't reference their own filename, but check for any `config_name` references.

- [ ] **Step 3: Remove empty `configs/experiment/` directories**

After moving all experiment configs, the `configs/experiment/` tree should be empty. List what remains and ask user for confirmation before deleting.

```bash
find configs/experiment/ -type f 2>/dev/null  # Show any remaining files
# Only delete after user confirms:
# rm -rf configs/experiment/
```

- [ ] **Step 4: Commit**

```bash
git add experiments/ configs/
git commit -m "refactor: co-locate experiment configs next to task code"
```

---

### Task 8: Move sandbox under experiments/

- [ ] **Step 1: Move sandbox directory**

```bash
git mv sandbox experiments/sandbox
```

- [ ] **Step 2: Update any absolute imports referencing sandbox**

Search for imports like `from sandbox.` or references to `sandbox/` paths. These should be rare since sandbox scripts are standalone.

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "refactor: move sandbox under experiments/"
```

---

### Task 9: Update `configs/base.yaml` for new output paths

The Hydra output directory needs to point to the experiment's local `outputs/` directory.

**Files:**
- Modify: `configs/base.yaml`

- [ ] **Step 1: Read current base.yaml**

- [ ] **Step 2: Update Hydra output dir**

The submit script will pass the output dir as an override, so base.yaml just needs a sensible default that can be overridden:

```yaml
hydra:
  job:
    chdir: true
    name: log
  run:
    dir: ${project_root}/outputs/experiments/${experiment_name}/${task}
  sweep:
    dir: ${project_root}/outputs/experiments/${experiment_name}/${task}
    subdir: ${hydra.job.override_dirname}
```

Change to a fallback default that still works if someone runs `run_task.py` directly (without submit):

```yaml
hydra:
  job:
    chdir: true
    name: log
  run:
    dir: ${project_root}/experiments/${experiment_name}/${task}/outputs
  sweep:
    dir: ${project_root}/experiments/${experiment_name}/${task}/outputs
    subdir: ${hydra.job.override_dirname}
```

The submit script overrides this with the exact path via `hydra.run.dir=<task_dir>/outputs`. The default is a reasonable fallback for direct invocation.

- [ ] **Step 3: Commit**

```bash
git add configs/base.yaml
git commit -m "refactor: require output dir from submit script in base config"
```

---

### Task 10: Update `scripts/submit`

The submit script needs to:
1. Find `config.yaml` next to the script (not in `configs/experiment/`)
2. Set output dir to `<experiment_dir>/outputs/`
3. Detect sandbox as `experiments/sandbox/` instead of `sandbox/`

**Files:**
- Modify: `scripts/submit`

- [ ] **Step 1: Read current submit script**

- [ ] **Step 2: Update `is_sandbox_script()`**

```python
def is_sandbox_script(script_path: Path) -> bool:
    """Check if script is in sandbox directory (now under experiments/)."""
    parts = script_path.parts
    return "experiments" in parts and "sandbox" in parts
```

- [ ] **Step 3: Update experiment detection and config resolution**

The key change: instead of computing config path from experiment_name/task_name mapping to `configs/experiment/<exp>/<task>.yaml`, look for `config.yaml` in the script's parent directory.

```python
# In the Hydra section of main():

# Detect experiment structure:
# experiments/<task>/run.py → experiments/<task>/config.yaml
# experiments/<exp>/<task>/run.py → experiments/<exp>/<task>/config.yaml
parts = original_script.parts
if "experiments" in parts:
    exp_idx = parts.index("experiments")
    remaining = parts[exp_idx + 1:]

    # The script is always run.py inside a task directory
    # Task dir is the script's parent
    task_dir = original_script.parent
    config_file = task_dir / "config.yaml"

    # Determine experiment_name and task_name from path
    # experiments/<task>/run.py → exp=task, task=task
    # experiments/<domain>/<task>/run.py → exp=domain, task=task
    remaining_dirs = [p for p in remaining if not p.endswith(".py")]
    if len(remaining_dirs) == 1:
        experiment_name = remaining_dirs[0]
        task_name = remaining_dirs[0]
    elif len(remaining_dirs) >= 2:
        experiment_name = remaining_dirs[0]
        task_name = remaining_dirs[-1]
```

- [ ] **Step 4: Update Hydra command construction**

```python
# Config is now next to the script
config_path = task_dir.resolve()
config_name = "config"

# Output dir is task_dir/outputs/
output_dir = task_dir / "outputs"

cmd = [
    "poetry", "run", "python",
    str(project_root / "scripts" / "run_task.py"),
    f"--config-path={config_path}",
    f"--config-name={config_name}",
    f"hydra.searchpath=[file://{project_root}/configs]",
    f"project_root={project_root}",
    f"+_module_path={module_path}",
    f"hydra.run.dir={output_dir}",
    f"hydra.sweep.dir={output_dir}",
]
```

- [ ] **Step 5: Update module path computation**

```python
# Module path from file structure
# experiments/<domain>/<task>/run.py → experiments.<domain>.<task>.run
# experiments/<task>/run.py → experiments.<task>.run
task_parts = list(remaining_dirs)
module_path = "experiments." + ".".join(task_parts) + ".run"
```

- [ ] **Step 6: Update sandbox output dir and session name**

Sandbox detection now uses `experiments/sandbox/` prefix. The output dir logic stays the same (timestamped under the script's parent `outputs/` dir).

```python
if is_sandbox:
    sandbox_idx = parts.index("sandbox")
    sandbox_parts = parts[sandbox_idx + 1:-1]
    session_name = "sandbox_" + "_".join(sandbox_parts)
```

- [ ] **Step 7: Update cleanup path**

```python
# Cleanup empty run folders -- no longer a single outputs/ dir
# Skip cleanup or make it per-experiment
```

- [ ] **Step 8: Verify with a dry-run**

```bash
poetry run python scripts/submit experiments/simulation/rank_detection/run.py --help
```

- [ ] **Step 9: Commit**

```bash
git add scripts/submit
git commit -m "refactor: update submit script for co-located experiment structure"
```

---

### Task 11: Update `scripts/run_task.py` module import paths

The module import paths change because task runners are now at `experiments/<domain>/<task>/run.py`.

**Files:**
- Modify: `scripts/run_task.py`

- [ ] **Step 1: Read current run_task.py**

- [ ] **Step 2: Update module resolution fallback**

```python
def _run_task(cfg: DictConfig, exp_name: str, task_name: str) -> None:
    module_path = cfg.get("_module_path")
    if module_path:
        module_paths = [module_path]
    else:
        # New structure: experiments/<domain>/<task>/run
        module_paths = [
            f"experiments.{exp_name}.{task_name}.run",
            f"experiments.{task_name}.run",
        ]
    # ... rest unchanged
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_task.py
git commit -m "refactor: update run_task module resolution for new structure"
```

---

### Task 12: Update `scripts/dash` and `scripts/jobs`

These scripts scan for `.status/job.json` files to find running/completed jobs. They need to search the new locations.

**Files:**
- Modify: `scripts/dash`
- Modify: `scripts/jobs`

- [ ] **Step 1: Read both scripts**

Identify where they scan for job.json files. They likely look under `outputs/` and `sandbox/`.

- [ ] **Step 2: Update scan paths**

Change scan paths to include `experiments/` (for both Hydra task outputs and sandbox outputs). Also keep scanning `outputs/` for historical jobs that haven't been moved.

Old scan paths (currently both scripts only scan `PROJECT_ROOT / "outputs"`):
- `outputs/`

New scan paths:
- `experiments/**/outputs/` (new location for all jobs)
- `outputs/` (backward compat for historical runs, can be removed later)

- [ ] **Step 3: Commit**

```bash
git add scripts/dash scripts/jobs
git commit -m "refactor: update dashboard and jobs CLI for new experiment paths"
```

---

### Task 13: Update CLAUDE.md

Update the project documentation to reflect the new structure.

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update directory structure section**

Replace the old structure with the new one. Update all path references, command examples, and the submit usage section.

Key changes:
- Directory structure diagram shows co-located experiments
- `configs/` section only shows shared configs
- Sandbox section shows `experiments/sandbox/`
- Color palette section uses new Tol Muted names
- Figure theme section updated (no more CMAP list, use named colors)

- [ ] **Step 2: Update color documentation**

Replace all `CMAP[0]`, `GRAY["dark"]` references with `ROSE`, `GRAY_DARK`, etc.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for new color system and experiment structure"
```

---

### Task 14: Integration test

- [ ] **Step 1: Run test suite**

```bash
poetry run pytest tests/ -v
```

- [ ] **Step 2: Test submit script with a fast experiment**

```bash
./scripts/submit experiments/test_job/run.py
```

Verify:
- Config loads from `experiments/test_job/config.yaml`
- Output goes to `experiments/test_job/outputs/`
- Job tracking works (`.status/job.json` created)

- [ ] **Step 3: Test sandbox submission**

```bash
./scripts/submit experiments/sandbox/simulation/rank_selection/run.py --bg
```

Verify:
- Timestamped output dir created under `experiments/sandbox/simulation/rank_selection/outputs/`
- Job appears in `dash`

- [ ] **Step 4: Test dashboard**

```bash
dash
```

Verify: Both Hydra experiment jobs and sandbox jobs appear.

- [ ] **Step 5: Fix any issues, commit**

---

### Task 15: Cleanup

- [ ] **Step 1: Remove old empty directories**

After all moves, these should be empty:
```bash
# Check what's left
ls configs/experiment/ 2>/dev/null
ls outputs/experiments/ 2>/dev/null
ls sandbox/ 2>/dev/null
```

Remove if empty (ask user first for outputs/ since it may contain historical runs).

- [ ] **Step 2: Remove old top-level config files that were moved**

Verify all flat experiment configs were moved:
```bash
ls configs/*.yaml  # Should only show base.yaml and shared configs
```

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "chore: clean up empty directories after restructuring"
```
