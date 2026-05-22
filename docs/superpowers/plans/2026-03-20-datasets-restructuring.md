# Datasets Experiment Restructuring + Pipeline Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `experiments/datasets/` into a clean pipeline: kappa rank estimation → cross-validation → consensus embedding → visualization. Flatten unnecessary nesting, merge plotting into task folders, archive old bounds.

**Architecture:** Each task folder follows `run.py` (Hydra) + optional `plot.py` (standalone) + `config.yaml` + `outputs/`. Pipeline steps read upstream outputs via relative paths. The pipeline flows: `ranks/kappa/ → ranks/cv/ → consensus/ → visualize/`.

**Tech Stack:** Python, Hydra, pysrf (SRF, cross_val_score, EnsembleEmbedding, AlignedConsensus), matplotlib/seaborn, PIL.

---

## Target Structure

```
experiments/datasets/
├── ranks/
│   ├── kappa/                  # Step 1: kappa coherence → k* per dataset
│   │   ├── run.py              # (existing, moved from coherence/kappa/)
│   │   ├── plot.py             # NEW: merged from plot_rank + plot_method_comparison
│   │   ├── config.yaml
│   │   ├── __init__.py
│   │   └── outputs/            # <dataset>.json (k*), <dataset>.npz (arrays)
│   ├── pct/                    # Alt method: PCT → k* per dataset
│   │   ├── run.py              # (existing, moved from coherence/pct/)
│   │   ├── plot.py             # NEW: PCT-specific plots
│   │   ├── config.yaml
│   │   ├── __init__.py
│   │   └── outputs/            # <dataset>.json, <dataset>.npz
│   ├── cv/                     # Step 2: cross-validation around k* → optimal rank
│   │   ├── run.py              # NEW: reads ../kappa/outputs/, estimates bounds inline
│   │   ├── config.yaml         # NEW
│   │   ├── __init__.py
│   │   └── outputs/            # <dataset>/rank_estimation.json, cv_results.csv, cv_curve.pdf
│   └── compare/                # Cross-method comparison plots
│       ├── plot.py             # NEW: reads ../kappa/ + ../pct/ + ../cv/
│       ├── __init__.py
│       └── outputs/
├── consensus/                  # Step 3: consensus embedding using rank from cv/
│   ├── run.py                  # UPDATED: reads ../ranks/cv/outputs/ for rank
│   ├── config.yaml
│   ├── __init__.py
│   └── outputs/                # <dataset>/embedding.npy, runs.npy, summary.json
└── visualize/                  # Step 4: top-k image grids per dimension
    ├── plot.py                 # NEW: adapted from experiments/figures/plot_dimensions/
    ├── __init__.py
    └── outputs/                # <dataset>/topk_images.pdf, dims/dim_01.png, ...
```

## Pipeline Data Flow

```
ranks/kappa/outputs/<ds>.json     →  ranks/cv/run.py reads k_star_kappa
                                      ↓
                                   ranks/cv/outputs/<ds>/rank_estimation.json  →  consensus/run.py reads optimal_rank
                                                                                  ↓
                                                                               consensus/outputs/<ds>/embedding.npy  →  visualize/plot.py
```

## Key Design Decisions

1. **`plot.py` = standalone scripts** (not Hydra). Use `Path(__file__).parent / "outputs"` for I/O. Run via `./scripts/submit <path>/plot.py`.
2. **Bounds computed inline** in `cv/run.py` via `estimate_sampling_bounds_fast()` (NOT `_ultra` which doesn't exist). The old `ranks/bounds/` folder is archived (outputs exist, code preserved in archive). This avoids a stale dependency.
3. **CV rank grid centered on kappa's k\***: for each dataset, build `range(max(step, k*-n*step), k*+n*step+1, step)` where `step` comes from dataset config's `rank_range[2]`. Falls back to full `rank_range` if kappa output missing. The kappa output filename may differ from the Hydra dataset name (e.g., kappa uses `nsd_subj01` but Hydra dataset is `nsd` + `subject_id=1`). The CV script must map between these conventions.
4. **Consensus saves**: `embedding.npy` (final consensus, n×k), `runs.npy` (all ensemble runs, n_runs×n×k), `summary.json` (metadata + agreement scores). No pipeline.joblib — it's large and rarely needed.
5. **Visualize** adapted from `experiments/figures/plot_dimensions/run.py` — uses `datasets.load_dataset()` for labels/images, reads embedding from `../consensus/outputs/`.
6. **Old directories archived**: `ranks/bounds/`, `ranks/coherence/`, `experiments/figures/plot_dimensions/` remain functional but the datasets pipeline is self-contained.
7. **`__init__.py` at all levels**: Ensure `experiments/datasets/__init__.py` and `experiments/datasets/ranks/__init__.py` exist for consistent package structure.
8. **CV outputs use per-dataset subdirectories**: `cv/run.py` writes to `outputs/<dataset_name>/` (not the Hydra cwd root), so consensus can find results by dataset name.
9. **Kappa name mapping for CV**: Kappa outputs use flat names like `nsd_subj01`, `things_100pct`. The CV script maps from Hydra's `(dataset.name, subject_id)` → kappa filename using: `f"{ds_name}_subj{subject_id:02d}"` for multi-subject, and a `kappa_name` config override for THINGS splits (e.g., `kappa_name=things_100pct`).
10. **`apply_theme()` removed**: This function doesn't exist in current `figure_theme.py`. Remove any calls when adapting plot code.
11. **`visualize/plot.py` uses `--` style args**: Since `./scripts/submit` passes extra args verbatim, use `--dataset mur92 --k 10` style (not Hydra `key=val` style) for standalone argparse scripts.

## Reference: Existing Code to Reuse

- **CV logic**: `experiments/estimate_rank.py` (in `.worktrees/coherence-extraction/`) — `run()` function with `cross_val_score()`, bounds loading, CV curve plotting. Copy and adapt.
- **Consensus logic**: `experiments/datasets/consensus/run.py` — update to read rank from `../ranks/cv/outputs/`.
- **Visualization logic**: `experiments/figures/plot_dimensions/run.py` — `_plot_topk_images()`, `_plot_individual_dimensions()`, `_plot_topk_text()`. Copy and adapt.
- **pysrf API**: `cross_val_score(similarity, param_grid={"rank": grid}, n_repeats=5, sampling_fraction=p, n_jobs=-1)`

---

### Task 1: Flatten coherence — move kappa and pct to ranks/

**Files:**
- Move: `ranks/coherence/kappa/{run.py,config.yaml,__init__.py,outputs/}` → `ranks/kappa/`
- Move: `ranks/coherence/pct/{run.py,config.yaml,__init__.py,outputs/}` → `ranks/pct/`

- [ ] **Step 1: Copy kappa and pct to new locations**

```bash
cd /LOCAL/fmahner/similarity-factorization/experiments/datasets

mkdir -p ranks/kappa ranks/pct

cp ranks/coherence/kappa/run.py ranks/kappa/run.py
cp ranks/coherence/kappa/config.yaml ranks/kappa/config.yaml
cp ranks/coherence/kappa/__init__.py ranks/kappa/__init__.py
cp -r ranks/coherence/kappa/outputs ranks/kappa/outputs

cp ranks/coherence/pct/run.py ranks/pct/run.py
cp ranks/coherence/pct/config.yaml ranks/pct/config.yaml
cp ranks/coherence/pct/__init__.py ranks/pct/__init__.py
cp -r ranks/coherence/pct/outputs ranks/pct/outputs

# Ensure intermediate __init__.py files exist
touch __init__.py        # experiments/datasets/__init__.py
touch ranks/__init__.py  # experiments/datasets/ranks/__init__.py
```

- [ ] **Step 2: Update config.yaml experiment_name in both**

`ranks/kappa/config.yaml`: change `experiment_name: coherence` → `experiment_name: ranks`
`ranks/pct/config.yaml`: change `experiment_name: coherence` → `experiment_name: ranks`

- [ ] **Step 3: Update usage comments in run.py docstrings**

`ranks/kappa/run.py` line 12: `./scripts/submit experiments/datasets/ranks/kappa/run.py --bg`
`ranks/pct/run.py` line 8: `./scripts/submit experiments/datasets/ranks/pct/run.py --bg`

- [ ] **Step 4: Commit**

```bash
git add experiments/datasets/ranks/kappa/ experiments/datasets/ranks/pct/
git commit -m "$(cat <<'EOF'
refactor: flatten kappa and pct into ranks/

Remove unnecessary coherence/ nesting layer.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Create kappa/plot.py

**Files:**
- Create: `ranks/kappa/plot.py`
- Source: `ranks/coherence/plot_rank/run.py` + `ranks/coherence/plot_method_comparison/run.py`

Standalone script. Reads from and writes to `outputs/`.

Functions to include:
- `plot_kstar_comparison` — bar chart of k* across datasets (from plot_rank)
- `plot_things_kstar_vs_data` — THINGS k* vs triplet % (from plot_rank)
- `plot_coherence_curves` — Iproj(p) per dataset (from plot_rank)
- `plot_eigenvalue_spectra` — eigenvalue overlay (from plot_rank)
- `plot_kappa_overlay` — normalized kappa curves (from plot_method_comparison)

- [ ] **Step 1: Write `ranks/kappa/plot.py`**

Structure:
```python
"""Plot kappa-based rank estimation results.

Reads JSON/NPZ from outputs/ and produces figures there.

Usage:
    ./scripts/submit experiments/datasets/ranks/kappa/plot.py
"""
from pathlib import Path
import json, logging, numpy as np, matplotlib.pyplot as plt, pandas as pd
from src.colors import ROSE, TEAL, CYAN, GRAY
from src.utils.figure_theme import CMAP, create_figure, despine, save_figure

log = logging.getLogger(__name__)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# ... _load_results, _load_npz, plot functions (adapted from originals) ...
# All plot functions take output_dir=OUTPUT_DIR as default
# Key change: replace all cfg.project_root path construction with OUTPUT_DIR

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    records = _load_results(OUTPUT_DIR)
    plot_kstar_comparison(records, OUTPUT_DIR)
    plot_things_kstar_vs_data(records, OUTPUT_DIR)
    plot_coherence_curves(records, OUTPUT_DIR, OUTPUT_DIR)
    plot_eigenvalue_spectra(records, OUTPUT_DIR, OUTPUT_DIR)
    plot_kappa_overlay(records, OUTPUT_DIR, OUTPUT_DIR)
```

- [ ] **Step 2: Verify**

```bash
./scripts/submit experiments/datasets/ranks/kappa/plot.py
ls experiments/datasets/ranks/kappa/outputs/*.pdf
```

- [ ] **Step 3: Commit**

```bash
git add experiments/datasets/ranks/kappa/plot.py
git commit -m "$(cat <<'EOF'
feat: add kappa/plot.py — self-contained kappa visualization

Merges plot_rank and kappa overlay into kappa task folder.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Create pct/plot.py

**Files:**
- Create: `ranks/pct/plot.py`
- Source: `ranks/coherence/plot_method_comparison/run.py` (PCT functions)

- [ ] **Step 1: Write `ranks/pct/plot.py`**

Functions:
- `plot_pct_overlay` — p-value curves all datasets (from plot_method_comparison `plot_pct_overlay`)
- `plot_pct_per_dataset` — per-dataset p-value scatter (adapted from `_plot_pct_panel`)

Same standalone pattern. Reads from `OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"`.

- [ ] **Step 2: Verify**

```bash
./scripts/submit experiments/datasets/ranks/pct/plot.py
```

- [ ] **Step 3: Commit**

```bash
git add experiments/datasets/ranks/pct/plot.py
git commit -m "$(cat <<'EOF'
feat: add pct/plot.py — self-contained PCT visualization

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Create ranks/cv/ — cross-validation using kappa k*

**Files:**
- Create: `ranks/cv/run.py`
- Create: `ranks/cv/plot.py`
- Create: `ranks/cv/config.yaml`
- Create: `ranks/cv/__init__.py`

This is the key new pipeline step. Reference: `experiments/estimate_rank.py` (worktree).

- [ ] **Step 1: Write `ranks/cv/config.yaml`**

```yaml
# Cross-validation rank estimation using kappa k* as center
#
# Reads k* from ../kappa/outputs/<dataset>.json
# Estimates bounds inline via estimate_sampling_bounds_fast
#
# Usage:
#   ./scripts/submit experiments/datasets/ranks/cv/run.py dataset=mur92
#   ./scripts/submit experiments/datasets/ranks/cv/run.py dataset=nsd subject_id=1
#   ./scripts/submit experiments/datasets/ranks/cv/run.py dataset=things_behavior kappa_name=things_100pct

defaults:
  - /base
  - /dataset: ???
  - _self_

experiment_name: ranks
task: cv

subject_id: null

# Override to map to kappa output filename when it differs from dataset.name
# e.g., kappa_name=things_100pct for dataset=things_behavior
kappa_name: null

cv:
  n_repeats: 5
  sampling_selection: mean   # mean/min/max of (pmin, pmax)
  n_ranks_around_kstar: 10   # search ±n_ranks_around_kstar * step from k*

common:
  n_jobs: -1
  random_state: 42
```

- [ ] **Step 2: Write `ranks/cv/run.py`**

```python
"""Cross-validation rank estimation centered on kappa k*.

Reads k* from ../kappa/outputs/<kappa_name>.json, builds a rank grid
centered on k*, estimates bounds inline, runs pysrf.cross_val_score.

Usage:
    ./scripts/submit experiments/datasets/ranks/cv/run.py dataset=mur92
    ./scripts/submit experiments/datasets/ranks/cv/run.py dataset=nsd subject_id=1
    ./scripts/submit experiments/datasets/ranks/cv/run.py dataset=things_behavior kappa_name=things_100pct
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from pysrf import cross_val_score
from pysrf.bounds import estimate_sampling_bounds_fast

from similarity import build_similarity
from src.utils.figure_theme import CMAP, GRAY, create_figure, despine, save_figure

log = logging.getLogger(__name__)

KAPPA_DIR = Path(__file__).resolve().parent.parent / "kappa" / "outputs"


def _resolve_kappa_name(cfg: DictConfig) -> str:
    """Map Hydra dataset config to kappa output filename.

    Kappa outputs use flat names: mur92, peterson_animals, nsd_subj01,
    things_100pct, etc. The mapping from (dataset.name, subject_id) is:
      - kappa_name override (if set): use directly
      - multi-subject datasets: {name}_subj{id:02d}
      - otherwise: dataset.name
    """
    if cfg.get("kappa_name"):
        return cfg.kappa_name
    ds_name = cfg.dataset.name
    subject_id = cfg.get("subject_id")
    if subject_id is not None:
        return f"{ds_name}_subj{subject_id:02d}"
    return ds_name


def _load_kappa_kstar(kappa_name: str) -> int | None:
    """Load k* from kappa outputs. Returns None if not found."""
    json_path = KAPPA_DIR / f"{kappa_name}.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)["k_star_kappa"]


def _build_rank_grid(
    ds_cfg: DictConfig, k_star: int | None, n_around: int,
) -> list[int]:
    """Build rank grid centered on kappa k*.

    Uses step from dataset rank_range config. Falls back to full rank_range
    if k_star is None.
    """
    start, stop, step = ds_cfg.rank_range
    if k_star is None:
        return list(range(start, stop + 1, step))

    lo = max(step, k_star - n_around * step)
    hi = k_star + n_around * step
    lo = max(lo, start)
    hi = min(hi, stop)
    grid = list(range(lo, hi + 1, step))
    if k_star not in grid and start <= k_star <= stop:
        grid.append(k_star)
        grid.sort()
    return grid


def run(cfg: DictConfig) -> None:
    subject_id = cfg.get("subject_id")
    ds_name = cfg.dataset.name

    # Per-dataset output subdirectory (so consensus can find by name)
    output_dir = Path.cwd() / ds_name
    if subject_id is not None:
        output_dir = output_dir / f"subj{subject_id:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve kappa filename and load k*
    kappa_name = _resolve_kappa_name(cfg)
    k_star = _load_kappa_kstar(kappa_name)
    if k_star is not None:
        log.info(f"Kappa k* = {k_star} (from {kappa_name}.json)")
    else:
        log.warning(f"No kappa output for {kappa_name}, using full rank_range")

    # Build rank grid
    n_around = cfg.cv.n_ranks_around_kstar
    rank_grid = _build_rank_grid(cfg.dataset, k_star, n_around)
    log.info(f"Rank grid: {rank_grid} ({len(rank_grid)} values)")

    # Load similarity
    log.info(f"Building similarity for {ds_name}...")
    similarity = build_similarity(cfg.dataset, subject_id=subject_id)
    log.info(f"Shape: {similarity.shape}")

    # Estimate bounds inline
    log.info("Estimating sampling bounds...")
    pmin, pmax, _ = estimate_sampling_bounds_fast(
        similarity, random_state=cfg.common.random_state, n_jobs=cfg.common.n_jobs,
    )
    sel = cfg.cv.sampling_selection
    p_star = {"min": pmin, "mean": 0.5 * (pmin + pmax), "max": pmax}[sel]
    log.info(f"Bounds: pmin={pmin:.4f}, pmax={pmax:.4f}, p*={p_star:.4f} ({sel})")

    # Cross-validation
    n_repeats = cfg.cv.n_repeats
    log.info(f"Running CV: {len(rank_grid)} ranks × {n_repeats} repeats...")
    cv_result = cross_val_score(
        similarity,
        param_grid={"rank": rank_grid},
        n_repeats=n_repeats,
        sampling_fraction=p_star,
        estimate_sampling_fraction=False,
        fit_final_estimator=False,
        random_state=cfg.common.random_state,
        n_jobs=cfg.common.n_jobs,
        verbose=1,
    )

    optimal_rank = cv_result.best_params_["rank"]
    log.info(f"Optimal rank: {optimal_rank} (CV score: {cv_result.best_score_:.6f})")

    # Aggregate
    df = cv_result.cv_results_
    cv_scores = {
        int(r): {"mean": float(g["score"].mean()), "std": float(g["score"].std())}
        for r, g in df.groupby("rank")
    }

    # Save to per-dataset subdirectory
    summary = {
        "dataset": ds_name,
        "subject_id": int(subject_id) if subject_id else None,
        "n_samples": int(similarity.shape[0]),
        "optimal_rank": int(optimal_rank),
        "best_cv_score": float(cv_result.best_score_),
        "kappa_k_star": k_star,
        "kappa_name": kappa_name,
        "rank_grid": [int(r) for r in rank_grid],
        "n_repeats": n_repeats,
        "pmin": float(pmin),
        "pmax": float(pmax),
        "p_star": float(p_star),
        "cv_scores": cv_scores,
    }
    (output_dir / "rank_estimation.json").write_text(json.dumps(summary, indent=2))
    df.to_csv(output_dir / "cv_results.csv", index=False)

    # Plot CV curve
    fig, ax = create_figure("single")
    ranks = sorted(cv_scores.keys())
    means = [cv_scores[r]["mean"] for r in ranks]
    stds = [cv_scores[r]["std"] for r in ranks]
    ax.errorbar(ranks, means, yerr=stds, fmt="o-", color=CMAP[1], capsize=3, markersize=4)
    ax.axvline(optimal_rank, color=CMAP[0], linestyle="--", linewidth=1.5,
               label=f"CV optimal: {optimal_rank}")
    if k_star is not None:
        ax.axvline(k_star, color=GRAY["medium"], linestyle=":", linewidth=1,
                   label=f"Kappa k*: {k_star}")
    ax.set_xlabel("Rank")
    ax.set_ylabel("CV Score (MSE)")
    ax.legend(fontsize=8)
    despine(ax)
    save_figure(fig, output_dir / "cv_curve.pdf")
    plt.close(fig)

    log.info(f"Saved to {output_dir}")
```

- [ ] **Step 3: Create `ranks/cv/__init__.py`**

Empty file.

- [ ] **Step 4: Verify with a small dataset**

```bash
./scripts/submit experiments/datasets/ranks/cv/run.py dataset=mur92
cat experiments/datasets/ranks/cv/outputs/rank_estimation.json
```

Expected: `optimal_rank` close to kappa's k*=2, CV curve PDF saved.

- [ ] **Step 5: Commit**

```bash
git add experiments/datasets/ranks/cv/
git commit -m "$(cat <<'EOF'
feat: add ranks/cv/ — cross-validation centered on kappa k*

Reads k* from kappa outputs, builds rank grid around it,
estimates sampling bounds inline, runs pysrf cross_val_score.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Create ranks/compare/ — cross-method comparison

**Files:**
- Create: `ranks/compare/plot.py`
- Create: `ranks/compare/__init__.py`
- Move: `ranks/coherence/plot_method_comparison/outputs/` → `ranks/compare/outputs/`
- Source: `ranks/coherence/plot_method_comparison/run.py`

- [ ] **Step 1: Create directory, move outputs**

```bash
cd /LOCAL/fmahner/similarity-factorization/experiments/datasets
mkdir -p ranks/compare
touch ranks/compare/__init__.py
cp -r ranks/coherence/plot_method_comparison/outputs ranks/compare/outputs
```

- [ ] **Step 2: Write `ranks/compare/plot.py`**

Standalone script reading from sibling dirs:
- `KAPPA_DIR = TASK_DIR.parent / "kappa" / "outputs"`
- `PCT_DIR = TASK_DIR.parent / "pct" / "outputs"`
- `CV_DIR = TASK_DIR.parent / "cv" / "outputs"`

Functions from `plot_method_comparison/run.py`:
- `plot_kstar_overview` — bar chart (kappa, activation, cluster, pct, + CV optimal)
- `plot_diagnostic_grid` — coherence / kappa / PCT panels per dataset
- `plot_things_scaling` — THINGS k* vs data across methods

Add CV optimal rank to the comparison if `cv/outputs/` exists.

- [ ] **Step 3: Verify**

```bash
./scripts/submit experiments/datasets/ranks/compare/plot.py
```

- [ ] **Step 4: Commit**

```bash
git add experiments/datasets/ranks/compare/
git commit -m "$(cat <<'EOF'
feat: add ranks/compare/ — cross-method comparison plots

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Update consensus/ — read rank from cv/

**Files:**
- Modify: `consensus/run.py`
- Modify: `consensus/config.yaml`

- [ ] **Step 1: Update `consensus/run.py`**

Replace `_load_rank_estimation()` to read from `../ranks/cv/outputs/<dataset>/rank_estimation.json` instead of the old `outputs/experiments/estimate_rank/` path.

Key changes:
```python
CV_DIR = Path(__file__).resolve().parent.parent / "ranks" / "cv" / "outputs"

def _load_rank_estimation(dataset_name: str, subject_id: int | None) -> dict:
    """Load rank from CV outputs. Path structure matches cv/run.py output."""
    base = CV_DIR / dataset_name
    if subject_id is not None:
        path = base / f"subj{subject_id:02d}" / "rank_estimation.json"
    else:
        path = base / "rank_estimation.json"
    if not path.exists():
        raise FileNotFoundError(
            f"CV results not found at {path}\n"
            f"Run CV first: ./scripts/submit experiments/datasets/ranks/cv/run.py dataset={dataset_name}"
        )
    with open(path) as f:
        return json.load(f)
```

Note: The CV output path is `cv/outputs/<dataset_name>/rank_estimation.json` (or `.../subj01/rank_estimation.json` for multi-subject). This matches because cv/run.py creates `Path.cwd() / ds_name / [subj0X/] rank_estimation.json` and Hydra sets cwd to `cv/outputs/`.

Also update what gets saved:
- `embedding.npy` — final consensus embedding (n × k)
- `runs.npy` — all ensemble runs (n_runs × n × k)
- `summary.json` — metadata, agreement scores, quality metrics

Remove `pipeline.joblib` and `ensemble_embeddings.npy` → replace with `runs.npy`.

- [ ] **Step 2: Verify with small dataset (requires cv/ outputs)**

```bash
./scripts/submit experiments/datasets/consensus/run.py dataset=mur92
ls experiments/datasets/consensus/outputs/mur92/
```

Expected: `embedding.npy`, `runs.npy`, `summary.json`

- [ ] **Step 3: Commit**

```bash
git add experiments/datasets/consensus/run.py experiments/datasets/consensus/config.yaml
git commit -m "$(cat <<'EOF'
refactor: consensus reads rank from ranks/cv/ outputs

Save embedding.npy + runs.npy + summary.json (drop pipeline.joblib).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Create visualize/ — top-k images per dimension

**Files:**
- Create: `visualize/plot.py`
- Create: `visualize/__init__.py`
- Source: `experiments/figures/plot_dimensions/run.py`

- [ ] **Step 1: Write `visualize/plot.py`**

Standalone script adapted from `plot_dimensions/run.py`. Reads embedding from `../consensus/outputs/<dataset>/embedding.npy`.

```python
"""Visualize top-k items per embedding dimension.

Reads consensus embedding from ../consensus/outputs/ and dataset
labels/images, produces per-dimension image grids and combined overview.

Usage:
    ./scripts/submit experiments/datasets/visualize/plot.py -- --dataset mur92
    ./scripts/submit experiments/datasets/visualize/plot.py -- --dataset things_behavior --k 15
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from datasets import load_dataset
from src.colors import CYCLE
from src.utils.figure_theme import despine, save_figure

log = logging.getLogger(__name__)

TASK_DIR = Path(__file__).resolve().parent
CONSENSUS_DIR = TASK_DIR.parent / "consensus" / "outputs"
OUTPUT_DIR = TASK_DIR / "outputs"
```

Functions copied from `plot_dimensions/run.py`:
- `_get_item_labels` — extract labels from dataset
- `_get_images` — extract images (array or paths)
- `_plot_topk_images` — combined grid (dims × items)
- `_plot_single_dimension_images` — one dim as square grid
- `_plot_topk_text` — bar chart fallback for text-only datasets
- `_plot_individual_dimensions` — loop over all dims

Main:
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., mur92, things_behavior)")
    parser.add_argument("--k", type=int, default=10, help="Top-k items per dimension")
    parser.add_argument("--subject-id", type=int, default=None)
    args = parser.parse_args()

    # Load embedding
    ds_dir = CONSENSUS_DIR / args.dataset
    if args.subject_id:
        ds_dir = ds_dir / f"subj{args.subject_id:02d}"
    embedding = np.load(ds_dir / "embedding.npy")

    # Load dataset for labels/images
    dataset = load_dataset(args.dataset)
    labels = _get_item_labels(dataset, embedding.shape[0])
    images = _get_images(dataset)

    # Output
    out = OUTPUT_DIR / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    if images is not None:
        _plot_topk_images(out / "topk_images.pdf", embedding, images, k=args.k)
    else:
        _plot_topk_text(out / "topk_items.pdf", embedding, labels, k=args.k)
    _plot_individual_dimensions(out, embedding, images, labels, k=args.k)
```

- [ ] **Step 2: Create `visualize/__init__.py`**

Empty file.

- [ ] **Step 3: Verify with small dataset (requires consensus outputs)**

```bash
./scripts/submit experiments/datasets/visualize/plot.py -- --dataset mur92
ls experiments/datasets/visualize/outputs/mur92/
```

Expected: `topk_items.pdf` (text, no images for mur92) and `dims/` folder.

- [ ] **Step 4: Commit**

```bash
git add experiments/datasets/visualize/
git commit -m "$(cat <<'EOF'
feat: add visualize/ — top-k image grids per dimension

Adapted from experiments/figures/plot_dimensions/.
Reads embedding from consensus/outputs/.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Archive old directories and clean up

**Files:**
- Delete: `ranks/coherence/` (all content moved to ranks/kappa/, ranks/pct/, ranks/compare/)
- Archive: `ranks/bounds/` (move to project archive)

- [ ] **Step 1: Verify no remaining references to old paths**

```bash
cd /LOCAL/fmahner/similarity-factorization
grep -rn "coherence/" --include="*.py" --include="*.yaml" experiments/datasets/ranks/{kappa,pct,cv,compare}/ experiments/datasets/consensus/ experiments/datasets/visualize/
grep -rn "bounds/" --include="*.py" experiments/datasets/ranks/{kappa,pct,cv,compare}/ experiments/datasets/consensus/
```

Expected: no hits to old paths.

- [ ] **Step 2: Delete coherence/**

```bash
rm -rf experiments/datasets/ranks/coherence/
```

- [ ] **Step 3: Move bounds/ to archive**

```bash
mv experiments/datasets/ranks/bounds/ archive/datasets_bounds/
```

- [ ] **Step 4: Commit**

```bash
git add -u experiments/datasets/ranks/coherence/
git add -u experiments/datasets/ranks/bounds/
git add archive/datasets_bounds/
git commit -m "$(cat <<'EOF'
refactor: archive old bounds/ and remove coherence/

All content moved to new structure. Bounds archived (outputs preserved).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Final verification

- [ ] **Step 1: Verify directory structure (scripts only)**

```bash
find experiments/datasets/ -type f -name "*.py" -o -name "*.yaml" | grep -v __pycache__ | sort
```

Expected:
```
experiments/datasets/__init__.py
experiments/datasets/consensus/__init__.py
experiments/datasets/consensus/config.yaml
experiments/datasets/consensus/run.py
experiments/datasets/ranks/__init__.py
experiments/datasets/ranks/compare/__init__.py
experiments/datasets/ranks/compare/plot.py
experiments/datasets/ranks/cv/__init__.py
experiments/datasets/ranks/cv/config.yaml
experiments/datasets/ranks/cv/run.py
experiments/datasets/ranks/kappa/__init__.py
experiments/datasets/ranks/kappa/config.yaml
experiments/datasets/ranks/kappa/plot.py
experiments/datasets/ranks/kappa/run.py
experiments/datasets/ranks/pct/__init__.py
experiments/datasets/ranks/pct/config.yaml
experiments/datasets/ranks/pct/plot.py
experiments/datasets/ranks/pct/run.py
experiments/datasets/visualize/__init__.py
experiments/datasets/visualize/plot.py
```

- [ ] **Step 2: Verify all existing outputs preserved**

```bash
ls experiments/datasets/ranks/kappa/outputs/*.json | wc -l   # expect 11
ls experiments/datasets/ranks/pct/outputs/*.json | wc -l     # expect 11
ls experiments/datasets/consensus/outputs/*/summary.json      # existing consensus outputs
```

- [ ] **Step 3: Run full pipeline on one small dataset**

```bash
# Step 1: kappa already done
cat experiments/datasets/ranks/kappa/outputs/mur92.json | python3 -c "import sys,json; print('k*=', json.load(sys.stdin)['k_star_kappa'])"

# Step 2: CV
./scripts/submit experiments/datasets/ranks/cv/run.py dataset=mur92

# Step 3: consensus
./scripts/submit experiments/datasets/consensus/run.py dataset=mur92

# Step 4: visualize
./scripts/submit experiments/datasets/visualize/plot.py -- --dataset mur92
```

- [ ] **Step 4: Final commit**

```bash
git add -A experiments/datasets/
git status
```

Verify only expected changes, then commit if needed.
