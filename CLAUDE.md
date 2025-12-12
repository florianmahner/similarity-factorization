# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Similarity-based Representation Factorization (SRF)** - Tools for modeling representations in minds, brains, and machines using symmetric non-negative matrix factorization with ADMM optimization.

Core library: `pysrf` (in `third_party/pysrf/`) - provides `SRF` model, cross-validation, consensus methods, and sampling bounds estimation.

## Sandbox vs Experiment Workflow

| Location | Purpose | Outputs |
|----------|---------|---------|
| `sandbox/<name>/` | Exploratory work, AI-generated code | Local: `sandbox/<name>/outputs/` |
| `experiments/<name>/` | Stable, versioned experiments | Central: `outputs/experiments/<name>/` |

### Sandbox Structure
```
sandbox/my_experiment/
├── run.py              # Main runner
├── config.yaml         # Minimal config (inherits /base)
├── plot.py             # Optional plotting script
└── outputs/            # Local timestamped outputs
```

### Experiment Structure (Flattened)
```
experiments/ppi/
├── __init__.py
├── models.py           # Shared utilities
├── utils.py
├── link_prediction.py  # Task entry point: run(cfg)
├── node_classification.py
└── ...
```

## Commands

```bash
# Run sandbox experiments
./scripts/submit sandbox/my_experiment/run.py

# Run stable experiments
./scripts/submit experiments/ppi/link_prediction.py

# Mode flags
./scripts/submit <script> -s              # Force sandbox mode (local outputs)
./scripts/submit <script> -e              # Force experiment mode (central outputs)

# Background execution
./scripts/submit <script> --bg

# SLURM submission
./scripts/submit <script> hydra/launcher=slurm

# Tests
poetry run pytest tests/
```

## Directory Structure

```
sandbox/                 # Exploratory work (outputs stay local)
├── <name>/run.py

experiments/             # Stable experiments (flattened structure)
├── <name>/*.py          # Task files and utilities at same level

outputs/                 # Central outputs for experiments
└── experiments/<name>/<task>/

configs/
├── base.yaml            # Global defaults
├── experiment/*.yaml    # Experiment-specific configs
├── mode/
│   ├── sandbox.yaml     # Local outputs (./outputs/...)
│   └── experiment.yaml  # Central outputs (${project_root}/outputs/...)
├── launcher/{local,slurm}.yaml
└── paths/local.yaml

src/
├── similarity/          # Dataset builders, similarity computation
├── datasets/            # Data loaders (NSD, THINGS, etc.)
├── tools/               # RSA, metrics, stats
└── utils/               # IO, plotting, graphs, simulation

third_party/
├── pysrf/               # Core SRF algorithm (local editable install)
└── OpenNE/              # Graph embedding baselines
```

## Key Components

**pysrf** - Core SRF algorithm (sklearn-compatible API):
```python
from pysrf import SRF, cross_val_score, EnsembleEmbedding, ClusterEmbedding

model = SRF(n_components=50, loss="frobenius", missing_values=np.nan)
model.fit(similarity_matrix)
embedding = model.embedding_

# Cross-validation for rank selection
scores = cross_val_score(similarity_matrix, ranks=[25, 50, 100], n_splits=5)
```

Loss functions: `frobenius`, `kullback-leibler`, `bce`

**OpenNE** (`third_party/OpenNE/`) - Graph embedding baselines (DeepWalk, Node2Vec, LINE)

## Experiments

Key domains in `experiments/`:
- `ppi/` - Protein-protein interaction networks
- `word_association/` - Semantic embeddings from behavioral data
- `things_behavior/` - Object similarity judgments
- `simulation/` - Synthetic data benchmarks
- `bounds/` - Sampling bounds estimation

## Coding Standards

- **Paths**: Always use `pathlib.Path`, reference data via `cfg.data_dir`
- **Outputs**: Use `Path.cwd()` (Hydra changes to output dir)
- **Parallelism**: `joblib.Parallel` locally, `hydra/launcher=slurm` for cluster
- **Plotting**: `seaborn`/`matplotlib`, save ONLY as `.pdf` (no PNG/SVG)

## Figure Theme (`src/utils/figure_theme.py`)

Use `create_figure()` and `save_figure()` for consistent publication-quality plots:

```python
from src.utils.figure_theme import CMAP, GRAY, create_figure, despine, save_figure

fig, ax = create_figure("single")  # Plotting area: 3.5" x 2.6"
ax.plot(x, y, color=CMAP[1])       # Blue from color palette
despine(ax)
save_figure(fig, output_path)      # Saves as PDF
```

**Fixed plotting area** - the data area is always the specified size, padding is added around it:
- `"single"` (2.7×2.2")
- `"square"` (2.2×2.2")
- `"wide"` (3.4×2.2")
- `"full_width"` (5.6×2.2")

**Padding** - `save_figure()` uses `bbox_inches='tight'` by default, so labels are never cut off:
```python
fig, ax = create_figure("single")
# ... plot ...
save_figure(fig, output_path)  # tight=True by default, prevents cutoff
```

For fixed dimensions (e.g., when aligning multiple figures), use `tight=False` and specify padding:
```python
fig, ax = create_figure("single", pad_left=0.8, pad_bottom=0.5, pad_right=0.2, pad_top=0.3)
save_figure(fig, output_path, tight=False)
```
Default padding: left=0.5, right=0.2, bottom=0.5, top=0.1

**Colors**:
- `CMAP[0]` red, `CMAP[1]` blue, `CMAP[2]` green, `CMAP[3]` purple
- `GRAY["dark"]`, `GRAY["medium"]`, `GRAY["light"]`, `GRAY["faint"]`

**Dual-axis plots** - use colored ylabels (not legend) to identify lines:
```python
ax1.set_ylabel("Left metric", color=CMAP[1])
ax2 = ax1.twinx()
ax2.set_ylabel("Right metric", color=CMAP[0])
```

## SRF Usage Notes

When using SRF for imputation with missing data, use **adaptive rho** based on sampling ratio:
- `obs_per_dof = n_observed / (n × k)` where k is rank
- For sparse data (ratio < 2): use `rho=0.01-0.05`
- For moderate data (ratio 2-5): use `rho=0.05-0.5`
- For dense data (ratio > 5): use `rho=0.5-3.0`

## Data

Data directory: `${project_root}/data/` (configured in `configs/paths/local.yaml`)
