# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Similarity-based Representation Factorization (SRF)** - Tools for modeling representations in minds, brains, and machines using symmetric non-negative matrix factorization with ADMM optimization.

Core library: `pysrf` (in `third_party/pysrf/`) - provides `SRF` model, cross-validation, consensus methods, and sampling bounds estimation.

## Critical: Development vs Production Workflow

**All new work MUST go into `development/`**. The `experiments/` folder contains stable, versioned code and should NOT be modified unless explicitly requested.

| Location | Purpose | Lifecycle |
|----------|---------|-----------|
| `development/<name>/` | New experiments, exploratory work | Create → Test → Delete or Promote |
| `experiments/<name>/` | Stable, canonical experiments | Read-only unless explicitly asked |

### Development Structure
```
development/my_experiment/
├── run.py              # Main runner with internal joblib parallelism
├── config.yaml         # Minimal config (inherits /base)
├── plot.py             # Auto-detected by submit script
└── outputs/            # Auto-generated timestamped runs
```

### Development Config Pattern
```yaml
# development/my_exp/config.yaml
defaults:
  - /base
  - _self_

experiment_name: my_exp
task: analysis

# Parameters (lists for internal parallel sweep)
sizes: [200, 500, 1000]
methods: [srf, node2vec]
n_jobs: -1

data_dir: ${paths.data_dir}/my_data
```

## Commands

```bash
# Run development experiments (primary workflow)
./scripts/submit development/my_experiment/run.py [hydra overrides]

# Parameter sweep (multirun)
./scripts/submit development/my_experiment/run.py -m size=200,500 method=srf,node2vec

# Background execution
./scripts/submit development/my_experiment/run.py --bg

# SLURM submission
./scripts/submit development/my_experiment/run.py hydra/launcher=slurm

# Run production experiments (only when needed)
./scripts/submit experiments/<name>/tasks/<task>.py [overrides]

# Tests
poetry run pytest tests/
```

## Architecture

### Execution Flow
```
./scripts/submit → [development: run.py | production: run_task.py] → run(cfg)
```

The `submit` script:
1. Resolves paths and injects `project_root` into Hydra config
2. Auto-cleans empty output folders (only logs) after completion
3. Runs `plot.py` automatically if present (development)

### Directory Structure

```
development/             # NEW WORK GOES HERE
├── <experiment>/run.py  # Self-contained experiments

configs/
├── base.yaml            # Global defaults (mode, paths, launcher)
├── experiment/*.yaml    # Production experiment definitions
├── mode/{development,production}.yaml
├── launcher/{local,slurm}.yaml
└── paths/local.yaml     # Data paths using ${project_root}

experiments/             # STABLE CODE - DO NOT MODIFY without asking
├── <name>/lib/          # Experiment-specific utilities
├── <name>/tasks/*.py    # Task entry points: run(cfg: DictConfig)
└── <name>/outputs/      # Production outputs (static paths)

src/
├── similarity/          # Dataset builders, similarity computation
├── datasets/            # Data loaders (NSD, THINGS, etc.)
├── tools/               # RSA, metrics, stats
└── utils/               # IO, plotting, graphs, simulation

third_party/
├── pysrf/               # Core SRF algorithm (local editable install)
└── OpenNE/              # Graph embedding baselines
```

### Key Components

**pysrf** - Core SRF algorithm (sklearn-compatible API):
```python
from pysrf import SRF, cross_val_score, EnsembleEmbedding, ClusterEmbedding

# Basic usage
model = SRF(n_components=50, loss="frobenius", missing_values=np.nan)
model.fit(similarity_matrix)
embedding = model.embedding_  # n_items x n_components

# Cross-validation for rank selection
scores = cross_val_score(similarity_matrix, ranks=[25, 50, 100], n_splits=5)

# Consensus embedding from multiple runs
ensemble = EnsembleEmbedding(n_runs=10, n_components=50)
ensemble.fit(similarity_matrix)
consensus = ClusterEmbedding().fit_transform(ensemble.embeddings_)
```

Loss functions: `frobenius`, `kullback-leibler`, `bce`

**OpenNE** (`third_party/OpenNE/`) - Graph embedding baselines (DeepWalk, Node2Vec, LINE)

### Production Experiments

Located in `experiments/`. Key domains:
- `ppi/` - Protein-protein interaction networks (link prediction, node classification)
- `word_association/` - Semantic embeddings from behavioral data
- `things_behavior/` - Object similarity judgments
- `simulation/` - Synthetic data benchmarks (denoising, rank detection)
- `bounds/` - Sampling bounds estimation

## Coding Standards

- **Development parallelism**: `run.py` handles parameter sweeps internally with `joblib.Parallel`
- **Production parallelism**: Use Hydra `-m` for multirun or `hydra/launcher=slurm` for cluster
- **Paths**: Always use `pathlib.Path`, reference data via `cfg.data_dir`
- **Outputs**: Timestamped folders in development (`./outputs/YYYY-MM-DD/HH-MM-SS/`)
- **Plotting**: `seaborn`/`matplotlib`, publication quality, save as `.png`

## Data

Data directory: `${project_root}/data/` (configured in `configs/paths/local.yaml`)
