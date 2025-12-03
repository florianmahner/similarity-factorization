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
- **Plotting**: `seaborn`/`matplotlib`, save as `.png`

## Data

Data directory: `${project_root}/data/` (configured in `configs/paths/local.yaml`)
