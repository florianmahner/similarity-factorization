# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL: Running Scripts

**NEVER run Python scripts directly with `poetry run python`.** Always use the submit script:

```bash
# Correct - ALWAYS use this:
./scripts/submit sandbox/simulation/example/run.py --bg
./scripts/submit experiments/rsa_comparison/spose.py --bg

# WRONG - Never do this:
poetry run python sandbox/simulation/example/run.py  # ❌ FORBIDDEN
```

This ensures proper job tracking, logging, and output directory management.

## CRITICAL: Third-Party Code

**NEVER modify files in `third_party/` without explicit user approval.** This includes `pysrf`, `OpenNE`, and any other third-party dependencies. Always ask before making changes to these directories.

## CRITICAL: Testing Code

**Always test code in sandbox scripts, not in interactive one-liners.** Create a sandbox script (e.g., `sandbox/<domain>/<name>/run.py`) for any non-trivial testing or exploration. This allows:
- Reuse and iteration on the code
- Proper logging and output management
- Easy cleanup if not needed

Only use `poetry run python -c "..."` for trivial checks (e.g., checking a file's shape or a single value).

## Project Overview

**Similarity-based Representation Factorization (SRF)** - Tools for modeling representations in minds, brains, and machines using symmetric non-negative matrix factorization with ADMM optimization.

Core library: `pysrf` (in `third_party/pysrf/`) - provides `SRF` model, cross-validation, consensus methods, and sampling bounds estimation.

## Sandbox vs Experiment Workflow

| Location | Purpose | Outputs |
|----------|---------|---------|
| `sandbox/<group>/<name>/` | Exploratory, iterative work | `sandbox/<group>/<name>/outputs/<timestamp>/` |
| `experiments/<domain>/` | Stable, versioned experiments | `outputs/experiments/<domain>/<task>/` |

### Sandbox Structure

Sandbox experiments are organized by domain and tracked via `./scripts/submit`:

```
sandbox/
├── nsd/                    # Neural data (NSD)
│   └── consensus_test/
│       ├── run.py          # Main script (uses OUTPUT_DIR from env)
│       ├── plot.py         # Optional plotting
│       └── outputs/
│           ├── 260115_113230/        # Flat timestamp
│           │   ├── .status/
│           │   │   ├── job.json      # Job tracking
│           │   │   └── log           # All output
│           │   └── <results>
│           ├── 260116_091500_v2/     # With optional suffix
│           └── latest -> 260116_091500_v2  # Symlink
├── things/                 # THINGS behavioral
├── ppi/                    # Graph/network
├── simulation/             # Method validation
├── semantic/               # Word embeddings
└── _archive/               # Old/unused experiments
```

**Running sandbox experiments:**
```bash
./scripts/submit sandbox/nsd/consensus_test/run.py --bg           # → outputs/260116_102500/
./scripts/submit sandbox/nsd/consensus_test/run.py name=v2 --bg   # → outputs/260116_102500_v2/
```

**Sandbox script pattern:**
```python
"""Brief description of experiment."""
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()

def main():
    # Save outputs to OUTPUT_DIR
    ...

if __name__ == "__main__":
    main()
```

**Creating a new sandbox experiment:**
1. Create folder: `sandbox/<group>/<name>/`
2. Add `run.py` with the pattern above
3. Run: `./scripts/submit sandbox/<group>/<name>/run.py --bg`
4. Monitor: `dash`

### Experiment Structure (Stable)

Stable experiments use Hydra configs and `run(cfg)` pattern:

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
# Run experiments (foreground)
./scripts/submit experiments/ppi/link_prediction.py
./scripts/submit experiments/consensus.py dataset=nsd subject_id=1

# Run experiments (background)
./scripts/submit experiments/ppi/link_prediction.py --bg

# Config overrides
./scripts/submit experiments/consensus.py dataset=swow n_jobs=64

# SLURM submission
./scripts/submit <script> hydra/launcher=slurm

# Tests
poetry run pytest tests/
```

## Job Monitoring

### Dashboard (`dash`)

Interactive terminal dashboard for monitoring experiments:

```bash
dash                    # Launch dashboard (alias configured in ~/.zshrc)
./scripts/dash          # Or run directly
```

| Key | Action |
|-----|--------|
| `←→` | Switch columns (Running/Completed/Failed) |
| `↑↓` | Navigate jobs |
| `l` | View log in pager |
| `t` | Tail log (live follow) |
| `o` | Open output directory in yazi |
| `K` | Kill job |
| `D` | Delete job and output directory |
| `r` | Refresh |
| `q` | Quit |

### Jobs CLI (`jobs`)

```bash
./scripts/jobs              # List all jobs
./scripts/jobs -a           # Active (running) only
./scripts/jobs -f           # Failed only
./scripts/jobs -c           # Completed only
./scripts/jobs <name>       # Show job details
./scripts/jobs <name> --tail   # Tail log file
```

### Job Tracking

All jobs (both `--bg` and foreground) create tracking files in `.status/`:

```
outputs/experiments/<name>/<task>/
├── .status/
│   ├── job.json          # Status, PID, timing, args
│   └── log               # All output (logging + stdout/stderr)
├── .hydra/               # Hydra config snapshot
└── <outputs>             # Experiment outputs
```

Job statuses: `running`, `completed`, `failed`, `aborted`, `dead`

Logs are stored alongside job outputs in `.status/log` (not in a separate directory).

## Directory Structure

```
scripts/
├── submit               # Job submission (./scripts/submit <script> [--bg])
├── dash                 # Interactive dashboard (curses-based)
├── jobs                 # Job status CLI
└── run_task.py          # Hydra task runner (called by submit)

sandbox/                 # Exploratory work (outputs stay local)
├── <name>/run.py

experiments/             # Stable experiments (flattened structure)
├── <name>/*.py          # Task files and utilities at same level

outputs/
└── experiments/<name>/<task>/
    ├── .status/
    │   ├── job.json     # Job tracking (status, PID, timing)
    │   └── log          # All output (logging + stdout/stderr)
    ├── .hydra/          # Hydra config snapshot
    └── <outputs>

configs/
├── base.yaml            # Global defaults (inherited by all)
├── consensus.yaml       # Consensus embedding generation
├── estimate_bounds.yaml # Sampling bounds estimation
├── dataset/             # Dataset definitions (nsd, swow, things_*, ...)
├── experiment/          # Experiment-specific configs
│   ├── ppi/
│   ├── simulation/
│   ├── things_behavior/
│   └── ...
└── paths/local.yaml     # Local paths (data_dir, etc.)

src/
├── similarity/          # Dataset builders, similarity computation
├── datasets/            # Data loaders (NSD, THINGS, etc.)
├── tools/               # RSA, metrics, stats
└── utils/               # IO, plotting, figure_theme

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

### Top-level experiments (flat config)

| Experiment | Config | Description |
|------------|--------|-------------|
| `consensus.py` | `consensus.yaml` | Generate consensus embeddings with CV rank selection |
| `estimate_bounds.py` | `estimate_bounds.yaml` | Estimate sampling bounds for datasets |
| `plot_dimensions.py` | `plot_dimensions.yaml` | Visualize embedding dimensions |

Usage: `./scripts/submit experiments/consensus.py dataset=nsd subject_id=1`

### Nested experiments (experiment-specific configs)

| Domain | Key tasks |
|--------|-----------|
| `ppi/` | `link_prediction`, `node_classification` |
| `simulation/` | `rank_detection`, `imputation`, `interpretability` |
| `things_behavior/` | `coherence`, `low_data`, `pairwise` |
| `rsa_comparison/` | `factorial`, `spose` |
| `swow/` | `predict_behavioral_properties` |

Usage: `./scripts/submit experiments/things_behavior/coherence.py`

### Datasets (via `dataset=` override)

Available in `configs/dataset/`: `nsd`, `swow`, `things_behavior`, `things_monkey_22k`, `mur92`, `cichy118`, `peterson`, `peterson_animals`, `peterson_various`, `vit`

## Coding Standards

- **Reuse existing code**: ALWAYS search `src/` first before writing new utilities
- **Paths**: Always use `pathlib.Path`, reference data via `cfg.data_dir`
- **Outputs**: Use `Path.cwd()` (Hydra changes to output dir)
- **Parallelism**: `joblib.Parallel` locally, `hydra/launcher=slurm` for cluster
- **Plotting**: `seaborn`/`matplotlib`, save as `.pdf` for experiments, `.png` for sandbox
- **Variables**: Always lowercase (`w`, `x`, `s`), never uppercase (`W`, `X`, `S`)
- **Atomic functions**: Break complex operations into small, reusable helpers prefixed with `_`
- **No complex one-liners**: Use explicit loops instead of dense list comprehensions

```python
# Good: explicit loop
r_obs = np.zeros(k)
for d in range(k):
    r_obs[d] = _correlation(w[:, d], x[:, d])

# Bad: dense one-liner
r_obs = np.array([_correlation(w[:, d], x[:, d]) for d in range(k)])
```

**Atomic functions example:**
```python
def _correlation(a, b, two_sided=True):
    r = pearsonr(a, b).statistic
    return np.abs(r) if two_sided else r

def _pvalue(obs, null):
    return (np.sum(null >= obs) + 1) / (len(null) + 1)

def permutation_test(a, b, permutations=1000, two_sided=True):
    r_obs = _correlation(a, b, two_sided)
    null = np.array([_correlation(a, rng.permutation(b), two_sided) for _ in range(permutations)])
    return _pvalue(r_obs, null), null, r_obs
```

**Statistical testing (`src/tools/rsa.py`):**

```python
from src.tools.rsa import alignment_test, mantel_test

# SRF alignment test
result = alignment_test(w, x, alignment="global", two_sided=True, fdr=True)
# Returns: {"r_obs", "raw_p", "corrected_p", "significant", "w_aligned"}

# RSA Mantel test
p, null, r = mantel_test(h, s, two_sided=True, permutations=1000)
```

- `two_sided=True`: Use |r| for sign ambiguity (default)
- `fdr=True`: Benjamini-Hochberg correction
- Null re-aligns each permutation (accounts for selection bias)

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

**Sandbox plots** - save as PNG (not PDF):
```python
fig.savefig(OUTPUT_DIR / "plot.png", dpi=300, bbox_inches='tight', facecolor='white')
```

## SRF Usage Notes

When using SRF for imputation with missing data, use **adaptive rho** based on sampling ratio:
- `obs_per_dof = n_observed / (n × k)` where k is rank
- For sparse data (ratio < 2): use `rho=0.01-0.05`
- For moderate data (ratio 2-5): use `rho=0.05-0.5`
- For dense data (ratio > 5): use `rho=0.5-3.0`

## Consensus Embeddings

For stable, interpretable embeddings, use `AlignedConsensus` with `aggregation="select"`:

```python
from sklearn.pipeline import Pipeline
from pysrf import SRF
from pysrf.consensus import EnsembleEmbedding, AlignedConsensus

pipeline = Pipeline([
    ("ensemble", EnsembleEmbedding(SRF(rank=k), n_runs=50, n_jobs=-1)),
    ("consensus", AlignedConsensus(rank=k, aggregation="select")),
])
embedding = pipeline.fit_transform(similarity)
```

**Why "select" not "mean/median"?** The factorization constraint is quadratic (`S ≈ WW^T`), but averaging is linear. Averaging breaks the factorization structure and destroys sparsity:
- W₁W₁ᵀ ≈ S ✓
- W₂W₂ᵀ ≈ S ✓  
- But: ((W₁+W₂)/2)((W₁+W₂)/2)ᵀ ≠ S (cross-terms break it)

**Aggregation methods:**
| Method | Use Case |
|--------|----------|
| `"select"` | **Recommended** — returns most central run, preserves interpretability |
| `"refine"` | Optimizes reconstruction at cost of sparsity |
| `"median"` | Stability analysis only (breaks factorization) |

**Agreement scores** (from `consensus.agreement_scores_`):
- **>0.9**: Stable — runs converge to same solution
- **0.7-0.9**: Some variability — consider more runs
- **<0.7**: Unreliable — multiple local minima

**Note on local minima:** Symmetric NMF has only permutation ambiguity (no rotation), but it's still non-convex. Different initializations CAN find genuinely different factorizations, not just permuted versions. High agreement scores indicate runs found the same basin.

## Data

Data directory: `${project_root}/data/` (configured in `configs/paths/local.yaml`)

**THINGS images**: `/SSD/datasets/things/`
- Full set: `/SSD/datasets/things/core/<class_name>/<class_name>_XXs.jpg`
- Behavioral 1854: `/SSD/datasets/things/behav1854/<class_name>/<class_name>_01b.jpg`
- Use for visualizing SRF dimensions (top-k images per dimension)

## Insights Knowledge Base

Document experimental findings and insights in `docs/insights/`. This builds a persistent knowledge base of what was tried and learned.

**When to add an insight:**
- Debugging reveals unexpected behavior
- Experiments show parameter sensitivity
- Comparisons reveal method differences
- Any finding that would be useful to remember

**Insight document structure:**
1. **Summary** - One-line description
2. **Background** - Context and motivation
3. **Evidence** - Data tables, experiment results, code snippets
4. **Tentative conclusions** - What the evidence suggests (with caveats)
5. **Open questions** - What remains unclear
6. **Reproducibility** - How to replicate the finding

**Guidelines:**
- Show evidence, not just conclusions
- Avoid definitive statements without strong support
- Note conditions under which findings hold
- Include references to scripts, data, sandbox experiments
- Date each insight for context

**Example topics:**
- `cv_sampling_fraction.md` - CV needs higher p than bounds estimation provides
- `factor_recovery_baseline.md` - Chance-level metrics for factor recovery
- `srf_statistical_testing.md` - LOO cross-validation for proper SRF testing
