# Coherence-Based Rank Estimation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a stable experiment that estimates dimensionality (k*) via eigenspace coherence for 7 datasets (mur92, peterson_animals, peterson_various, dinov3, swow, nsd subj01, nsd subj02), running sequentially in a single job with results saved as JSON per dataset.

**Architecture:** Single Hydra experiment script loops through a config-defined dataset list. For each dataset: load RSM via `dispatch_dataset_builder`, run `compute_incremental_coherence_multi_k_eig_anisotropic`, extract k* via three methods (activation counting, kappa changepoint, cluster consensus), save JSON + NPZ. Monkey 22k commented out (estimated ~3h).

**Tech Stack:** NumPy, SciPy (eigsh), joblib, Hydra/OmegaConf. Source: `src/coherence.py`, `src/similarity/datasets.py`.

---

## File Structure

### New files to create

| File | Responsibility |
|------|---------------|
| `experiments/coherence/__init__.py` | Package marker (empty) |
| `experiments/coherence/estimate_rank.py` | Main experiment: load datasets, run coherence, extract k*, save results |
| `configs/experiment/coherence/estimate_rank.yaml` | Dataset list, coherence params, Hydra output config |

---

## Task 1: Create Hydra config

**Files:**
- Create: `configs/experiment/coherence/estimate_rank.yaml`

- [ ] **Step 1: Create config file**

```yaml
# Coherence-based rank estimation across datasets
#
# Usage:
#   ./scripts/submit experiments/coherence/estimate_rank.py --bg

defaults:
  - /base
  - _self_

experiment_name: coherence
task: estimate_rank

coherence:
  B: 50
  B_null: 30
  n_p: 25
  p_min: 0.05
  p_max: 0.95
  alpha_tau: 0.95
  ci_level: 0.95
  random_state: 42

datasets:
  - name: mur92
    config: mur92
    k_max: 80
    k_step: 2
  - name: peterson_animals
    config: peterson_animals
    k_max: 100
    k_step: 2
  - name: peterson_various
    config: peterson_various
    k_max: 100
    k_step: 2
  - name: dinov3
    config: dinov3
    k_max: 200
    k_step: 5
  - name: swow
    config: swow
    k_max: 200
    k_step: 5
  - name: nsd_subj01
    config: nsd
    subject_id: 1
    k_max: 200
    k_step: 5
  - name: nsd_subj02
    config: nsd
    subject_id: 2
    k_max: 200
    k_step: 5
  # Commented out: ~3h runtime
  # - name: things_monkey_22k
  #   config: things_monkey_22k_f
  #   k_max: 200
  #   k_step: 5
```

- [ ] **Step 2: Commit**

---

## Task 2: Create experiment script

**Files:**
- Create: `experiments/coherence/__init__.py`
- Create: `experiments/coherence/estimate_rank.py`

The script:
1. Iterates `cfg.datasets`, loading each dataset config from `configs/dataset/{config}.yaml`
2. Builds RSM via `dispatch_dataset_builder`
3. Runs coherence with `compute_incremental_coherence_multi_k_eig_anisotropic` (visualize=False)
4. Extracts k* via three methods:
   - **Activation**: count dimensions whose CI lower bound exceeds null threshold (tau)
   - **Kappa changepoint**: largest jump in leakage rate kappa
   - **Cluster consensus**: boundary of first signal cluster
5. Saves `{name}.json` (k* estimates, params, runtime) and `{name}.npz` (raw arrays)
6. Prints summary table

Key implementation details:
- Load dataset yaml with `OmegaConf.load`, resolve `${paths.data_dir}` via parent config merge
- Use `dispatch_dataset_builder(dataset_cfg, subject_id)` to get RSM
- Import `compute_incremental_coherence_multi_k_eig_anisotropic`, `_estimate_kappa_hat`, `kappa_changepoint`, `analyze_cluster_consensus_across_p` from `src.coherence`
- k_list = `range(1, k_max+1, k_step)` per dataset
- p_list = `np.linspace(p_min, p_max, n_p)`
- JSON must be fully serializable (convert numpy to python types)

- [ ] **Step 1: Create `experiments/coherence/__init__.py`** (empty)

- [ ] **Step 2: Write `experiments/coherence/estimate_rank.py`**

See implementation section below.

- [ ] **Step 3: Verify imports work**

```bash
poetry run python -c "
from src.coherence import (
    compute_incremental_coherence_multi_k_eig_anisotropic,
    _estimate_kappa_hat,
    kappa_changepoint,
    analyze_cluster_consensus_across_p,
)
print('All imports OK')
"
```

- [ ] **Step 4: Dry run on mur92 only (smallest dataset)**

```bash
./scripts/submit experiments/coherence/estimate_rank.py --bg
```

Monitor with `dash`, check JSON output.

- [ ] **Step 5: Commit**

---

## Estimated runtimes (144 cores)

| Dataset | RSM build | Coherence | Total |
|---------|-----------|-----------|-------|
| mur92 | instant | <1s | <1s |
| peterson_animals | instant | <1s | <1s |
| peterson_various | instant | <1s | <1s |
| dinov3 | ~10s | ~1 min | ~1 min |
| swow | ~5s | ~30 min | ~30 min |
| nsd_subj01 | ~5 min | ~42 min | ~47 min |
| nsd_subj02 | ~5 min | ~42 min | ~47 min |
| **Total** | | | **~2h** |
