# Alpha=0 Low-Data Rerun Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch the THINGS triplet RSM construction from alpha=1 (Laplace smoothing) to alpha=0 (raw proportions) as the default, rerun kappa rank estimation and the lowdata SRF comparison, and compare accuracy and rank estimates against the existing alpha=1 results.

**Architecture:** Change the default alpha in `configs/dataset/things_behavior.yaml` from implicit 1.0 to explicit 0.0. Keep existing alpha=1 lowdata results in a nested subfolder for comparison. Recompute kappa ranks for all THINGS percentages (they don't currently exist in the kappa pipeline -- requires adding triplet partition support to kappa/run.py), recompute PCT ranks, then rerun the lowdata SRF experiment with the new ranks.

**Tech Stack:** pysrf, joblib, Hydra configs, coherence estimation pipeline

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `configs/dataset/things_behavior.yaml` | Modify | Add `alpha: 0.0` |
| `experiments/datasets/ranks/kappa/run.py` | Modify | Add triplet partition loading (mirrors PCT's approach) |
| `experiments/datasets/ranks/kappa/config.yaml` | Modify | Add THINGS percentages (5,10,20,50,100) + `things_behavior` entry |
| `experiments/analyses/things_behavior/ranks.py` | Verify | Confirm `load_things_ranks(method="kappa")` key matches kappa output |
| `experiments/datasets/ranks/pct/run.py` | Verify | Already uses alpha=0 -- no change |
| `experiments/analyses/things_behavior/lowdata/srf/coherence_ranks.py` | Modify | Change alpha=1.0 to alpha=0.0 |
| `experiments/analyses/things_behavior/lowdata/comparison/run.py` | Modify | Change alpha=1.0 to alpha=0.0 |
| `experiments/analyses/things_behavior/lowdata/srf/outputs/` | Rename | Archive to `outputs_alpha1/` |
| `experiments/analyses/things_behavior/lowdata/comparison/outputs/` | Rename | Archive to `outputs_alpha1/` |

---

### Task 1: Archive existing alpha=1 results

- [ ] **Step 1: Move existing SRF lowdata results**

```bash
cd /LOCAL/fmahner/similarity-factorization
mv experiments/analyses/things_behavior/lowdata/srf/outputs \
   experiments/analyses/things_behavior/lowdata/srf/outputs_alpha1
```

- [ ] **Step 2: Move existing comparison results**

```bash
mv experiments/analyses/things_behavior/lowdata/comparison/outputs \
   experiments/analyses/things_behavior/lowdata/comparison/outputs_alpha1
```

- [ ] **Step 3: Verify archives exist**

```bash
ls experiments/analyses/things_behavior/lowdata/srf/outputs_alpha1/results.csv
ls experiments/analyses/things_behavior/lowdata/comparison/outputs_alpha1/lowdata_comparison.csv
```

---

### Task 2: Set alpha=0 as default for THINGS config

**Files:**
- Modify: `configs/dataset/things_behavior.yaml`

- [ ] **Step 1: Add alpha field to config**

Add `alpha: 0.0` after `n_objects`:

```yaml
# @package dataset
# THINGS Behavioral Triplet Similarity

name: things_behavior
type: triplet
path: ${paths.data_dir}/things

# Loading parameters
triplet_number: "4.7mio"
n_objects: 1854
alpha: 0.0

# Cross-experiment references
bounds_task: things_behavior
```

- [ ] **Step 2: Verify build_similarity picks up new alpha**

```bash
poetry run python -c "
from similarity import build_similarity
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/dataset/things_behavior.yaml')
parent = OmegaConf.create({'paths': {'data_dir': 'data'}, 'dataset': cfg})
OmegaConf.resolve(parent)
print(f'alpha = {parent.dataset.get(\"alpha\", 1.0)}')
s = build_similarity(parent.dataset)
import numpy as np
offdiag = s[~numpy.eye(s.shape[0], dtype=bool)]
print(f'RSM mean={offdiag.mean():.4f} (expect ~0.33 for alpha=0)')
"
```

- [ ] **Step 3: Commit**

```bash
git add configs/dataset/things_behavior.yaml
git commit -m "config: set alpha=0 (no Laplace smoothing) for things_behavior"
```

---

### Task 3: Update lowdata scripts to use alpha=0

**Files:**
- Modify: `experiments/analyses/things_behavior/lowdata/srf/coherence_ranks.py:111`
- Modify: `experiments/analyses/things_behavior/lowdata/comparison/run.py`

- [ ] **Step 1: Update coherence_ranks.py line 111**

```python
# Change from:
rsm = compute_similarity_matrix_from_triplets(N_OBJECTS, train, alpha=1.0)
# To:
rsm = compute_similarity_matrix_from_triplets(N_OBJECTS, train, alpha=0.0)
```

- [ ] **Step 2: Update comparison/run.py**

Find `compute_similarity_matrix_from_triplets(N_OBJECTS, train_triplets, alpha=1.0)` and change to `alpha=0.0`.

- [ ] **Step 3: Commit**

```bash
git add experiments/analyses/things_behavior/lowdata/srf/coherence_ranks.py
git add experiments/analyses/things_behavior/lowdata/comparison/run.py
git commit -m "feat: use alpha=0 for THINGS triplet RSM in lowdata experiments"
```

---

### Task 4: Add triplet partition support to kappa run.py

**CRITICAL:** The kappa `run.py` currently uses `build_similarity(dataset_cfg)` which always loads the full dataset. It has no `triplet_pct` support. The PCT `run.py` has `_load_things_triplet_rsm()` (line 40-47) that handles this.

**Files:**
- Modify: `experiments/datasets/ranks/kappa/run.py`

- [ ] **Step 1: Add triplet partition loading to kappa/run.py**

In `_process_dataset()`, add triplet partition support mirroring PCT's approach. Before `build_similarity()` call (~line 65), add:

```python
triplet_pct = ds_entry.get("triplet_pct", None)
if triplet_pct is not None:
    from utils.helpers import compute_similarity_matrix_from_triplets
    from utils.io import load_triplets
    data_dir = project_root / "data" / "things"
    n_objects = 1854
    triplet_partition = ds_entry.get("triplet_partition", 0)
    if triplet_pct == 100:
        triplets, _ = load_triplets(data_dir)
    else:
        path = data_dir / "partitions" / f"{triplet_pct}pct_part{triplet_partition}" / "train_90.txt"
        triplets = np.loadtxt(path).astype(int)
    alpha = dataset_cfg.get("alpha", 0.0)
    similarity = compute_similarity_matrix_from_triplets(n_objects, triplets, alpha=alpha)
    log.info(f"Loaded THINGS triplets: {triplet_pct}% (alpha={alpha}), {len(triplets)} triplets")
else:
    similarity = build_similarity(dataset_cfg, subject_id=subject_id)
```

- [ ] **Step 2: Fix kappa output key to match ranks.py expectations**

The kappa `run.py` saves rank as `"k_star"` (line 146), but `ranks.py:load_things_ranks(method="kappa")` looks for `"k_star_kappa"`.

In `_process_dataset()`, change the output key from `"k_star"` to `"k_star_kappa"`:

```python
# Line ~146, change:
"k_star": int(k_star),
# To:
"k_star_kappa": int(k_star),
```

**Also add `"k_star": int(k_star)` as an alias** so existing code that reads `k_star` still works.

- [ ] **Step 3: Commit**

```bash
git add experiments/datasets/ranks/kappa/run.py
git commit -m "feat: add triplet partition support to kappa rank estimation"
```

---

### Task 5: Add THINGS entries to kappa config and run

**Files:**
- Modify: `experiments/datasets/ranks/kappa/config.yaml`

- [ ] **Step 1: Add THINGS entries to kappa config**

Append to `datasets:` list:

```yaml
  # --- THINGS behavioral triplets (alpha=0, raw proportions) ---
  - name: things_5pct
    config: things_behavior
    triplet_pct: 5
    triplet_partition: 0
    k_max: 60
    k_step: 1
    center: false
  - name: things_10pct
    config: things_behavior
    triplet_pct: 10
    triplet_partition: 0
    k_max: 60
    k_step: 1
    center: false
  - name: things_20pct
    config: things_behavior
    triplet_pct: 20
    triplet_partition: 0
    k_max: 60
    k_step: 1
    center: false
  - name: things_50pct
    config: things_behavior
    triplet_pct: 50
    triplet_partition: 0
    k_max: 60
    k_step: 1
    center: false
  - name: things_100pct
    config: things_behavior
    triplet_pct: 100
    k_max: 80
    k_step: 1
    center: false
```

- [ ] **Step 2: Run kappa**

```bash
./scripts/submit experiments/datasets/ranks/kappa/run.py skip_existing=true --bg
```

Expected runtime: ~10 min per percentage. Monitor with `dash`.

- [ ] **Step 3: Verify kappa outputs and key format**

```bash
for f in experiments/datasets/ranks/kappa/outputs/things_*pct.json; do
  echo "$(basename $f): $(python3 -c "import json; d=json.load(open('$f')); print(f'k_star_kappa={d.get(\"k_star_kappa\", \"MISSING\")}')")"
done
```

Expected: `k_star_kappa` key exists in each file. For 100%: k* ~29.

---

### Task 6: Verify PCT ranks (no rerun needed)

PCT already uses alpha=0 (`pct/run.py:47`).

- [ ] **Step 1: Verify existing PCT outputs**

```bash
for f in experiments/datasets/ranks/pct/outputs/things_*pct.json; do
  echo "$(basename $f): $(python3 -c "import json; print(f'k*={json.load(open(\"$f\"))[\"k_star_pct\"]}')")"
done
```

Expected: k* values for all 5 percentages (4, 10, 14, 20, 25). No rerun needed.

---

### Task 7: Verify rank loading and run lowdata SRF

**Depends on:** Tasks 4, 5, 6

- [ ] **Step 1: Verify rank loading works**

```bash
poetry run python -c "
from experiments.analyses.things_behavior.ranks import load_things_ranks
kappa = load_things_ranks(method='kappa')
pct = load_things_ranks(method='pct')
print(f'Kappa ranks: {kappa}')
print(f'PCT ranks: {pct}')
"
```

Expected: both dicts populated for all 5 percentages.

- [ ] **Step 2: Run SRF lowdata**

```bash
./scripts/submit experiments/analyses/things_behavior/lowdata/srf/coherence_ranks.py --run --bg
```

Expected runtime: ~30-60 min (joblib parallelized).

- [ ] **Step 3: Verify outputs**

```bash
poetry run python -c "
import pandas as pd
df = pd.read_csv('experiments/analyses/things_behavior/lowdata/srf/outputs/results.csv')
print(df.groupby(['method', 'pct', 'rank'])['val_acc'].agg(['mean', 'std', 'count']).round(4))
"
```

---

### Task 8: Rerun lowdata comparison (SRF vs VICE)

**Depends on:** Task 7

- [ ] **Step 1: Run comparison**

```bash
./scripts/submit experiments/analyses/things_behavior/lowdata/comparison/run.py --bg
```

- [ ] **Step 2: Generate plots**

```bash
poetry run python experiments/analyses/things_behavior/lowdata/comparison/plot.py
```

- [ ] **Step 3: Compare alpha=0 vs alpha=1**

```bash
poetry run python -c "
import pandas as pd
df0 = pd.read_csv('experiments/analyses/things_behavior/lowdata/comparison/outputs/lowdata_comparison.csv')
df1 = pd.read_csv('experiments/analyses/things_behavior/lowdata/comparison/outputs_alpha1/lowdata_comparison.csv')

print('=== Alpha=0 (new) ===')
for pct in [5, 10, 20, 50, 100]:
    for model in ['SRF', 'VICE']:
        vals = df0[(df0['model']==model) & (df0['pct']==pct)]['val_acc']
        if len(vals) > 0:
            print(f'  {model} {pct:>3d}%: {vals.mean():.4f} +/- {vals.std():.4f} (n={len(vals)})')

print()
print('=== Alpha=1 (old) ===')
for pct in [5, 10, 20, 50, 100]:
    for model in ['SRF', 'VICE']:
        vals = df1[(df1['model']==model) & (df1['pct']==pct)]['val_acc']
        if len(vals) > 0:
            print(f'  {model} {pct:>3d}%: {vals.mean():.4f} +/- {vals.std():.4f} (n={len(vals)})')
"
```

---

### Task 9: Compare kappa rank trajectories (key result)

**Depends on:** Task 5 (kappa outputs) and Task 8 (accuracy data)

- [ ] **Step 1: Rank comparison table**

```bash
poetry run python -c "
import json
from pathlib import Path

kappa_dir = Path('experiments/datasets/ranks/kappa/outputs')
pct_dir = Path('experiments/datasets/ranks/pct/outputs')
archive_dir = Path('archive/outputs_arxiv_200326/experiments/coherence/kappa')

print(f'{\"Pct\":>5s} {\"kappa(a=0)\":>12s} {\"kappa(a=1)\":>12s} {\"PCT(a=0)\":>10s}')
for pct in [5, 10, 20, 50, 100]:
    name = f'things_{pct}pct'
    kf = kappa_dir / f'{name}.json'
    k_new = json.loads(kf.read_text()).get('k_star_kappa', '?') if kf.exists() else '?'
    af = archive_dir / f'{name}.json'
    k_old = json.loads(af.read_text()).get('k_star_kappa', '?') if af.exists() else '?'
    pf = pct_dir / f'{name}.json'
    k_pct = json.loads(pf.read_text())['k_star_pct'] if pf.exists() else '?'
    print(f'{pct:>5d} {str(k_new):>12s} {str(k_old):>12s} {str(k_pct):>10s}')
"
```

**This is the key result:** do alpha=0 kappa ranks grow more smoothly with data fraction?

---

## Dependency Graph

```
Task 1 (archive) ─────────────────────────────────────┐
Task 2 (config) ──┬── Task 4 (kappa code fix) ─┐      │
Task 3 (code)  ───┤                             │      │
                  ├── Task 5 (kappa config+run) ┤      │
                  │   (depends on Task 4)       │      │
                  │                             ├── Task 7 (SRF lowdata) ── Task 8 (comparison) ── Task 9
                  └── Task 6 (PCT verify) ──────┘
```

Tasks 1, 2, 3 can run in parallel.
Task 4 needs Task 2 done (config must have alpha=0 before kappa reads it).
Task 5 needs Task 4 done (code fix before running).
Tasks 5 and 6 can run in parallel.
Task 7 needs Tasks 5 and 6 done.
Task 8 needs Task 7 done.
Task 9 needs Tasks 5 and 8 done.

---

## Out of Scope (for now)

These use default alpha=1.0 but are not critical for the lowdata comparison:
- `experiments/analyses/things_behavior/reliability/run.py` (line 76, default alpha)
- `experiments/analyses/things_behavior/similarity_48/run.py` (line 28, default alpha)
- `experiments/datasets/consensus/run.py` (uses build_similarity, now alpha=0 via config)

Consensus will automatically pick up alpha=0 from the config change. Reliability and similarity_48 use direct calls with default alpha -- leave for a follow-up if alpha=0 results are promising.
