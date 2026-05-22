# SWOW PPMI Cleanup + Coherence Rank Selection

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up SWOW to use PPMI only (delete RW code), visualize a rank-20 embedding, and run coherence diagnostics to find optimal rank.

**Architecture:** (1) Remove RW code from src/datasets/swow.py and simplify config, (2) sandbox script to fit rank-20 using the exact stored similarity.npy, (3) sandbox script to run v2 coherence on the stored PPMI matrix.

**Tech Stack:** pysrf (SRF), v2 coherence from sandbox/coherence/kachun/v2, matplotlib via figure_theme.

---

## File Structure

```
Modified:
  src/datasets/swow.py              # Delete RW functions, keep PPMI only
  src/datasets/loaders.py            # Update load_swow to remove RW dispatch
  src/similarity/datasets.py         # Simplify build_word_association
  configs/dataset/swow.yaml          # PPMI only, remove RW params

Created:
  sandbox/swow/ppmi_embedding/run.py      # Fit rank-20, visualize
  sandbox/swow/ppmi_coherence/run.py      # V2 coherence diagnostic
```

---

## Task 1: Clean up SWOW config and delete RW code

**Files:**
- Modify: `configs/dataset/swow.yaml`
- Modify: `src/datasets/swow.py`
- Modify: `src/datasets/loaders.py`
- Modify: `src/similarity/datasets.py`

- [ ] **Step 1: Update swow.yaml to PPMI-only**

Replace the entire file with:

```yaml
# @package dataset
# SWOW Word Association Similarity (PPMI)
#
# Positive Pointwise Mutual Information from word association counts.
# Reference: De Deyne et al. (2019) "The Small World of Words"

name: swow
type: word_association
path: ${paths.data_dir}/small-world-of-words

use_all_responses: true
top_n_words: null
min_word_length: 1
symmetrization: sum
bidirectional_only: false

bounds_task: swow
rank_range: [5, 80, 5]
```

- [ ] **Step 2: Delete RW functions from src/datasets/swow.py**

Remove these functions and their imports:
- `_l1_normalize` (lines 109-113)
- `_ppmi_rw` (lines 116-137)
- `_katz_walk` (lines 140-148)
- `compute_rw_similarity` (lines 151-180)
- `load_swow_rw` (lines 311-380)
- `load_swow_similarity` (lines 383-444)

Remove the imports that are only needed for RW:
- `from scipy.linalg import solve`
- `from sklearn.metrics.pairwise import cosine_similarity`

Update the module docstring to remove RW references.

The file should keep: `load_swow_data`, `filter_by_word_length`, `_select_vocabulary`,
`compute_ppmi`, `make_ppmi_graph`, `load_swow_ppmi`.

- [ ] **Step 3: Update src/datasets/loaders.py**

Change the import (line 21) from:
```python
from .swow import load_swow_ppmi, load_swow_similarity
```
to:
```python
from .swow import load_swow_ppmi
```

Update `load_swow` function (around line 491) to remove the `similarity_method`
dispatch and always use `load_swow_ppmi` directly.

- [ ] **Step 4: Simplify build_word_association in src/similarity/datasets.py**

Remove the `similarity_method` and `alpha` kwargs from the `load_dataset` call
(lines 94-104). The function should just pass PPMI parameters.

- [ ] **Step 5: Verify nothing is broken**

Run: `poetry run python -c "from src.datasets.swow import load_swow_ppmi; print('OK')"`

Run: `poetry run python -c "
from src.similarity import build_similarity
from omegaconf import OmegaConf
cfg = OmegaConf.create({'name': 'swow', 'type': 'word_association', 'path': 'data/small-world-of-words', 'use_all_responses': True, 'symmetrization': 'sum', 'bidirectional_only': False, 'min_word_length': 1})
s = build_similarity(cfg)
print(f'OK: {s.shape}')
"`

- [ ] **Step 6: Commit**

```
git add src/datasets/swow.py src/datasets/loaders.py src/similarity/datasets.py configs/dataset/swow.yaml
git commit -m "refactor: remove RW similarity from SWOW, keep PPMI only"
```

---

## Task 2: Sandbox rank-20 PPMI embedding

**Files:**
- Create: `sandbox/swow/ppmi_embedding/run.py`

- [ ] **Step 1: Write the script**

Load the stored `similarity.npy` from `outputs/experiments/word_association/generate_embedding/50/`.
This is the exact PPMI matrix (8593x8593, diagonal=NaN) used in production.
Fit SRF at rank=20 with the exact same parameters: `rho=3.0, max_outer=1000, missing_values=np.nan`.
Load existing rank-50 embedding for comparison.
Plot: top 10 words for first 5 dimensions (rank-20 and rank-50), sparsity per dimension, eigenspectrum.

- [ ] **Step 2: Run**

`./scripts/submit sandbox/swow/ppmi_embedding/run.py --bg`

- [ ] **Step 3: Inspect outputs and commit**

---

## Task 3: Coherence diagnostic on SWOW PPMI

**Files:**
- Create: `sandbox/swow/ppmi_coherence/run.py`

- [ ] **Step 1: Write the script**

Load the stored `similarity.npy` (8593x8593 PPMI).
Run v2 coherence: `compute_incremental_coherence_multi_k_eig_anisotropic`.

Key parameters:
- k_list: `list(range(5, 101, 5)) + list(range(120, 201, 20))` (fine grid to 100, coarse to 200)
- p_list: `np.linspace(0.05, 0.95, 15)` (15 points, coarser for speed)
- B=20, B_null=20 (reduced for the large matrix)

Extract per-component activation, fraction-above-null, excess at p=0.50 and p=0.80.

This matrix has diagonal=NaN (8593 missing values) and 98.7% zeros.
The v2 code handles NaN via `_symmetrize_with_nan` and `_prepare_observation_mask`.
The observation rate q will be ~0.9999 (only diagonal missing), so the reference
eigenspace will use exact eigh (not randomized).

Important: the eigendecomposition of this 8593x8593 matrix for k=200 will be
expensive. Each eigh call takes ~10s. With B=20, P=15: 300 eigendecompositions
= ~50 minutes. Plus null computation.

Expected runtime: 1-3 hours.

- [ ] **Step 2: Run**

`./scripts/submit sandbox/swow/ppmi_coherence/run.py --bg`

- [ ] **Step 3: Inspect results**

Check:
- Does the activation profile show a clear cutoff?
- How does it compare to the RW result (which showed all 300 dims at p=0.05)?
- Does the PPMI matrix give a more discriminating coherence test?
