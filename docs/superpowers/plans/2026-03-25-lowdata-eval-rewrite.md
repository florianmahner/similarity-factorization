# Lowdata Evaluation Rewrite Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Single principled evaluation script that computes triplet accuracy for both SRF and VICE from stored embeddings, using the correct train/val split, with VICE ReLU applied. One CSV output, reproducible.

**Architecture:** Separate computation (SRF fitting) from evaluation (triplet accuracy). The SRF script stores embeddings. A new comparison script loads ALL embeddings (SRF + VICE), evaluates them with one shared accuracy function on one shared val set, and writes one CSV. No accuracy numbers baked into CSVs from training -- everything re-evaluated from embeddings.

**Tech Stack:** numpy, pandas, joblib, pysrf

---

## Key Design Decisions

1. **One val set everywhere:** `data/things/triplets_47/test_10.txt` (412k triplets, 10% of trainset, identical across all partition dirs)
2. **One accuracy function:** hard argmax on dot-product similarities
3. **VICE embeddings:** always `np.maximum(pruned_q_mu, 0)` (ReLU, matches VICE's own eval)
4. **SRF RSM for 100%:** built from `train_90.txt` (3.71M), never `trainset.txt` (4.12M)
5. **Outputs:** single `lowdata_comparison.csv` with columns: `model, pct, part, seed, rank, val_acc`

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `experiments/analyses/things_behavior/lowdata/comparison/run.py` | **Rewrite** | Evaluate all embeddings, produce one CSV |
| `experiments/analyses/things_behavior/lowdata/comparison/plot.py` | **Update** | Read new CSV, generate figures |
| `experiments/analyses/things_behavior/lowdata/srf/run_alpha_comparison.py` | **Keep** | Produces SRF embeddings (already fixed for 100% leak) |

---

### Task 1: Rewrite comparison/run.py

**Files:**
- Rewrite: `experiments/analyses/things_behavior/lowdata/comparison/run.py`

The script must:
- Load the shared val set once
- Find all SRF embeddings (from `srf/outputs/kappa_alpha1/embeddings/`)
- Find all VICE embeddings (from `vice/outputs/models/`)
- Evaluate every embedding with the same function
- Save one CSV

- [ ] **Step 1: Write the new run.py**

```python
"""Evaluate SRF and VICE embeddings on the shared THINGS triplet val set.

Loads stored embeddings, applies the same accuracy function to all.
VICE: ReLU applied to pruned_q_mu (matches VICE's own evaluation).
SRF: embeddings used as-is (already non-negative).

Outputs:
    lowdata_comparison.csv  -- model, pct, part, seed, rank, val_acc
"""
```

Core logic:
- `_load_val_triplets()` -> loads `triplets_47/test_10.txt` once
- `_triplet_accuracy(embedding, triplets)` -> shared accuracy function
- `_collect_srf(srf_dir, val)` -> glob `srf_*pct_*.npz`, parse name, eval
- `_collect_vice(vice_dir, val)` -> glob `vice_*pct_*/parameters.npz`, parse name, ReLU, eval
- `run(cfg)` -> calls both collectors, concats, saves CSV

- [ ] **Step 2: Verify it runs**

```bash
./scripts/submit experiments/analyses/things_behavior/lowdata/comparison/run.py --bg
```

- [ ] **Step 3: Validate output**

Check that the CSV has correct counts per (model, pct) and that VICE has ReLU applied.

- [ ] **Step 4: Commit**

---

### Task 2: Run SRF alpha comparison with fixed 100% split

The `run_alpha_comparison.py` is already fixed (uses `train_90.txt` for 100%). But the outputs were wiped. Rerun for alpha=1 only (the main comparison).

- [ ] **Step 1: Launch**

```bash
./scripts/submit experiments/analyses/things_behavior/lowdata/srf/run_alpha_comparison.py --alpha 1.0 --bg
```

- [ ] **Step 2: Verify outputs**

```
outputs/kappa_alpha1/ranks.json
outputs/kappa_alpha1/results.csv
outputs/kappa_alpha1/embeddings/ (380 files)
```

---

### Task 3: Run comparison evaluation

After Task 2 completes, run the new comparison script.

- [ ] **Step 1: Update paths in run.py to point to `kappa_alpha1/`**
- [ ] **Step 2: Run and verify CSV**
- [ ] **Step 3: Commit**

---

### Task 4: Update plot.py

- [ ] **Step 1: Update to read new CSV format**
- [ ] **Step 2: Generate figures**
- [ ] **Step 3: Commit**

---

## Dependency Graph

```
Task 1 (rewrite run.py) ─┐
Task 2 (SRF rerun)  ─────┼── Task 3 (run comparison) ── Task 4 (plot)
```

Tasks 1 and 2 can run in parallel.
