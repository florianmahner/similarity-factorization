# SRF Low-Data at Coherence-Estimated Ranks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run SRF on THINGS low-data partitions at coherence-estimated ranks (kappa and PCT), producing results and comparison plots against existing VICE and SRF-at-VICE-rank baselines.

**Architecture:** Two new experiment scripts in `experiments/things_behavior/lowdata/` following the exact pattern of `srf_seeds.py`. One runs at kappa ranks, one at PCT ranks. A plot script produces comparison figures. All use joblib parallelism across (pct, partition, seed) tasks.

**Tech Stack:** pysrf (SRF model), joblib (parallelism), numpy, pandas, matplotlib/seaborn.

---

## Reference: Rank configurations

| pct | kappa | PCT | VICE | SRF-existing |
|-----|-------|-----|------|-------------|
| 5% | 6 | 4 | 11 | 10 |
| 10% | 11 | 10 | 20 | 19 |
| 20% | 11 | 14 | 41 | 31 |
| 50% | 21 | 20 | 69 | 53 |
| 100% | 26 | 25 | -- | 66 |

## Reference: Existing code pattern

`experiments/things_behavior/lowdata/srf_seeds.py`:
- Loads partitions from `data/things/partitions/{pct}pct_part{i}/train_90.txt` and `test_10.txt`
- Builds RSM via `compute_similarity_matrix_from_triplets(N, train, alpha=1.0)`
- Fits `SRF(rank=rank, random_state=seed, max_outer=100, max_inner=30)`
- Computes triplet accuracy via softmax odd-one-out
- Saves to CSV with columns: `model, pct, part, seed, rank, val_acc, train_acc, n_train, n_val`
- Parallelizes via `Parallel(n_jobs=n_jobs)` across tasks
- Partitions: {5: 20, 10: 10, 20: 5, 50: 2}, seeds 0-9

## File Structure

### New files to create

| File | Responsibility |
|------|---------------|
| `experiments/things_behavior/lowdata/srf_coherence_ranks.py` | Run SRF at both kappa and PCT ranks across all partitions and seeds |
| `experiments/things_behavior/lowdata/plot_coherence_comparison.py` | Comparison plots: accuracy vs data% for all methods |

### Existing files to read (NOT modify)

| File | Used for |
|------|----------|
| `experiments/things_behavior/lowdata/srf_seeds.py` | Pattern to follow (copy structure) |
| `outputs/experiments/things_behavior/srf_lowdata/results.csv` | Existing SRF-at-VICE-rank results |
| `outputs/experiments/things_behavior/lowdata_comparison/lowdata_comparison.csv` | Existing VICE results |

---

## Task 1: Create SRF coherence ranks experiment

**Files:**
- Create: `experiments/things_behavior/lowdata/srf_coherence_ranks.py`

- [ ] **Step 1: Write the experiment script**

Follow the exact pattern of `srf_seeds.py` but with two rank configs:

```python
KAPPA_RANKS = {5: 6, 10: 11, 20: 11, 50: 21, 100: 26}
PCT_RANKS = {5: 4, 10: 10, 20: 14, 50: 20, 100: 25}
```

Key differences from `srf_seeds.py`:
- Run TWO methods (kappa and PCT) per task, stored in a `method` column
- Add 100% data (full triplets, 1 partition, 10 seeds)
- Output to `outputs/experiments/things_behavior/srf_coherence_ranks/`
- CSV columns: `method, model, pct, part, seed, rank, val_acc, train_acc, n_train, n_val`
- Use `Parallel(n_jobs=-1)` for all tasks
- Incremental saving per batch

For 100% data loading:
```python
if pct == 100:
    from src.utils.io import load_triplets
    train, _ = load_triplets(Path("data/things"))
    # Use the 50pct_part0 test set for validation (consistent with other percentages)
    val = np.loadtxt(PARTITION_DIR / "50pct_part0" / "test_10.txt", dtype=float).astype(int)
```

Total tasks: (20+10+5+2+1) partitions x 10 seeds x 2 methods = 760 tasks.

- [ ] **Step 2: Verify script is syntactically correct**

```bash
poetry run python -c "import ast; ast.parse(open('experiments/things_behavior/lowdata/srf_coherence_ranks.py').read()); print('OK')"
```

- [ ] **Step 3: Run the experiment**

```bash
./scripts/submit experiments/things_behavior/lowdata/srf_coherence_ranks.py --run --n-jobs -1 --bg
```

Expected runtime: ~30-60 min (760 tasks, each ~30-60s, parallelized across 144 cores).

- [ ] **Step 4: Verify results**

```bash
poetry run python -c "
import pandas as pd
df = pd.read_csv('outputs/experiments/things_behavior/srf_coherence_ranks/results.csv')
print(df.groupby(['method', 'pct', 'rank'])['val_acc'].agg(['mean', 'std', 'count']).round(4))
"
```

Expected: 760 rows, mean val_acc similar to sandbox quick-test results.

- [ ] **Step 5: Commit**

```bash
git add experiments/things_behavior/lowdata/srf_coherence_ranks.py
git commit -m "feat: SRF low-data experiment at coherence-estimated ranks (kappa + PCT)"
```

---

## Task 2: Create comparison plot script

**Files:**
- Create: `experiments/things_behavior/lowdata/plot_coherence_comparison.py`

- [ ] **Step 1: Write the plot script**

Loads three data sources:
1. `outputs/experiments/things_behavior/srf_coherence_ranks/results.csv` (new: kappa + PCT ranks)
2. `outputs/experiments/things_behavior/srf_lowdata/results.csv` (existing: VICE-matched ranks)
3. `outputs/experiments/things_behavior/lowdata_comparison/lowdata_comparison.csv` (existing: VICE results)

Produces two plots:

**Plot A: `accuracy_vs_data.pdf`** -- Main comparison figure
- x-axis: training data percentage (5, 10, 20, 50, 100)
- y-axis: validation accuracy (%)
- Lines with error bars (mean +/- std across partitions x seeds):
  - SRF @ PCT rank (blue, solid, circles) -- new
  - SRF @ kappa rank (green, solid, triangles) -- new
  - SRF @ VICE rank (purple, dashed, squares) -- existing
  - VICE (red, dashed, diamonds) -- existing
- Reference lines: chance (33.3%), noise ceiling (66.7%)
- Annotate PCT points with rank used (k=4, k=10, etc.)
- Use `create_figure("single")` from figure_theme

**Plot B: `rank_vs_data.pdf`** -- Rank comparison
- x-axis: training data percentage
- y-axis: rank used
- Lines: PCT, kappa, VICE, SRF-existing
- Shows how different methods select rank

Output directory: same as the experiment results.

- [ ] **Step 2: Run the plot script**

```bash
poetry run python experiments/things_behavior/lowdata/plot_coherence_comparison.py
```

- [ ] **Step 3: Verify plots look correct**

Check that:
- SRF @ PCT/kappa beats VICE at all percentages
- SRF @ PCT/kappa accuracy is monotone (increases with data)
- Rank plot shows PCT/kappa are much lower than VICE

- [ ] **Step 4: Commit**

```bash
git add experiments/things_behavior/lowdata/plot_coherence_comparison.py
git commit -m "feat: comparison plots for SRF at coherence ranks vs VICE"
```
