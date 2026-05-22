# Kappa Rank Detection Simulation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean experiment comparing coherence kappa rank detection against standard heuristics (BIC, elbow, variance-90%) on a large simulation grid, producing publication-quality detected-vs-true scatter plots.

**Architecture:** Single experiment script `experiments/simulation/kappa_rank_detection.py` generates a CSV with detected ranks for each method across a grid of (true_rank x alpha x SNR x seed). A separate plotting script produces two figure panels: vary-alpha (fixed SNR) and vary-SNR (fixed alpha). All computation parallelized with joblib.

**Tech Stack:** numpy, pandas, joblib, pysrf (SRF for BIC), kneed (elbow detection), src/coherence (kappa), matplotlib/seaborn for plots via src/utils/figure_theme.

---

## Cleanup

Before starting, remove all the hacky files created during exploration:

- Delete: `experiments/simulation/kappa_rank_grid.py`
- Delete: `experiments/simulation/plot_kappa_grid.py`
- Delete: `sandbox/simulation/kappa_rank_grid/` (entire directory)
- Delete: `outputs/experiments/simulation/kappa_rank_grid/` (entire directory)

## File Structure

| File | Purpose |
|------|---------|
| `experiments/simulation/kappa_rank_detection.py` | Simulation grid: compute detected rank for each method per condition. Saves CSV. |
| `experiments/simulation/plot_kappa_rank_detection.py` | Load CSV, produce publication-quality PDF figures. |

## Simulation Design

**Grid (n=300, all combinations):**
- `true_ranks`: [5, 10, 15, 20, 25, 30, 35, 40]
- `alphas`: [0.1, 0.5, 1.0, 2.0, 5.0]
- `snrs`: [0.3, 0.5, 0.7, 1.0]
- `n_seeds`: 5

Total: 8 x 5 x 4 x 5 = 800 conditions.

**Methods:**
1. **Kappa (ours):** Coherence changepoint via eigenspace bootstrap. No model fitting.
2. **BIC:** Fit SRF at each candidate rank on full matrix, pick rank minimizing BIC. Standard NMF approach.
3. **Elbow:** Compute eigenvalues, find knee point using `kneed.KneeLocator` on the scree curve (eigenvalue vs index). Very common heuristic.
4. **Variance 90%:** Pick smallest rank where cumulative eigenvalue variance > 90%. Common PCA/factor analysis default.

**CSV columns:** `true_rank, alpha, snr, seed, rank_kappa, rank_bic, rank_elbow, rank_var90, error_kappa, error_bic, error_elbow, error_var90`

## Figure Design

Two panels showing detected rank (y) vs true rank (x) as scatter plots with identity line:

**Panel a: Vary alpha (SNR = 1.0)**
- Rows of subpanels, one per alpha value (0.1, 1.0, 5.0 -- three representative values)
- Each subpanel: detected vs true, with kappa (blue circles) and best baseline (red squares) overlaid
- Identity line (gray diagonal)
- Jittered individual points (transparent) + median line (solid)

**Panel b: Vary SNR (alpha = 5.0 -- the regime where baselines struggle)**
- Same format, one subpanel per SNR (0.3, 0.5, 0.7, 1.0)

Use `src/utils/figure_theme` with `create_figure`, `despine`, `save_figure`. PDF only.

---

### Task 1: Cleanup old files

- [ ] **Step 1: Remove hacky exploration files**

```bash
rm experiments/simulation/kappa_rank_grid.py
rm experiments/simulation/plot_kappa_grid.py
rm -rf sandbox/simulation/kappa_rank_grid/
rm -rf outputs/experiments/simulation/kappa_rank_grid/
```

- [ ] **Step 2: Verify cleanup**

```bash
ls experiments/simulation/kappa_rank* 2>/dev/null  # should be empty
ls sandbox/simulation/kappa_rank* 2>/dev/null  # should be empty
```

---

### Task 2: Write the simulation script

**Files:**
- Create: `experiments/simulation/kappa_rank_detection.py`

The script reuses `_make_similarity` and `_select_rank_variance` patterns from `experiments/simulation/rank_detection.py` but is standalone (no Hydra, no CV).

- [ ] **Step 1: Write `experiments/simulation/kappa_rank_detection.py`**

Key functions:
- `_make_similarity(n, k, alpha, snr, seed)` -- Dirichlet simulation + noise
- `_select_kappa(similarity, max_k)` -- coherence bootstrap + kappa changepoint
- `_select_bic(similarity, candidate_ranks, seed)` -- fit SRF at each rank, pick min BIC
- `_select_elbow(eigvals)` -- `kneed.KneeLocator` on eigenvalue scree curve
- `_select_var90(eigvals)` -- cumulative variance > 90%
- `_run_one(true_rank, alpha, snr, seed)` -- run all 4 methods on one condition, return dict
- `main()` -- build grid, run with `Parallel(n_jobs=-1)`, save CSV

Grid constants at top of file:
```python
N = 300
N_SEEDS = 5
TRUE_RANKS = [5, 10, 15, 20, 25, 30, 35, 40]
ALPHAS = [0.1, 0.5, 1.0, 2.0, 5.0]
SNRS = [0.3, 0.5, 0.7, 1.0]
OUTPUT_DIR = Path("outputs/experiments/simulation/kappa_rank_detection")
```

For elbow detection:
```python
from kneed import KneeLocator
def _select_elbow(eigvals):
    x = np.arange(1, len(eigvals) + 1)
    kn = KneeLocator(x, eigvals, curve="convex", direction="decreasing")
    return int(kn.knee) if kn.knee is not None else 1
```

For BIC, use candidate_ranks = range(max(2, true_rank - 15), true_rank + 16, 2), ensuring true_rank is included.

For kappa, use `compute_incremental_coherence_multi_k_eig_anisotropic` with `B=50, B_null=0, compute_null=False, n_jobs=1` (inner parallelism off since outer loop is parallelized).

- [ ] **Step 2: Run the experiment**

```bash
./scripts/submit experiments/simulation/kappa_rank_detection.py --bg
```

- [ ] **Step 3: Verify CSV output**

```bash
wc -l outputs/experiments/simulation/kappa_rank_detection/kappa_rank_detection.csv
# Expected: 801 (800 data rows + header)
```

- [ ] **Step 4: Commit**

```bash
git add experiments/simulation/kappa_rank_detection.py
git commit -m "feat: kappa rank detection simulation grid (n=300, 800 conditions)"
```

---

### Task 3: Write the plotting script

**Files:**
- Create: `experiments/simulation/plot_kappa_rank_detection.py`

- [ ] **Step 1: Write `experiments/simulation/plot_kappa_rank_detection.py`**

Two functions:
- `plot_vary_alpha(df, output_dir)` -- 3 subpanels (alpha=0.1, 1.0, 5.0), SNR fixed at 1.0. Detected vs true scatter for kappa + BIC + elbow.
- `plot_vary_snr(df, output_dir)` -- 4 subpanels (SNR=0.3, 0.5, 0.7, 1.0), alpha fixed at 5.0. Same format.

Each subpanel:
- Gray identity line
- Jittered scatter points per method (alpha=0.3 transparency)
- Median line per method connecting the rank medians
- Legend in first panel only
- Square aspect ratio, `despine(ax)`
- PDF output via `save_figure`

Methods to show (pick 3 for visual clarity, drop var90 if it's always terrible -- can mention in text):
- Kappa: blue circles
- BIC: red squares
- Elbow: green diamonds

- [ ] **Step 2: Run plotting**

```bash
poetry run python experiments/simulation/plot_kappa_rank_detection.py
```

- [ ] **Step 3: Verify PDFs exist**

```bash
ls outputs/experiments/simulation/kappa_rank_detection/*.pdf
```

- [ ] **Step 4: Commit**

```bash
git add experiments/simulation/plot_kappa_rank_detection.py
git commit -m "feat: publication-quality rank detection plots (kappa vs BIC vs elbow)"
```
