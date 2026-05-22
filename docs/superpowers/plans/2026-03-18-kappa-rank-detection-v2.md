# Kappa Rank Detection v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace BIC with three principled baselines (parallel analysis, cophenetic correlation, elbow) and increase kappa bootstrap parameters for more robust rank detection. Rerun the 800-condition grid and produce publication-quality scatter plots.

**Architecture:** Modify the existing `experiments/simulation/kappa_rank_detection.py` to replace `_select_bic` with `_select_parallel_analysis` and `_select_cophenetic`. Keep elbow. Increase kappa's `B` from 50 to 100 and `hi_band_quantile` from 0.85 to 0.90. Rerun and replot.

**Tech Stack:** numpy, scipy (cophenetic correlation, linkage), pysrf (SRF for cophenetic), kneed (elbow), src/coherence (kappa), matplotlib via src/utils/figure_theme.

---

## Method Descriptions

### 1. Kappa (ours) -- improved parameters
Same coherence changepoint but with `B=100` (was 50) for more stable bootstrap estimates and `hi_band_quantile=0.90` (was 0.85) for a stricter high-p band.

### 2. Parallel Analysis (Horn, 1965)
Very common in psychology/neuroscience for factor analysis. Algorithm:
1. Compute eigenvalues of the observed similarity matrix
2. Generate `n_iter` random matrices of the same size (shuffle each column independently to preserve marginal distributions, OR generate from Wishart null)
3. Compute eigenvalues of each random matrix
4. Take the 95th percentile of random eigenvalues at each position
5. Select rank = number of observed eigenvalues exceeding the random threshold

For similarity matrices (S = WW^T), the null should be a random positive semidefinite matrix. Simplest approach: generate random W_null with same dimensions and compute S_null = W_null @ W_null^T, then eigendecompose.

### 3. Cophenetic Correlation (Brunet et al., 2004)
The standard NMF stability approach:
1. For each candidate rank k, fit SRF `n_runs` times with different seeds
2. Compute connectivity matrix: C_ij = 1 if items i,j share the same dominant dimension
3. Average connectivity across runs to get consensus matrix
4. Compute cophenetic correlation of the consensus matrix (how well hierarchical clustering preserves pairwise distances)
5. Select rank with highest cophenetic correlation

This requires fitting SRF many times per candidate rank, so it's expensive. Use a reduced set of candidate ranks (step=2 or 3) and fewer runs (n_runs=10).

### 4. Elbow (unchanged)
KneeLocator on eigenvalue scree curve. Keep as-is.

## CSV columns (updated)
`true_rank, alpha, snr, seed, rank_kappa, rank_parallel, rank_cophenetic, rank_elbow, error_kappa, error_parallel, error_cophenetic, error_elbow`

---

### Task 1: Update simulation script with new methods

**Files:**
- Modify: `experiments/simulation/kappa_rank_detection.py`

- [ ] **Step 1: Read the existing script**

Read `experiments/simulation/kappa_rank_detection.py` to understand the current structure.

- [ ] **Step 2: Replace `_select_bic` with `_select_parallel_analysis`**

```python
def _select_parallel_analysis(eigvals: np.ndarray, n: int, n_iter: int = 100, seed: int = 0) -> int:
    """Parallel analysis: keep eigenvalues exceeding random null (95th percentile)."""
    rng = np.random.default_rng(seed)
    k_max = len(eigvals)
    null_eigvals = np.zeros((n_iter, k_max))
    for i in range(n_iter):
        # Random PSD matrix of same size: X @ X.T where X is n x n with shuffled entries
        x_rand = rng.standard_normal((n, n))
        s_rand = x_rand @ x_rand.T / n
        eig_rand = np.linalg.eigvalsh(s_rand)[::-1]
        null_eigvals[i] = eig_rand[:k_max]
    threshold = np.percentile(null_eigvals, 95, axis=0)
    n_above = np.sum(eigvals[:k_max] > threshold)
    return max(1, int(n_above))
```

- [ ] **Step 3: Add `_select_cophenetic`**

```python
def _select_cophenetic(similarity: np.ndarray, candidate_ranks: list[int], n_runs: int = 10, seed: int = 0) -> int:
    """Cophenetic correlation: pick rank with highest consensus stability."""
    from scipy.cluster.hierarchy import linkage, cophenet
    from scipy.spatial.distance import squareform

    best_rank = candidate_ranks[0]
    best_coph = -1.0
    n = similarity.shape[0]

    for rank in candidate_ranks:
        # Run SRF n_runs times, compute connectivity matrices
        consensus = np.zeros((n, n))
        for r in range(n_runs):
            model = SRF(rank=rank, random_state=seed + r, max_outer=50, max_inner=20)
            model.fit(similarity)
            w = model.embedding_
            assignments = np.argmax(w, axis=1)
            connectivity = (assignments[:, None] == assignments[None, :]).astype(float)
            consensus += connectivity
        consensus /= n_runs

        # Cophenetic correlation
        dist = 1.0 - consensus
        np.fill_diagonal(dist, 0)
        dist = np.maximum(dist, 0)  # numerical safety
        dist_vec = squareform(dist, checks=False)
        Z = linkage(dist_vec, method="average")
        coph_corr, _ = cophenet(Z, dist_vec)

        if coph_corr > best_coph:
            best_coph = coph_corr
            best_rank = rank

    return best_rank
```

- [ ] **Step 4: Update kappa parameters**

Change in `_select_kappa`:
- `B=50` -> `B=100`
- `hi_band_quantile=0.85` -> `hi_band_quantile=0.90`

- [ ] **Step 5: Update `_run_one` to use new methods**

Replace BIC calls with parallel analysis and cophenetic. For cophenetic, use candidate_ranks with step=3 to keep it feasible:
```python
candidate_ranks = list(range(max(2, true_rank - 12), true_rank + 13, 3))
if true_rank not in candidate_ranks:
    candidate_ranks.append(true_rank)
candidate_ranks = sorted(set(candidate_ranks))
```

Update return dict to have: `rank_kappa, rank_parallel, rank_cophenetic, rank_elbow` (and corresponding errors).

- [ ] **Step 6: Update CSV column names in main()**

Replace all BIC references with the new method names.

- [ ] **Step 7: Commit**

```bash
git add experiments/simulation/kappa_rank_detection.py
git commit -m "feat: replace BIC with parallel analysis + cophenetic baselines, increase kappa B=100"
```

---

### Task 2: Run the experiment

- [ ] **Step 1: Delete old results**

```bash
rm -f outputs/experiments/simulation/kappa_rank_detection/kappa_rank_detection.csv
```

- [ ] **Step 2: Submit the job**

```bash
./scripts/submit experiments/simulation/kappa_rank_detection.py --bg
```

- [ ] **Step 3: Monitor until CSV appears**

Check `outputs/experiments/simulation/kappa_rank_detection/kappa_rank_detection.csv` exists and has 801 lines.

---

### Task 3: Update plotting script

**Files:**
- Modify: `experiments/simulation/plot_kappa_rank_detection.py`

- [ ] **Step 1: Update METHODS list**

Replace BIC with parallel analysis and cophenetic:
```python
METHODS = [
    ("Coherence ($\\kappa$)", "rank_kappa"),
    ("Parallel analysis", "rank_parallel"),
    ("Cophenetic", "rank_cophenetic"),
    ("Elbow", "rank_elbow"),
]
```

Use 4 colors from CMAP: blue (kappa), red (parallel), purple (cophenetic), green (elbow).

- [ ] **Step 2: Update figure layout for 4 panels**

Change `plt.subplots(1, 3, ...)` to `plt.subplots(1, 4, ...)` or keep 3 panels if we drop one method that's clearly useless (decide after seeing results).

Alternatively: show only 3 methods (kappa + best 2 baselines) for visual clarity. Decide based on results.

- [ ] **Step 3: Run plotting and verify PDFs**

```bash
poetry run python experiments/simulation/plot_kappa_rank_detection.py
ls outputs/experiments/simulation/kappa_rank_detection/*.pdf
```

- [ ] **Step 4: Commit**

```bash
git add experiments/simulation/plot_kappa_rank_detection.py
git commit -m "feat: update plots with parallel analysis + cophenetic baselines"
```
