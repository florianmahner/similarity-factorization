# Masked Parallel Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a statistically principled dimensionality estimator based on parallel analysis adapted to the entry-masking setting, with rigorous mathematical documentation, and validate on THINGS behavioral data across triplet percentages.

**Architecture:** Two deliverables: (1) A math document (`docs/insights/masked_parallel_analysis.md`) deriving the method step-by-step from first principles, and (2) a sandbox script (`sandbox/coherence/parallel_analysis/run.py`) that implements the test on THINGS at 5%/10%/20%/50%/100% triplet splits and produces diagnostic plots. The core algorithm lives in `src/coherence.py` as a new public function `masked_parallel_analysis()`.

**Tech Stack:** NumPy, SciPy (`eigsh`), joblib (parallelization), matplotlib/seaborn (plots). Existing infrastructure: `src/coherence.py`, `src/utils/figure_theme.py`, `src/utils/helpers.py`.

---

## File Structure

### New files to create

| File | Responsibility |
|------|---------------|
| `docs/insights/masked_parallel_analysis.md` | Mathematical derivation: null model, test statistic, Type I error control, connection to Davis-Kahan and Marchenko-Pastur |
| `sandbox/coherence/parallel_analysis/run.py` | Validation script: run on THINGS at multiple triplet percentages, produce plots |

### Files to modify

| File | Change |
|------|--------|
| `src/coherence.py` | Add `masked_parallel_analysis()` function (~80 lines) |

---

## Task 1: Write the mathematical derivation document

**Files:**
- Create: `docs/insights/masked_parallel_analysis.md`

This document is the theoretical foundation. It must be precise enough for a top-tier ML paper methods section.

- [ ] **Step 1: Write the document**

The document must cover these sections in order:

**1. Setting and notation**
- S is n x n symmetric PSD similarity matrix
- Eigendecomposition: S = U Lambda U^T, lambda_1 >= ... >= lambda_n >= 0
- Bernoulli(p) entry masking: observe each off-diagonal (i,j) with probability p, rescale by 1/p for unbiasedness
- Masked estimator: S_tilde(p) = D + (1/p)(S - D) . M, where D = diag(S), M_ij ~ Bernoulli(p) for i != j
- E[S_tilde] = S (unbiased)

**2. The question**
- We want to determine k*: the number of eigenvalues of S that represent signal rather than noise
- Signal eigenvalue: carries recoverable structure that survives entry masking
- Noise eigenvalue: arises from finite sampling / marginal entry distribution, not from latent structure

**3. Null model: entry-permutation**
- H0: the eigenvalue structure of S is no stronger than what arises by chance from entries with the observed marginal distribution
- Construction: randomly permute the n(n-1)/2 upper-triangle off-diagonal entries of S, fill symmetrically, preserve diagonal
- This preserves: marginal distribution of entries, mean, variance, diagonal values
- This destroys: all pairwise structure, transitivity, latent factor structure
- The null matrix S_pi is symmetric but generally NOT PSD (this is expected and informative)

**4. Test statistic**
- For the observed data: apply Bernoulli(p) masking to S, compute top-K eigenvalues via eigsh
- For each null replicate j = 1,...,J: apply same masking to S_pi^(j), compute top-K eigenvalues
- At each position k and masking rate p:
  - Observed: lambda_k(S_tilde(p)) averaged (or median) across B bootstraps
  - Null distribution: {lambda_k(S_pi_tilde^(j)(p)) : j = 1,...,J}
- Test: lambda_k(observed, p) > quantile_{1-alpha}(lambda_k(null, p))

**5. Multiple testing and decision rule**
- For each k, test across P masking rates: this is a family of tests
- Conservative approach: dimension k is signal if it passes the test at ALL p >= p_0 (intersection test)
- Liberal approach: dimension k is signal if it passes at ANY p (union test, with Bonferroni correction)
- Recommended: require passing at the majority of p values above p_0 (e.g., at all p >= median(p_list))
- k* = max k such that dimension k is classified as signal

**6. Connection to Davis-Kahan (B1)**
- Davis-Kahan sin(Theta) theorem: top-k subspace is recoverable when lambda_k - lambda_{k+1} > ||E||_2
- This is the gap-based test (B1 / kappa)
- Parallel analysis (B2) is complementary: tests whether lambda_k itself is above noise, not whether the gap is large
- For gradual eigenvalue decay (small consecutive gaps but all above noise), B2 detects more dimensions than B1
- Combined: [k_B1, k_B2] gives a principled interval for the true dimensionality

**7. Computational considerations**
- Generating null matrices: O(n^2) per replicate (permute upper triangle)
- Eigendecomposition: O(n * K * n_iter) per masked matrix via eigsh
- Total cost per masking rate p: O(J * n * K * n_iter) for J null replicates
- Parallelization: across (null replicate, p) pairs via joblib
- For n=1854, K=120, J=100, P=25: ~2500 eigendecompositions, each ~1s = ~40 min with 64 cores

- [ ] **Step 2: Review the document for mathematical correctness**

Read through every equation and claim. Verify:
- The unbiasedness of S_tilde
- The permutation null preserves the stated properties
- The Type I error control argument is valid
- The connection to Davis-Kahan is stated correctly (not over-claimed)

- [ ] **Step 3: Commit**

```bash
git add docs/insights/masked_parallel_analysis.md
git commit -m "docs: mathematical derivation of masked parallel analysis for rank estimation"
```

---

## Task 2: Implement `masked_parallel_analysis()` in `src/coherence.py`

**Files:**
- Modify: `src/coherence.py` (add function at end of file, before the IprojAnalysisResult class at line ~2576)

- [ ] **Step 1: Write the function**

The function signature and implementation:

```python
def masked_parallel_analysis(
    S: np.ndarray,
    k_max: int,
    p_list: np.ndarray,
    B: int = 50,
    J: int = 100,
    alpha: float = 0.05,
    random_state: int = 42,
    n_jobs: int | None = None,
    show_progress: bool = True,
) -> dict:
    """Masked parallel analysis for rank estimation.

    Tests whether each eigenvalue of S exceeds what would be expected
    from a matrix with the same marginal entry distribution but no
    pairwise structure, under Bernoulli(p) entry masking.

    Parameters
    ----------
    S : (n, n) array
        Symmetric similarity matrix (may contain NaN).
    k_max : int
        Maximum rank to test.
    p_list : (P,) array
        Masking fractions to evaluate.
    B : int
        Bootstrap replicates per masking rate (for the observed data).
    J : int
        Number of null (permuted) replicates.
    alpha : float
        Significance level for the test.
    random_state : int
        Random seed.
    n_jobs : int or None
        Number of parallel workers. None = cpu_count - 1.
    show_progress : bool
        Show tqdm progress bar.

    Returns
    -------
    dict with keys:
        k_star : int
            Estimated rank (max k passing test at all p >= median(p_list)).
        pvalues : (k_max,) array
            Per-dimension p-value (fraction of null replicates with
            eigenvalue >= observed, averaged across high-p masking rates).
        evals_observed : (k_max, P, B) array
            Bootstrap eigenvalues of masked data.
        evals_null : (k_max, P, J) array
            Eigenvalues of masked null matrices.
        thresholds : (k_max, P) array
            The (1-alpha) quantile of null eigenvalues at each (k, p).
        p_list : (P,) array
            Masking fractions used.
        params : dict
            Input parameters for reproducibility.
    """
```

Implementation outline (pseudocode in the plan, exact code in implementation):

1. Symmetrize S (handle NaN via `_symmetrize_with_nan`)
2. Compute reference eigenvalues of full S (top k_max via `eigh`)
3. Generate J null matrices by permuting upper-triangle off-diagonal entries
4. For each (p, b) pair: mask S with Bernoulli(p), compute top-k_max eigenvalues
5. For each (p, j) pair: mask null matrix j with Bernoulli(p), compute top-k_max eigenvalues
6. Compute thresholds: quantile_{1-alpha} of null eigenvalues at each (k, p)
7. Compute p-values: for each k, fraction of null replicates exceeding observed eigenvalue (averaged over high-p masking rates)
8. Determine k*: max k where observed > threshold at all p >= median(p_list)

Key implementation details:
- Use `_symmetrize_with_nan` and `_prepare_observation_mask` from existing coherence code for NaN handling
- For eigenvalues: use `scipy.linalg.eigh` with `subset_by_index` for exact top-k (more reliable than eigsh for this)
- Parallelize over the (p, replicate) pairs using joblib
- The null permutation must preserve diagonal and NaN positions

- [ ] **Step 2: Verify the function is importable**

```bash
poetry run python -c "from src.coherence import masked_parallel_analysis; print('OK')"
```

- [ ] **Step 3: Quick smoke test on a small synthetic matrix**

```bash
poetry run python -c "
import numpy as np
from src.coherence import masked_parallel_analysis

# Rank-3 signal + noise
rng = np.random.default_rng(42)
W = rng.dirichlet(np.ones(3), size=50)
S = W @ W.T + 0.01 * rng.standard_normal((50, 50))
S = (S + S.T) / 2

result = masked_parallel_analysis(
    S, k_max=10, p_list=np.linspace(0.3, 0.95, 10),
    B=20, J=30, alpha=0.05, n_jobs=-1, show_progress=True,
)
print(f'k* = {result[\"k_star\"]}')
print(f'p-values (first 10): {result[\"pvalues\"][:10].round(3)}')
# Expected: k* should be close to 3
"
```

- [ ] **Step 4: Commit**

```bash
git add src/coherence.py
git commit -m "feat: add masked_parallel_analysis() for principled rank estimation"
```

---

## Task 3: Validation script on THINGS behavioral data

**Files:**
- Create: `sandbox/coherence/parallel_analysis/run.py`

- [ ] **Step 1: Write the validation script**

The script must:
1. Load THINGS triplet data at 5%, 10%, 20%, 50%, 100% (same splits as `experiments/things_behavior/lowdata_comparison`)
2. For each percentage: compute similarity matrix (alpha=0, no smoothing), run `masked_parallel_analysis`
3. Save results as JSON per percentage + summary CSV
4. Produce three plots:
   - **Plot A**: k* vs triplet percentage (line plot, showing how estimated rank increases with data)
   - **Plot B**: eigenvalue spectrum (observed vs null threshold) for 100% data
   - **Plot C**: per-dimension p-values at 100% (bar plot, with alpha threshold line)

Parameters:
- `k_max = 120` (same as existing coherence experiments)
- `p_list = np.linspace(0.05, 0.95, 25)`
- `B = 50` (bootstrap replicates for observed data)
- `J = 100` (null permutation replicates)
- `alpha = 0.05`

Script structure:
```python
"""Masked parallel analysis on THINGS behavioral data across triplet percentages."""

from src.coherence import masked_parallel_analysis
from src.utils import get_output_dir
from src.utils.helpers import compute_similarity_matrix_from_triplets
from src.utils.io import load_triplets
from src.utils.figure_theme import CMAP, GRAY, create_figure, despine, save_figure

OUTPUT_DIR = get_output_dir()
DATA_DIR = Path("data/things")
N_OBJECTS = 1854
PERCENTAGES = [5, 10, 20, 50, 100]
```

- [ ] **Step 2: Run on THINGS 100% only first (quick validation)**

```bash
./scripts/submit sandbox/coherence/parallel_analysis/run.py --bg
```

Monitor with `dash`. Check that k* is in a reasonable range (expect 30-60 for full THINGS data).

- [ ] **Step 3: Inspect outputs and plots**

Check:
- Does k* increase monotonically with triplet percentage?
- Is the eigenvalue spectrum plot showing clear separation between signal and null at 100%?
- Are the p-values showing a clean transition from significant to non-significant?

- [ ] **Step 4: If k* looks unreasonable, iterate**

Possible adjustments:
- If too conservative: check that `eigh` eigenvalues are correct (not reversed); verify null permutation is destroying structure
- If too liberal: increase J (more null replicates for tighter quantiles); check that diagonal is truly preserved in permutation
- If noisy: increase B (more bootstrap replicates)

- [ ] **Step 5: Run full sweep across all percentages**

Once validated at 100%, run the full sweep. Expected runtime:
- 5%: ~15 min (sparse matrix, eigsh might need more iterations)
- 10-50%: ~10-20 min each
- 100%: ~20-30 min
- Total: ~1-2 hours

- [ ] **Step 6: Commit**

```bash
git add sandbox/coherence/parallel_analysis/
git commit -m "feat: validate masked parallel analysis on THINGS across triplet percentages"
```

---

## Task 4: Compare B1 (kappa) and B2 (parallel analysis) results

**Files:**
- Modify: `sandbox/coherence/parallel_analysis/run.py` (add comparison section)

- [ ] **Step 1: Add comparison plot**

After running both methods, produce a combined figure:
- x-axis: triplet percentage
- y-axis: estimated k*
- Lines: kappa (B1), parallel analysis (B2), and optionally activation (current)
- This is the key figure for the paper showing the [k_B1, k_B2] interval

Use the already-computed kappa results from `outputs/experiments/coherence/estimate_rank/things_*.json`.

- [ ] **Step 2: Add comparison table to log output**

Print a summary table:
```
pct  k_activation  k_kappa  k_parallel_analysis
5%   113           3        ??
10%  106           5        ??
20%  106           11       ??
50%  105           22       ??
100% 97            23       ??
```

- [ ] **Step 3: Commit**

```bash
git add sandbox/coherence/parallel_analysis/
git commit -m "feat: compare B1 (kappa) vs B2 (parallel analysis) on THINGS data"
```

---

## Expected outcomes

For THINGS at 100%:
- Activation (current): k* ~ 97 (too high)
- Kappa / B1 (gap-based): k* ~ 23 (conservative)
- Parallel analysis / B2 (noise-floor): k* ~ 40-60 (expected)

The parallel analysis result should:
- Be strictly between activation and kappa
- Increase monotonically with triplet percentage
- Show a clean p-value transition from significant to non-significant around k*
