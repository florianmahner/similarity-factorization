# Consensus Methods for NMF: Literature Review

## The Problem

NMF is non-convex, so different random initializations can yield different solutions. How do we combine multiple runs to get a robust result?

## Key Insight: Two Different Goals

The literature distinguishes between:

1. **Consensus for clustering** (most common): Aggregate cluster *assignments* across runs
2. **Consensus for factors** (less studied): Aggregate the actual *factor matrices* W, H

Most literature focuses on (1), which doesn't directly apply to our use case.

---

## Approach 1: Connectivity Matrix (Brunet et al., 2004)

**Source**: [Metagenes and molecular pattern discovery using matrix factorization](https://www.pnas.org/doi/10.1073/pnas.0308531101)

**Method**:
- For each run, compute connectivity matrix C where `c_ij = 1` if samples i,j in same cluster
- Average connectivity matrices across runs: `C̄ = mean(C_1, C_2, ..., C_n)`
- C̄ entries range [0,1] = probability samples cluster together

**Stability metric**: Cophenetic correlation
- Perfect consensus (all 0s and 1s): correlation = 1
- Scattered entries: correlation < 1

**Limitation**: Only captures cluster assignments, not factor values. Doesn't preserve factorization.

---

## Approach 2: cNMF - Consensus NMF (Kotliar et al., 2019)

**Source**: [Identifying gene expression programs with single-cell RNA-Seq](https://elifesciences.org/articles/43803)

**Method**:
1. Run NMF many times (e.g., 100 runs) with different seeds
2. Pool all factors from all runs into one big matrix
3. **Cluster the factors** (not the samples) using k-means
4. Take cluster centroids as consensus factors
5. Filter outliers based on silhouette score / density

**Key insight**: Factors that appear consistently across runs will form tight clusters. Unstable factors scatter.

**Stability metric**: Silhouette score of factor clusters

**Limitation**:
- Designed for standard NMF (V ≈ WH), not symmetric NMF (S ≈ WW^T)
- Centroid averaging may break factorization constraint

---

## Approach 3: Our Current Method (AlignedConsensus)

**Method**:
1. Run symmetric NMF many times
2. Align dimensions across runs via Hungarian matching (cosine similarity)
3. Compute element-wise median across aligned runs
4. **Select** the run closest to median (preserves factorization)

**Why "select" not "median"**:
- Median breaks symmetry: `median(W) @ median(W)^T ≠ S`
- Individual runs satisfy: `W_i @ W_i^T ≈ S`

**Stability metrics**:
- `centrality_scores_`: distance of each run to median (lower = more central)
- `agreement_scores_`: mean cosine similarity with other runs (higher = more agreed)

---

## The Fundamental Issue

**Assumption**: All runs find the same solution up to permutation.

**Reality**: With noise, weak signal, or overlapping factors, runs can find **genuinely different** factorizations.

When this happens:
- Hungarian alignment pairs unrelated dimensions
- "Consensus" becomes meaningless
- We're comparing apples to oranges

**Detection**: Low agreement scores (<0.7) indicate runs found different solutions.

---

## More Principled Alternatives (Ideas)

### 1. Cluster Runs First
Before aligning, check if runs actually agree:
```
similarity(run_i, run_j) = corr(W_i @ W_i^T, W_j @ W_j^T)
```
Cluster runs by this similarity. If multiple clusters exist, report "solution not unique."

### 2. Dimension-wise Stability Selection
Build consensus dimension-by-dimension:
- For each dimension slot, measure consistency across runs
- Keep only dimensions that appear in >80% of runs
- Result: variable-rank embedding with confidence scores

### 3. Reconstruction-Based Selection
Instead of geometric centrality, select based on generalization:
- Hold out entries of S
- Pick run that best predicts held-out entries
- More principled than "closest to median"

### 4. Weighted Consensus
Weight runs by their agreement with others:
- Outlier runs (different local minimum) get downweighted
- Consensus = weighted median

### 5. Report Uncertainty
Don't hide disagreement:
- Return point estimate + confidence intervals
- Flag dimensions with low stability
- Report number of distinct solutions found

---

## Recommendations

1. **Always check `agreement_scores_`** before trusting consensus
2. **Use `aggregation="select"`** for symmetric NMF (preserves factorization)
3. **High agreement (>0.9)**: Consensus is meaningful
4. **Low agreement (<0.7)**: Multiple solutions exist, interpret with caution
5. **Consider reconstruction-based selection** for more principled approach

---

## References

- Brunet, J. P., et al. (2004). Metagenes and molecular pattern discovery using matrix factorization. PNAS.
- Kotliar, D., et al. (2019). Identifying gene expression programs of cell-type identity and cellular activity with single-cell RNA-Seq. eLife.
- [Symmetric NMF: A systematic review](https://www.sciencedirect.com/science/article/abs/pii/S0925231223008445) (2023)
- [cNMF GitHub](https://github.com/dylkot/cNMF)
- [NMF R Package - Consensus](https://nmf.r-forge.r-project.org/connectivity.html)
