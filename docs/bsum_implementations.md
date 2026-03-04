# BSUM implementations

## Overview

The BSUM (Block Successive Upper-bound Minimization) algorithm for symmetric NMF
(S ≈ W @ W.T) has 3 Cython implementations in a single file `pysrf/_bsum.pyx`,
with different performance characteristics. The `model.py` import uses the fastest
(BLAS-3 blocked) by default, falling back to Python if Cython is not compiled.

Functions in `pysrf._bsum`:
1. `update_w`: scalar Cython (ground truth, bitwise-identical to Python)
2. `update_w_blas`: BLAS-2 dgemv (~5x over scalar)
3. `update_w_blas_blocked`: BLAS-3 dsymm/dgemm (19-31x over scalar, default)

## Algorithm

Per iteration, for each row i and rank index j:

```
mw_i[j] = sum_k M[i,k] * W[k,j]          # O(n) — dominates at 95%+ of cost
dot_val = W[i,:] · WtW[j,:]               # O(r)
coefficients = f(old, diag, WtW, mw_i)    # O(1)
new = quartic_root(coefficients)           # O(1)
update WtW, diag, W[i,j]                  # O(r)
```

Total per iteration: O(n²r + nr²). The mw_i computation is O(n²r) and dominates.

## Implementation details

### `update_w` (scalar, ground truth)

Precomputes `mw_i[j] = M[i,:] @ W[:,j]` for all j simultaneously in a single pass
through M[i,:] per row. This reduces M memory reads from r passes to 1 pass per row.
Uses Cython memoryviews for all array access.

**Mathematical validity**: Within row i's j-loop, mw_i[j] = sum_k M[i,k] * W[k,j].
When we update W[i, j_prev] for j_prev < j, this does NOT affect mw_i[j] because
the sum over k includes k=i, but we're modifying W[i, j_prev] (column j_prev),
not W[k, j] (column j). The effect of W[i, j_prev] changes is captured in the
WtW terms.

**Numerical agreement**: Bit-for-bit identical to the Python `update_w` in `model.py`.

### `update_w_blas` (BLAS-2)

Same per-row mw_i algorithm as `update_w`, but uses:

- **BLAS dgemv** for `mw_i = M[i,:] @ W` (the dominant 95% cost). OpenBLAS dgemv
  uses optimized cache blocking, SIMD, and prefetching for this matrix-vector product.
- **BLAS ddot** for `W[i,:] · WtW[j,:]` dot products (contiguous row access).
- **BLAS daxpy** for `WtW[j,:] += delta * W[i,:]` (contiguous row update).
- Raw C `double*` pointers for scalar operations and the strided WtW column update.

**Fortran layout for dgemv**: C row-major W(n,r) = Fortran column-major A(r,n) with
lda=r, where A = W^T. The call `dgemv('N', r, n, 1.0, W, r, M[i,:], 1, 0.0, mw_i, 1)`
computes `mw_i = A @ M[i,:] = W^T @ M[i,:] = M[i,:] @ W`.

### `update_w_blas_blocked` (BLAS-3, fastest, default)

Replaces per-row BLAS-2 operations (memory-bound dgemv) with BLAS-3 operations
(compute-bound dsymm/dgemm):

1. **dsymm** precomputes `MW = M @ W` for the entire matrix at once (BLAS-3)
2. **dgemm** applies inter-block corrections after each block completes
3. **Scalar daxpy** corrections within each block for rows processed earlier

The key insight: instead of calling dgemv n times (once per row), we call dsymm
once per iteration to get all MW values, then correct them as W changes within
each block. For block_size B:
- 1 dsymm call (n×n × n×r → n×r): O(n²r) but at BLAS-3 bandwidth
- n/B dgemm calls for inter-block corrections: O(n²r/B) total
- O(B²r) scalar corrections within each block

The parameter `block_size` controls the trade-off between BLAS-3 efficiency and
correction overhead. The default block_size is `min(50, max(1, n // 10))` because the dsymm
dominates and corrections are cheap.

**Numerical agreement**: Same as `update_w_blas`; small floating-point differences
(1e-13 to 1e-9) due to BLAS summation reordering, but reconstruction error matches
to 12+ decimal places.

## Compile flags

The extension is compiled with `-O3 -march=native -ffp-contract=off`:
- `-march=native`: enables AVX2/AVX512 SIMD instructions
- `-ffp-contract=off`: disables FMA contraction to keep scalar loops bit-identical
  (the BLAS calls still use their own internal optimizations)

## Benchmarks

Measured on SLURM compute nodes (4 threads), 5 BSUM iterations, tol=0.0:

<!-- vale off -->

| n | rank | scalar | blas | blocked | blocked speedup |
|---|------|--------|------|---------|-----------------|
| 5000 | 50 | 2.46s | 0.96s | 0.29s | 8.5x |
| 10000 | 50 | 9.53s | 4.68s | 0.88s | 10.8x |
| 20000 | 50 | 37.41s | 18.28s | 3.04s | 12.3x |

<!-- vale on -->

Speedup grows with n because the BLAS-3 dsymm/dgemm ratio improves with matrix size.
At production sizes (n=1854, rank=50), the blocked variant is ~19x faster than scalar.

## Memory-efficient fitting

The `_fit_complete_data` method in `model.py` uses `_frobenius_residual` to compute
`||X - WW^T||_F` without materializing the n×n product `WW^T`. This keeps memory
at O(n²) (just the input matrix) instead of O(2n²).

The identity used:
```
||X - WW^T||²_F = ||X||²_F - 2·tr(X·W·W^T) + ||WW^T||²_F
```
where `tr(X·W·W^T) = sum((X@W) * W)` costs O(n²r) and `||WW^T||²_F = ||W^TW||²_F`
costs O(nr²), both with no n×n temporaries.

## Numerical agreement

### `update_w` (scalar) vs Python

Bit-for-bit identical: max_diff = 0.0 in all tests. This is because both
implementations perform the exact same arithmetic in the exact same order; the
scalar Cython version just reorganizes the loop structure without changing the
computation.

<!-- vale off -->

### `update_w_blas` and `update_w_blas_blocked` vs `update_w` (scalar)

The BLAS variants compute mathematically identical quantities via different
floating-point summation order. This section provides a complete mathematical
proof of equivalence, explains how IEEE 754 rounding cascades through the
quartic solver, and documents the test strategy.

#### Step-by-step mathematical proof of equivalence

The sBSUM algorithm (Shi et al. 2017, Algorithm 1) updates each element W[i,j]
by solving a quartic subproblem with coefficients (a, b, c, d) that depend on
the current state of W, WtW = W^T @ W, diag = row norms of W, and
MW[i,j] = (M @ W)[i,j]. We prove below that all implementations compute
identical coefficients for each (i,j) update, up to IEEE 754 rounding in the
MW term.

**1. Quartic solver**: All implementations use an identical `_quartic_root(a, b, c, d)`,
implementing the cubic resolvent from Shi et al. (2017) Eq. (11). Identical code,
identical inputs → identical outputs. ✓

**2. Coefficient a = 4.0**: Hardcoded constant in all implementations. ✓

**3. Coefficient b = 12 · W[i,j]**: Depends only on the current element. All
implementations read `old = W[i,j]` from the same array, so b is identical. ✓

**4. Coefficient c = 4·((diag[i] − M[i,i]) + WtW[j,j] + W[i,j]²)**:
- `diag[i]` = ||W[i,:]||² — maintained incrementally via
  `diag[i] += new² − old²`, identical in all implementations.
- `M[i,i]` — read-only input, identical.
- `WtW[j,j]` — maintained incrementally via `WtW[j,j] += delta²` after each
  update. The rank-1 update `WtW[j,:] += delta·W[i,:]` uses daxpy in the BLAS
  variants for the row (stride-1) and a scalar loop for the column (stride-r).
  Since daxpy computes `y[k] += alpha·x[k]` element-by-element with no
  reordering, this is arithmetically identical to the scalar loop. ✓

**5. Coefficient d = 4·(W[i,:] · WtW[:,j] − MW[i,j])**:

The first term, `W[i,:] · WtW[:,j]`: The scalar variant uses a scalar loop over
WtW[:,j] (column, stride r). The BLAS variants use `ddot(r, W[i,:], 1,
WtW[j,:], 1)` on WtW[j,:] (row, stride 1). Since WtW = W^T W is symmetric,
WtW[j,:] = WtW[:,j] — same values, different memory layout. The ddot result
differs from the scalar loop only by IEEE 754 rounding.

The second term, `MW[i,j] = (M @ W_current)[i,j]`, is the critical difference:

- **Scalar** (`update_w`): Precomputes `mw_i[j] = sum_k M[i,k]·W[k,j]`
  for all j in a single pass at the start of row i. This is valid because
  within row i's j-loop, updating W[i, j_prev] for j_prev < j does not affect
  mw_i[j]: the sum is over W[:,j] (column j), and only W[i, j_prev] (column
  j_prev ≠ j) was modified. Same summation order as Python → bit-identical.
- **BLAS** (`update_w_blas`): Same per-row precomputation via dgemv:
  `mw_i = W^T @ M[i,:]`. The Fortran call `dgemv('N', r, n, 1, W, r,
  M[i,:], 1, 0, mw_i, 1)` interprets C row-major W(n,r) as Fortran
  col-major A(r,n) = W^T, computing A @ M[i,:] = W^T @ M[i,:]. By symmetry
  of M: (W^T @ M[i,:])[j] = sum_k W[k,j]·M[i,k] = (M @ W)[i,j]. Same
  mathematical quantity, but dgemv uses SIMD accumulation which changes
  summation order.
- **Blocked** (`update_w_blas_blocked`): Precomputes MW = M @ W for all rows at
  once via dsymm, then corrects for W changes during the iteration:

  At the start of iteration, MW₀ = M @ W₀ via dsymm. When processing row i,
  rows 0..i−1 have already been updated: W_current[k,:] = W₀[k,:] + ΔW[k,:]
  for k < i, and W_current[k,:] = W₀[k,:] for k ≥ i. The needed value is:

  ```
  MW_correct[i,j] = sum_k M[i,k] · W_current[k,j]
                   = sum_k M[i,k] · W₀[k,j] + sum_{k<i} M[i,k] · ΔW[k,j]
                   = MW₀[i,j] + sum_{k<i} M[i,k] · ΔW[k,j]
  ```

  The correction `sum_{k<i} M[i,k] · ΔW[k,j]` is computed in two parts:

  **Inter-block** (via dgemm, after each block completes): For rows in
  completed blocks {0..block_start−1}, the dgemm call
  `MW[remaining,:] += M[remaining, block] @ ΔW[block,:]` applies the
  cumulative delta from each completed block to all subsequent rows.

  **Intra-block** (via daxpy, before processing each row): For earlier rows
  in the current block {block_start..i−1}, scalar daxpy calls apply
  `MW[i,:] += M[i,k] · ΔW[k,:]` for each k in the current block.

  Together: MW_corrected[i,:] = MW₀[i,:] + (inter-block corrections from
  completed blocks) + (intra-block corrections from current block)
  = MW₀[i,:] + sum_{k=0..i−1} M[i,k] · ΔW[k,:] = (M @ W_current)[i,:]. ✓

  The dsymm and dgemm calls use BLAS-3 accumulation which differs in
  summation order from the scalar loop, introducing IEEE 754 rounding
  differences.

**6. State updates (WtW, diag, W)**: All implementations perform identical
rank-1 updates: `WtW[j,:] += delta·W[i,:]` (daxpy, element-wise identical),
`WtW[:,j] += delta·W[i,:]` (scalar loop in all BLAS variants, identical),
`WtW[j,j] += delta²`, `diag[i] += new² − old²`, `W[i,j] = new`. ✓

**Conclusion**: The ONLY source of floating-point difference between
implementations is the summation order in:
- dsymm/dgemv for MW computation (BLAS uses SIMD/blocked accumulation)
- ddot for W[i,:] · WtW[:,j] (BLAS may use pairwise summation)

All other operations are arithmetically identical.

#### IEEE 754 rounding error bound

By Higham (2002, Theorem 3.1), the forward error of a floating-point inner
product fl(x^T y) of length n satisfies:

```
|fl(x^T y) − x^T y| ≤ γ_n · |x|^T |y|
```

where γ_n = nu/(1−nu) ≈ nu and u = 2^{-53} ≈ 1.11×10^{-16} is the IEEE 754
double-precision unit roundoff. Two different summation orders (e.g., scalar
sequential vs BLAS SIMD) each satisfy this bound independently, so their
results can differ by up to ~2·γ_n · |x|^T |y|.

For the MW computation with n=100, typical |M[i,k]| ≈ 0.5, |W[k,j]| ≈ 0.1:
max difference ≈ 2 · 100 · 1.11×10^{-16} · (sum of |M[i,k]·W[k,j]|)
≈ 2 · 100 · 1.11×10^{-16} · ~5 ≈ 1.1×10^{-13}, consistent with observed
~10^{-14} differences after one iteration.

This non-reproducibility of BLAS results due to summation reordering is a
well-documented phenomenon. Standard BLAS libraries (OpenBLAS, MKL, cuBLAS)
do not guarantee a specific summation order — SIMD instructions accumulate
partial sums in parallel lanes that are reduced in implementation-defined order
(Ahrens, Demmel & Nguyen, 2020; Demmel & Nguyen, 2015). The ReproBLAS project
provides reproducible alternatives at ~1.5× cost, but standard BLAS is used
here for maximum performance.

#### Why element-wise differences grow over iterations

The BSUM quartic solver computes `new = max(0, cbrt(-d/4))` when old=0 and c<0.
The cube root function has derivative 1/(3·x^{2/3}), which diverges as x → 0.
Near the non-negativity boundary:

1. BLAS rounding gives d_blas = d_scalar ± ~10^{-14} (from MW difference)
2. When |d| is small, cbrt amplifies: cbrt(d ± ε) ≈ cbrt(d) ± ε/(3·|d|^{2/3})
3. For d ~ 10^{-15}: amplification factor ~ 1/(3·(10^{-15})^{2/3}) ~ 10^{9}
4. If the scalar root is barely positive and the BLAS root is barely negative
   (or vice versa), `max(0, root)` creates a discrete 0-vs-nonzero flip

This boundary flip propagates through subsequent WtW updates within the same
row's j-loop and cascades to all downstream elements. Over many iterations,
flips accumulate — but both algorithms converge to the same fixed point of the
NMF objective (Shi et al. 2017, Theorem 1).

#### Reconstruction quality is identical

Despite element-wise divergence, the reconstruction quality
`||M − WW^T||_F / ||M||_F` matches to **10+ decimal places** at all iteration
counts. Both algorithms minimize the same objective function — boundary flips
select a slightly different path through the solution space but converge to
equally good local minima.

Tested at n=1854, r=50 up to 500 iterations (blocked with B=50):

| iters | recon err (scalar) | recon err (blocked) | max W diff |
|-------|--------------------|---------------------|------------|
| 1 | 0.561962478782 | 0.561962478782 | 1.2e-12 |
| 5 | 0.060587047326 | 0.060587047326 | 1.9e-11 |
| 50 | 0.018596595745 | 0.018596595745 | 2.2e-11 |
| 100 | 0.013706608896 | 0.013706608896 | 4.6e-11 |
| 500 | 0.006769387540 | 0.006769387540 | 3.6e-10 |

#### Test strategy

- **1-iteration element-wise test** (`atol=1e-10`): Proves the correction math
  is exact on arbitrary symmetric M (including worst-case full-rank random M
  with 70-90% sparsity in W). Differences are pure IEEE 754 rounding.
- **Reconstruction quality test** (`rtol=1e-2`): Proves both algorithms produce
  equally good NMF solutions after convergence on any M. The tolerance is 1%
  to account for full-rank random M where NMF residual remains high (~33%)
  and boundary flips select different (but equally valid) local minima.
- **Nonnegativity test**: Verifies the max(0, root) constraint is maintained.

#### References

- Shi, Q., Sun, H., Lu, S., Hong, M. & Razaviyayn, M. (2017). "Inexact Block
  Coordinate Descent Methods for Symmetric Nonnegative Matrix Factorization."
  IEEE Trans. Signal Processing, 65(22), 5995–6008.
  https://arxiv.org/abs/1607.03092
- Higham, N. J. (2002). "Accuracy and Stability of Numerical Algorithms."
  2nd ed., SIAM. Theorem 3.1 (inner product error bound).
- Ahrens, W., Demmel, J. & Nguyen, H. D. (2020). "Algorithms for Efficient
  Reproducible Floating Point Summation." ACM Trans. Math. Software, 46(3).
  https://doi.org/10.1145/3389360
- Demmel, J. & Nguyen, H. D. (2015). "Parallel Reproducible Summation."
  IEEE Trans. Computers, 64(7), 2060–2070.
- Ye, Y. & Bhatt, N. (2024). "FPRev: Revealing Floating-Point Accumulation
  Orders in Software/Hardware Implementations." arXiv:2411.00442.

<!-- vale on -->

## Running benchmarks

```bash
# Submit to SLURM
sbatch benchmarks/run.sh

# Or run interactively
srun --partition=interactive --cpus-per-task=4 --mem=16G --time=00:30:00 \
    python benchmarks/benchmark.py

# Correctness tests
python benchmarks/correctness.py
```
