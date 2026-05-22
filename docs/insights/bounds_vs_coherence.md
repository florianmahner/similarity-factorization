# Bounds Estimation vs Projected Coherence: Why the Mean Underestimates

**Date:** 2026-02-11
**Context:** Investigating why pysrf's bound estimation (pmin, pmax) sometimes underestimates the recoverable dimensionality, and how Ka Chun's projected coherence framework provides a more nuanced view.

## Summary

The pysrf bound estimation uses a single "effective dimension" computed from `ceil((||S||_F / ||S||_2)^2)` and a mean Stieltjes transform (isotropic VDE) to determine the bulk edge. This approach is coarse: it answers "how many eigenvalues exceed the average noise floor?" but ignores per-dimension anisotropy. The projected coherence framework instead measures each dimension's recoverability continuously, revealing that many more dimensions are recoverable than the bounds suggest.

## Background

### pysrf bounds (`pysrf/bounds.py`)
- **pmin**: Bernstein concentration bound using worst-case row norms and max entries of S
- **pmax**: Largest p where exactly `eff_dim` eigenvalues satisfy `p * lambda_k > lambda_bulk(p)`
- **VDE**: Solves `m_i(w) = -1/(w + sum_j V_ij m_j(w))` where `V = p(1-p)(S∘S)`, then uses `Im(mean(m))` to detect the bulk edge

### Ka Chun's coherence (`sandbox/coherence/kachun/`)
- **I^proj_k(p)**: How much of the k-th reference eigenvector is captured by the top-k empirical eigenspace of the masked matrix A(p) = (M(p)/p) ∘ S
- **Null threshold tau_k(p)**: 95th percentile of the same statistic computed with random directions instead of the true eigenvector
- **Activation point p_act(k)**: Smallest p where the lower CI of I^proj_k exceeds tau_k

## Evidence

### 1. Effective dimension is too crude

The formula `eff_dim = ceil((||S||_F / ||S||_2)^2)` produces a single integer. For mur92 (92x92), this can be as low as 1-2, even though the coherence analysis shows 5-10 dimensions activating at moderate p. The ceiling operation introduces further discretization error (see `FIXME` comment at `bounds.py:37`).

### 2. pmin uses worst-case quantities

The Bernstein-based pmin uses `max(row sums of S^2)` and `max(|S_ij|)` — worst-case quantities that are dominated by the matrix's most extreme entries. For heterogeneous matrices (hub structure, heavy-tailed degree distributions), this bound is very loose.

### 3. VDE mean is isotropic

The current implementation computes `Im(mean(m(z)))` — averaging over all n entries of the VDE solution. This gives the bulk edge for the *average* spectral density. But the variance profile V(p) = p(1-p)(S∘S) is anisotropic — different rows of S have different noise levels.

From Ka Chun's theory (PDF, equations q-aniso / A), the correct per-direction functional is:

```
q_k(z) = sum_i (u_{k,i}^ref)^2 * m_i(z)   (anisotropic)
```

rather than:

```
q(z) = (1/n) sum_i m_i(z)   (isotropic, currently used)
```

The anisotropic functional weighs VDE entries by the squared components of the k-th eigenvector. This means eigenvectors concentrated on high-S rows are easier to recover than the isotropic bound suggests, and vice versa.

### 4. Binary vs continuous detection

The bound estimation uses `count_out(p) = sum(p * lambda > edge)` — a binary criterion. The coherence curves show that dimensions gradually become recoverable. A dimension might be 60% recoverable (I^proj = 0.6) at the current pmax but is counted as "not detectable" by the binary criterion.

## Relationship between approaches

```
pysrf bounds:     pmin ←————————→ pmax       (single feasibility window)
                                              uses eff_dim + isotropic VDE

coherence:        p_act(1) .. p_act(k*) .. p_act(K)   (per-dimension activation)
                  |<-- signal -->|<--- noise --->|      uses anisotropic projectors

The activation curve IS the fine-grained version of the bounds.
```

Signal dimensions activate at low p (eigenvectors survive even heavy masking). Noise dimensions activate at high p or never. The transition between the two gives the true rank — more precisely than `eff_dim`.

## Practical implications

1. **For cross-validation sampling fraction**: Using `mean(pmin, pmax)` can set p too low when pmin is loose. The coherence activation curve suggests using a p where "most signal dimensions have activated" — e.g., `p = max(p_act(k)) for signal k`.

2. **For rank estimation**: The coherence activation curve provides a direct, principled rank estimate without needing to fit SRF at many different ranks. The largest gap in the activation points separates signal from noise dimensions.

3. **For the VDE bulk edge**: Switching to anisotropic functionals `q_k(z)` would tighten the per-dimension bounds. However, the PDF notes this is computationally expensive (requires solving a separate outlier equation per dimension).

## Open questions

- How sensitive is the coherence-based rank estimate to n_boot and the p-grid resolution?
- Can the anisotropic VDE functional be computed cheaply enough to replace the isotropic one in pysrf bounds?
- Does the gap-based rank selection from activation points agree with CV-based rank selection on real datasets?
- How does the coherence rank estimate behave for datasets with slowly decaying spectra (no clear gap)?

## Reproducibility

- Theory: `sandbox/coherence/kachun/Numerical Iproj summary v5.pdf`
- Notebook (original): `sandbox/coherence/kachun/test_bulk_coherence_eig_v5.ipynb`
- Clean code: `sandbox/coherence/kachun/run.py`
- Existing projected coherence: `sandbox/coherence/projected_iproj/run.py`
- pysrf bounds: `third_party/pysrf/pysrf/bounds.py`
- Related insight: `docs/insights/cv_sampling_fraction.md`
