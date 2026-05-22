# Recipe K — overview

> See [`RECIPE_K_THESIS.md`](../../src_coherence/results/RECIPE_K_THESIS.md)
> for the full manuscript with proofs, diagnostics, and empirical validation.

## Motivation

Given a symmetric similarity matrix $S\in\mathbb{R}^{n\times n}$ (kernel,
covariance, RSM, affinity, …) we want to estimate the **representation
dimension**. Naive entrywise $k_{\rm cv}$-fold CV has a hidden problem: the
effective training fraction $(k_{\rm cv}-1)/k_{\rm cv}$ depends on $k_{\rm cv}$, so the
selected dimension drifts with the fold count. Recipe K fixes this by
calibrating an *operating sampling probability* $p^\star$ from the spectrum
of $S$, then inflating it to $p_{\rm cv}=p^\star\cdot k_{\rm cv}/(k_{\rm cv}-1)$ so that
the population training fraction is $p^\star$ at every $k_{\rm cv}\ge 2$.

## The two layers

| Layer | Question | Output |
|---|---|---|
| Spectral calibration | *How many stable spectral directions does $S$ contain?* | $k_{\rm cut}$ |
| Model selection | *What model dimension does CV select at the calibrated point?* | $\hat r_{\mathcal F}^{\rm CV}$ |

These two need not agree: $k_{\rm cut}$ is a property of $S$'s spectrum;
$\hat r_{\mathcal F}^{\rm CV}$ is a property of CV on the chosen representation
family $\mathcal F$ (PCA, NMF, PPCA, SoftImpute, SymmNMF, …).

## Algorithm (one-pass summary)

1. **Bootstrap leakage profile.** For each rank $r$ and probability $p$ on a
   grid, draw $B$ symmetric Bernoulli masks, form $A(p)=M\circ S/p$ (keeping
   the diagonal), and record the projector-overlap statistic
   $I_r^{\rm proj}(p)=\|P_{1:r}^{\rm masked}(p)\,u_r^{\rm ref}\|^2$.
2. **Per-rank leakage rate.** Aggregate at the top of the $p$ grid:
   $\hat\kappa_r\approx(1-I_r^{\rm proj}(p_{\rm high}))\,p_{\rm high}/(1-p_{\rm high})$.
3. **Spectral cutoff $k_{\rm cut}$.** F-statistic two-segment changepoint of
   the $\hat\kappa_r$ profile. (Cliff and smooth detectors run in parallel
   for diagnostics.)
4. **Rayleigh-trace fidelity.** $\mathrm{VE}^D(k_{\rm cut},p)=\mathrm{tr}(P^{\rm masked}_{k_{\rm cut}}(p)\,S)/\mathrm{tr}(S)$,
   computed exactly via one $S\!\cdot\!U_{k_{\rm cut}}$ matvec per $(p,b)$.
5. **Empirical deficit and inversion.** $\delta_{\rm emp}(p)=1-\mathrm{VE}^D/\mathrm{VE}^{\rm ref}$,
   monotonized by PAV, then $p^\star_{\rm raw}=\inf\{p:\delta_{\rm iso}(p)\le\delta\}$.
6. **Wigner-proxy operator-norm safety floor.** $p_{\rm floor}=\lambda_{k+1}^2/(\lambda_k^2+\lambda_{k+1}^2)$;
   the operating point is $p^\star=\max(p^\star_{\rm raw},p_{\rm floor})$.
7. **Fold inflation.** $p_{\rm cv}=p^\star\cdot k_{\rm cv}/(k_{\rm cv}-1)$, capped at
   $p_{\max}$.  This makes the marginal training fraction equal to $p^\star$
   for every $k_{\rm cv}$.

Diagnostic flags — `bulk_edge_plausible`, `gap_ratio`, `bulk_flatness`,
`status ∈ {accepted, borderline, rejected_smooth, unknown}` — accompany the
output and should be consulted before trusting $k_{\rm cut}/p^\star$.

## Outputs returned by `recipe_K(...)`

The dict returned by `_common.recipe_K(spectral_out, …)` includes:

- `status`, `k_cut`, `delta`, `k_cv`
- `p_star_raw`, `p_floor`, `floor_binding`, `p_star`, `p_target`
- `p_cv`, `p_cv_unclipped`, `cap_binding`, `p_train_eff`, `delta_eff`
- `k_cv_min_unclipped`, `p_max`, `M_min`, `n`
- Bulk-edge diagnostics: `lambda_k`, `lambda_kp1`, `lambda_kp5`,
  `gap_ratio`, `bulk_flatness`, `bulk_edge_plausible`
- Curve diagnostics: `delta_emp_raw`, `delta_emp`, `delta_emp_iso`,
  `p_grid`, `n_monotonicity_violations`
- Provenance: `used_rayleigh_trace`, `tr_S_mode`, `tr_S`, `VE_ref_k`

See [`api.md`](api.md) for the full signature and field reference.

## Suggested reading order

1. This file.
2. [`tutorial.md`](tutorial.md) — guided demo on a 10-block synthetic.
3. [`experiments.md`](experiments.md) — how to reproduce manuscript figures.
4. [`api.md`](api.md) — function-by-function reference.
5. [`RECIPE_K_THESIS.md`](../../src_coherence/results/RECIPE_K_THESIS.md) — the
   full manuscript for theoretical background.
