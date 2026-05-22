# Tutorial — running Recipe K end-to-end

This guide walks through Recipe K on a small synthetic 10-block similarity
matrix. The notebook [`tutorial/recipe_K_demo.ipynb`](../tutorial/recipe_K_demo.ipynb)
does the same in Jupyter and is the recommended starting point.

## 1. Set up

```python
import sys, os
RECIPE_K_DIR = "/path/to/RECIPE_K"
sys.path.insert(0, os.path.join(RECIPE_K_DIR, "src"))

import numpy as np
from _common import spectral_pass, recipe_K
from _generate_examples import simulation, _rbf_kernel
```

Both `_common` and `_generate_examples` live in `RECIPE_K/src/`. Adding
that directory to `sys.path` is the only path manipulation you need.

## 2. Build a synthetic test matrix

```python
n, K = 200, 10
rng = np.random.default_rng(0)
D = simulation(n, K, ndict=10) + rng.random((n, K)) * 0.5
S = _rbf_kernel(D, bw=1.0)
print(S.shape)  # (200, 200)
```

`simulation(n, K, ndict=10)` generates random block-membership coordinates in
$K=10$ latent dimensions; `_rbf_kernel` converts these to a positive
semidefinite similarity matrix. The ground-truth rank is $K=10$.

## 3. Spectral pass (estimate $k_{\rm cut}$)

```python
spec = spectral_pass(S, B=20, smooth_window=10, show_progress=False)
print("k_cut =", spec["k_cut"], "  rho =", round(spec["rho"], 2),
      "  detectability flag =", spec["flag"])
```

`spec` is a dict containing the bootstrap leakage profile $\hat\kappa_r$, the
reference eigenvalues, the F-statistic changepoint, and bulk-edge diagnostic
flags. For this 10-block matrix you should see `k_cut = 10`.

## 4. Recipe K (calibrate $p^\star$ and $p_{\rm cv}$)

```python
out = recipe_K(spec,
               delta=0.10,
               k_cv=5,
               p_floor=0.5,
               p_floor_mode="adaptive",
               M_min=2000)
print(f"p* = {out['p_star']:.3f}")
print(f"p_cv (5-fold inflated) = {out['p_cv']:.3f}")
print(f"status = {out['status']}")
print(f"floor binding = {out['floor_binding']},  cap binding = {out['cap_binding']}")
print(f"bulk_edge_plausible = {out['bulk_edge_plausible']}")
```

A typical result for the 10-block synthetic:

```
p* ≈ 0.85,  p_cv ≈ 0.95,  status = accepted
```

`p_floor_mode="adaptive"` enables the Wigner-proxy operator-norm safety floor
$p_{\rm floor}=\lambda_{k+1}^2/(\lambda_k^2+\lambda_{k+1}^2)$. To reproduce the
pre-2026-05-11 floor formula use `"adaptive_legacy"`; to bypass the data-driven
floor use `"constant"` with `p_floor=0.5`.

## 5. Fold-invariant cross-validation

The protocol now provides a calibrated $p_{\rm cv}$. Run ordinary $k_{\rm cv}$-fold
CV on the outer-masked matrix:

```python
from sklearn.utils import check_random_state
from joblib import Parallel, delayed
from _common import split_omega_into_folds, fit_admm_score
from symmnmf.cross_validation import mask_missing_entries

rng = check_random_state(0)
M_outer = mask_missing_entries(S, out["p_cv"], rng, missing_values=np.nan)
val_masks = split_omega_into_folds(M_outer, k_inner=5, rng=rng)

ranks = list(range(1, 25))
bounds = (float(np.nanmin(S)), float(np.nanmax(S)))
results = Parallel(n_jobs=4)(
    delayed(fit_admm_score)(S, V_i | M_outer, V_i, M_outer, V_i | M_outer,
                            r, bounds, seed=1000 + fi * 100 + ri)
    for fi, V_i in enumerate(val_masks)
    for ri, r in enumerate(ranks)
)
```

Recall that *changing* `k_inner` in Recipe K only changes variance reduction,
not the population training fraction (which stays at $p^\star$).

## 6. Sanity-check the diagnostics

Before reporting $\hat r_{\mathcal F}^{\rm CV}$, *always* inspect the
diagnostic flags:

| Field | Action if … |
|---|---|
| `status == "rejected_smooth"` | spectrum is continuous; do **not** treat $k_{\rm cut}/p^\star$ as committed. |
| `bulk_edge_plausible == False` | the safety floor is unreliable; consider increasing $B$, switching to `p_floor_mode="constant"`, or rerunning with a different $p$-grid. |
| `cap_binding == True` | $p_{\rm cv}$ hit $p_{\max}$; use `k_cv_min_unclipped` to pick a larger fold count. |
| `n_monotonicity_violations > 0` | bootstrap deficit was non-monotone; PAV handled it but inspect `delta_emp_raw` vs `delta_emp_iso`. |

## 7. Next steps

- Run a full manuscript experiment: `python exp/experiment_A_REDO.py` or
  `python exp/experiment_B_full_sweep.py`. Outputs go under
  `output/experiment_A/` and `output/experiment_B/`.
- Inspect the kappa-sweep diagnostic: `python exp/recipe_K_kcv_sweep.py`.
- See `experiments.md` for the full table of manuscript-reproducing scripts.
