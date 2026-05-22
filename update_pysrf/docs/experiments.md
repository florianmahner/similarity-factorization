# Reproducing the manuscript experiments

Every script in [`RECIPE_K/exp/`](../exp/) is a *copy* of the canonical
manuscript script with two minor edits:

1. `sys.path.insert(0, SRC)` where `SRC = RECIPE_K/src/`.
2. Output directories rewritten to `RECIPE_K/output/<experiment>/`.

No algorithmic logic was touched. Random seeds, dataset registries, p-grids,
and recipe-K defaults are identical to the originals.

## Pre-requisites

```bash
# Activate the dependency-complete conda env
conda activate SymmNMF
```

The code requires `numpy`, `scipy`, `scikit-learn`, `joblib`, `matplotlib`,
`tqdm`, `plotly`, `kneed`, and `pyclustering`. The
[`SymmNMF` conda env](https://github.com/.../README.md) at
`/home/lamk5/miniconda3/envs/SymmNMF/` already has all of these.

## Canonical experiments (manuscript §6)

### Experiment A — self-consistency table (§6.1)

```bash
cd RECIPE_K
python exp/experiment_A_REDO.py
```

- 33 unique seeded datasets × 11 benchmarks × 10 seeds.
- Records per-seed $\hat\kappa_r$, eigenvalues, F-statistic $k_{\rm cut}$,
  full `recipe_K` diagnostics, and every benchmark's selected rank.
- **Outputs** in `output/experiment_A/`:
  - `run.log`, `env.txt`, `results.pkl`, `summary.json`
  - `figures/AGGREGATE_heatmap.png`, `figures/AGGREGATE_metrics.png`
  - `figures/kappa/kappa_<label>.png` per dataset
- **Wall time** on a 96-core node: ~25 min.

### Experiment B — full sweep (§6.2–6.4)

```bash
python exp/experiment_B_full_sweep.py
```

- 49 cells × 6 protocols (P0a/P0b/P0c at fixed `p ∈ {0.5,0.75,0.9}`, P1 at
  `p=p*` with `n_reps=1`, P2 at `p=p*` with `n_reps=10`, P3 = 5-fold nested
  CV at `p_cv`).
- Includes the two real-data cells (`CLIP_RBF`, `THINGS`) loaded from
  `data/`.
- **Outputs** in `output/experiment_B/`:
  - `run.log`, `summary.json`
  - `raw/<label>.pkl` per cell
  - `figures/<label>.png` per cell
- **Wall time** on a 96-core node: ~10 hours (variable; the high-D cells
  dominate).

### Recipe K kappa-sweep (§6.5)

```bash
python exp/recipe_K_kcv_sweep.py
```

- 8 small datasets × `k_cv ∈ {3, 5, 10}`.
- Quick sanity run for the manuscript's "small-n calibration" claim.
- **Outputs** in `output/recipe_K_kcv_sweep/`.
- **Wall time** ~5-15 min.

### A3 n-scan (§1.2 / §5.5a motivating table)

```bash
python exp/recipe_K_cvfold_A3_nscan.py       # run
python exp/plot_A3_n300_diagnostic.py        # render single-n diagnostic
python exp/plot_A3_nscan_combined.py         # render combined 3x3 figure
```

Demonstrates that plain $k$-fold drifts with $n$ at fixed $k_{\rm cv}$, while
Recipe K stays at the true rank.

## Gap-plug companion experiments

### Legacy operator-edge convergence (§6.6)

```bash
python exp/experiment_B_pstar_correctness.py
```

Sweeps $n \in \{400, 800, 1600\}$ and $\lambda_k = c_k\cdot 2\sigma\sqrt n$,
tests whether $p^\star_{\rm raw}(n) \to 1/c_{\min}^2$ (legacy benchmark) at
rate $O(n^{-1/2})$.

### κ̂ vs eigengap diagnostic

```bash
python exp/experiment_B_kappa_vs_eigengap.py
```

Compares the leakage rate to the spectral gap; loads CLIP_RBF and THINGS from
`data/`.

### Real-data SymmNMF gap (§6.7.2)

```bash
python exp/experiment_B_pstar_gap_C_symmnmf.py        # SymmNMF on CLIP and THINGS
python exp/experiment_B_pstar_gap_C_clip_extended.py  # SoftImpute, extended rank grid on CLIP
python exp/experiment_B_pstar_gap_C_things_fixed.py   # SoftImpute on THINGS with NaN-imputed input
python exp/figure_pstar_real_symmnmf.py               # combined figure
```

All three write into `output/pstar_gap/`.

## CLI / configuration

The experiment scripts are intentionally executable with no arguments — the
JSON files in `configs/` are reference manifests, not runtime arguments.
If you want a parameter different from the manuscript default, edit the
constant at the top of the script (e.g. `N_SEEDS`, `DELTA`, `K_CVS`).

## Output discipline

- Each experiment writes only into `output/<experiment>/`, never into
  `data/` or `src/`.
- `run.log` is appended (most scripts open with `mode='w'`, so re-running
  overwrites the previous log; rename it first if you want to keep history).
- Raw per-cell pickles live under `raw/`; figures under `figures/`.
- `env.txt` (when emitted) records the numpy/sklearn/python versions for the
  run, so you can audit reproducibility.
