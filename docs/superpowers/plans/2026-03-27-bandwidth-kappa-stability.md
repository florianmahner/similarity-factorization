# Bandwidth-Kappa Stability Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether kappa changepoint sharpness (cheap, no factorization) predicts which RBF bandwidth produces the most stable SRF embedding on DINOv3 features.

**Architecture:** Single sandbox script with three stages: (1) compute kappa curves at 5 bandwidth multipliers, extracting changepoint SNR, (2) run 5 SRF fits per bandwidth at the kappa-estimated rank, compute pairwise embedding stability via Hungarian-aligned correlations, (3) compare SNR ranking to stability ranking. All stages parallelize via joblib.

**Tech Stack:** numpy, scipy (linear_sum_assignment, median filter), sklearn (pairwise_distances, pairwise_kernels), pysrf (SRF), joblib, pandas, existing coherence functions from src/coherence.py.

---

### Task 1: Create scaffold and build RBF similarity matrices

**Files:**
- Create: `experiments/sandbox/things/bandwidth_kappa_stability/run.py`

- [ ] **Step 1: Create the script scaffold**

```python
"""Joint bandwidth-rank selection via kappa sharpness and profile stability.

Tests whether kappa changepoint SNR predicts factorization stability across
RBF bandwidth multipliers on DINOv3 features (1854 images).
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances, pairwise_kernels

from pysrf import SRF
from src.coherence import (
    _estimate_kappa_hat,
    _mad,
    _smooth_median,
    compute_incremental_coherence_multi_k_eig_anisotropic,
    kappa_changepoint,
)
from src.utils import get_output_dir

log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()

FEATURES_PATH = Path("experiments/sandbox/things/dino_extract/outputs/latest/dinov3_features.npy")
SIGMA_MULTS = [0.2, 0.4, 0.6, 0.8, 1.0]

# Coherence parameters (reduced for speed)
B = 20
B_NULL = 15
N_P = 20
K_MAX = 100
HI_BAND_QUANTILE = 0.85
SMOOTH_WINDOW = 5

# Stability parameters
N_SRF_RUNS = 5


def _build_rbf(features: np.ndarray, sigma_mult: float) -> tuple[np.ndarray, float]:
    """Build RBF similarity matrix with given multiplier of the median heuristic."""
    dist = pairwise_distances(features, metric="euclidean")
    median_dist = float(np.median(dist[np.triu_indices(len(dist), k=1)]))
    sigma = median_dist * sigma_mult
    gamma = 1 / (2 * sigma**2)
    s = pairwise_kernels(features, metric="rbf", gamma=gamma)
    return s, sigma
```

- [ ] **Step 2: Verify the scaffold imports work**

Run: `poetry run python -c "from experiments.sandbox.things.bandwidth_kappa_stability.run import _build_rbf; print('OK')"`
Expected: `OK`

---

### Task 2: Implement Stage 1 -- kappa changepoint sharpness

**Files:**
- Modify: `experiments/sandbox/things/bandwidth_kappa_stability/run.py`

- [ ] **Step 1: Add the kappa SNR computation function**

Add after `_build_rbf`:

```python
def _compute_kappa_snr(s: np.ndarray) -> dict:
    """Compute kappa curve and changepoint SNR for a similarity matrix.

    Returns dict with k_star, kappa_hat, snr, and diagnostics.
    """
    n = s.shape[0]
    k_list = list(range(1, min(K_MAX, n - 1) + 1))
    p_list = np.linspace(0.05, 0.95, N_P)

    result = compute_incremental_coherence_multi_k_eig_anisotropic(
        s,
        k_list=k_list,
        p_list=p_list,
        B=B,
        random_state=42,
        compute_null=True,
        B_null=B_NULL,
        alpha_tau=0.95,
        ci_level=0.95,
        use_baseline_correction=False,
        n_jobs=os.cpu_count() - 1,
        show_progress=False,
        visualize=False,
    )

    diag = result["diagnostics"]
    x_median = diag["x_median"]
    k_arr = result["k_list"]
    p_arr = result["p"]

    kappa_hat, kappa_info = _estimate_kappa_hat(x_median, p_arr, HI_BAND_QUANTILE)
    k_star, cp_info = kappa_changepoint(kappa_hat, k_arr, smooth_window=SMOOTH_WINDOW)

    # Changepoint SNR: max |delta_kappa_smooth| / MAD(kappa_hat)
    kappa_smooth = _smooth_median(kappa_hat, SMOOTH_WINDOW)
    delta = np.diff(kappa_smooth)
    mad_val = _mad(kappa_hat)
    snr = float(np.max(np.abs(delta)) / mad_val)

    return {
        "k_star": int(k_star),
        "kappa_hat": kappa_hat,
        "kappa_smooth": kappa_smooth,
        "snr": snr,
        "mad": mad_val,
        "max_abs_delta": float(np.max(np.abs(delta))),
        "k_arr": k_arr,
    }
```

- [ ] **Step 2: Add the Stage 1 driver that loops over bandwidths**

Add after `_compute_kappa_snr`:

```python
def run_stage1(features: np.ndarray) -> tuple[list[dict], dict]:
    """Stage 1: Compute kappa changepoint SNR for each bandwidth.

    Returns list of record dicts and dict of kappa curves keyed by sigma_mult.
    """
    records = []
    kappa_curves = {}

    for mult in SIGMA_MULTS:
        log.info(f"Stage 1: sigma_mult={mult}")
        s, sigma = _build_rbf(features, mult)
        sim_mean = float(s.mean())
        sim_min = float(s.min())
        log.info(f"  sigma={sigma:.2f}, sim_mean={sim_mean:.4f}, sim_min={sim_min:.4f}")

        kappa_result = _compute_kappa_snr(s)
        log.info(f"  k*={kappa_result['k_star']}, SNR={kappa_result['snr']:.3f}")

        records.append({
            "sigma_mult": mult,
            "sigma": sigma,
            "k_star": kappa_result["k_star"],
            "snr": kappa_result["snr"],
            "mad": kappa_result["mad"],
            "max_abs_delta": kappa_result["max_abs_delta"],
            "sim_mean": sim_mean,
            "sim_min": sim_min,
        })

        kappa_curves[str(mult)] = kappa_result["kappa_hat"]

    return records, kappa_curves
```

---

### Task 3: Implement Stage 2 -- profile stability

**Files:**
- Modify: `experiments/sandbox/things/bandwidth_kappa_stability/run.py`

- [ ] **Step 1: Add the Hungarian-aligned reliability function**

Add after `run_stage1`:

```python
def _align_and_correlate(w_a: np.ndarray, w_b: np.ndarray) -> np.ndarray:
    """Align two embedding matrices via Hungarian matching on absolute correlation.

    Returns per-dimension correlation after optimal alignment.
    """
    k = w_a.shape[1]
    corr_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            r = np.corrcoef(w_a[:, i], w_b[:, j])[0, 1]
            corr_matrix[i, j] = r if np.isfinite(r) else 0.0

    # Hungarian on negative absolute correlation (minimize cost = maximize matching)
    row_idx, col_idx = linear_sum_assignment(-np.abs(corr_matrix))

    per_dim_corr = np.zeros(k)
    for i, j in zip(row_idx, col_idx):
        per_dim_corr[i] = abs(corr_matrix[i, j])

    return per_dim_corr


def _compute_stability(s: np.ndarray, rank: int) -> dict:
    """Run N_SRF_RUNS independent SRF fits and compute pairwise embedding stability.

    Returns dict with mean/min stability and per-dimension reliability.
    """
    embeddings = Parallel(n_jobs=-1)(
        delayed(_fit_one_srf)(s, rank, seed) for seed in range(N_SRF_RUNS)
    )

    # All pairwise correlations (C(5,2) = 10 pairs)
    n_runs = len(embeddings)
    k = rank
    all_per_dim = []

    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            per_dim = _align_and_correlate(embeddings[i], embeddings[j])
            all_per_dim.append(per_dim)

    all_per_dim = np.array(all_per_dim)  # (n_pairs, k)

    # Average in Fisher-z space for correctness
    z_scores = np.arctanh(np.clip(all_per_dim, -0.999, 0.999))
    mean_z = np.mean(z_scores, axis=0)
    reliability_per_dim = np.tanh(mean_z)

    return {
        "stability_mean": float(np.mean(reliability_per_dim)),
        "stability_min": float(np.min(reliability_per_dim)),
        "reliability_per_dim": reliability_per_dim,
    }


def _fit_one_srf(s: np.ndarray, rank: int, seed: int) -> np.ndarray:
    """Fit a single SRF model and return the embedding."""
    model = SRF(rank=rank, random_state=seed)
    model.fit(s)
    return model.w_
```

- [ ] **Step 2: Add the Stage 2 driver**

Add after `_fit_one_srf`:

```python
def run_stage2(features: np.ndarray, stage1_records: list[dict]) -> list[dict]:
    """Stage 2: Compute profile stability for each bandwidth at its kappa-estimated rank."""
    stability_records = []

    for rec in stage1_records:
        mult = rec["sigma_mult"]
        k_star = rec["k_star"]
        log.info(f"Stage 2: sigma_mult={mult}, rank={k_star}")

        s, _ = _build_rbf(features, mult)
        stab = _compute_stability(s, k_star)
        log.info(f"  stability_mean={stab['stability_mean']:.3f}, "
                 f"stability_min={stab['stability_min']:.3f}")

        stability_records.append({
            "sigma_mult": mult,
            "k_star": k_star,
            "stability_mean": stab["stability_mean"],
            "stability_min": stab["stability_min"],
            "reliability_per_dim": stab["reliability_per_dim"],
        })

    return stability_records
```

---

### Task 4: Implement Stage 3 and main -- validation and output

**Files:**
- Modify: `experiments/sandbox/things/bandwidth_kappa_stability/run.py`

- [ ] **Step 1: Add the main function with all three stages and output saving**

Add at the end of the file:

```python
def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    log.info(f"Loading features from {FEATURES_PATH}")
    features = np.load(FEATURES_PATH)
    log.info(f"Features shape: {features.shape}")

    # Stage 1: Kappa changepoint sharpness
    log.info("\n=== Stage 1: Kappa changepoint sharpness ===")
    stage1_records, kappa_curves = run_stage1(features)

    # Stage 2: Profile stability
    log.info("\n=== Stage 2: Profile stability ===")
    stage2_records = run_stage2(features, stage1_records)

    # Merge records
    rows = []
    stability_per_dim = {}
    for s1, s2 in zip(stage1_records, stage2_records):
        rows.append({
            "sigma_mult": s1["sigma_mult"],
            "sigma": s1["sigma"],
            "k_star": s1["k_star"],
            "snr": s1["snr"],
            "mad": s1["mad"],
            "max_abs_delta": s1["max_abs_delta"],
            "sim_mean": s1["sim_mean"],
            "sim_min": s1["sim_min"],
            "stability_mean": s2["stability_mean"],
            "stability_min": s2["stability_min"],
        })
        stability_per_dim[str(s1["sigma_mult"])] = s2["reliability_per_dim"]

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    log.info(f"\nSaved results to {OUTPUT_DIR / 'results.csv'}")

    # Save kappa curves and per-dim stability for diagnostics
    np.savez(OUTPUT_DIR / "kappa_curves.npz", **kappa_curves)
    np.savez(OUTPUT_DIR / "stability_per_dim.npz", **stability_per_dim)

    # Stage 3: Validation -- compare rankings
    log.info("\n=== Stage 3: Validation ===")
    snr_rank = np.argsort(-df["snr"].values)  # descending
    stab_rank = np.argsort(-df["stability_mean"].values)  # descending

    snr_best = df.iloc[snr_rank[0]]["sigma_mult"]
    stab_best = df.iloc[stab_rank[0]]["sigma_mult"]

    from scipy.stats import spearmanr
    rho, p_val = spearmanr(df["snr"].values, df["stability_mean"].values)

    log.info(f"Best by SNR:       sigma_mult={snr_best}")
    log.info(f"Best by stability: sigma_mult={stab_best}")
    log.info(f"Spearman rho(SNR, stability) = {rho:.3f} (p={p_val:.4f})")
    log.info(f"Argmax agreement: {'YES' if snr_best == stab_best else 'NO'}")

    log.info(f"\n=== Summary table ===")
    log.info(df.to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create empty `__init__.py`**

Create `experiments/sandbox/things/bandwidth_kappa_stability/__init__.py` (empty file).

- [ ] **Step 3: Run the experiment**

Run: `./scripts/submit experiments/sandbox/things/bandwidth_kappa_stability/run.py --bg`

Monitor via: `./scripts/jobs bandwidth_kappa_stability --tail`

- [ ] **Step 4: Commit**

```bash
git add experiments/sandbox/things/bandwidth_kappa_stability/
git commit -m "feat: add bandwidth-kappa stability experiment for DINOv3

Tests whether kappa changepoint SNR predicts factorization stability
across RBF bandwidth multipliers [0.2, 0.4, 0.6, 0.8, 1.0].
Stage 1: kappa sharpness (no factorization needed).
Stage 2: 5-run SRF stability via Hungarian-aligned correlations.
Stage 3: Spearman rank correlation between SNR and stability."
```
