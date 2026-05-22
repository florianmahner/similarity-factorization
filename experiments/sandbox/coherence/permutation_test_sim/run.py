"""Simulation study: spectral permutation test under controlled conditions.

Diagnoses the non-monotonicity issue: does k* increase with observed data?
Tests the effect of bootstrap aggregation in the null (single-mask vs B-aggregated).

Ground truth: rank-k matrix with known spectrum.
Vary: fraction of observed entries (simulating THINGS 5%-100%).
"""

import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import eigh

from src.colors import ROSE, TEAL, GRAY_LIGHT
from src.utils import get_output_dir
from src.utils.figure_theme import create_figure, despine

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

OUTPUT_DIR = get_output_dir()

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

N = 200
TRUE_RANK = 10
ALPHA_DIR = 0.5
SNR = 1.0
OBS_FRACS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
P_LIST = np.linspace(0.1, 0.95, 15)
K_MAX = 30
B = 30
J = 100
ALPHA = 0.05
SEED = 42


# ---------------------------------------------------------------------------
# Generate ground truth
# ---------------------------------------------------------------------------


def make_similarity(n, k, alpha, snr, rng):
    """Generate rank-k similarity matrix with noise."""
    w = rng.dirichlet(np.ones(k) * alpha, size=n)
    s_signal = w @ w.T
    noise = np.abs(rng.standard_normal((n, n))) * 0.01
    noise = (noise + noise.T) / 2
    s = snr * s_signal + (1 - snr) * noise
    s = (s + s.T) / 2
    return s


def introduce_missing(s, obs_frac, rng):
    """Randomly set a fraction of off-diagonal entries to NaN."""
    n = s.shape[0]
    s_out = s.copy()
    if obs_frac >= 1.0:
        return s_out
    iu = np.triu_indices(n, k=1)
    n_pairs = len(iu[0])
    mask = rng.random(n_pairs) < obs_frac
    missing = ~mask
    s_out[iu[0][missing], iu[1][missing]] = np.nan
    s_out[iu[1][missing], iu[0][missing]] = np.nan
    return s_out


# ---------------------------------------------------------------------------
# Spectral permutation test (both variants)
# ---------------------------------------------------------------------------


def _worker(idx, p_list, is_null, diag, triu_vals, n, k_max, seed_base, aggregate_B):
    """Compute eigenvalues for one replicate across all p values.

    If aggregate_B > 1, draw aggregate_B masks per p and return MEDIAN eigenvalue.
    If aggregate_B == 1, draw one mask per p (current null behavior).
    """
    rng = np.random.default_rng(seed_base)
    iu = np.triu_indices(n, k=1)
    m = len(iu[0])

    if is_null:
        vals = rng.permutation(triu_vals)
    else:
        vals = triu_vals

    A_base = np.zeros((n, n), dtype=np.float64)
    A_base[iu] = vals
    A_base = A_base + A_base.T
    np.fill_diagonal(A_base, diag)

    p_list = np.asarray(p_list, dtype=np.float64)
    out = np.empty((k_max, len(p_list)), dtype=np.float64)

    for j, p in enumerate(p_list):
        if aggregate_B > 1:
            evals_all = np.empty((k_max, aggregate_B), dtype=np.float64)
            for b_idx in range(aggregate_B):
                U = rng.random(m)
                mask = U < p
                A_masked = np.zeros((n, n), dtype=np.float64)
                np.fill_diagonal(A_masked, diag)
                inv_p = 1.0 / p
                masked_vals = vals[mask] * inv_p
                A_masked[iu[0][mask], iu[1][mask]] = masked_vals
                A_masked[iu[1][mask], iu[0][mask]] = masked_vals
                ev = eigh(A_masked, eigvals_only=True)
                evals_all[:, b_idx] = ev[-k_max:][::-1]
            out[:, j] = np.median(evals_all, axis=1)
        else:
            U = rng.random(m)
            mask = U < p
            A_masked = np.zeros((n, n), dtype=np.float64)
            np.fill_diagonal(A_masked, diag)
            inv_p = 1.0 / p
            masked_vals = vals[mask] * inv_p
            A_masked[iu[0][mask], iu[1][mask]] = masked_vals
            A_masked[iu[1][mask], iu[0][mask]] = masked_vals
            ev = eigh(A_masked, eigvals_only=True)
            out[:, j] = ev[-k_max:][::-1]

    return idx, out


def spectral_permutation_test(
    s, k_max, p_list, B, J, alpha, random_state, null_aggregate_B=1,
):
    """Run the spectral permutation test.

    Args:
        null_aggregate_B: if 1, null uses single mask (current).
                         if >1, null uses median of this many masks (proposed fix).
    """
    n = s.shape[0]
    s_sym = (s + s.T) / 2
    nan_mask = np.isnan(s_sym)
    s0 = np.where(nan_mask, 0.0, s_sym)

    diag = np.diag(s0).copy()
    iu = np.triu_indices(n, k=1)
    triu_vals = s0[iu].copy()

    p_list = np.asarray(p_list, dtype=np.float64)
    P = len(p_list)
    n_jobs = max(1, __import__("os").cpu_count() - 1)

    # Dispatch all tasks
    tasks = []
    # Observed: B replicates, each with single mask (we aggregate via median after)
    for b in range(B):
        seed = (random_state + 7919 * b) & 0xFFFFFFFF
        tasks.append((b, p_list, False, diag, triu_vals, n, k_max, seed, 1))

    # Null: J replicates
    for j in range(J):
        seed = (random_state + 104729 * (j + 1)) & 0xFFFFFFFF
        tasks.append((j, p_list, True, diag, triu_vals, n, k_max, seed, null_aggregate_B))

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_worker)(*t) for t in tasks
    )

    # Separate observed and null results
    obs_evals = np.empty((k_max, P, B), dtype=np.float64)
    null_evals = np.empty((k_max, P, J), dtype=np.float64)

    for idx, evals in results[:B]:
        obs_evals[:, :, idx] = evals
    for idx, evals in results[B:]:
        null_evals[:, :, idx] = evals

    # Observed: median across bootstraps
    obs_median = np.median(obs_evals, axis=2)

    # Thresholds
    thresholds = np.quantile(null_evals, 1 - alpha, axis=2)

    # P-values (intersection test over p >= median(p_list))
    p0 = np.median(p_list)
    high_p_mask = p_list >= p0
    pvalues = np.zeros(k_max)
    for k in range(k_max):
        worst_pv = 0.0
        for pi in range(P):
            if not high_p_mask[pi]:
                continue
            n_exceed = np.sum(null_evals[k, pi, :] >= obs_median[k, pi])
            pv = (1 + n_exceed) / (1 + J)
            worst_pv = max(worst_pv, pv)
        pvalues[k] = worst_pv

    significant = np.where(pvalues < alpha)[0]
    k_star = int(significant[-1] + 1) if len(significant) > 0 else 0

    return {
        "k_star": k_star,
        "pvalues": pvalues,
        "obs_median": obs_median,
        "null_evals": null_evals,
        "thresholds": thresholds,
    }


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def main():
    rng = np.random.default_rng(SEED)

    # Generate ground truth matrix (fully observed)
    s_full = make_similarity(N, TRUE_RANK, ALPHA_DIR, SNR, rng)
    evals_full = np.sort(eigh(s_full, eigvals_only=True))[::-1]
    log.info(f"Ground truth: n={N}, true_rank={TRUE_RANK}")
    log.info(f"Top eigenvalues: {evals_full[:15].round(3)}")

    records_v1 = []  # single-mask null (current)
    records_v2 = []  # B-aggregated null (proposed fix)

    for obs_frac in OBS_FRACS:
        s_obs = introduce_missing(s_full, obs_frac, rng)
        n_nan = np.isnan(s_obs).sum() // 2
        n_pairs = N * (N - 1) // 2
        log.info(f"\n--- obs_frac={obs_frac:.0%}, missing={n_nan}/{n_pairs} ({n_nan/n_pairs:.1%}) ---")

        # V1: single-mask null (current behavior)
        t0 = time.time()
        r1 = spectral_permutation_test(
            s_obs, K_MAX, P_LIST, B, J, ALPHA, SEED, null_aggregate_B=1,
        )
        t1 = time.time()
        log.info(f"  V1 (single-mask null): k*={r1['k_star']}, time={t1-t0:.1f}s")
        records_v1.append({"obs_frac": obs_frac, "k_star": r1["k_star"], "pvalues": r1["pvalues"].tolist()})

        # V2: B-aggregated null (proposed fix)
        t0 = time.time()
        r2 = spectral_permutation_test(
            s_obs, K_MAX, P_LIST, B, J, ALPHA, SEED, null_aggregate_B=B,
        )
        t2 = time.time()
        log.info(f"  V2 (B-aggregated null): k*={r2['k_star']}, time={t2-t0:.1f}s")
        records_v2.append({"obs_frac": obs_frac, "k_star": r2["k_star"], "pvalues": r2["pvalues"].tolist()})

    # Save results
    with open(OUTPUT_DIR / "simulation_results.json", "w") as f:
        json.dump({"v1_single_mask": records_v1, "v2_aggregated": records_v2,
                    "true_rank": TRUE_RANK, "n": N}, f, indent=2)

    # Summary table
    log.info(f"\n{'='*60}")
    log.info(f"SIMULATION SUMMARY (true rank = {TRUE_RANK})")
    log.info(f"{'='*60}")
    log.info(f"{'obs%':>5s}  {'V1 (single)':>12s}  {'V2 (aggregated)':>16s}")
    for r1, r2 in zip(records_v1, records_v2):
        log.info(f"{r1['obs_frac']:>4.0%}  {r1['k_star']:>12d}  {r2['k_star']:>16d}")

    # Plots
    _plot_comparison(records_v1, records_v2)
    _plot_pvalue_heatmap(records_v1, records_v2)

    log.info(f"\nOutputs saved to {OUTPUT_DIR}")


def _plot_comparison(records_v1, records_v2):
    """k* vs observation fraction for both variants."""
    fig, ax = create_figure("single")
    fracs = [r["obs_frac"] for r in records_v1]
    k1 = [r["k_star"] for r in records_v1]
    k2 = [r["k_star"] for r in records_v2]

    ax.axhline(TRUE_RANK, color=GRAY_LIGHT, linestyle="--", linewidth=1, label=f"True rank ({TRUE_RANK})")
    ax.plot(fracs, k1, marker="o", color=ROSE, linewidth=2, label="V1: single-mask null")
    ax.plot(fracs, k2, marker="s", color=TEAL, linewidth=2, label="V2: aggregated null")

    ax.set_xlabel("Fraction of observed entries")
    ax.set_ylabel("Estimated rank (k*)")
    ax.set_xticks(fracs)
    ax.set_xticklabels([f"{f:.0%}" for f in fracs])
    ax.legend(fontsize=7)
    despine(ax)

    fig.savefig(OUTPUT_DIR / "kstar_vs_obs_frac.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved kstar_vs_obs_frac.png")


def _plot_pvalue_heatmap(records_v1, records_v2):
    """Per-dimension p-values for both variants across obs fractions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, records, title in [(axes[0], records_v1, "V1: single-mask null"),
                                (axes[1], records_v2, "V2: aggregated null")]:
        pv_matrix = np.array([r["pvalues"][:20] for r in records])
        im = ax.imshow(pv_matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.2)
        ax.set_xlabel("Dimension k")
        ax.set_ylabel("Obs fraction")
        ax.set_yticks(range(len(OBS_FRACS)))
        ax.set_yticklabels([f"{f:.0%}" for f in OBS_FRACS])
        ax.set_xticks(range(0, 20, 2))
        ax.set_xticklabels(range(1, 21, 2))
        ax.set_title(title, fontsize=10)
        # Mark alpha threshold
        ax.contour(pv_matrix, levels=[ALPHA], colors=["black"], linewidths=[1.5])

    plt.colorbar(im, ax=axes, label="p-value", shrink=0.8)
    fig.savefig(OUTPUT_DIR / "pvalue_heatmap.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved pvalue_heatmap.png")


if __name__ == "__main__":
    main()
