"""End-to-end check: coherence k* and p* should match cross-validation minimum.

Creates a synthetic low-rank similarity matrix, estimates both the rank (k*)
and operating point (p*) from eigenspace coherence, then feeds p* as the
sampling fraction into cross-validation. The CV minimum should recover k*.

Usage:
    poetry run python tests/checks/check_coherence.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pysrf import SRF, cross_val_score
from pysrf.coherence import estimate_rank


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def _make_low_rank_similarity(n: int, true_rank: int, noise: float, seed: int):
    """Synthetic non-negative similarity matrix with known rank."""
    rng = np.random.default_rng(seed)
    w = np.abs(rng.standard_normal((n, true_rank)))
    s = w @ w.T
    e = np.abs(rng.standard_normal((n, n)))
    e = (e + e.T) / 2
    s += noise * e
    return s, w


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    seed = 42
    n = 150
    true_rank = 8
    noise = 0.02

    print(f"Simulation: n={n}, true_rank={true_rank}, noise={noise}")
    s, _ = _make_low_rank_similarity(n, true_rank, noise, seed)

    # -- Step 1: coherence → k* and p* --
    print("\n--- Coherence rank estimation ---")
    result = estimate_rank(
        s,
        k_max=20,
        p_list=np.linspace(0.1, 0.95, 20),
        n_boot=30,
        random_state=seed,
        n_jobs=-1,
    )

    k_star = result["k_star"]
    p_star = result["p_star"]
    kappa = result["kappa"]
    iproj_median = result["iproj_median"]
    p_list = result["p_list"]
    k_list = result["k_list"]
    signal_curve = result["signal_curve"]
    noise_curve = result["noise_curve"]

    print(f"k* = {k_star}  (true rank = {true_rank})")
    print(f"p* = {p_star:.3f}  (signal-noise lift-off)")

    # -- Step 2: feed p* into cross-validation --
    print(f"\n--- Cross-validation (sampling_fraction = p* = {p_star:.3f}) ---")
    ranks_to_test = list(range(2, 18, 2))
    grid = cross_val_score(
        s,
        estimator=SRF(random_state=seed, max_outer=50),
        param_grid={"rank": ranks_to_test},
        sampling_fraction=p_star,
        n_repeats=10,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )

    cv_results = grid.cv_results_
    mean_scores = cv_results.groupby("rank")["score"].mean()
    std_scores = cv_results.groupby("rank")["score"].std()
    k_cv = int(mean_scores.idxmin())

    print(f"CV best rank = {k_cv}")
    print(f"CV scores:\n{mean_scores.to_string()}")

    # -- Step 3: bounds as independent sanity check --
    from pysrf.bounds import estimate_sampling_bounds_fast

    pmin, pmax, _ = estimate_sampling_bounds_fast(s, random_state=seed, n_jobs=-1)
    p_bounds = 0.5 * (pmin + pmax)
    print("\n--- Bounds (sanity check) ---")
    print(f"pmin = {pmin:.3f}, pmax = {pmax:.3f}, mean = {p_bounds:.3f}")
    print(f"coherence p* = {p_star:.3f}")

    # -- Step 4: check agreement --
    print("\n--- Result ---")
    match = k_star == k_cv
    print(
        f"Coherence k* = {k_star}, CV best = {k_cv}, "
        f"true rank = {true_rank}, match = {match}"
    )

    # -- Step 5: plot --
    plot_dir = Path(__file__).parent / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Panel 1: Iproj curves
    ax = axes[0, 0]
    for ki in range(0, len(k_list), 4):
        ax.plot(p_list, iproj_median[ki], label=f"k={k_list[ki]}")
    ax.axvline(p_star, color="k", linestyle="--", linewidth=1, label=f"p*={p_star:.2f}")
    ax.set_xlabel("p (masking probability)")
    ax.set_ylabel("Iproj (median)")
    ax.set_title("Eigenspace coherence curves")
    ax.legend(fontsize=8)

    # Panel 2: Kappa + changepoint
    ax = axes[0, 1]
    ax.plot(k_list, kappa, "o-", markersize=4)
    ax.axvline(k_star + 0.5, color="r", linestyle="--", label=f"k*={k_star}")
    ax.axvline(true_rank + 0.5, color="gray", linestyle=":", label=f"true={true_rank}")
    ax.set_xlabel("k")
    ax.set_ylabel("kappa (scaled leakage)")
    ax.set_title("Kappa changepoint")
    ax.legend(fontsize=8)

    # Panel 3: signal-noise lift-off
    ax = axes[1, 0]
    ax.plot(
        p_list, signal_curve, "b-", linewidth=2, label=f"signal median (k*={k_star})"
    )
    ax.plot(p_list, noise_curve, "r--", linewidth=2, label="noise upper envelope")
    ax.fill_between(
        p_list,
        noise_curve,
        signal_curve,
        where=signal_curve > noise_curve,
        alpha=0.15,
        color="green",
    )
    ax.axvline(p_star, color="k", linestyle="--", linewidth=1, label=f"p*={p_star:.2f}")
    ax.set_xlabel("p (masking probability)")
    ax.set_ylabel("Iproj")
    ax.set_title("Signal-noise lift-off")
    ax.legend(fontsize=8)

    # Panel 4: CV curve
    ax = axes[1, 1]
    ax.errorbar(
        mean_scores.index,
        mean_scores.values,
        yerr=std_scores.values,
        fmt="o-",
        capsize=3,
    )
    ax.axvline(k_star, color="r", linestyle="--", label=f"coherence k*={k_star}")
    ax.axvline(k_cv, color="b", linestyle=":", label=f"CV best={k_cv}")
    ax.axvline(
        true_rank, color="gray", linestyle=":", alpha=0.5, label=f"true={true_rank}"
    )
    ax.set_xlabel("rank")
    ax.set_ylabel("MSE (validation)")
    ax.set_title(f"Cross-validation (p*={p_star:.2f} from coherence)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(plot_dir / "check_coherence.png", dpi=150)
    print(f"\nPlot saved to {plot_dir / 'check_coherence.png'}")
    plt.close()


if __name__ == "__main__":
    main()
