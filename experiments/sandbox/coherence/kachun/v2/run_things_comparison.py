"""Compare three rank-selection methods from coherence on THINGS behavioral.

Methods:
  1. Count-based activation (stage 1 diagnostic) -- known to be unstable
  2. Kappa changepoint + signal-noise liftoff (stage 2) -- expected stable
  3. Coherence CV curve: I_k(p_max) vs null threshold tau_k -- per-dimension

Runs coherence once at k_max=120, then subsets results to simulate
different k_max values and compares all three methods.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import logging

from src.similarity import build_similarity
from src.colors import ROSE, TEAL, CYAN, GRAY, GRAY_DARK, GRAY_LIGHT
from src.utils import get_output_dir
from src.utils.figure_theme import create_figure, despine, save_figure
from omegaconf import OmegaConf

from _compute_coherence import (
    compute_incremental_coherence_multi_k_eig_anisotropic,
    analyze_iproj_signal_alignment_v3,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()


def _count_based_activation(diag, k_arr, p_list):
    """Recompute count-based activation from diagnostics subset."""
    tau_kp = diag["tau_kp"]
    x_ci_lo = diag["x_ci_lo"]
    K = len(k_arr)

    A = (x_ci_lo[:K] >= tau_kp[:K])
    N = A.sum(axis=0)

    activation_idx = np.full(K, -1, int)
    activation_p = np.full(K, np.nan, float)
    for r in range(K):
        idxs = np.where(N >= (r + 1))[0]
        if len(idxs) > 0:
            activation_idx[r] = int(idxs[0])
            activation_p[r] = float(p_list[idxs[0]])

    valid = ~np.isnan(activation_p)
    if np.any(valid):
        k_star = int(k_arr[np.where(valid)[0][-1]])
    else:
        k_star = None
    return k_star, activation_p, valid


def _cv_curve_kstar(i_boot, tau_kp, k_arr):
    """k* from coherence CV curve: last k where I_k(p_max) > tau_k(p_max)."""
    K = len(k_arr)
    i_at_maxp = np.median(i_boot[:K, -1, :], axis=1)
    tau_at_maxp = tau_kp[:K, -1] if tau_kp is not None else None

    if tau_at_maxp is None:
        return None, i_at_maxp, None

    above = i_at_maxp > tau_at_maxp
    if np.any(above):
        k_star = int(k_arr[np.where(above)[0][-1]])
    else:
        k_star = None
    return k_star, i_at_maxp, tau_at_maxp


def main():
    cfg = OmegaConf.create(
        {
            "name": "things_behavior",
            "type": "triplet",
            "path": "data/things",
            "triplet_number": "4.7mio",
            "n_objects": 1854,
        }
    )
    s = build_similarity(cfg)
    n = s.shape[0]
    log.info(f"THINGS behavioral: {s.shape}, NaN={np.sum(np.isnan(s))}")

    k_list_full = list(range(5, 121, 5))
    p_list = np.linspace(0.05, 0.95, 25)
    k_arr_full = np.array(k_list_full)

    log.info(f"Running coherence: k_list={k_list_full[0]}..{k_list_full[-1]}, B=100")
    result = compute_incremental_coherence_multi_k_eig_anisotropic(
        s,
        k_list_full,
        p_list,
        B=100,
        use_baseline_correction=False,
        clip_baseline=False,
        plot_tau=False,
        mark_activation=True,
        compute_null=True,
        visualize=False,
    )
    log.info("Coherence computation done.")

    diag = result["diagnostics"]
    i_boot = result["I_boot"]

    # -- Compare methods across k_max subsets --
    k_max_values = [40, 50, 60, 70, 80, 90, 100, 110, 120]

    results_activation = []
    results_kappa = []
    results_cv = []

    for km in k_max_values:
        idx_end = np.searchsorted(k_arr_full, km, side="right")
        k_sub = k_arr_full[:idx_end]
        k_list_sub = k_sub.tolist()

        # Method 1: Count-based activation
        k_act, _, _ = _count_based_activation(diag, k_sub, p_list)
        results_activation.append(k_act)

        # Method 2: Kappa changepoint + liftoff
        analysis = analyze_iproj_signal_alignment_v3(
            Iproj_boot=i_boot[:idx_end],
            k_list=k_list_sub,
            p_list=p_list,
            K_sig_mode="kappa_changepoint",
            signal_reference="boundary",
            signal_curve_mode="lower_ci",
            noise_curve_mode="upper_ci",
            smooth_window=3,
            noise_quantile=0.90,
            lift_margin=0.0,
            require_consecutive=1,
            make_plots=False,
        )
        k_kappa = analysis.summary["K_sig_details"]["k_cut"]
        p_kappa = analysis.p_star["recommended_global"]
        results_kappa.append((k_kappa, p_kappa))

        # Method 3: Coherence CV curve
        k_cv, _, _ = _cv_curve_kstar(i_boot, diag["tau_kp"], k_sub)
        results_cv.append(k_cv)

    # -- Print comparison table --
    log.info("\n=== COMPARISON TABLE ===")
    log.info(f"{'k_max':>6} | {'Activation k*':>15} | {'Kappa k*':>10} | {'Kappa p*':>10} | {'CV curve k*':>12}")
    log.info("-" * 65)
    for i, km in enumerate(k_max_values):
        k_act = results_activation[i]
        k_kap, p_kap = results_kappa[i]
        k_cv = results_cv[i]
        act_str = str(k_act) if k_act is not None else "None"
        p_str = f"{p_kap:.3f}" if p_kap is not None else "None"
        cv_str = str(k_cv) if k_cv is not None else "None"
        log.info(f"{km:>6} | {act_str:>15} | {k_kap:>10} | {p_str:>10} | {cv_str:>12}")

    # =====================================================================
    # PLOT 1: k* vs k_max for all three methods
    # =====================================================================
    fig, ax = create_figure("wide")

    k_act_arr = np.array([v if v is not None else np.nan for v in results_activation])
    k_kap_arr = np.array([v[0] for v in results_kappa])
    k_cv_arr = np.array([v if v is not None else np.nan for v in results_cv])

    ax.plot(k_max_values, k_act_arr, marker="o", markersize=5, linewidth=1.5,
            color=ROSE, label="Count-based activation")
    ax.plot(k_max_values, k_kap_arr, marker="s", markersize=5, linewidth=1.5,
            color=TEAL, label="Kappa changepoint")
    ax.plot(k_max_values, k_cv_arr, marker="^", markersize=5, linewidth=1.5,
            color=CYAN, label="CV curve")

    ax.plot(k_max_values, k_max_values, linestyle=":", linewidth=0.8, color=GRAY_LIGHT,
            label="k* = k_max (ceiling)")
    ax.set_xlabel("Maximum tested rank $k_{\\max}$")
    ax.set_ylabel("Estimated $k^*$")
    ax.set_title("Rank estimate stability across $k_{\\max}$")
    ax.legend(fontsize=6)
    despine(ax)
    fig.savefig(
        OUTPUT_DIR / "kstar_vs_kmax.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)

    # =====================================================================
    # PLOT 2: Kappa changepoint p* vs k_max
    # =====================================================================
    fig, ax = create_figure("wide")
    p_kap_arr = np.array([v[1] if v[1] is not None else np.nan for v in results_kappa])
    valid_p = ~np.isnan(p_kap_arr)
    ax.plot(np.array(k_max_values)[valid_p], p_kap_arr[valid_p],
            marker="s", markersize=5, linewidth=1.5, color=TEAL)
    ax.set_xlabel("Maximum tested rank $k_{\\max}$")
    ax.set_ylabel("Estimated $p^*$")
    ax.set_title("Liftoff $p^*$ stability across $k_{\\max}$")
    ax.set_ylim(0, 1.02)
    despine(ax)
    fig.savefig(
        OUTPUT_DIR / "pstar_vs_kmax.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)

    # =====================================================================
    # PLOT 3: The three methods side-by-side at k_max=120
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    # -- Panel A: Count-based activation --
    ax = axes[0]
    _, act_p, act_valid = _count_based_activation(diag, k_arr_full, p_list)
    ax.plot(k_arr_full[act_valid], act_p[act_valid],
            marker="o", markersize=3, linewidth=1, color=ROSE)
    ax.set_xlabel("Rank $k$")
    ax.set_ylabel("Activation $p$")
    ax.set_ylim(0, 1.02)
    ax.set_title("Count-based activation", fontsize=8)
    k_act_120 = results_activation[-1]
    if k_act_120 is not None:
        ax.axvline(k_act_120, linestyle=":", linewidth=0.8, color=GRAY)
        ax.text(k_act_120 + 2, 0.15, f"$k^*$={k_act_120}", fontsize=6, color=GRAY_DARK)
    despine(ax)

    # -- Panel B: Kappa changepoint --
    ax = axes[1]
    analysis_full = analyze_iproj_signal_alignment_v3(
        Iproj_boot=i_boot,
        k_list=k_list_full,
        p_list=p_list,
        K_sig_mode="kappa_changepoint",
        signal_reference="boundary",
        signal_curve_mode="lower_ci",
        noise_curve_mode="upper_ci",
        smooth_window=3,
        noise_quantile=0.90,
        lift_margin=0.0,
        require_consecutive=1,
        make_plots=False,
    )
    kappa = analysis_full.summary["kappa_hat"]
    k_cut = analysis_full.summary["K_sig_details"]["k_cut"]
    k_sig_mask = k_arr_full <= k_cut

    ax.plot(k_arr_full, kappa, marker=".", markersize=3, linewidth=0.8, color=GRAY_DARK)
    ax.scatter(k_arr_full[k_sig_mask], kappa[k_sig_mask], s=20, color=TEAL, zorder=5)
    ax.scatter(k_arr_full[~k_sig_mask], kappa[~k_sig_mask], s=10, color=GRAY_LIGHT, zorder=4)
    ax.axvline(k_cut + 2.5, linestyle="--", linewidth=1, color=TEAL)
    ax.text(k_cut + 4, kappa.max() * 0.9, f"$k^*$={k_cut}", fontsize=6, color=TEAL)
    ax.set_xlabel("Dimension $k$")
    ax.set_ylabel("$\\hat{\\kappa}_k$")
    ax.set_title("Kappa changepoint", fontsize=8)
    despine(ax)

    # -- Panel C: Coherence CV curve --
    ax = axes[2]
    k_cv_120, i_at_maxp, tau_at_maxp = _cv_curve_kstar(i_boot, diag["tau_kp"], k_arr_full)
    i_lo_maxp = np.quantile(i_boot[:, -1, :], 0.05, axis=1)
    i_hi_maxp = np.quantile(i_boot[:, -1, :], 0.95, axis=1)

    ax.fill_between(k_arr_full, i_lo_maxp, i_hi_maxp, color=CYAN, alpha=0.15)
    ax.plot(k_arr_full, i_at_maxp, marker="o", markersize=3, linewidth=1, color=CYAN)
    if tau_at_maxp is not None:
        ax.plot(k_arr_full, tau_at_maxp, linestyle="--", linewidth=1.2, color=ROSE,
                label="Null $\\tau$")
    if k_cv_120 is not None:
        ax.axvline(k_cv_120, linestyle=":", linewidth=0.8, color=GRAY)
        ax.text(k_cv_120 + 2, i_at_maxp.max() * 0.9, f"$k^*$={k_cv_120}", fontsize=6,
                color=GRAY_DARK)
    ax.set_xlabel("Dimension $k$")
    ax.set_ylabel(f"$I_k(p={p_list[-1]:.2f})$")
    ax.set_title("Coherence CV curve", fontsize=8)
    ax.legend(fontsize=5, loc="upper right")
    despine(ax)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "three_methods_comparison.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)

    # =====================================================================
    # PLOT 4: Coherence heatmap with all three k* lines
    # =====================================================================
    i_med = analysis_full.summary["I_median"]

    fig, ax = create_figure("wide")
    im = ax.imshow(
        i_med,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[p_list[0], p_list[-1], k_arr_full[0] - 2.5, k_arr_full[-1] + 2.5],
        cmap="magma",
    )
    colors_lines = [ROSE, TEAL, CYAN]
    labels = [
        f"Activation $k^*$={k_act_120}" if k_act_120 else "Activation: None",
        f"Kappa $k^*$={k_cut}",
        f"CV curve $k^*$={k_cv_120}" if k_cv_120 else "CV curve: None",
    ]
    k_vals = [k_act_120, k_cut, k_cv_120]
    for kv, c, lab in zip(k_vals, colors_lines, labels):
        if kv is not None:
            ax.axhline(kv + 2.5, color=c, linewidth=1.5, linestyle="--", label=lab)

    ax.set_xlabel("Sampling fraction $p$")
    ax.set_ylabel("Dimension $k$")
    ax.set_title("$I^{\\mathrm{proj}}_k(p)$ with rank estimates")
    ax.legend(fontsize=6, loc="upper left")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(
        OUTPUT_DIR / "heatmap_all_methods.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)

    # =====================================================================
    # PLOT 5: Signal-noise liftoff at k_max=120 (shows p*)
    # =====================================================================
    signal_ref = analysis_full.summary["signal_ref_curve"]
    noise_ref = analysis_full.summary["noise_ref_curve"]
    ref_gap = analysis_full.summary["ref_gap_curve"]
    p_star = analysis_full.p_star["recommended_global"]

    fig, ax = create_figure("wide")
    ax.plot(p_list, signal_ref, linewidth=1.5, color=CYAN,
            label=f"Signal ref ($k$={k_cut})")
    ax.plot(p_list, noise_ref, linewidth=1.5, linestyle="--", color=ROSE,
            label="Noise ref")
    ax.plot(p_list, ref_gap, linewidth=1, linestyle=":", color=GRAY_DARK,
            label="Gap")
    ax.axhline(0.0, color=GRAY_LIGHT, linewidth=0.5)
    if p_star is not None:
        ax.axvline(p_star, color="k", linewidth=1, linestyle="--",
                    label=f"$p^*$ = {p_star:.3f}")
    ax.set_xlabel("Sampling fraction $p$")
    ax.set_ylabel("Coherence")
    ax.set_title(f"Signal-noise liftoff ($k^*$={k_cut})")
    ax.legend(fontsize=6)
    despine(ax)
    fig.savefig(
        OUTPUT_DIR / "signal_noise_liftoff.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)

    # -- Final summary --
    log.info("\n=== FINAL SUMMARY (k_max=120) ===")
    log.info(f"  Count-based activation: k* = {k_act_120}")
    log.info(f"  Kappa changepoint:      k* = {k_cut}, p* = {p_star}")
    log.info(f"  Coherence CV curve:     k* = {k_cv_120}")
    log.info(f"\nPlots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
