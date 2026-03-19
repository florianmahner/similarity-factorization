"""Coherence analysis on Peterson animals to find optimal k and p*."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from src.datasets import load_dataset
from src.colors import ROSE, TEAL, CYAN, GRAY, GRAY_DARK, GRAY_LIGHT
from src.utils import get_output_dir
from src.utils.figure_theme import create_figure, despine, save_figure

from _compute_coherence import (
    compute_incremental_coherence_multi_k_eig_anisotropic,
    analyze_iproj_signal_alignment_v3,
)

OUTPUT_DIR = get_output_dir()


def main():
    ds = load_dataset("peterson-animals", root="/SSD/datasets/similarity_datasets/peterson")
    s = ds.rsm
    n = s.shape[0]
    print(f"Peterson animals RSM: {s.shape}, range [{s.min():.2f}, {s.max():.2f}]")

    k_list = list(range(1, min(n - 1, 61)))
    p_list = np.linspace(0.05, 0.95, 25)

    result = compute_incremental_coherence_multi_k_eig_anisotropic(
        s,
        k_list,
        p_list,
        B=100,
        use_baseline_correction=False,
        clip_baseline=False,
        plot_tau=False,
        mark_activation=True,
        visualize=False,
    )

    analysis = analyze_iproj_signal_alignment_v3(
        Iproj_boot=result["I_boot"],
        k_list=k_list,
        p_list=p_list,
        K_sig_mode="kappa_changepoint",
        signal_reference="boundary",
        signal_curve_mode="lower_ci",
        noise_curve_mode="upper_ci",
        smooth_window=5,
        noise_quantile=0.90,
        lift_margin=0.0,
        require_consecutive=1,
        make_plots=False,
    )

    # -- Null-calibrated activation (from stage 1) --
    diag = result["diagnostics"]
    act_p = diag["activation_p"]
    tau_kp = diag["tau_kp"]
    print("\n--- Null-calibrated activation (stage 1) ---")
    print(f"activation_p (first p where >= r dims active):")
    for r in range(min(20, len(act_p))):
        val = act_p[r]
        label = "nan" if val < 0 or np.isnan(p_list[int(val)] if val >= 0 else np.nan) else f"{p_list[int(diag['activation_idx'][r])]:.3f}"
        if diag["activation_idx"][r] >= 0:
            print(f"  r={r+1:2d}: p_act = {p_list[int(diag['activation_idx'][r])]:.3f}")
        else:
            print(f"  r={r+1:2d}: not activated")

    # -- Signal-noise liftoff (from stage 2) --
    k_opt = analysis.summary["K_sig_k_values"][-1]
    p_star = analysis.p_star["recommended_global"]
    p_star_signals = analysis.p_star["global_signals"]
    print(f"\n--- Signal-noise liftoff (stage 2) ---")
    print(f"Estimated optimal k: {k_opt}")
    print(f"p* (recommended_global): {p_star}")
    print(f"p* (global_signals): {p_star_signals}")
    print(f"p* by k (liftoff): {analysis.p_star['by_k'][:k_opt + 5]}")

    if p_star is None:
        p_star = p_star_signals

    k_arr = np.array(k_list)
    i_med = analysis.summary["I_median"]
    i_lo = analysis.summary["I_ci_low"]
    i_hi = analysis.summary["I_ci_high"]
    kappa = analysis.summary["kappa_hat"]
    k_cut = analysis.summary["K_sig_details"]["k_cut"]
    k_sig_mask = k_arr <= k_cut
    noise_ref = analysis.summary["noise_ref_curve"]
    signal_ref = analysis.summary["signal_ref_curve"]

    # -- Plot 1: Iproj heatmap --
    fig, ax = create_figure("wide")
    im = ax.imshow(
        i_med,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[p_list[0], p_list[-1], k_arr[0] - 0.5, k_arr[-1] + 0.5],
        cmap="magma",
    )
    ax.axhline(k_cut + 0.5, color="white", linewidth=1.5, linestyle="--")
    ax.text(
        p_list[1], k_cut + 1.5, f"k* = {k_cut}", color="white", fontsize=8, va="bottom"
    )
    ax.set_xlabel("Sampling fraction $p$")
    ax.set_ylabel("Dimension $k$")
    ax.set_title("Projected incremental coherence $I^{\\mathrm{proj}}_k(p)$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(OUTPUT_DIR / "coherence_heatmap.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # -- Plot 2: Iproj curves for selected k values --
    fig, ax = create_figure("wide")
    sel_k = [1, 2, 3, k_cut - 1, k_cut, k_cut + 1, k_cut + 3, min(k_cut + 10, k_arr[-1])]
    sel_k = sorted(set(k for k in sel_k if 1 <= k <= k_arr[-1]))
    for k in sel_k:
        idx = k - 1
        is_signal = k <= k_cut
        color = TEAL if is_signal else GRAY
        alpha = 0.9 if is_signal else 0.5
        ax.plot(p_list, i_med[idx], label=f"k={k}", color=color, alpha=alpha, linewidth=1.5)
        ax.fill_between(p_list, i_lo[idx], i_hi[idx], color=color, alpha=0.08)
    ax.plot(p_list, noise_ref, linestyle="--", linewidth=2, color=ROSE, label="Noise ref")
    ax.plot(p_list, signal_ref, linewidth=2, color=CYAN, label=f"Signal ref (k={k_cut})")
    if p_star is not None:
        ax.axvline(p_star, color=GRAY_DARK, linewidth=1, linestyle=":", label=f"p* = {p_star:.3f}")
    ax.set_xlabel("Sampling fraction $p$")
    ax.set_ylabel("$I^{\\mathrm{proj}}_k(p)$")
    ax.set_title("Signal vs. noise coherence curves")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    despine(ax)
    fig.savefig(OUTPUT_DIR / "coherence_curves.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # -- Plot 3: Kappa changepoint --
    fig, ax = create_figure("wide")
    ax.plot(k_arr, kappa, marker=".", markersize=4, linewidth=1, color=GRAY_DARK)
    ax.scatter(k_arr[k_sig_mask], kappa[k_sig_mask], s=30, color=TEAL, zorder=5, label="Signal")
    ax.scatter(k_arr[~k_sig_mask], kappa[~k_sig_mask], s=15, color=GRAY_LIGHT, zorder=4, label="Noise")
    ax.axvline(k_cut + 0.5, linestyle="--", linewidth=1.5, color=ROSE, label=f"k* = {k_cut}")
    ax.set_xlabel("Dimension $k$")
    ax.set_ylabel("$\\hat{\\kappa}_k$")
    ax.set_title("Leakage difficulty (kappa changepoint)")
    ax.legend(fontsize=7)
    despine(ax)
    fig.savefig(OUTPUT_DIR / "kappa_changepoint.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # -- Plot 4: Signal-noise liftoff --
    ref_gap = analysis.summary["ref_gap_curve"]
    fig, ax = create_figure("wide")
    ax.plot(p_list, signal_ref, linewidth=2, color=CYAN, label=f"Signal ref (k={k_cut})")
    ax.plot(p_list, noise_ref, linewidth=2, linestyle="--", color=ROSE, label="Noise ref")
    ax.plot(p_list, ref_gap, linewidth=1.5, linestyle=":", color=GRAY_DARK, label="Gap (signal - noise)")
    ax.axhline(0.0, color=GRAY_LIGHT, linewidth=0.8)
    if p_star is not None:
        ax.axvline(p_star, color="k", linewidth=1.2, linestyle="--", label=f"p* = {p_star:.3f}")
    ax.set_xlabel("Sampling fraction $p$")
    ax.set_ylabel("Coherence / gap")
    ax.set_title("Signal-noise liftoff")
    ax.legend(fontsize=7)
    despine(ax)
    fig.savefig(OUTPUT_DIR / "signal_noise_liftoff.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # -- Plot 5: p* by k --
    p_star_by_k = np.array(
        [np.nan if v is None else v for v in analysis.p_star["by_k"]], dtype=float
    )
    fig, ax = create_figure("wide")
    ax.plot(k_arr, p_star_by_k, marker="o", markersize=3, linewidth=1, color=GRAY_DARK)
    ax.scatter(
        k_arr[k_sig_mask], p_star_by_k[k_sig_mask], s=30, color=TEAL, zorder=5, label="Signal"
    )
    if p_star is not None:
        ax.axhline(p_star, linestyle="--", linewidth=1.2, color=ROSE, label=f"p* = {p_star:.3f}")
    ax.axvline(k_cut + 0.5, linestyle=":", linewidth=1, color=GRAY)
    ax.set_xlabel("Dimension $k$")
    ax.set_ylabel("$p^*_k$")
    ax.set_ylim(0, 1.02)
    ax.set_title("Per-dimension optimal sampling fraction")
    ax.legend(fontsize=7)
    despine(ax)
    fig.savefig(OUTPUT_DIR / "pstar_by_k.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # -- Plot 6: Activation-based p* vs k --
    act_idx = diag["activation_idx"]
    act_p_vals = np.array([p_list[int(j)] if j >= 0 else np.nan for j in act_idx])
    valid_act = ~np.isnan(act_p_vals)
    n_active = np.sum(valid_act)

    fig, ax = create_figure("wide")
    ax.plot(k_arr[valid_act], act_p_vals[valid_act], marker="o", markersize=3, linewidth=1, color=TEAL, label="Activation $p$ (null test)")
    if p_star is not None:
        ax.axhline(p_star, linestyle="--", linewidth=1, color=ROSE, alpha=0.7, label=f"Liftoff p* = {p_star:.3f}")
    ax.axvline(k_cut + 0.5, linestyle=":", linewidth=1, color=GRAY, label=f"k* = {k_cut}")
    ax.set_xlabel("Rank $r$ (number of dimensions)")
    ax.set_ylabel("Activation $p$")
    ax.set_ylim(0, 1.02)
    ax.set_title("Null-calibrated activation: min $p$ for $r$ active dimensions")
    ax.legend(fontsize=7)
    despine(ax)
    fig.savefig(OUTPUT_DIR / "activation_p_by_rank.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # -- Final summary --
    if n_active > 0:
        k_act_max = int(k_arr[valid_act][-1])
        p_act_at_kcut = act_p_vals[k_cut - 1] if k_cut <= len(act_p_vals) and valid_act[k_cut - 1] else np.nan
        print(f"\n=== FINAL ESTIMATES ===")
        print(f"  k* = {k_cut}  (kappa changepoint)")
        print(f"  p* = {p_act_at_kcut:.3f}  (activation at k={k_cut})")
        print(f"  Max active rank = {k_act_max} (at p={act_p_vals[k_act_max - 1]:.3f})")

    print(f"\nPlots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
