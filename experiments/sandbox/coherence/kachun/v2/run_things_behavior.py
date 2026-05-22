"""Coherence analysis on THINGS behavioral similarity (from triplets, has NaN)."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

from src.similarity import build_similarity
from src.colors import ROSE, TEAL, GRAY, GRAY_DARK, GRAY_LIGHT
from src.utils import get_output_dir
from src.utils.figure_theme import create_figure, despine
from omegaconf import OmegaConf

from _compute_coherence import (
    compute_incremental_coherence_multi_k_eig_anisotropic,
    analyze_iproj_signal_alignment_v3,
)

OUTPUT_DIR = get_output_dir()


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
    n_nan = np.sum(np.isnan(s))
    obs_frac = 1.0 - n_nan / s.size
    print(f"THINGS behavioral: {s.shape}, NaN count={n_nan}, obs={obs_frac:.6f}")

    k_list = list(range(5, 121, 5))
    p_list = np.linspace(0.05, 0.95, 25)

    print(f"k_list: {k_list}")
    print(f"p_list: {len(p_list)} values from {p_list[0]:.2f} to {p_list[-1]:.2f}")

    result = compute_incremental_coherence_multi_k_eig_anisotropic(
        s,
        k_list,
        p_list,
        B=100,
        use_baseline_correction=False,
        clip_baseline=False,
        plot_tau=False,
        mark_activation=True,
        compute_null=True,
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
        smooth_window=3,
        noise_quantile=0.90,
        lift_margin=0.0,
        require_consecutive=1,
        make_plots=False,
    )

    diag = result["diagnostics"]
    act_idx = diag["activation_idx"]
    k_arr = np.array(k_list)

    # activation: for rank r, first p where >= r dims active against null
    act_p_vals = np.array(
        [p_list[int(j)] if j >= 0 else np.nan for j in act_idx]
    )
    valid_act = ~np.isnan(act_p_vals)

    print("\n--- Null-calibrated activation ---")
    for r in range(len(k_arr)):
        k = k_arr[r]
        if valid_act[r]:
            print(f"  k={k:3d}: p_act = {act_p_vals[r]:.3f}")
        else:
            print(f"  k={k:3d}: not activated")

    if np.any(valid_act):
        max_active_idx = np.where(valid_act)[0][-1]
        k_final = int(k_arr[max_active_idx])
        p_final = act_p_vals[max_active_idx]
        print(f"\n=== FINAL: k* = {k_final}, p* = {p_final:.3f} ===")
    else:
        k_final = None
        p_final = None
        print("\nNo dimensions activated.")

    k_cut = analysis.summary["K_sig_details"]["k_cut"]
    kappa = analysis.summary["kappa_hat"]
    i_med = analysis.summary["I_median"]
    i_lo = analysis.summary["I_ci_low"]
    i_hi = analysis.summary["I_ci_high"]
    noise_ref = analysis.summary["noise_ref_curve"]
    signal_ref = analysis.summary["signal_ref_curve"]
    k_sig_mask = k_arr <= k_cut

    print(f"\nKappa changepoint: k_cut = {k_cut}")
    print(f"Liftoff p* (recommended): {analysis.p_star['recommended_global']}")

    # -- Plot 1: Coherence heatmap --
    fig, ax = create_figure("wide")
    im = ax.imshow(
        i_med,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[p_list[0], p_list[-1], k_arr[0] - 2.5, k_arr[-1] + 2.5],
        cmap="magma",
    )
    if k_final is not None:
        ax.axhline(k_final + 2.5, color="white", linewidth=1.5, linestyle="--")
        ax.text(p_list[1], k_final + 4, f"k* = {k_final}", color="white", fontsize=8)
    ax.set_xlabel("Sampling fraction $p$")
    ax.set_ylabel("Dimension $k$")
    ax.set_title("$I^{\\mathrm{proj}}_k(p)$ -- THINGS behavioral")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(
        OUTPUT_DIR / "coherence_heatmap.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)

    # -- Plot 2: Iproj curves --
    fig, ax = create_figure("wide")
    for r, k in enumerate(k_arr):
        is_signal = valid_act[r]
        color = TEAL if is_signal else GRAY
        alpha = 0.9 if is_signal else 0.4
        ax.plot(p_list, i_med[r], label=f"k={k}", color=color, alpha=alpha, linewidth=1.2)
        ax.fill_between(p_list, i_lo[r], i_hi[r], color=color, alpha=0.06)
    ax.plot(p_list, noise_ref, linestyle="--", linewidth=2, color=ROSE, label="Noise ref")
    ax.set_xlabel("Sampling fraction $p$")
    ax.set_ylabel("$I^{\\mathrm{proj}}_k(p)$")
    ax.set_title("Coherence curves -- THINGS behavioral")
    ax.legend(fontsize=5, ncol=3, loc="upper left")
    despine(ax)
    fig.savefig(
        OUTPUT_DIR / "coherence_curves.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)

    # -- Plot 3: Kappa --
    fig, ax = create_figure("wide")
    ax.plot(k_arr, kappa, marker=".", markersize=5, linewidth=1, color=GRAY_DARK)
    ax.scatter(k_arr[k_sig_mask], kappa[k_sig_mask], s=30, color=TEAL, zorder=5, label="Signal")
    ax.scatter(k_arr[~k_sig_mask], kappa[~k_sig_mask], s=15, color=GRAY_LIGHT, zorder=4, label="Noise")
    ax.axvline(k_cut + 2.5, linestyle="--", linewidth=1.5, color=ROSE, label=f"k_cut = {k_cut}")
    ax.set_xlabel("Dimension $k$")
    ax.set_ylabel("$\\hat{\\kappa}_k$")
    ax.set_title("Kappa changepoint -- THINGS behavioral")
    ax.legend(fontsize=7)
    despine(ax)
    fig.savefig(
        OUTPUT_DIR / "kappa_changepoint.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)

    # -- Plot 4: Activation p by rank --
    fig, ax = create_figure("wide")
    ax.plot(
        k_arr[valid_act], act_p_vals[valid_act],
        marker="o", markersize=4, linewidth=1.2, color=TEAL, label="Activation $p$",
    )
    if k_final is not None:
        ax.axvline(k_final + 2.5, linestyle=":", linewidth=1, color=GRAY)
        ax.annotate(
            f"k* = {k_final}\np* = {p_final:.3f}",
            xy=(k_final, p_final), xytext=(k_final + 8, p_final - 0.15),
            fontsize=7, arrowprops=dict(arrowstyle="->", color=GRAY_DARK),
        )
    ax.set_xlabel("Rank $k$")
    ax.set_ylabel("Activation $p$")
    ax.set_ylim(0, 1.02)
    ax.set_title("Null-calibrated activation -- THINGS behavioral")
    ax.legend(fontsize=7)
    despine(ax)
    fig.savefig(
        OUTPUT_DIR / "activation_p_by_rank.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)

    # -- Plot 5: Coherence "CV curve" -- I_k at max p with null threshold --
    # Incremental coherence at highest p: how recoverable is each dimension
    # when you have almost all the data. Analog of a CV reconstruction curve.
    i_boot = result["I_boot"]  # (K, P, B)
    tau_kp = diag["tau_kp"]  # (K, P) null threshold

    # Use the highest p value
    i_at_maxp = np.median(i_boot[:, -1, :], axis=1)  # (K,)
    i_lo_maxp = np.quantile(i_boot[:, -1, :], 0.05, axis=1)
    i_hi_maxp = np.quantile(i_boot[:, -1, :], 0.95, axis=1)
    tau_at_maxp = tau_kp[:, -1] if tau_kp is not None else None

    fig, ax = create_figure("wide")
    ax.fill_between(k_arr, i_lo_maxp, i_hi_maxp, color=TEAL, alpha=0.15)
    ax.plot(k_arr, i_at_maxp, marker="o", markersize=4, linewidth=1.5, color=TEAL, label="$I_k(p_{\\max})$")
    if tau_at_maxp is not None:
        ax.plot(k_arr, tau_at_maxp, linestyle="--", linewidth=1.5, color=ROSE, label="Null threshold $\\tau$")
    if k_final is not None:
        ax.axvline(k_final, linestyle=":", linewidth=1, color=GRAY_DARK, label=f"k* = {k_final}")
    ax.set_xlabel("Dimension $k$")
    ax.set_ylabel("Incremental coherence at $p = {:.2f}$".format(p_list[-1]))
    ax.set_title("Coherence rank selection -- THINGS behavioral")
    ax.legend(fontsize=7)
    despine(ax)
    fig.savefig(
        OUTPUT_DIR / "coherence_cv_curve.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)

    print(f"\nPlots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
