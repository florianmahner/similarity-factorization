"""Publication plots for THINGS coherence rank detection across triplet percentages.

Four panels:
A) k* vs triplet percentage for all three methods + matrix completeness
B) Kappa profiles across percentages (why kappa works)
C) Coherence heatmap at 100% with k* marked
D) Select coherence curves at 100% showing signal vs noise separation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path

from src.colors import ROSE, TEAL, CYAN, SAND, PURPLE, GRAY, setup_style
from src.utils.figure_theme import despine

OUTPUT_DIR = Path("sandbox/coherence/things/outputs/latest")
DIAG_DIR = OUTPUT_DIR / "diagnostics"
df = pd.read_csv(OUTPUT_DIR / "rank_detection.csv")

df_pct = df[df["alpha_tau"] == 0.95].sort_values("pct")

PCTS = [5, 10, 20, 50, 100]
PCT_COLORS = {5: ROSE, 10: PURPLE, 20: SAND, 50: CYAN, 100: TEAL}


def _load_diag(pct, tau=0.95):
    tag = f"pct{pct}_tau{tau:.2f}"
    return np.load(DIAG_DIR / tag / "coherence.npz")


def main():
    setup_style()

    fig = plt.figure(figsize=(7.2, 5.8))
    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[1, 1],
        width_ratios=[1, 1],
        hspace=0.5, wspace=0.45,
        left=0.09, right=0.96, top=0.94, bottom=0.08,
    )

    pcts = df_pct["pct"].values
    k_act = df_pct["k_activation"].values
    k_kap = df_pct["k_kappa"].values
    k_clu = df_pct["k_cluster"].values
    obs_frac = df_pct["obs_frac"].values
    missing_pct = (1.0 - obs_frac) * 100

    # =========================================================================
    # Panel A: k* vs percentage -- all three methods + missing %
    # =========================================================================
    ax_comp = fig.add_subplot(gs[0, 0])

    method_style = {
        "Null threshold": (k_act, ROSE, "o", "-"),
        "Kappa changepoint": (k_kap, TEAL, "s", "-"),
        "Curve clustering": (k_clu, SAND, "D", "-"),
    }

    for name, (vals, color, marker, ls) in method_style.items():
        ax_comp.plot(pcts, vals, ls, color=color, linewidth=1.2, alpha=0.8, zorder=2)
        ax_comp.scatter(pcts, vals, s=35, color=color, marker=marker,
                        edgecolors="white", linewidths=0.4, zorder=4)

    # Missing % as secondary axis -- dashed line
    ax_miss = ax_comp.twinx()
    ax_miss.plot(pcts, missing_pct, "--", color=GRAY, linewidth=0.9, alpha=0.6, zorder=0)
    ax_miss.scatter(pcts, missing_pct, s=15, color=GRAY, alpha=0.6, zorder=0)
    ax_miss.set_ylabel("Missing (%)", color=GRAY, fontsize=7)
    ax_miss.tick_params(axis="y", labelcolor=GRAY, labelsize=7)
    ax_miss.set_ylim(-5, 100)
    ax_miss.spines["top"].set_visible(False)

    ax_comp.set_xlabel("Triplets used (%)")
    ax_comp.set_ylabel("Detected rank $\\hat{k}$")
    ax_comp.set_xlim(-5, 110)
    ax_comp.set_ylim(0, 130)
    despine(ax_comp)

    method_handles = [
        Line2D([], [], marker=m, linestyle=ls, color=c, markersize=5,
               markeredgecolor="white", markeredgewidth=0.3, linewidth=1, label=n)
        for n, (_, c, m, ls) in method_style.items()
    ]
    method_handles.append(
        Line2D([], [], linestyle="--", color=GRAY, linewidth=0.9, label="Missing %")
    )
    ax_comp.legend(
        handles=method_handles, fontsize=6.5, loc="center right",
        handletextpad=0.3, labelspacing=0.25,
    )

    ax_comp.text(-0.14, 1.05, "A", transform=ax_comp.transAxes,
                 fontsize=13, fontweight="bold", va="bottom")

    # =========================================================================
    # Panel B: Kappa profiles across percentages -- legend box
    # =========================================================================
    ax_kappa = fig.add_subplot(gs[0, 1])

    for pct in PCTS:
        d = _load_diag(pct)
        kappa = d["kappa"]
        k_list = d["k_list"]
        color = PCT_COLORS[pct]

        row = df_pct[df_pct["pct"] == pct].iloc[0]
        k_cut = int(row["k_kappa"])

        ax_kappa.plot(k_list, kappa, color=color, linewidth=1.2, alpha=0.85,
                      label=f"{pct}%")

        if 0 < k_cut <= k_list[-1]:
            idx = np.searchsorted(k_list, k_cut)
            if idx < len(kappa):
                ax_kappa.plot(k_cut, kappa[idx], marker="v", color=color,
                              markersize=5, markeredgecolor="white",
                              markeredgewidth=0.4, zorder=5)

    ax_kappa.set_xlabel("Dimension $k$")
    ax_kappa.set_ylabel("Leakage $\\hat{\\kappa}_k$")
    ax_kappa.set_xlim(0, 80)
    despine(ax_kappa)

    ax_kappa.legend(
        title="Triplets", fontsize=6.5, title_fontsize=7,
        loc="center right",
        handlelength=1.2, handletextpad=0.3, labelspacing=0.2,
    )

    ax_kappa.text(-0.14, 1.05, "B", transform=ax_kappa.transAxes,
                  fontsize=13, fontweight="bold", va="bottom")

    # =========================================================================
    # Panel C: Coherence heatmap at 100%
    # =========================================================================
    ax_heat = fig.add_subplot(gs[1, 0])

    d100 = _load_diag(100)
    x_med = d100["x_median"]
    k_list = d100["k_list"]
    p_list = d100["p_list"]

    k_mask = k_list <= 80
    x_show = x_med[k_mask]

    im = ax_heat.imshow(
        x_show, aspect="auto", origin="lower", interpolation="bilinear",
        extent=[p_list[0], p_list[-1], k_list[k_mask][0] - 0.5, k_list[k_mask][-1] + 0.5],
        cmap="magma", vmin=0, vmax=1,
    )

    k_cut_100 = int(df_pct[df_pct["pct"] == 100].iloc[0]["k_kappa"])
    ax_heat.axhline(k_cut_100, color="white", linewidth=1, linestyle="--", alpha=0.8)
    ax_heat.text(p_list[1], k_cut_100 + 2, f"$\\hat{{k}}$ = {k_cut_100}",
                 color="white", fontsize=7.5, va="bottom")

    ax_heat.set_xlabel("Sampling fraction $p$")
    ax_heat.set_ylabel("Dimension $k$")

    cb = plt.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02, aspect=25)
    cb.set_label("$I^{\\mathrm{proj}}_k(p)$", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax_heat.text(-0.14, 1.05, "C", transform=ax_heat.transAxes,
                 fontsize=13, fontweight="bold", va="bottom")

    # =========================================================================
    # Panel D: Coherence curves at 100% -- signal vs noise, spaced labels
    # =========================================================================
    ax_curves = fig.add_subplot(gs[1, 1])

    x_med = d100["x_median"]
    tau = d100["tau_kp"]
    k_list = d100["k_list"]
    p_list = d100["p_list"]

    # Each curve gets a unique color from a continuous colormap
    show_k = [3, 10, 23, 40, 80]
    cmap_curves = plt.cm.viridis_r
    k_norm = np.linspace(0.1, 0.9, len(show_k))

    for i, k in enumerate(show_k):
        idx = np.searchsorted(k_list, k)
        if idx >= len(k_list):
            continue
        color = cmap_curves(k_norm[i])
        is_signal = k <= k_cut_100
        lw = 1.4 if is_signal else 1.0
        ls = "-" if is_signal else "--"
        ax_curves.plot(p_list, x_med[idx], color=color, linewidth=lw,
                       linestyle=ls, label=f"$k$={k}")

    if tau is not None and tau.shape[0] > 0:
        tau_ref = np.median(tau[:40], axis=0)
        ax_curves.fill_between(p_list, 0, tau_ref, color=ROSE, alpha=0.07, zorder=0)
        ax_curves.plot(p_list, tau_ref, "-", color=ROSE, linewidth=0.7, alpha=0.4,
                       label="Null $\\tau$")

    ax_curves.set_xlabel("Sampling fraction $p$")
    ax_curves.set_ylabel("$I^{\\mathrm{proj}}_k(p)$")
    ax_curves.set_xlim(p_list[0] - 0.02, p_list[-1] + 0.02)
    ax_curves.set_ylim(-0.02, 1.05)
    ax_curves.legend(fontsize=6, loc="center right", labelspacing=0.2, handlelength=1.5)
    despine(ax_curves)

    ax_curves.text(-0.14, 1.05, "D", transform=ax_curves.transAxes,
                   fontsize=13, fontweight="bold", va="bottom")

    fig.savefig(OUTPUT_DIR / "things_coherence.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / "things_coherence.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR / 'things_coherence.png'}")


if __name__ == "__main__":
    main()
