"""Presentation figure: coherence-informed rank estimation for peterson-animals.

Row 1: I^proj curves | Activation curve p_act(k) | Cumulative activations vs CV k*
Row 2: CV error curves at four sampling fractions
"""
from __future__ import annotations

from pathlib import Path

import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.colors import ROSE, TEAL, CYAN, SAND, GRAY, GRAY_DARK, GRAY_LIGHT
from src.utils.figure_theme import despine

DATA_DIR = Path(__file__).parent / "outputs" / "260211_124520_peterson_v8"
OUTPUT_DIR = DATA_DIR

P_COLORS = {0.33: ROSE, 0.5: CYAN, 0.65: SAND, 0.8: GRAY_DARK}


def main():
    coh = np.load(DATA_DIR / "coherence_results.npz")
    activation_p = coh["activation_p"]
    k_list = coh["k_list"]
    x_mean = coh["x_mean"]
    x_ci_lo = coh["x_ci_lo"]
    x_ci_hi = coh["x_ci_hi"]
    p_list = coh["p"]

    with open(DATA_DIR / "rank_estimation.json") as f:
        summary = json.load(f)
    cv_sweep = summary["cv_results_sweep"]
    n_act = summary.get("n_activating", 9)

    fig = plt.figure(figsize=(14.0, 7.5))
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.4, height_ratios=[1, 0.85])

    # Top row: 3 panels
    top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.35)
    # Bottom row: 4 panels
    bot = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.35)

    # ===== Panel A: I^proj curves =====
    ax_a = fig.add_subplot(top[0])
    n_show = 15

    signal_colors = {}
    label_y_raw = []
    label_info = []
    for i in range(min(n_act, n_show)):
        shade = 0.3 + 0.6 * (i / max(n_act - 1, 1))
        c = plt.cm.Purples(shade)
        signal_colors[i] = c
        ax_a.plot(p_list, x_mean[i], color=c, linewidth=1.3)
        ax_a.fill_between(p_list, x_ci_lo[i], x_ci_hi[i], color=c, alpha=0.05)
        label_y_raw.append(x_mean[i, -1])
        label_info.append((f"k={i+1}", c))

    # De-overlap labels: enforce minimum vertical spacing
    min_gap = 0.055
    label_y = np.array(label_y_raw)
    order = np.argsort(label_y)
    sorted_y = label_y[order].copy()
    for j in range(1, len(sorted_y)):
        if sorted_y[j] - sorted_y[j - 1] < min_gap:
            sorted_y[j] = sorted_y[j - 1] + min_gap
    adjusted_y = np.empty_like(label_y)
    adjusted_y[order] = sorted_y

    for i in range(min(n_act, n_show)):
        text, c = label_info[i]
        ax_a.annotate(text, (1.01, adjusted_y[i]), fontsize=6.5, color=c,
                      va="center", annotation_clip=False)

    for i in range(n_act, n_show):
        ax_a.plot(p_list, x_mean[i], color=GRAY, linewidth=0.5, alpha=0.5)
        ax_a.fill_between(p_list, x_ci_lo[i], x_ci_hi[i], color=GRAY, alpha=0.03)

    ax_a.annotate("noise", (1.01, 0.0), fontsize=6.5, color=GRAY,
                  va="center", annotation_clip=False)

    ax_a.axhline(0, color=GRAY, linewidth=0.5)
    ax_a.set_xlabel("Sampling fraction p")
    ax_a.set_ylabel("Subspace coherence\n(baseline-corrected)")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(-0.15, 1.05)
    ax_a.set_title("A   Coherence curves", fontsize=11, loc="left", fontweight="bold")
    despine(ax_a)

    # ===== Panel B: Activation curve p_act(k) =====
    ax_b = fig.add_subplot(top[1])
    valid = np.isfinite(activation_p)
    k_valid = k_list[valid]
    p_valid = activation_p[valid]
    k_invalid = k_list[~valid]

    ax_b.plot(k_valid, p_valid, color=TEAL, linewidth=2, marker="o", markersize=6,
              zorder=3)
    n_show_invalid = min(len(k_invalid), 15)
    ax_b.scatter(k_invalid[:n_show_invalid],
                 np.ones(n_show_invalid) * 1.03,
                 color=GRAY_LIGHT, marker="x", s=30, zorder=2)

    ax_b.annotate(f"{n_act} signal\ndimensions", (n_act / 2, 0.48),
                  fontsize=9, color=TEAL, ha="center", va="center",
                  fontstyle="italic", alpha=0.7)
    ax_b.annotate("not detectable", (15, 1.07), fontsize=8, color=GRAY,
                  ha="center")

    ax_b.set_xlabel("Dimension k")
    ax_b.set_ylabel(r"Detection threshold $p_{\rm act}(k)$")
    ax_b.set_xlim(0, 22)
    ax_b.set_ylim(0, 1.15)
    ax_b.set_title("B   Per-dimension threshold", fontsize=11, loc="left", fontweight="bold")
    despine(ax_b)

    # ===== Panel C: N_act(p) vs CV k* =====
    ax_c = fig.add_subplot(top[2])
    valid_act = activation_p[np.isfinite(activation_p)]
    p_dense = np.linspace(0, 1, 200)
    n_act_curve = np.array([np.sum(valid_act <= p) for p in p_dense])

    ax_c.fill_between(p_dense, 0, n_act_curve, color=TEAL, alpha=0.1, step="post")
    ax_c.step(p_dense, n_act_curve, color=TEAL, linewidth=1.8, where="post",
              label="# activated dims")

    for p_val, color in P_COLORS.items():
        p_str = str(p_val)
        if p_str not in cv_sweep:
            continue
        best_k = cv_sweep[p_str]["best_rank"]
        ax_c.scatter([p_val], [best_k], color=color, s=90, zorder=5,
                     marker="s", edgecolors="white", linewidths=0.8)
        ax_c.annotate(f"k*={best_k}", (p_val + 0.02, best_k + 0.6),
                      fontsize=8, color=color, fontweight="bold")

    cv_marker = Line2D([0], [0], marker="s", color="none", markerfacecolor=GRAY_DARK,
                       markeredgecolor="white", markersize=8, label="CV optimum")
    ax_c.legend(handles=[ax_c.get_lines()[0], cv_marker], fontsize=8, frameon=False,
                loc="upper left")

    ax_c.set_xlabel("Sampling fraction p")
    ax_c.set_ylabel("Number of dimensions k")
    ax_c.set_xlim(0, 0.95)
    ax_c.set_ylim(0, 28)
    ax_c.set_title("C   Prediction vs CV optimum", fontsize=11, loc="left", fontweight="bold")
    despine(ax_c)

    # ===== Panel D: CV curves at each p =====
    p_vals_sorted = sorted([p for p in P_COLORS if str(p) in cv_sweep])
    for col_idx, p_val in enumerate(p_vals_sorted):
        ax = fig.add_subplot(bot[col_idx])
        color = P_COLORS[p_val]
        cv_res = cv_sweep[str(p_val)]
        ranks = np.array(cv_res["ranks"])
        mean_mse = np.array(cv_res["mean_mse"])
        std_mse = np.array(cv_res["std_mse"])
        se = std_mse / np.sqrt(cv_res["n_repeats"])

        ax.plot(ranks, mean_mse, color=color, linewidth=1.5, marker="o", markersize=4)
        ax.fill_between(ranks, mean_mse - se, mean_mse + se, color=color, alpha=0.12)

        best_idx = np.argmin(mean_mse)
        best_k = ranks[best_idx]
        ax.axvline(best_k, color=color, linestyle="-", linewidth=1.5, alpha=0.25)
        ax.scatter([best_k], [mean_mse[best_idx]], color=color, s=60, zorder=5,
                   edgecolors="white", linewidths=0.8)

        ax.set_xlabel("Rank k")
        if col_idx == 0:
            ax.set_ylabel("CV MSE")

        prefix = "D   " if col_idx == 0 else ""
        ax.set_title(f"{prefix}p={p_val:.2f}  →  k*={best_k}", fontsize=10,
                     color=color, fontweight="bold", loc="left" if col_idx == 0 else "center")
        despine(ax)

    dataset_name = summary.get("dataset", "peterson-animals")
    fig.suptitle(f"Coherence-informed rank estimation: {dataset_name}",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(OUTPUT_DIR / "presentation_figure.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR / 'presentation_figure.png'}")


if __name__ == "__main__":
    main()
