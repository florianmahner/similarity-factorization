"""Comprehensive diagnostic figure for A3_anisotropic at n=300 (2026-05-11).

Loads the saved n=300 results from `recipe_K_cvfold_A3_nscan_2026-05-11`
and produces a single multi-panel figure showing:

  (1) Top eigenvalue spectrum (log scale, with k_cut, true_r markers)
  (2) Per-rank coherence profile κ̂_r (with F-stat k_cut marker)
  (3) Empirical deficit δ_emp(p) (with δ target, p★_raw, p_floor, p★ markers)
  (4) Plain k-fold V CV-MSE curves (k_inner ∈ {2, 5, 10})
  (5) Pre-mask + k-fold V CV-MSE curves (k_inner ∈ {2, 5, 10})
  (6) Summary metadata table

Output: figures/A3_n300_diagnostic.png in the same results dir.
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))   # RECIPE_K/exp/
ROOT = os.path.dirname(HERE)                          # RECIPE_K/
A3_DIR = os.path.join(ROOT, "output", "pstar_gap", "cvfold_A3_nscan")
PKL = os.path.join(A3_DIR, "n300.pkl")
OUT = os.path.join(A3_DIR, "A3_n300_diagnostic.png")

with open(PKL, "rb") as f:
    data = pickle.load(f)

rows = data["rows"]
spec = data["spec"]
recK = data["recK"]

n = rows[0]["n"]
true_r = 5
k_cut = int(spec["k_cut"])
p_star = float(recK["p_star"])
p_star_raw = float(recK["p_star_raw"])
p_floor = float(recK["p_floor"])
delta_target = float(recK["delta"])
floor_binding = bool(recK["floor_binding"])
cap_binding = bool(recK["cap_binding"])
ranks = rows[0]["ranks"]
evals_ref = np.asarray(spec["evals_ref"])
kappa_hat = np.asarray(spec["kappa_hat"])
k_list = list(spec["k_list"])
delta_emp = list(recK["delta_emp"])
p_grid = list(recK["p_grid"])

K_COLORS = {2: "C3", 5: "C0", 10: "C2"}
K_VALUES = [2, 5, 10]

fig = plt.figure(figsize=(17, 11))
gs = GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.30,
              height_ratios=[1, 1, 1])

# ---------- Panel 1: eigenvalue spectrum ----------
ax1 = fig.add_subplot(gs[0, 0])
top = min(40, len(evals_ref))
xs = np.arange(1, top + 1)
ax1.plot(xs, evals_ref[:top], "o-", color="C0", markersize=4, alpha=0.85)
ax1.axvline(true_r, color="C2", linestyle=":", linewidth=1.4, label=f"true r={true_r}")
ax1.axvline(k_cut, color="k", linestyle="--", linewidth=1.4, label=f"k_cut={k_cut}")
ax1.set_xlabel("rank r"); ax1.set_ylabel(r"$\lambda_r^{\rm ref}$")
ax1.set_title(f"Top-{top} eigenvalue spectrum")
ax1.set_yscale("log"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

# ---------- Panel 2: kappa profile ----------
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(k_list, kappa_hat, "o-", color="C0", markersize=4, alpha=0.85)
ax2.axvline(k_cut, color="k", linestyle="--", linewidth=1.4, label=f"k_cut={k_cut}")
ax2.axvline(true_r, color="C2", linestyle=":", linewidth=1.4, label=f"true r={true_r}")
ax2.axhline(1.0, color="gray", linestyle=":", alpha=0.4)
ax2.set_xlabel("rank r"); ax2.set_ylabel(r"$\hat\kappa_r$")
ax2.set_title("Per-rank leakage rate (F-stat at k_cut)")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

# ---------- Panel 3: delta_emp(p) ----------
ax3 = fig.add_subplot(gs[0, 2])
p_arr = np.asarray(p_grid); d_arr = np.asarray(delta_emp)
ax3.plot(p_arr, d_arr, "o-", color="C0", markersize=4, label=r"$\delta_{\rm emp}(p)$ (PAV)")
ax3.axhline(delta_target, color="gray", linestyle=":", alpha=0.7,
            label=fr"target $\delta={delta_target}$")
ax3.axvline(p_star_raw, color="C3", linestyle="-", linewidth=1.6,
            label=fr"$p^\star_{{\rm raw}}={p_star_raw:.3f}$")
ax3.axvline(p_floor, color="C1", linestyle="-.", linewidth=1.6,
            label=fr"$p_{{\rm floor}}={p_floor:.3f}$")
ax3.axvline(p_star, color="C2", linestyle="--", linewidth=1.6,
            label=fr"$p^\star={p_star:.3f}$")
ax3.set_xlabel("p"); ax3.set_ylabel(r"$\delta_{\rm emp}$")
ax3.set_title(f"Empirical deficit; floor binding={floor_binding}")
ax3.set_ylim(-0.05, max(1.0, d_arr.max() * 1.1))
ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

# ---------- Panel 4: Plain k-fold V curves ----------
ax4 = fig.add_subplot(gs[1, :])
for r in rows:
    k = r["k_inner"]; c = K_COLORS[k]
    cv = np.asarray(r["plain_curve"])
    sem = np.asarray(r["plain_sem"])
    ax4.plot(ranks, cv, color=c, marker=".", markersize=4,
             label=f"k={k}: train=(k-1)/k={(k-1)/k:.2f}, "
                   f"argmin={r['plain_argmin']} (1SE={r['plain_1se']})")
    # Shade ±1 SE
    ax4.fill_between(ranks, cv - sem, cv + sem, color=c, alpha=0.12)
    # Mark argmin
    am_idx = ranks.index(r["plain_argmin"]) if r["plain_argmin"] in ranks else None
    if am_idx is not None:
        ax4.scatter([r["plain_argmin"]], [cv[am_idx]], s=130, marker="*",
                    color=c, edgecolor="k", zorder=5)
ax4.axvline(true_r, color="C2", linestyle=":", alpha=0.6, label=f"true r={true_r}")
ax4.axvline(k_cut, color="k", linestyle="--", alpha=0.4, label=f"k_cut={k_cut}")
ax4.set_xlabel("rank r"); ax4.set_ylabel("V CV-MSE")
ax4.set_title(f"Plain k-fold (NO pre-mask) — training fraction = (k-1)/k varies with k. "
              f"True p★={p_star:.2f}; under-train for k=2, lucky-ish for k=5, over-train for k=10.")
ax4.set_yscale("log")
ax4.legend(fontsize=8, loc="upper right"); ax4.grid(alpha=0.3)

# ---------- Panel 5: Pre-mask k-fold V curves ----------
ax5 = fig.add_subplot(gs[2, :])
for r in rows:
    k = r["k_inner"]; c = K_COLORS[k]
    cv = np.asarray(r["pre_curve"])
    sem = np.asarray(r["pre_sem"])
    ax5.plot(ranks, cv, color=c, marker=".", markersize=4,
             label=f"k={k}: p_cv={r['p_cv']:.2f}, train≈p★={p_star:.2f}, "
                   f"argmin={r['pre_argmin']} (1SE={r['pre_1se']})")
    ax5.fill_between(ranks, cv - sem, cv + sem, color=c, alpha=0.12)
    am_idx = ranks.index(r["pre_argmin"]) if r["pre_argmin"] in ranks else None
    if am_idx is not None:
        ax5.scatter([r["pre_argmin"]], [cv[am_idx]], s=130, marker="*",
                    color=c, edgecolor="k", zorder=5)
ax5.axvline(true_r, color="C2", linestyle=":", alpha=0.6, label=f"true r={true_r}")
ax5.axvline(k_cut, color="k", linestyle="--", alpha=0.4, label=f"k_cut={k_cut}")
ax5.set_xlabel("rank r"); ax5.set_ylabel("V CV-MSE")
ax5.set_title(f"Pre-mask + k-fold (Recipe K) — training fraction held at p★={p_star:.2f} "
              f"for every k_cv. Same model, same argmin.")
ax5.set_yscale("log")
ax5.legend(fontsize=8, loc="upper right"); ax5.grid(alpha=0.3)

# ---------- suptitle with summary ----------
plain_amin = [r["plain_argmin"] for r in rows]
pre_amin   = [r["pre_argmin"]   for r in rows]
sup = (f"A3_anisotropic (n={n}, true r={true_r}): "
       f"k_cut={k_cut}, p★_raw={p_star_raw:.3f}, p_floor={p_floor:.2f} (binding={floor_binding}), "
       f"p★={p_star:.3f}, cap_binding={cap_binding}, k_cv_min={recK.get('k_cv_min_unclipped')}\n"
       f"Plain argmin across k∈{{2,5,10}}: {plain_amin}   |   "
       f"Pre-mask argmin across k∈{{2,5,10}}: {pre_amin}   "
       f"(Recipe K is k_cv-invariant)")
fig.suptitle(sup, fontsize=11)

plt.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"saved: {OUT}")
