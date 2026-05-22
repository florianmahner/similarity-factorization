"""Combined 3×3 diagnostic figure for A3_anisotropic n-scan (2026-05-11).

Loads n300.pkl, n600.pkl, n1000.pkl from the A3 n-scan run and produces a
single multi-panel figure for cross-n comparison.

Layout (3 rows × 3 columns):
  Row 1: eigenvalue spectra (one per n)
  Row 2: Plain k-fold V-MSE curves (one per n, three k_inner colored)
  Row 3: Pre-mask + k-fold V-MSE curves (one per n)

Output: A3_nscan_combined.png alongside the per-n pkls.
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))   # RECIPE_K/exp/
ROOT = os.path.dirname(HERE)                          # RECIPE_K/
DIR = os.path.join(ROOT, "output", "pstar_gap", "cvfold_A3_nscan")
NS = [300, 600, 1000]
K_COLORS = {2: "C3", 5: "C0", 10: "C2"}

# Load
datas = {}
for n in NS:
    p = os.path.join(DIR, f"n{n}.pkl")
    if not os.path.exists(p):
        print(f"skip n={n}: {p} missing"); continue
    with open(p, "rb") as f:
        datas[n] = pickle.load(f)
NS_have = sorted(datas.keys())

fig, axes = plt.subplots(3, len(NS_have), figsize=(6 * len(NS_have), 13),
                         squeeze=False)
for col, n in enumerate(NS_have):
    d = datas[n]
    rows = d["rows"]; spec = d["spec"]; recK = d["recK"]
    evals_ref = np.asarray(spec["evals_ref"])
    k_cut = int(spec["k_cut"]); p_star = float(recK["p_star"])
    p_star_raw = float(recK["p_star_raw"]); p_floor = float(recK["p_floor"])
    floor_b = bool(recK["floor_binding"]); cap_b = bool(recK["cap_binding"])
    ranks = rows[0]["ranks"]

    # Row 1: eigenvalues (top 40)
    ax = axes[0, col]
    top = min(40, len(evals_ref))
    ax.plot(np.arange(1, top + 1), evals_ref[:top], "o-", color="C0",
            markersize=4, alpha=0.85)
    ax.axvline(5, color="C2", linestyle=":", linewidth=1.4, label="true r=5")
    ax.axvline(k_cut, color="k", linestyle="--", linewidth=1.4,
               label=f"k_cut={k_cut}")
    ax.set_xlabel("rank r"); ax.set_ylabel(r"$\lambda_r^{\rm ref}$")
    ax.set_title(f"n={n}: top-{top} spectrum\n"
                 f"p_raw={p_star_raw:.3f}, p_floor={p_floor:.2f} (bind={floor_b}), "
                 f"p★={p_star:.2f}")
    ax.set_yscale("log"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Row 2: Plain k-fold curves
    ax = axes[1, col]
    for r in rows:
        k = r["k_inner"]; c = K_COLORS[k]
        cv = np.asarray(r["plain_curve"]); sem = np.asarray(r["plain_sem"])
        ax.plot(ranks, cv, color=c, marker=".", markersize=3,
                label=f"k={k} (train={(k-1)/k:.2f}): "
                      f"argmin={r['plain_argmin']} (1SE={r['plain_1se']})")
        ax.fill_between(ranks, cv - sem, cv + sem, color=c, alpha=0.10)
        am_idx = ranks.index(r["plain_argmin"]) if r["plain_argmin"] in ranks else None
        if am_idx is not None:
            ax.scatter([r["plain_argmin"]], [cv[am_idx]], s=80, marker="*",
                       color=c, edgecolor="k", zorder=5)
    ax.axvline(5, color="C2", linestyle=":", alpha=0.5)
    ax.axvline(k_cut, color="k", linestyle="--", alpha=0.4)
    ax.set_xlabel("rank r"); ax.set_ylabel("V CV-MSE")
    ax.set_title(f"Plain k-fold (no pre-mask), n={n}")
    ax.set_yscale("log"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Row 3: Pre-mask k-fold curves
    ax = axes[2, col]
    for r in rows:
        k = r["k_inner"]; c = K_COLORS[k]
        cv = np.asarray(r["pre_curve"]); sem = np.asarray(r["pre_sem"])
        ax.plot(ranks, cv, color=c, marker=".", markersize=3,
                label=f"k={k}: p_cv={r['p_cv']:.2f}, "
                      f"argmin={r['pre_argmin']} (1SE={r['pre_1se']})")
        ax.fill_between(ranks, cv - sem, cv + sem, color=c, alpha=0.10)
        am_idx = ranks.index(r["pre_argmin"]) if r["pre_argmin"] in ranks else None
        if am_idx is not None:
            ax.scatter([r["pre_argmin"]], [cv[am_idx]], s=80, marker="*",
                       color=c, edgecolor="k", zorder=5)
    ax.axvline(5, color="C2", linestyle=":", alpha=0.5)
    ax.axvline(k_cut, color="k", linestyle="--", alpha=0.4)
    ax.set_xlabel("rank r"); ax.set_ylabel("V CV-MSE")
    ax.set_title(f"Pre-mask + k-fold (Recipe K), n={n}, p★={p_star:.2f}")
    ax.set_yscale("log"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

fig.suptitle("A3_anisotropic n-scan: plain k-fold drifts with k_cv as n grows; "
             "Recipe K's pre-mask k-fold stays at argmin=5 (= true r) for every k_cv",
             fontsize=13)
plt.tight_layout()
out = os.path.join(DIR, "A3_nscan_combined.png")
plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
print(f"saved: {out}")
