"""
6-panel evidence figures for SymmNMF on CLIP_RBF + THINGS.

Mirrors figure_pstar_real.py but loads the *_symmnmf.pkl files. Recipe K
diagnostics (spectrum, kappa, delta_emp) are dataset-properties so unchanged;
only panels D/E/F (CV) differ between fitters.
"""
import os, sys, pickle, time
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))   # RECIPE_K/exp/
ROOT = os.path.dirname(HERE)                          # RECIPE_K/
SRC = os.path.join(ROOT, "src")
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, SRC)

from _common import spectral_pass, recipe_K

# Source the raw pkls from the SymmNMF pstar-gap run; write figures alongside.
PKL_DIR = os.path.join(ROOT, "output", "pstar_gap", "pstar_gap_C_symmnmf", "raw")
FIG_DIR = os.path.join(ROOT, "output", "pstar_gap", "pstar_gap_C_symmnmf", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_clip():
    Df = np.load(os.path.join(DATA_DIR, "model_features", "clip_vit_b_32.npy"))
    dist = cdist(Df, Df, "euclidean")
    bw = float(np.median(dist))
    return np.exp(-(dist / bw) ** 2)


def load_things():
    A = np.load(os.path.join(DATA_DIR, "things_behavior_similarity.npy"))
    A = (A + A.T) / 2
    diag = np.eye(A.shape[0], dtype=bool)
    med = float(np.nanmedian(A[~diag]))
    return np.where(np.isnan(A), med, A)


loaders = {"CLIP_RBF": load_clip, "THINGS": load_things}


def render_dataset(label):
    pkl_path = os.path.join(PKL_DIR, f"{label}_symmnmf.pkl")
    if not os.path.exists(pkl_path):
        print(f"  {label}_symmnmf: pkl not found"); return None
    with open(pkl_path, "rb") as f:
        cell = pickle.load(f)
    n = cell["n"]
    print(f"\n=== {label} SymmNMF (n={n}) ===")

    t0 = time.time()
    S = loaders[label]()
    spec = spectral_pass(S, B=20, smooth_window=10)
    spec["k_cut"] = spec["k_smooth"]
    recK = recipe_K(spec, delta=0.10, k_cv=5, p_floor=0.5,
                       p_floor_mode="adaptive", M_min=2000)
    print(f"  spectral_pass + recipe_K in {time.time()-t0:.1f}s")

    k_cut = int(spec["k_smooth"])
    evals_ref = np.asarray(spec["evals_ref"])
    kappa_hat = np.asarray(spec["kappa_hat"])
    p_star_raw = float(recK["p_star_raw"])
    p_floor_used = float(recK["p_floor"])
    p_star = float(recK["p_star"])
    p_cv = float(recK["p_cv"])
    delta_emp_iso = np.asarray(recK["delta_emp"])
    delta_emp_raw = np.asarray(recK["delta_emp_raw"])
    p_grid_sorted = np.asarray(recK["p_grid"])
    sigma_hat = recK.get("sigma_hat", None)

    p_sweep_done = sorted(cell["sweep"].keys())
    ranks = cell["ranks"]
    print(f"  argmin range: {min(cell['sweep'][p]['argmin'] for p in p_sweep_done)} – "
          f"{max(cell['sweep'][p]['argmin'] for p in p_sweep_done)}")

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28)

    # (A) Spectrum
    ax = fig.add_subplot(gs[0, 0])
    top = min(100, len(evals_ref))
    ax.semilogy(np.arange(1, top + 1), evals_ref[:top], "o-", color="C0", ms=4)
    ax.axvline(k_cut, color="red", ls="--", lw=2, label=f"$k_{{cut}}={k_cut}$")
    if sigma_hat is not None:
        edge = 2 * sigma_hat * np.sqrt(n)
        ax.axhline(edge, color="gray", ls=":", lw=1.5,
                   label=fr"Wigner edge $2\hat\sigma\sqrt{{n}}={edge:.2f}$")
    ax.set_xlabel("eigenvalue index $r$")
    ax.set_ylabel(r"$\lambda_r$ (log)")
    ax.set_title(f"(A) Spectrum of {label} (n={n})\n$\\hat\\sigma={sigma_hat:.3f}$")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which="both")

    # (B) kappa_hat
    ax = fig.add_subplot(gs[0, 1])
    k_list = np.arange(1, len(kappa_hat) + 1)
    ax.semilogy(k_list, np.maximum(kappa_hat, 1e-6), "o-", color="C2", ms=4)
    ax.axvline(k_cut, color="red", ls="--", lw=2, label=f"$\\hat k^\\star={k_cut}$")
    ax.set_xlabel("rank $r$"); ax.set_ylabel(r"$\hat\kappa_r$ (log)")
    ax.set_title("(B) Per-rank coherence $\\hat\\kappa_r$\n"
                 f"signal regime $r\\leq {k_cut}$ vs bulk regime")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which="both")

    # (C) delta_emp
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(p_grid_sorted, delta_emp_raw, "o-", color="C3", ms=4, alpha=0.5, label=r"$\hat\delta_{\rm raw}$")
    ax.plot(p_grid_sorted, delta_emp_iso, "s-", color="C3", ms=5, label=r"$\hat\delta_{\rm iso}$ (PAV)")
    ax.axhline(0.10, color="k", ls="--", lw=2, label=r"$\delta=0.10$")
    ax.axvline(p_star_raw, color="C1", ls="-", lw=2, label=f"$p^*_{{raw}}={p_star_raw:.3f}$")
    if abs(p_star - p_star_raw) > 1e-4:
        ax.axvline(p_star, color="C4", ls="-", lw=2, label=f"$p^*={p_star:.3f}$")
    ax.axvline(p_floor_used, color="C5", ls=":", lw=1.5, label=f"$p_{{floor}}^{{adapt}}={p_floor_used:.3f}$")
    ax.set_xlabel("$p$"); ax.set_ylabel(r"$\hat\delta_{\rm emp}(p)$")
    ax.set_title("(C) Empirical deficit (Recipe K diagnostic)")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, max(1.0, float(np.nanmax(delta_emp_raw)) * 1.1))

    # (D) CV-MSE overlay (SymmNMF)
    ax = fig.add_subplot(gs[1, 0])
    cmap = plt.cm.viridis
    p_norm = (np.array(p_sweep_done) - min(p_sweep_done)) / (max(p_sweep_done) - min(p_sweep_done) + 1e-12)
    for p, color_val in zip(p_sweep_done, p_norm):
        mean = cell["sweep"][p]["mean"]
        am = cell["sweep"][p]["argmin"]
        ax.plot(ranks, mean, "-", color=cmap(color_val), lw=1.2, alpha=0.7)
        if am in ranks:
            i = ranks.index(am)
            ax.scatter([am], [mean[i]], s=40, color=cmap(color_val), zorder=10,
                       edgecolors="black", linewidths=0.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(p_sweep_done), max(p_sweep_done)))
    cbar = plt.colorbar(sm, ax=ax); cbar.set_label("sampling $p$")
    ax.set_xlabel("rank $r$"); ax.set_ylabel("CV-MSE (off-diagonal)")
    ax.set_title(f"(D) SymmNMF CV-MSE at all swept $p$ — argmin marked\n"
                 f"Recipe K $p_{{cv}}={p_cv:.3f}$")
    ax.grid(True, alpha=0.3); ax.set_xscale("log")

    # (E) argmin vs p
    ax = fig.add_subplot(gs[1, 1])
    am_list = [cell["sweep"][p]["argmin"] for p in p_sweep_done]
    a1se_list = [cell["sweep"][p]["argmin_1se"] for p in p_sweep_done]
    ax.plot(p_sweep_done, am_list, "o-", color="C0", ms=8, label="argmin")
    ax.plot(p_sweep_done, a1se_list, "s--", color="C1", ms=6, alpha=0.7, label="1-SE rank")
    ax.axvline(p_cv, color="red", ls="--", lw=2, label=f"$p_{{cv}}^{{RecK}}={p_cv:.3f}$")
    ax.axvline(p_star, color="C5", ls=":", lw=1.5, label=f"$p^*={p_star:.3f}$")
    ax.axhline(k_cut, color="gray", ls=":", lw=1.5, label=f"$\\hat k^\\star={k_cut}$")
    ax.set_xlabel("$p$"); ax.set_ylabel("selected rank")
    ax.set_title("(E) SymmNMF selected rank vs $p$")
    ax.legend(fontsize=9, loc="best"); ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # (F) 1-SE band width
    ax = fig.add_subplot(gs[1, 2])
    span = np.array(a1se_list) - np.array(am_list)
    ax.plot(p_sweep_done, span, "o-", color="C2", ms=8)
    ax.axvline(p_cv, color="red", ls="--", lw=2, label=f"$p_{{cv}}^{{RecK}}={p_cv:.3f}$")
    ax.set_xlabel("$p$"); ax.set_ylabel("1-SE − argmin (parsimony bonus)")
    ax.set_title("(F) 1-SE rule parsimony (negative = simpler model preferred)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    am_at_pcv = cell['sweep'].get(min(p_sweep_done, key=lambda x: abs(x-p_cv)), {}).get('argmin', '?')
    fig.suptitle(f"Recipe K + **SymmNMF** on real data: {label} (n={n})\n"
                 f"$\\hat k^\\star = {k_cut}$, $p^* = {p_star:.3f}$, "
                 f"$p_{{cv}}={p_cv:.3f}$, argmin at $p_{{cv}}$ = {am_at_pcv}",
                 fontsize=14, y=0.995)
    out_path = os.path.join(FIG_DIR, f"{label}_symmnmf_6panel.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")

    return dict(label=label, n=n, k_cut=k_cut, p_star=p_star, p_cv=p_cv,
                argmins=am_list, a1se=a1se_list, p_sweep=p_sweep_done)


results = {}
for label in ["CLIP_RBF", "THINGS"]:
    r = render_dataset(label)
    if r is not None:
        results[label] = r


# Cross-fitter comparison (SoftImpute vs SymmNMF) — argmin vs p
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
for i, label in enumerate(["CLIP_RBF", "THINGS"]):
    ax = axs[i]
    # SymmNMF (just rendered)
    r = results.get(label)
    if r is not None:
        ax.plot(r["p_sweep"], r["argmins"], "o-", color="C0", ms=8,
                label=f"SymmNMF argmin")
        ax.plot(r["p_sweep"], r["a1se"], "s--", color="C1", ms=6, alpha=0.7,
                label=f"SymmNMF 1-SE")
    # SoftImpute (from SoftImpute pkl)
    si_label = label if label == "THINGS" else "CLIP_RBF_ext150"
    si_pkl = os.path.join(PKL_DIR, f"{si_label}.pkl")
    if os.path.exists(si_pkl):
        with open(si_pkl, "rb") as f: cell_si = pickle.load(f)
        ps = sorted(cell_si["sweep"].keys())
        ams_si = [cell_si["sweep"][p]["argmin"] for p in ps]
        ax.plot(ps, ams_si, "^-", color="C2", ms=8, alpha=0.7,
                label=f"SoftImpute argmin (rmax={cell_si['ranks'][-1]})")
    if r is not None:
        ax.axvline(r["p_cv"], color="red", ls="--", lw=2,
                   label=f"Recipe K $p_{{cv}}={r['p_cv']:.3f}$")
        ax.axhline(r["k_cut"], color="gray", ls=":", lw=1.5,
                   label=f"$\\hat k^\\star={r['k_cut']}$")
    ax.set_xlabel("$p$"); ax.set_ylabel("selected rank")
    ax.set_title(f"{label}: SymmNMF vs SoftImpute argmin")
    ax.legend(fontsize=9, loc="best"); ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fitter_comparison.png"), dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nsaved fitter_comparison.png")
print(f"\nDone. Figures in {FIG_DIR}")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"  {f}")
