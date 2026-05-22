"""
Supplementary experiment: Recipe K's κ̂_r vs pure-eigengap detectors.

For each of three regimes (planted-spike BBP-matched / boundary / real-data),
compute four detector outputs:
  (D1) Recipe K's full machinery: F-stat on κ̂_r → k_cut + δ_emp inversion → p*
  (D2) Maximum eigenvalue ratio: k_DG_ratio = argmax_r λ_r/λ_{r+1}
  (D3) Donoho-Gavish-style edge count: k_DG = #{r : λ_r > 2σ̂√n}
  (D4) Recipe K variant where κ_r is replaced by the leading-order
       formula κ_r^{eig} = sum_{s≠r} λ_s² / (λ_r - λ_s)² (no μ², deterministic).

For (D2)-(D4) we also compute a BBP-derived operating point
p_BBP = 4σ̂²n/λ̂_{k}² for direct comparison with Recipe K's p_star.

Output:
  - results/experiment_B_v3_kappa_vs_eigengap/comparison.csv
  - figures/comparison.png (per-cell k_cut table + p_star scatter)
"""
import os, sys, time, pickle, warnings
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))   # RECIPE_K/exp/
ROOT = os.path.dirname(HERE)                          # RECIPE_K/
SRC = os.path.join(ROOT, "src")
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, SRC)

from _common import spectral_pass, recipe_K, estimate_k_cut_fstat

OUT_DIR = os.path.join(ROOT, "output", "pstar_gap", "kappa_vs_eigengap")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "run.log")
LOG = open(LOG_PATH, "w", buffering=1)
def log(msg):
    print(msg, flush=True); LOG.write(msg + "\n")


# ============================================================
# Cell definitions
# ============================================================
def make_bbp_planted(n, r, lambdas, sigma, seed):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, r))
    U, _ = np.linalg.qr(U)
    sig = U @ np.diag(lambdas) @ U.T
    G = rng.standard_normal((n, n))
    E = sigma * (G + G.T) / np.sqrt(2)
    return sig + E


def cell_bbp_supercrit(n, seed=0):
    edge = 2 * np.sqrt(n)
    lambdas = [c * edge for c in [1.6, 1.8, 2.0, 2.4, 3.0]]
    return make_bbp_planted(n, 5, lambdas, 1.0, seed), 5


def cell_boundary(n, seed=0):
    edge = 2 * np.sqrt(n)
    lambdas = [c * edge for c in [1.2, 1.4, 1.6, 2.0, 2.6]]
    return make_bbp_planted(n, 5, lambdas, 1.0, seed), 5


def cell_heavy_tail(n, seed=0):
    """Synthetic heavy-tail: top-5 spikes + power-law bulk decay."""
    rng = np.random.default_rng(seed)
    r_top = 5
    edge = 2 * np.sqrt(n)
    spikes = [edge * c for c in [3.0, 2.5, 2.0, 1.8, 1.6]]
    bulk_ranks = np.arange(1, n - r_top + 1)
    bulk_lams = 5.0 * bulk_ranks ** (-0.5)
    all_lam = np.concatenate([spikes, bulk_lams])
    U = rng.standard_normal((n, n)); U, _ = np.linalg.qr(U)
    sig = U @ np.diag(all_lam) @ U.T
    G = rng.standard_normal((n, n)); E = 0.5 * (G + G.T) / np.sqrt(2)
    return sig + E, r_top


def load_clip():
    Df = np.load(os.path.join(DATA_DIR, "model_features", "clip_vit_b_32.npy"))
    dist = cdist(Df, Df, "euclidean")
    bw = float(np.median(dist))
    return np.exp(-(dist / bw) ** 2), None


def load_things():
    A = np.load(os.path.join(DATA_DIR, "things_behavior_similarity.npy"))
    A = (A + A.T) / 2
    diag = np.eye(A.shape[0], dtype=bool)
    med = float(np.nanmedian(A[~diag]))
    return np.where(np.isnan(A), med, A), None


CELLS = [
    ("BBP_n400",       lambda: cell_bbp_supercrit(400)),
    ("BBP_n800",       lambda: cell_bbp_supercrit(800)),
    ("BBP_n1600",      lambda: cell_bbp_supercrit(1600)),
    ("Boundary_n800",  lambda: cell_boundary(800)),
    ("Boundary_n1600", lambda: cell_boundary(1600)),
    ("Heavy_n800",     lambda: cell_heavy_tail(800)),
    ("Heavy_n1600",    lambda: cell_heavy_tail(1600)),
    ("CLIP_RBF",       load_clip),
    ("THINGS",         load_things),
]


# ============================================================
# Detectors
# ============================================================
def detector_recipe_K(S, spec, recK):
    """D1: Recipe K's k_cut from F-stat on κ̂_r, p* from δ_emp inversion."""
    return dict(
        k_cut=int(spec["k_smooth"]),
        p_star_raw=float(recK["p_star_raw"]),
        p_star=float(recK["p_star"]),
        p_floor=float(recK["p_floor"]),
        sigma_hat=float(recK.get("sigma_hat", 0)) if recK.get("sigma_hat") else None,
    )


def detector_eigratio(spec):
    """D2: Maximum eigenvalue ratio λ_r/λ_{r+1}, r >= 2 to avoid trivial r=1."""
    evals = np.asarray(spec["evals_ref"])
    ratios = evals[:-1] / np.maximum(evals[1:], 1e-12)   # length K-1
    # Restrict to r >= 2 (i.e., index >= 1 in 0-based)
    valid = np.arange(1, len(ratios))
    if len(valid) == 0:
        return dict(k_cut=-1)
    best = int(valid[np.argmax(ratios[valid])])
    k_cut = best + 1  # 1-based r
    return dict(k_cut=k_cut, ratio=float(ratios[best]))


def detector_DG_edge(spec, n, sigma_hat):
    """D3: Donoho-Gavish-style: count eigenvalues above 2σ̂√n."""
    evals = np.asarray(spec["evals_ref"])
    edge = 2 * sigma_hat * np.sqrt(n)
    k_cut = int(np.sum(evals > edge))
    return dict(k_cut=k_cut, edge=float(edge))


def kappa_r_leading_order(evals_ref, k_max=None):
    """Compute κ_r^eig = sum_{s≠r} λ_s²/(λ_r-λ_s)² (leading-order formula,
    no μ² factor — purely spectral)."""
    K = len(evals_ref) if k_max is None else min(k_max, len(evals_ref))
    kappa = np.zeros(K)
    for r in range(K):
        lam_r = evals_ref[r]
        lam_s = np.concatenate([evals_ref[:r], evals_ref[r+1:]])
        denom = (lam_r - lam_s) ** 2
        # Avoid division by zero for degenerate spectra
        denom = np.maximum(denom, 1e-12 * lam_r ** 2)
        kappa[r] = float(np.sum(lam_s ** 2 / denom))
    return kappa


def detector_kappa_eig(spec):
    """D4: Recipe-K-style F-stat detector using κ_r^eig (leading-order, no bootstrap)."""
    evals = np.asarray(spec["evals_ref"])
    K = len(evals)
    kappa_eig = kappa_r_leading_order(evals)
    k_list = np.arange(1, K + 1)
    k_cut, F_max, _, _ = estimate_k_cut_fstat(kappa_eig, list(k_list))
    return dict(k_cut=int(k_cut) if k_cut else -1, kappa_eig=kappa_eig.tolist(),
                F_max=float(F_max) if F_max else float("nan"))


def operating_point_BBP(n, sigma_hat, lam_kcut):
    """p_BBP = 4σ̂²n/λ_{k_cut}² — operating point from BBP threshold alone."""
    if lam_kcut <= 1e-12: return float("nan")
    p = 4.0 * sigma_hat ** 2 * n / (lam_kcut ** 2)
    return float(min(max(p, 0.0), 1.0))


# ============================================================
# Run
# ============================================================
log(f"=== κ̂_r vs eigengap supplementary — start {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
rows = []
for label, loader in CELLS:
    log(f"\n--- {label} ---")
    t0 = time.time()
    res = loader()
    if isinstance(res, tuple) and len(res) == 2:
        S, true_r = res
    else:
        S = res; true_r = None
    n = S.shape[0]
    log(f"  shape={S.shape}, true_r={true_r}")

    spec = spectral_pass(S, B=20, smooth_window=10)
    spec["k_cut"] = spec["k_smooth"]
    recK = recipe_K(spec, delta=0.10, k_cv=5, p_floor=0.5,
                       p_floor_mode="adaptive", M_min=2000)
    sigma_hat = float(recK.get("sigma_hat") or 0.0)
    evals = np.asarray(spec["evals_ref"])

    D1 = detector_recipe_K(S, spec, recK)
    D2 = detector_eigratio(spec)
    D3 = detector_DG_edge(spec, n, sigma_hat)
    D4 = detector_kappa_eig(spec)

    # BBP-derived p* for each k_cut
    p_BBP_D1 = operating_point_BBP(n, sigma_hat, evals[D1["k_cut"] - 1]) if D1["k_cut"] >= 1 else float("nan")
    p_BBP_D2 = operating_point_BBP(n, sigma_hat, evals[D2["k_cut"] - 1]) if D2["k_cut"] >= 1 else float("nan")
    p_BBP_D3 = operating_point_BBP(n, sigma_hat, evals[D3["k_cut"] - 1]) if D3["k_cut"] >= 1 else float("nan")
    p_BBP_D4 = operating_point_BBP(n, sigma_hat, evals[D4["k_cut"] - 1]) if D4["k_cut"] >= 1 else float("nan")

    row = dict(label=label, n=n, true_r=true_r, sigma_hat=sigma_hat,
                 D1_kcut=D1["k_cut"], D1_p_star=D1["p_star"], D1_p_star_raw=D1["p_star_raw"], D1_p_BBP=p_BBP_D1,
                 D2_kcut=D2["k_cut"], D2_p_BBP=p_BBP_D2, D2_ratio=D2.get("ratio", np.nan),
                 D3_kcut=D3["k_cut"], D3_p_BBP=p_BBP_D3, D3_edge=D3.get("edge", np.nan),
                 D4_kcut=D4["k_cut"], D4_p_BBP=p_BBP_D4,
                 elapsed=time.time() - t0)
    rows.append(row)
    log(f"  D1 Recipe K:       k={D1['k_cut']:>3}, p*_raw={D1['p_star_raw']:.4f}, p*={D1['p_star']:.4f}, p_BBP={p_BBP_D1:.4f}")
    log(f"  D2 max-eigratio:   k={D2['k_cut']:>3}, (ratio={D2.get('ratio',0):.2f}), p_BBP={p_BBP_D2:.4f}")
    log(f"  D3 DG edge count:  k={D3['k_cut']:>3}, edge={D3.get('edge',0):.2f}, p_BBP={p_BBP_D3:.4f}")
    log(f"  D4 κ_eig F-stat:   k={D4['k_cut']:>3}, p_BBP={p_BBP_D4:.4f}")

# Save table
import json
with open(os.path.join(OUT_DIR, "comparison.json"), "w") as f:
    json.dump(rows, f, indent=2, default=str)

# Markdown table
md_path = os.path.join(OUT_DIR, "comparison.md")
with open(md_path, "w") as f:
    f.write("# κ̂_r (Recipe K) vs eigengap-based detectors — supplementary\n\n")
    f.write("| Cell | n | true_r | σ̂ | **D1: Recipe K** k | D1 p\\*_raw | D1 p\\* | **D2: max λ_r/λ_{r+1}** k | **D3: DG edge** k | **D4: κ_eig F-stat** k |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        true_r = r.get("true_r") or "—"
        f.write(f"| {r['label']} | {r['n']} | {true_r} | {r['sigma_hat']:.3f} | "
                f"**{r['D1_kcut']}** | {r['D1_p_star_raw']:.3f} | {r['D1_p_star']:.3f} | "
                f"{r['D2_kcut']} | {r['D3_kcut']} | {r['D4_kcut']} |\n")
    f.write("\n## Operating-point comparison (where applicable)\n\n")
    f.write("| Cell | D1 Recipe K p\\* | D2 p_BBP(k_D2) | D3 p_BBP(k_D3) | D4 p_BBP(k_D4) |\n")
    f.write("|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['label']} | {r['D1_p_star']:.3f} | "
                f"{r['D2_p_BBP']:.3f} | {r['D3_p_BBP']:.3f} | {r['D4_p_BBP']:.3f} |\n")
log(f"\nsaved {md_path}")

# Figure: k_cut comparison + p_star scatter
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

ax = axs[0]
labels = [r["label"] for r in rows]
x = np.arange(len(labels))
w = 0.2
ax.bar(x - 1.5*w, [r["D1_kcut"] for r in rows], w, label="D1 Recipe K (κ̂_r)", color="C0")
ax.bar(x - 0.5*w, [r["D2_kcut"] for r in rows], w, label="D2 max λ_r/λ_{r+1}", color="C1")
ax.bar(x + 0.5*w, [r["D3_kcut"] for r in rows], w, label="D3 DG edge count", color="C2")
ax.bar(x + 1.5*w, [r["D4_kcut"] for r in rows], w, label="D4 κ_eig F-stat", color="C3")
# True r overlays
for i, r in enumerate(rows):
    if r["true_r"]:
        ax.scatter([x[i]], [r["true_r"]], s=140, marker="*", color="black",
                     zorder=10, edgecolors="white", linewidths=1.5)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("$k_{cut}$")
ax.set_yscale("log")
ax.set_title("Detector $k_{cut}$ comparison (★ = true_r where known)")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3, axis="y", which="both")

ax = axs[1]
for r in rows:
    if not (np.isnan(r["D1_p_star"]) or np.isnan(r["D3_p_BBP"])):
        ax.scatter(r["D3_p_BBP"], r["D1_p_star"], s=80, label=r["label"])
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Recipe K = BBP")
ax.set_xlabel("$p_{BBP}$ (Donoho-Gavish-derived)")
ax.set_ylabel("$p^*$ (Recipe K)")
ax.set_title("Operating point: Recipe K vs BBP-only")
ax.legend(fontsize=8, loc="best", ncol=1)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "comparison.png"), dpi=120, bbox_inches="tight")
plt.close(fig)
log(f"saved figure {FIG_DIR}/comparison.png")

LOG.close()
print(f"\nDONE. See {md_path} and {FIG_DIR}/comparison.png")
