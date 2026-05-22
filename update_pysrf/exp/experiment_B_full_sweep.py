"""
Comprehensive Experiment B v3 sweep — single-mask vs nested k-fold across
all manuscript datasets (Experiment A taxonomy + group expansion + real data).

Six protocols per (dataset, n) cell:
  P0a: single-mask n_reps=1, p = 0.5
  P0b: single-mask n_reps=1, p = 0.75
  P0c: single-mask n_reps=1, p = 0.9
  P1 : single-mask n_reps=1, p = p_star (Recipe K's training fraction)
  P2 : single-mask n_reps=10 (n_reps=5 for n>=1500), p = p_star (with SE bars)
  P3 : 5-fold nested CV n_reps=2, outer p_cv (= p_star * k/(k-1) capped at p_max)

Datasets: 49 cells total
  - A-suite (12): A1..A8, A4h, A5h, A7h, A8h
  - E-suite (4) : E1..E4
  - R-suite (4) : R1..R4
  - R4 variants (4): R4a..R4d
  - HD-suite (4): HD1..HD4 at n=2000
  - X-suite (5) : X_10block, X_DCSBM, X_Multiscale, X_Temporal, X_Hornfail
  - Group 4 expansion (7)
  - Group 3 high-rank (3)
  - Group 5 high-K (4) at n=1600-2000
  - Real (2): CLIP_RBF, THINGS

Output folder: results/experiment_B_v3_full_sweep/
Per dataset: <label>.pkl (full curves) + <label>.png (figure).
Master JSON: summary.json with argmins per protocol.
"""
import os, sys, time, pickle, json, warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))   # RECIPE_K/exp/
ROOT = os.path.dirname(HERE)                          # RECIPE_K/
SRC = os.path.join(ROOT, "src")
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, SRC)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.utils import check_random_state
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

from symmnmf.cross_validation import mask_missing_entries
from _common import spectral_pass, recipe_K, split_omega_into_folds
from _representations import fit_symmnmf_admm
from _synthetic_extras import (
    ADVERSARIAL_DATASETS_SEEDED, EGAP_ADVERSARIAL_SEEDED,
    RECIPE_K_ADVERSARIAL_SEEDED, R4_VARIANTS_SEEDED,
    HIGH_DIM_SEEDED, X_SUITE_SEEDED,
    GROUP4_EXPANSION_SEEDED, GROUP3_HIGHRANK_SEEDED, GROUP5_HIGHRANK_SEEDED,
)

OUT_DIR = os.path.join(ROOT, "output", "experiment_B")
FIG_DIR = os.path.join(OUT_DIR, "figures")
PKL_DIR = os.path.join(OUT_DIR, "raw")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PKL_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUT_DIR, "run.log")
LOG = open(LOG_PATH, "w", buffering=1)
def log(msg):
    print(msg, flush=True); LOG.write(msg + "\n")

SEED = 0
DELTA = 0.10
P_FLOOR = 0.5
M_MIN = 2000
N_JOBS = 95
P_FIXED_LIST = [0.5, 0.75, 0.9]


def get_clip():
    Df = np.load(os.path.join(DATA_DIR, "model_features", "clip_vit_b_32.npy"))
    dist = cdist(Df, Df, metric="euclidean")
    bw = float(np.median(dist))
    return np.exp(-(dist / bw) ** 2), None


def get_things():
    A = np.load(os.path.join(DATA_DIR, "things_behavior_similarity.npy"))
    A = (A + A.T) / 2
    # The raw THINGS file contains 8 NaN entries in the off-diagonal. Impute
    # with the off-diagonal median so the V-MSE objective on single-mask
    # protocols isn't all-NaN (cf. experiment_B_pstar_gap_C_things_fixed.py).
    diag = np.eye(A.shape[0], dtype=bool)
    med = float(np.nanmedian(A[~diag]))
    A = np.where(np.isnan(A), med, A)
    return A, None


# Build flat list: (label, generator, group_label)
ALL_DATASETS = []
for label, gen in ADVERSARIAL_DATASETS_SEEDED: ALL_DATASETS.append((label, gen, "A"))
for label, gen in EGAP_ADVERSARIAL_SEEDED:     ALL_DATASETS.append((label, gen, "E"))
for label, gen in RECIPE_K_ADVERSARIAL_SEEDED: ALL_DATASETS.append((label, gen, "R"))
for label, gen in R4_VARIANTS_SEEDED:          ALL_DATASETS.append((label, gen, "R4v"))
for label, gen in X_SUITE_SEEDED:              ALL_DATASETS.append((label, gen, "X"))
for label, gen in GROUP4_EXPANSION_SEEDED:     ALL_DATASETS.append((label, gen, "G4"))
for label, gen in GROUP3_HIGHRANK_SEEDED:      ALL_DATASETS.append((label, gen, "G3"))
for label, gen in HIGH_DIM_SEEDED:             ALL_DATASETS.append((label, gen, "HD"))
for label, gen in GROUP5_HIGHRANK_SEEDED:      ALL_DATASETS.append((label, gen, "G5"))
ALL_DATASETS.append(("CLIP_RBF", lambda seed: get_clip(),  "real"))
ALL_DATASETS.append(("THINGS",   lambda seed: get_things(), "real"))


# ============================================================
# CV scoring helpers
# ============================================================
def score_off_diag(S, S_hat, mask):
    n = S.shape[0]
    diag = np.eye(n, dtype=bool)
    sm = mask & ~diag
    if sm.sum() == 0:
        return float("nan")
    return float(np.mean((S_hat[sm] - S[sm]) ** 2))


def _fit_score_one(S, fitter, r, holdout_mask, V_mask):
    try:
        S_hat = fitter(S, r, holdout_mask, seed=SEED)
    except Exception:
        return float("nan")
    if not np.all(np.isfinite(S_hat)):
        return float("nan")
    return score_off_diag(S, S_hat, V_mask)


def single_mask_curve(S, fitter, ranks, p, n_reps, seed=0, n_jobs=N_JOBS):
    rng = check_random_state(seed)
    cv = np.full((len(ranks), n_reps), np.nan)
    for rep in range(n_reps):
        val_mask = mask_missing_entries(S, float(p), rng,
                                          missing_values=np.nan)
        jobs = [(ri, val_mask, r) for ri, r in enumerate(ranks)]
        scores = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_fit_score_one)(S, fitter, r, vm, vm)
            for (_, vm, r) in jobs
        )
        for (ri, _, _), s in zip(jobs, scores):
            cv[ri, rep] = s
    mean = np.nanmean(cv, axis=1)
    if n_reps > 1:
        se = np.nanstd(cv, axis=1, ddof=1) / np.sqrt(np.maximum(
            np.sum(~np.isnan(cv), axis=1), 1))
    else:
        se = np.zeros_like(mean)
    return mean, se


def kfold_premask_curve(S, fitter, ranks, p_cv, k_inner=5, n_reps=2,
                          seed=0, n_jobs=N_JOBS):
    rng = check_random_state(seed)
    cv = np.full((len(ranks), n_reps, k_inner), np.nan)
    for rep in range(n_reps):
        M_outer = mask_missing_entries(S, float(p_cv), rng,
                                         missing_values=np.nan)
        if (~M_outer).sum() < k_inner * 2: continue
        val_masks = split_omega_into_folds(M_outer, k_inner, rng)
        if val_masks is None: continue
        jobs = []
        for fi, V_i in enumerate(val_masks):
            holdout = V_i | M_outer
            for ri, r in enumerate(ranks):
                jobs.append((ri, fi, holdout, V_i, r))
        scores = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_fit_score_one)(S, fitter, r, hm, vm)
            for (_, _, hm, vm, r) in jobs)
        for (ri, fi, _, _, _), s in zip(jobs, scores):
            cv[ri, rep, fi] = s
    flat = cv.reshape(len(ranks), -1)
    mean = np.nanmean(flat, axis=1)
    se = np.nanstd(flat, axis=1, ddof=1) / np.sqrt(np.maximum(
        np.sum(~np.isnan(flat), axis=1), 1))
    return mean, se


def select_argmin(curve, ranks):
    finite = np.where(~np.isnan(curve))[0]
    if len(finite) == 0: return -1
    return int(ranks[finite[np.argmin(curve[finite])]])


def select_1se(mean_v, se_v, ranks):
    finite = np.where(~np.isnan(mean_v))[0]
    if len(finite) == 0: return -1
    am = int(finite[np.argmin(mean_v[finite])])
    threshold = mean_v[am] + se_v[am]
    for i in finite:
        if mean_v[i] <= threshold:
            return int(ranks[i])
    return int(ranks[am])


def get_rank_max(n, k_cut, true_r):
    """Adaptive rank ceiling: enough to bracket true_r and k_cut."""
    base = max(int(true_r) if true_r else 0, int(k_cut))
    return min(n // 4, max(60, int(base * 2 + 20)))


# ============================================================
# Main loop
# ============================================================
log(f"=== Experiment B v3 full sweep — start {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
log(f"Total datasets: {len(ALL_DATASETS)}")
log(f"Output: {OUT_DIR}")

summary = {}
t_global = time.time()

for ds_idx, (label, gen, group) in enumerate(ALL_DATASETS):
    pkl_path = os.path.join(PKL_DIR, f"{label}.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            cell = pickle.load(f)
        log(f"\n[{ds_idx+1}/{len(ALL_DATASETS)}] {label} ({group}) — already done, "
            f"skipping (loaded from cache)")
        summary[label] = cell["summary"]
        continue

    log(f"\n{'='*72}")
    log(f"[{ds_idx+1}/{len(ALL_DATASETS)}] {label} ({group})  "
        f"(global elapsed: {(time.time()-t_global)/60:.1f}min)")
    log(f"{'='*72}")
    t_ds = time.time()
    try:
        S, true_r = gen(SEED)
    except Exception as e:
        log(f"  GENERATOR FAILED: {e}")
        continue
    n = S.shape[0]
    log(f"  S shape={S.shape}, range=[{np.nanmin(S):.3g}, {np.nanmax(S):.3g}]")

    # Spectral pass + Recipe K
    try:
        spec = spectral_pass(S, B=20, smooth_window=10)
        spec["k_cut"] = spec["k_smooth"]
        recK = recipe_K(spec, delta=DELTA, k_cv=5,
                        p_floor=P_FLOOR, p_floor_mode="adaptive",
                        M_min=M_MIN)
    except Exception as e:
        log(f"  SPECTRAL_PASS / Recipe K FAILED: {e}")
        continue
    p_star = recK["p_star"]
    p_cv = recK["p_cv"]
    k_cut = int(spec["k_smooth"])
    rmax = get_rank_max(n, k_cut, true_r)
    ranks = list(range(1, rmax + 1))
    log(f"  k_cut={k_cut}, true_r={true_r}, "
        f"p_star={p_star:.4f}, p_cv={p_cv:.4f}, ranks=[1..{rmax}]")

    # Adaptive n_reps for P2 to keep wall-clock manageable on big n
    n_reps_p2 = 5 if n >= 1500 else 10

    cell = {
        "label": label, "group": group, "n": int(n), "true_r": true_r,
        "k_cut": k_cut, "p_star": float(p_star), "p_cv": float(p_cv),
        "rmax": int(rmax),
        "summary": {
            "label": label, "group": group, "n": int(n),
            "true_r": true_r, "k_cut": k_cut,
            "p_star": float(p_star), "p_cv": float(p_cv),
        },
    }

    # Fixed-p single-mask protocols
    for p_fixed in P_FIXED_LIST:
        t0 = time.time()
        m, se = single_mask_curve(S, fit_symmnmf_admm, ranks, p_fixed,
                                     n_reps=1, seed=SEED, n_jobs=N_JOBS)
        t = time.time() - t0
        a = select_argmin(m, ranks)
        key = f"P0_p{int(p_fixed*100):02d}"
        cell[key] = dict(mean=m, se=se, argmin=a, time=t,
                          p=float(p_fixed), n_reps=1)
        cell["summary"][key] = dict(argmin=a, time=t, p=float(p_fixed))
        log(f"  {key} (p={p_fixed:.2f}, n_reps=1): argmin={a:>3d}  ({t:.1f}s)")

    # P1: single-mask n_reps=1 at p_star
    t0 = time.time()
    m1, se1 = single_mask_curve(S, fit_symmnmf_admm, ranks, p_star,
                                  n_reps=1, seed=SEED, n_jobs=N_JOBS)
    t_p1 = time.time() - t0
    a_p1 = select_argmin(m1, ranks)
    cell["P1"] = dict(mean=m1, se=se1, argmin=a_p1, time=t_p1,
                        p=float(p_star), n_reps=1)
    cell["summary"]["P1"] = dict(argmin=a_p1, time=t_p1, p=float(p_star))
    log(f"  P1 (p_star={p_star:.3f}, n_reps=1): argmin={a_p1:>3d}  ({t_p1:.1f}s)")

    # P2: single-mask n_reps adaptive at p_star
    t0 = time.time()
    m2, se2 = single_mask_curve(S, fit_symmnmf_admm, ranks, p_star,
                                  n_reps=n_reps_p2, seed=SEED, n_jobs=N_JOBS)
    t_p2 = time.time() - t0
    a_p2 = select_argmin(m2, ranks)
    a_p2_1se = select_1se(m2, se2, ranks)
    cell["P2"] = dict(mean=m2, se=se2, argmin=a_p2, argmin_1se=a_p2_1se,
                        time=t_p2, p=float(p_star), n_reps=n_reps_p2)
    cell["summary"]["P2"] = dict(argmin=a_p2, argmin_1se=a_p2_1se,
                                   time=t_p2, p=float(p_star), n_reps=n_reps_p2)
    log(f"  P2 (p_star={p_star:.3f}, n_reps={n_reps_p2}): "
        f"argmin={a_p2:>3d}  1SE={a_p2_1se:>3d}  ({t_p2:.1f}s)")

    # P3: 5-fold n_reps=2 with p_cv outer
    t0 = time.time()
    m3, se3 = kfold_premask_curve(S, fit_symmnmf_admm, ranks, p_cv,
                                     k_inner=5, n_reps=2, seed=SEED,
                                     n_jobs=N_JOBS)
    t_p3 = time.time() - t0
    a_p3 = select_argmin(m3, ranks)
    a_p3_1se = select_1se(m3, se3, ranks)
    cell["P3"] = dict(mean=m3, se=se3, argmin=a_p3, argmin_1se=a_p3_1se,
                        time=t_p3, p=float(p_cv), n_reps=2)
    cell["summary"]["P3"] = dict(argmin=a_p3, argmin_1se=a_p3_1se,
                                   time=t_p3, p=float(p_cv), n_reps=2)
    log(f"  P3 (p_cv={p_cv:.3f}, k_inner=5, n_reps=2): "
        f"argmin={a_p3:>3d}  1SE={a_p3_1se:>3d}  ({t_p3:.1f}s)")
    cell["ranks"] = ranks

    # Save per-dataset
    with open(pkl_path, "wb") as f:
        pickle.dump(cell, f)

    # Per-dataset figure
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        COLORS = {
            "P0_p50": ("C5", "p=0.5"),
            "P0_p75": ("C6", "p=0.75"),
            "P0_p90": ("C7", "p=0.9"),
            "P1":     ("C0", "p=p_star, n_reps=1"),
            "P2":     ("C2", f"p=p_star, n_reps={n_reps_p2}"),
            "P3":     ("C1", "k-fold + p_cv"),
        }
        for key, (color, lbl) in COLORS.items():
            v = cell[key]
            am_str = f"argmin={v['argmin']}"
            if "argmin_1se" in v: am_str += f" 1SE={v['argmin_1se']}"
            ax.plot(ranks, v["mean"], "o-", color=color, markersize=2.5,
                    label=f"{lbl} {am_str} ({v['time']:.0f}s)")
            if v.get("n_reps", 1) > 1:
                ax.fill_between(ranks, v["mean"] - v["se"], v["mean"] + v["se"],
                                 color=color, alpha=0.10)
        if true_r is not None:
            ax.axvline(true_r, color="C3", linewidth=2, alpha=0.7,
                       label=f"true_r={true_r}")
        ax.axvline(k_cut, color="grey", linewidth=1.0, linestyle=":",
                   alpha=0.6, label=f"k_cut={k_cut}")
        ax.set_xlabel("rank r"); ax.set_ylabel("V-MSE")
        ax.set_title(f"{label} ({group})  n={n}  p_star={p_star:.3f}  p_cv={p_cv:.3f}")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"{label}.png"),
                    dpi=110, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log(f"  figure save FAILED: {e}")

    summary[label] = cell["summary"]
    # Incremental summary
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log(f"  total: {time.time() - t_ds:.1f}s")

log(f"\n=== DONE  total elapsed {(time.time()-t_global)/60:.1f}min ===")
LOG.close()
