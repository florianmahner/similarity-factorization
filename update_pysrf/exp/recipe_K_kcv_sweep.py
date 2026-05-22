"""
Recipe K test (with CV): direct inversion of empirical VE^D curve, BBP-floor.

For each dataset, fix delta=0.10 and sweep k_cv in {3, 5, 10}.

For each (dataset, k_cv):
  1. Spectral pass (B=20, cached per-dataset)
  2. Recipe K to find p* (with BBP-floor, n-adaptive p_max)
  3. Run nested CV at p_cv (k_inner=k_cv)
  4. Score V and M (skip U)
  5. Save per-(dataset, k_cv) pkl + 4-panel figure

Order: 10-block n=100, n=200, n=400, DCSBM, Horn-fail, Multiscale, Temporal,
       10-block n=800.
"""
import os, sys, time, json, pickle, warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))   # RECIPE_K/exp/
ROOT = os.path.dirname(HERE)                          # RECIPE_K/
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.utils import check_random_state
from joblib import Parallel, delayed

from _common import (spectral_pass, recipe_K, variance_weighted_kappa,
                     env_info, estimate_incoherence, split_omega_into_folds,
                     fit_admm_score, argmin_safe, argmin_1se_safe)
from _generate_examples import (
    _rbf_kernel, simulation,
    make_similarity_dcsbm, make_similarity_multiscale_manifolds,
    make_similarity_temporal, make_correlation_horn_fail_challenging,
)

RESULTS_DIR = os.path.join(ROOT, "output", "recipe_K_kcv_sweep")
os.makedirs(RESULTS_DIR, exist_ok=True)
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LOG = open(os.path.join(RESULTS_DIR, "run.log"), "w", buffering=1)
def log(msg):
    print(msg, flush=True); LOG.write(msg + "\n")

log(f"=== Recipe K test (k_cv sweep, delta=0.10) — start {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
with open(os.path.join(RESULTS_DIR, "env.txt"), "w") as f:
    json.dump(env_info(), f, indent=2)


# ============================================================
# Custom nested CV: V, M only (no U)
# ============================================================
def nested_cv_VM(S, p_outer, ranks, k_inner=5, n_reps=1, seed=0, n_jobs=4):
    from symmnmf.cross_validation import mask_missing_entries
    n = S.shape[0]
    rng = check_random_state(seed)
    bounds = (float(np.nanmin(S)), float(np.nanmax(S)))
    already_nan = np.isnan(S)

    cv_v = np.full((len(ranks), n_reps, k_inner), np.nan)
    cv_m = np.full((len(ranks), n_reps, k_inner), np.nan)

    for rep in range(n_reps):
        M_outer = mask_missing_entries(S, float(p_outer), rng, missing_values=np.nan)
        if (~M_outer).sum() < k_inner * 2:
            continue
        val_masks = split_omega_into_folds(M_outer, k_inner, rng)
        if val_masks is None:
            continue

        jobs = []
        for fold_idx, V_i in enumerate(val_masks):
            holdout = V_i | M_outer
            V_valid = V_i & ~already_nan
            M_valid = M_outer & ~already_nan
            U_valid = holdout & ~already_nan
            for ri, r in enumerate(ranks):
                seed_rfk = int(seed + 1000 * (rep + 1) + 100 * fold_idx + ri)
                jobs.append((ri, fold_idx, holdout, V_valid, M_valid, U_valid, r, seed_rfk))

        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(fit_admm_score)(S, hm, vm, mm, um, r, bounds, sd)
            for (_, _, hm, vm, mm, um, r, sd) in jobs
        )
        for (ri, fold_idx, *_), (mv, mm_, _) in zip(jobs, results):
            cv_v[ri, rep, fold_idx] = mv
            cv_m[ri, rep, fold_idx] = mm_

    mean_v = np.nanmean(cv_v.reshape(len(ranks), -1), axis=1)
    mean_m = np.nanmean(cv_m.reshape(len(ranks), -1), axis=1)
    # SEM across the n_reps * k_inner folds (2026-05-11, round-1):
    flat_v = cv_v.reshape(len(ranks), -1)
    flat_m = cv_m.reshape(len(ranks), -1)
    n_finite_v = np.sum(np.isfinite(flat_v), axis=1)
    n_finite_m = np.sum(np.isfinite(flat_m), axis=1)
    sem_v = np.nanstd(flat_v, axis=1, ddof=1) / np.sqrt(np.maximum(n_finite_v, 1))
    sem_m = np.nanstd(flat_m, axis=1, ddof=1) / np.sqrt(np.maximum(n_finite_m, 1))
    return dict(ranks=list(ranks), cv_v=mean_v, cv_m=mean_m,
                sem_v=sem_v, sem_m=sem_m)


# ============================================================
# Datasets — n=100 added at start
# ============================================================
def get_10block(n, seed=0):
    rng = np.random.default_rng(seed)
    D = simulation(n, 10, ndict=10) + rng.random((n, 10)) * 0.5
    return _rbf_kernel(D, bw=1.0), 10

datasets = [
    ("10-block n=100", lambda: get_10block(n=100)),
    ("10-block n=200", lambda: get_10block(n=200)),
    ("10-block n=400", lambda: get_10block(n=400)),
    ("DCSBM",          lambda: make_similarity_dcsbm(n=300, K=5, seed=0)[:2]),
    ("Horn-fail",      lambda: (make_correlation_horn_fail_challenging(n=200, r_true=3, n_clusters=20, seed=0)[0], 3)),
    ("Multiscale",     lambda: make_similarity_multiscale_manifolds(n1=200, n2=200, seed=1)[:2]),
    ("Temporal",       lambda: make_similarity_temporal(n=400, regimes=4, seed=2)[:2]),
    ("10-block n=800", lambda: get_10block(n=800)),
]

DELTA = 0.10
K_CVS = [3, 5, 10]
N_REPS = 1
N_JOBS = 95
P_FLOOR = 0.5
M_MIN = 2000

KCV_COLOR = {3: "C3", 5: "C0", 10: "C2"}


# ============================================================
# Per-(dataset, k_cv) processing
# ============================================================
def process_dataset_at_kcv(label, S, true, spec, tk, mu_hat, k_cv,
                            ds_idx, total_ds, sw_idx, total_sw):
    log(f"\n{'='*70}\n[sweep {sw_idx+1}/{total_sw} k_cv={k_cv}]"
        f"  [{ds_idx+1}/{total_ds}] {label}\n{'='*70}")
    n = S.shape[0]
    log(f"  k_cut={spec['k_smooth']}, k_cliff={spec['k_cliff']}, "
        f"κ̃={tk:.4f}, μ̂={mu_hat:.3f}")

    rank_max = min(n // 4, max(spec["k_smooth"] + 30, 60))
    rank_grid = list(range(1, rank_max + 1, 1 if n <= 400 else 2))

    # Recipe K with BBP-floor and n-adaptive p_max
    recK = recipe_K(spec, delta=DELTA, k_cv=k_cv, p_floor=P_FLOOR, M_min=M_MIN)
    p_I = float(tk / (tk + DELTA))

    log(f"  [k_cv={k_cv}]: Recipe I p*={p_I:.4f} | Recipe K p_star_raw={recK['p_star_raw']:.4f}, "
        f"p_star(after floor)={recK['p_star']:.4f}, floor_binding={recK['floor_binding']}, "
        f"p_cv={recK['p_cv']:.4f}, p_train={recK['p_train']:.4f}, "
        f"p_max(n={n})={recK['p_max']:.4f}")

    t0 = time.time()
    cv = nested_cv_VM(S, p_outer=recK["p_cv"], ranks=rank_grid,
                      k_inner=k_cv, n_reps=N_REPS, seed=0, n_jobs=N_JOBS)
    am_v = argmin_safe(cv["cv_v"], rank_grid)
    am_m = argmin_safe(cv["cv_m"], rank_grid)
    am_v_1se = argmin_1se_safe(cv["cv_v"], rank_grid, sem=cv["sem_v"])
    am_m_1se = argmin_1se_safe(cv["cv_m"], rank_grid, sem=cv["sem_m"])
    log(f"    CV: V_argmin={am_v} (1SE={am_v_1se}), "
        f"M_argmin={am_m} (1SE={am_m_1se})  ({time.time()-t0:.1f}s)")

    row = dict(
        label=label, n=n, true_rank=true,
        k_cut=spec["k_smooth"], k_cliff=spec["k_cliff"],
        rho=spec["rho"], alpha=spec["alpha"],
        tilde_kappa=tk, mu_hat=mu_hat,
        delta=DELTA,
        k_cv=int(k_cv),
        p_I=p_I,
        p_K_raw=recK["p_star_raw"],
        p_K=recK["p_star"],
        floor_binding=recK["floor_binding"],
        p_cv=recK["p_cv"], p_train=recK["p_train"],
        p_max_used=recK["p_max"],
        argmin_v=am_v, argmin_m=am_m,
        argmin_v_1se=am_v_1se, argmin_m_1se=am_m_1se,
        cv_v=list(cv["cv_v"]), cv_m=list(cv["cv_m"]),
        sem_v=list(cv["sem_v"]), sem_m=list(cv["sem_m"]),
        rank_grid=list(rank_grid),
        delta_emp=recK["delta_emp"], p_grid=recK["p_grid"],
    )

    # Save per-(dataset, k_cv) pkl
    safe_label = label.replace(" ", "_").replace("=", "")
    kcv_tag = f"kcv{k_cv:02d}"
    ds_pkl = os.path.join(FIG_DIR, f"{safe_label}_{kcv_tag}.pkl")
    with open(ds_pkl, "wb") as f:
        pickle.dump(dict(label=label, n=n, true_rank=true, spectral=spec,
                         tk=tk, mu_hat=mu_hat, row=row, k_cv=k_cv), f)
    log(f"  saved per-(dataset,k_cv) pkl: {ds_pkl}")
    return row


def plot_dataset_kcv_sweep(label, rows, spec, true_rank, tk, mu_hat, safe_label):
    """4 panels: kappa profile, delta_emp curve, V curves, M curves —
    one curve per k_cv value."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel 1: kappa profile
    ax = axes[0, 0]
    ax.plot(spec["k_list"], spec["kappa_hat"], marker=".", alpha=0.5, color="C0")
    ax.axvline(spec["k_smooth"], color="k", linestyle="--", linewidth=1.4,
               label=f"k_cut={spec['k_smooth']}")
    if spec["k_cliff"] is not None:
        ax.axvline(spec["k_cliff"], color="C1", linewidth=1.4,
                   label=f"k_cliff={spec['k_cliff']}")
    if true_rank is not None:
        ax.axvline(true_rank, color="C2", linestyle=":", linewidth=1.3,
                   label=f"true={true_rank}")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlabel("k"); ax.set_ylabel(r"$\hat\kappa_k$")
    ax.set_title(f"{label} κ profile (κ̃={tk:.3f}, μ̂={mu_hat:.2f})", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: delta_emp curve + per-k_cv markers
    ax = axes[0, 1]
    p_grid = rows[0]["p_grid"]
    delta_emp = rows[0]["delta_emp"]
    ax.plot(p_grid, delta_emp, "o-", color="C0", label=r"$\delta_{\rm emp}(p)$ (Recipe K)")
    delta_pred = tk * (1 - np.array(p_grid)) / np.array(p_grid)
    ax.plot(p_grid, delta_pred, "--", color="C1",
            label=r"$\delta_{\rm pred}(p)=\tilde\kappa(1-p)/p$ (Recipe I)")
    ax.axhline(DELTA, color="gray", linestyle=":", alpha=0.5, label=f"δ={DELTA}")
    for r in rows:
        c = KCV_COLOR[r["k_cv"]]
        ax.scatter([r["p_K"]], [DELTA], color=c, marker="o", s=80, zorder=5,
                   label=f"k_cv={r['k_cv']}: p_K*={r['p_K']:.3f}, p_cv={r['p_cv']:.3f}"
                         + (" (FLOOR)" if r["floor_binding"] else ""))
        ax.axvline(r["p_cv"], color=c, linestyle=":", alpha=0.4)
    ax.axvline(P_FLOOR, color="red", linestyle="-.", alpha=0.4, label=f"BBP floor p={P_FLOOR}")
    ax.axvline(rows[0]["p_max_used"], color="purple", linestyle="-.", alpha=0.4,
               label=f"p_max(n)={rows[0]['p_max_used']:.3f}")
    ax.set_xlabel("p"); ax.set_ylabel(r"$\delta$")
    ax.set_title("Empirical vs predicted deficit curve")
    ax.set_ylim(-0.05, max(1.0, max(delta_emp) * 1.1))
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Panel 3: V curves
    ax = axes[1, 0]
    for r in rows:
        c = KCV_COLOR[r["k_cv"]]
        ax.plot(r["rank_grid"], r["cv_v"], color=c,
                label=f"k_cv={r['k_cv']}: p_cv={r['p_cv']:.3f}, V_am={r['argmin_v']}")
        if r["argmin_v"] in r["rank_grid"]:
            i = r["rank_grid"].index(r["argmin_v"])
            ax.scatter([r["argmin_v"]], [r["cv_v"][i]], s=80, marker="*",
                       color=c, edgecolor="k", zorder=5)
    if true_rank is not None: ax.axvline(true_rank, color="C2", linestyle=":", alpha=0.5)
    ax.axvline(spec["k_smooth"], color="k", linestyle="--", alpha=0.5)
    if spec["k_cliff"] is not None: ax.axvline(spec["k_cliff"], color="C1", alpha=0.5)
    ax.set_xlabel("rank"); ax.set_ylabel("V MSE")
    ax.set_title("V CV curves"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Panel 4: M curves
    ax = axes[1, 1]
    for r in rows:
        c = KCV_COLOR[r["k_cv"]]
        ax.plot(r["rank_grid"], r["cv_m"], color=c,
                label=f"k_cv={r['k_cv']}: p_cv={r['p_cv']:.3f}, M_am={r['argmin_m']}")
        if r["argmin_m"] in r["rank_grid"]:
            i = r["rank_grid"].index(r["argmin_m"])
            ax.scatter([r["argmin_m"]], [r["cv_m"][i]], s=80, marker="*",
                       color=c, edgecolor="k", zorder=5)
    if true_rank is not None: ax.axvline(true_rank, color="C2", linestyle=":", alpha=0.5)
    ax.axvline(spec["k_smooth"], color="k", linestyle="--", alpha=0.5)
    if spec["k_cliff"] is not None: ax.axvline(spec["k_cliff"], color="C1", alpha=0.5)
    ax.set_xlabel("rank"); ax.set_ylabel("M MSE")
    ax.set_title("M CV curves"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle(f"Recipe K, k_cv sweep — {label}, δ={DELTA}\n"
                 f"p_floor={P_FLOOR}, M_min={M_MIN}, n-adaptive p_max, B=20",
                 fontsize=12)
    plt.tight_layout()
    fname = os.path.join(FIG_DIR, f"{safe_label}_kcv_sweep.png")
    fig.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log(f"  saved sweep figure: {fname}")


# ============================================================
# Main loop: for each dataset, sweep k_cv. Cache spectral pass.
# ============================================================
spectral_cache = {}
all_rows = []

for ds_idx, (label, build_fn) in enumerate(datasets):
    log(f"\n--- Loading + spectral pass: {label} ---")
    out = build_fn()
    if isinstance(out, tuple):
        S, true = out
    else:
        S = out; true = None
    t0 = time.time()
    spec = spectral_pass(S, B=20, smooth_window=10)
    spec["k_cut"] = spec["k_smooth"]
    log(f"  spectral pass {time.time()-t0:.1f}s, k_cut={spec['k_smooth']}, k_cliff={spec['k_cliff']}")
    tk = variance_weighted_kappa(spec["kappa_hat"], spec["evals_ref"], spec["k_list"], spec["k_cut"])
    k_max_for_mu = min(int(2 * spec["k_cut"] + 1), S.shape[0] // 4, 100)
    mu_hat, _ = estimate_incoherence(S, k_max_for_mu)
    spectral_cache[label] = (S, true, spec, tk, mu_hat)

    rows_this_ds = []
    for k_idx, k_cv in enumerate(K_CVS):
        row = process_dataset_at_kcv(
            label, S, true, spec, tk, mu_hat, k_cv,
            ds_idx, len(datasets), k_idx, len(K_CVS)
        )
        rows_this_ds.append(row)
        all_rows.append(row)

        # Save running combined pkl
        with open(os.path.join(RESULTS_DIR, "recipe_K_results.pkl"), "wb") as f:
            pickle.dump(all_rows, f)
        slim = [{k: v for k, v in r.items()
                 if k not in ("cv_v", "cv_m", "rank_grid", "delta_emp", "p_grid")}
                for r in all_rows]
        with open(os.path.join(RESULTS_DIR, "recipe_K_summary.json"), "w") as f:
            json.dump(slim, f, indent=2, default=str)

    # Plot combined k_cv sweep figure for this dataset
    safe_label = label.replace(" ", "_").replace("=", "")
    plot_dataset_kcv_sweep(label, rows_this_ds, spec, true, tk, mu_hat, safe_label)

    log(f"\n>>> DATASET {label} COMPLETE (all k_cv) <<<")


# ============================================================
# Final table
# ============================================================
log(f"\n=== Recipe K test done {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
log(f"\nFinal results table:")
log(f"{'dataset':18s} | {'n':4s} | {'k_cut':5s} | {'k_cv':4s} | {'κ̃':>7s} | "
    f"{'p_K_raw':>7s} | {'p_K':>6s} | {'floor':>5s} | {'p_max':>6s} | {'p_cv':>6s} | "
    f"{'V':>4s} | {'V_1SE':>5s} | {'M':>4s} | {'M_1SE':>5s}")
log("-" * 145)
for r in all_rows:
    log(f"{r['label']:18s} | {r['n']:4d} | {r['k_cut']:5d} | {r['k_cv']:4d} | "
        f"{r['tilde_kappa']:7.4f} | {r['p_K_raw']:7.4f} | {r['p_K']:6.4f} | "
        f"{'YES' if r['floor_binding'] else 'no':>5s} | {r['p_max_used']:6.4f} | "
        f"{r['p_cv']:6.4f} | {r['argmin_v']:>4d} | {r['argmin_v_1se']:>5d} | "
        f"{r['argmin_m']:>4d} | {r['argmin_m_1se']:>5d}")

LOG.close()
