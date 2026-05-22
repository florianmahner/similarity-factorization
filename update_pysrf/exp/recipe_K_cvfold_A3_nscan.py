"""CV-fold sensitivity demo — A3_anisotropic n-scan (2026-05-11 follow-up).

Replicates the A3 result from `recipe_K_cvfold_sensitivity_demo.py` at
n ∈ {300 (baseline), 600, 1000}. Tests whether the "plain k-fold drifts
with k_cv, PreK is invariant" pattern survives at larger n, or whether the
n=300 result was a small-sample artifact.

Setup matches the original demo exactly except for n.
Output: results/recipe_K_cvfold_A3_nscan_2026-05-11/
"""
import os, sys, time, json, pickle, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))   # RECIPE_K/exp/
ROOT = os.path.dirname(HERE)                          # RECIPE_K/
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from _common import (spectral_pass, recipe_K, env_info,
                     split_omega_into_folds, argmin_safe, argmin_1se_safe)
from _synthetic_extras import make_anisotropic
from symmnmf.cross_validation import mask_missing_entries
from sklearn.utils import check_random_state
from joblib import Parallel, delayed

OUT_DIR = os.path.join(ROOT, "output", "pstar_gap", "cvfold_A3_nscan")
os.makedirs(OUT_DIR, exist_ok=True)
LOG = open(os.path.join(OUT_DIR, "run.log"), "w", buffering=1)
def log(msg):
    print(msg, flush=True); LOG.write(msg + "\n")


def fit_softimpute(S, r, holdout_mask, seed=0, max_iter=80, tol=1e-4):
    X = S.copy(); X[holdout_mask] = 0.0
    obs = ~holdout_mask
    for _ in range(max_iter):
        U, sv, Vt = np.linalg.svd(X, full_matrices=False)
        sv_t = sv.copy(); sv_t[r:] = 0.0
        X_new = (U * sv_t) @ Vt
        X_new[obs] = S[obs]
        if np.linalg.norm(X_new - X) / max(np.linalg.norm(X), 1e-12) < tol:
            X = X_new; break
        X = X_new
    return 0.5 * (X + X.T)


def score_off_diag(S, S_hat, mask):
    n = S.shape[0]; diag = np.eye(n, dtype=bool); sm = mask & ~diag
    if sm.sum() == 0:
        return float("nan")
    return float(np.mean((S_hat[sm] - S[sm]) ** 2))


def _fit_score(S, r, holdout, V, seed):
    try:
        Sh = fit_softimpute(S, r, holdout, seed=seed)
    except Exception:
        return float("nan")
    return score_off_diag(S, Sh, V)


def plain_kfold(S, ranks, k_inner, seed=0, n_jobs=16):
    rng = check_random_state(seed)
    M_empty = np.zeros_like(S, dtype=bool)
    vals = split_omega_into_folds(M_empty, k_inner, rng)
    if vals is None:
        return None, None
    jobs = [(ri, fold, V_i, r)
            for fold, V_i in enumerate(vals)
            for ri, r in enumerate(ranks)]
    scores = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_fit_score)(S, r, V_i, V_i, seed) for (_, _, V_i, r) in jobs)
    cv = np.full((len(ranks), k_inner), np.nan)
    for (ri, fold, _, _), s in zip(jobs, scores):
        cv[ri, fold] = s
    return (np.nanmean(cv, axis=1),
            np.nanstd(cv, axis=1, ddof=1) / np.sqrt(
                np.maximum(np.sum(np.isfinite(cv), axis=1), 1)))


def prekfold(S, ranks, k_inner, p_star, p_max=0.95, seed=0, n_jobs=16):
    rng = check_random_state(seed)
    p_cv = min(p_star * k_inner / max(k_inner - 1, 1), p_max)
    M_outer = mask_missing_entries(S, float(p_cv), rng, missing_values=np.nan)
    vals = split_omega_into_folds(M_outer, k_inner, rng)
    if vals is None:
        return None, None, p_cv
    jobs = [(ri, fold, V_i | M_outer, V_i, r)
            for fold, V_i in enumerate(vals)
            for ri, r in enumerate(ranks)]
    scores = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_fit_score)(S, r, hm, vm, seed) for (_, _, hm, vm, r) in jobs)
    cv = np.full((len(ranks), k_inner), np.nan)
    for (ri, fold, _, _, _), s in zip(jobs, scores):
        cv[ri, fold] = s
    return (np.nanmean(cv, axis=1),
            np.nanstd(cv, axis=1, ddof=1) / np.sqrt(
                np.maximum(np.sum(np.isfinite(cv), axis=1), 1)),
            p_cv)


N_VALUES = [300, 600, 1000]
K_VALUES = [2, 5, 10]


def run_one_n(n_val, seed=0):
    log(f"\n=== A3_anisotropic n={n_val}, seed={seed} ===")
    S, true_r = make_anisotropic(n=n_val, r=5, snr=4.0, rho=0.7, seed=seed)
    log(f"  generated S {S.shape}, true_r={true_r}")
    t = time.time()
    spec = spectral_pass(S, B=20, show_progress=False)
    log(f"  spectral_pass {time.time()-t:.1f}s, k_cut={spec['k_cut']}, flag={spec['flag']}")
    recK = recipe_K(spec, delta=0.10, k_cv=5, p_floor=0.5, M_min=2000)
    p_star = float(recK["p_star"])
    log(f"  p_star={p_star:.3f}, floor_binding={recK['floor_binding']}, "
        f"cap_binding={recK['cap_binding']}, k_cv_min_unclipped={recK['k_cv_min_unclipped']}")

    rmax = min(n_val // 4, max(40, spec["k_cut"] + 20, true_r * 2 + 10))
    ranks = list(range(1, rmax + 1))

    rows = []
    for k_inner in K_VALUES:
        train_plain = (k_inner - 1) / k_inner
        t = time.time()
        m_p, sem_p = plain_kfold(S, ranks, k_inner, seed=seed)
        am_p = argmin_safe(m_p, ranks); am_p_1se = argmin_1se_safe(m_p, ranks, sem=sem_p)
        log(f"  Plain k={k_inner}: train={train_plain:.2f} (Δp*={train_plain-p_star:+.2f}), "
            f"argmin={am_p}, 1SE={am_p_1se}  ({time.time()-t:.1f}s)")

        t = time.time()
        m_pre, sem_pre, p_cv = prekfold(S, ranks, k_inner, p_star, seed=seed)
        am_pre = argmin_safe(m_pre, ranks); am_pre_1se = argmin_1se_safe(m_pre, ranks, sem=sem_pre)
        log(f"  PreK  k={k_inner}: p_cv={p_cv:.3f}, train~p*={p_star:.2f}, "
            f"argmin={am_pre}, 1SE={am_pre_1se}  ({time.time()-t:.1f}s)")

        rows.append(dict(n=n_val, k_inner=k_inner, train_plain=train_plain,
                         p_star=p_star, p_cv=p_cv,
                         plain_argmin=am_p, plain_1se=am_p_1se,
                         pre_argmin=am_pre, pre_1se=am_pre_1se,
                         plain_curve=list(m_p), pre_curve=list(m_pre),
                         plain_sem=list(sem_p), pre_sem=list(sem_pre),
                         ranks=ranks))
    return rows, spec, recK


def plot_n_scan(all_rows):
    """3-row figure: each row = one n. Left col = plain, right col = pre."""
    fig, axes = plt.subplots(len(N_VALUES), 2, figsize=(14, 4 * len(N_VALUES)))
    K_COLORS = {2: "C3", 5: "C0", 10: "C2"}

    for i, n_val in enumerate(N_VALUES):
        rows_n = [r for r in all_rows if r["n"] == n_val]
        ax_l = axes[i, 0]; ax_r = axes[i, 1]
        ranks = rows_n[0]["ranks"]

        for r in rows_n:
            k = r["k_inner"]; c = K_COLORS[k]
            ax_l.plot(ranks, r["plain_curve"], color=c, marker=".",
                      label=f"k={k}: train={r['train_plain']:.2f}, "
                            f"argmin={r['plain_argmin']} (1SE={r['plain_1se']})")
            am_idx = ranks.index(r["plain_argmin"]) if r["plain_argmin"] in ranks else None
            if am_idx is not None:
                ax_l.scatter([r["plain_argmin"]], [r["plain_curve"][am_idx]],
                             s=80, marker="*", color=c, edgecolor="k", zorder=5)

            ax_r.plot(ranks, r["pre_curve"], color=c, marker=".",
                      label=f"k={k}: p_cv={r['p_cv']:.2f}, "
                            f"argmin={r['pre_argmin']} (1SE={r['pre_1se']})")
            am_idx = ranks.index(r["pre_argmin"]) if r["pre_argmin"] in ranks else None
            if am_idx is not None:
                ax_r.scatter([r["pre_argmin"]], [r["pre_curve"][am_idx]],
                             s=80, marker="*", color=c, edgecolor="k", zorder=5)

        for ax, title in [(ax_l, "Plain k-fold"), (ax_r, "Pre-mask k-fold (Recipe K)")]:
            ax.axvline(5, color="C2", linestyle=":", alpha=0.5, label="true_r=5")
            ax.set_xlabel("rank r"); ax.set_ylabel("V CV-MSE")
            ax.set_title(f"{title} — A3 n={n_val}, p*={rows_n[0]['p_star']:.3f}")
            ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_yscale("log")

    fig.suptitle("A3_anisotropic n-scan: plain k-fold drifts with k_cv,\n"
                 "Recipe K's pre-mask k-fold gives the same answer at every k_cv", fontsize=12)
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, "A3_nscan.png")
    fig.savefig(fname, dpi=110, bbox_inches="tight"); plt.close(fig)
    log(f"\nsaved: {fname}")


def main():
    t0 = time.time()
    log(f"=== A3_anisotropic n-scan — {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    with open(os.path.join(OUT_DIR, "env.txt"), "w") as f:
        json.dump(env_info(), f, indent=2)

    all_rows = []
    for n_val in N_VALUES:
        rows, spec, recK = run_one_n(n_val, seed=0)
        all_rows.extend(rows)
        with open(os.path.join(OUT_DIR, f"n{n_val}.pkl"), "wb") as f:
            pickle.dump(dict(rows=rows, spec=spec, recK=recK), f)

    log("\n\n=== SUMMARY ===")
    log(f"{'n':>5}|{'k_in':>5}|{'plain_train':>11}|{'p_star':>7}|"
        f"{'plain_argmin':>12}|{'plain_1SE':>9}|"
        f"{'pre_argmin':>10}|{'pre_1SE':>8}")
    log("-" * 100)
    for r in all_rows:
        log(f"{r['n']:>5}|{r['k_inner']:>5}|{r['train_plain']:>11.2f}|"
            f"{r['p_star']:>7.3f}|{r['plain_argmin']:>12}|{r['plain_1se']:>9}|"
            f"{r['pre_argmin']:>10}|{r['pre_1se']:>8}")
    plot_n_scan(all_rows)
    log(f"\nTotal wall-clock: {time.time()-t0:.1f}s")
    log(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
