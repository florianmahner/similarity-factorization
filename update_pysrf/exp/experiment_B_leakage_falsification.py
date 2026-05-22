"""
Leakage falsification experiment per EXPERIMENT_B_AUDIT.md §7.6.

Tests whether plain k-fold CV's V-MSE is downward-biased relative to true
generalization MSE when (k-1)/k > p^star, while pre-mask + k-fold CV and
Monte Carlo at p_star remain calibrated. Includes MC-matchedK as a
confounder control to dissociate train fraction from V-mask size effects.

Hypotheses (directional, per-cell):
  H1 (level bias) — b_k(r) = E[V_k(r)] - MSE_true(r). Plain k-fold:
    b_k < 0 when (k-1)/k > p_star. PreKfold + MC-pstar: bounded.
  H2 (argmin shift) — Delta r between plain and pre-mask non-zero,
    magnitude grows with k past p_star.
  H3 (monotone in k) — Plain-k-fold argmin monotone in k on leakage side.
  H4 (placebo) — When (k-1)/k <= p_star, plain and pre-mask agree.
    If H4 fails, leakage is NOT the diagnosis.

Cells (selected from partial sweep results to span p_star):
  Low p* ≈ 0.5  : A3_anisotropic, R2_heteroscedastic, G4_blockhetero
  Mid p* ~ 0.7  : R3_extreme_coh, X_10block_n200
  High p* ≥ 0.9 : A5_bbp_twin, E4_nested16, X_DCSBM_K20_subset

Output: results/experiment_B_v3_leakage_falsification/
"""
import os, sys, time, pickle, json, warnings
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

from symmnmf.cross_validation import mask_missing_entries
from _common import spectral_pass, recipe_K, split_omega_into_folds
from _representations import fit_symmnmf_admm
from _synthetic_extras import (
    make_anisotropic, make_heteroscedastic, make_block_hetero,
    make_extreme_coherence, _x_10block, make_bbp_twin,
    make_nested_blocks, make_DCSBM_K20,
)

OUT_DIR = os.path.join(ROOT, "output", "experiment_B_leakage_falsification")
PKL_DIR = os.path.join(OUT_DIR, "raw")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PKL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUT_DIR, "run.log")
LOG = open(LOG_PATH, "a", buffering=1)
def log(msg):
    print(msg, flush=True); LOG.write(msg + "\n")

# Configuration
N_SEEDS = 5                            # paired within-cell; ≥ 8 needed for 1-rank shift detection
K_VALUES = [2, 5, 10]                  # plain k-fold and pre-mask k-fold
N_REPS_MC_LIST = [1, 5, 20]            # Monte Carlo at p_star
DELTA = 0.10
P_FLOOR = 0.5
M_MIN = 2000
N_JOBS = 95


# ----- Cells -----
# (label, generator(seed)→S, group_label)
CELLS = [
    # Low p_star (≈0.5) — k=5,10 should leak in plain k-fold per claim
    ("A3_anisotropic",     lambda seed: make_anisotropic(n=300, r=5, snr=4.0, rho=0.7, seed=seed),     "low"),
    ("R2_heteroscedastic", lambda seed: make_heteroscedastic(n=300, r=5, snr=4.0, sigma_lo=0.3, sigma_hi=1.5, seed=seed), "low"),
    ("G4_blockhetero",     lambda seed: make_block_hetero(n=400, r=5, snr=4.0, seed=seed),             "low"),
    # Medium p_star (0.7-0.8) — k=10 should leak; k=5 borderline
    ("R3_extreme_coh",     lambda seed: make_extreme_coherence(n=300, r=5, snr=5.0, n_active=10, seed=seed), "mid"),
    ("X_10block_n200",     lambda seed: _x_10block(n=200, seed=seed),                                  "mid"),
    # High p_star (≥0.9) — H4 placebo: plain k-fold legal, should agree with pre-mask
    ("A5_bbp_twin",        lambda seed: make_bbp_twin(n=400, lam_inside=1.6, lam_outside=0.7, seed=seed), "high"),
    ("E4_nested16",        lambda seed: make_nested_blocks(n=400, super_K=4, sub_K=4, seed=seed),      "high"),
]


# ============================================================
# CV scoring
# ============================================================
def score_off_diag(S_eval, S_hat, mask):
    """V-MSE on entries where mask is True, off-diagonal."""
    n = S_eval.shape[0]
    diag = np.eye(n, dtype=bool)
    sm = mask & ~diag
    if sm.sum() == 0:
        return float("nan")
    return float(np.mean((S_hat[sm] - S_eval[sm]) ** 2))


def _fit(S, fitter, r, holdout_mask, seed):
    try:
        S_hat = fitter(S, r, holdout_mask, seed=seed)
    except Exception:
        return None
    if not np.all(np.isfinite(S_hat)):
        return None
    return S_hat


def _fit_score(S, fitter, r, holdout_mask, V_mask, seed, S_eval=None):
    """Fit on holdout-masked S, score on V_mask of S_eval (or S if None)."""
    S_hat = _fit(S, fitter, r, holdout_mask, seed)
    if S_hat is None:
        return float("nan")
    target = S if S_eval is None else S_eval
    return score_off_diag(target, S_hat, V_mask)


def true_gen_curve(S, S_fresh, fitter, ranks, seed, n_jobs=N_JOBS):
    """True generalization MSE: fit on full S, score on full S_fresh."""
    n = S.shape[0]
    empty_mask = np.zeros((n, n), dtype=bool)
    full_mask = np.ones((n, n), dtype=bool)
    jobs = [(ri, r) for ri, r in enumerate(ranks)]

    def _one(r):
        S_hat = _fit(S, fitter, r, empty_mask, seed)
        return score_off_diag(S_fresh, S_hat, full_mask) if S_hat is not None else float("nan")

    out = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_one)(r) for (_, r) in jobs)
    return np.array(out)


def plain_kfold_curve(S, fitter, ranks, k_inner, n_reps=1, seed=0,
                       n_jobs=N_JOBS):
    """Plain k-fold (no outer pre-mask). Train (k-1)/k, test 1/k."""
    n = S.shape[0]
    rng = check_random_state(seed)
    no_mask = np.zeros((n, n), dtype=bool)
    cv = np.full((len(ranks), n_reps, k_inner), np.nan)
    for rep in range(n_reps):
        val_masks = split_omega_into_folds(no_mask, k_inner, rng)
        if val_masks is None: continue
        jobs = []
        for fi, V_i in enumerate(val_masks):
            for ri, r in enumerate(ranks):
                jobs.append((ri, fi, V_i, V_i, r))
        scores = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_fit_score)(S, fitter, r, hm, vm, seed)
            for (_, _, hm, vm, r) in jobs)
        for (ri, fi, _, _, _), s in zip(jobs, scores):
            cv[ri, rep, fi] = s
    flat = cv.reshape(len(ranks), -1)
    mean = np.nanmean(flat, axis=1)
    se = np.nanstd(flat, axis=1, ddof=1) / np.sqrt(np.maximum(
        np.sum(~np.isnan(flat), axis=1), 1))
    return mean, se


def prekfold_curve(S, fitter, ranks, k_inner, p_star, n_reps=1, seed=0,
                     n_jobs=N_JOBS):
    """Pre-mask + k-fold. p_cv = p_star * k/(k-1) capped at 0.99.
    Per-fold training fraction = p_star exactly."""
    n = S.shape[0]
    rng = check_random_state(seed)
    p_cv_target = float(p_star) * k_inner / max(k_inner - 1, 1)
    if p_cv_target > 0.99:
        return np.full(len(ranks), np.nan), np.full(len(ranks), np.nan), p_cv_target
    cv = np.full((len(ranks), n_reps, k_inner), np.nan)
    for rep in range(n_reps):
        M_outer = mask_missing_entries(S, p_cv_target, rng,
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
            delayed(_fit_score)(S, fitter, r, hm, vm, seed)
            for (_, _, hm, vm, r) in jobs)
        for (ri, fi, _, _, _), s in zip(jobs, scores):
            cv[ri, rep, fi] = s
    flat = cv.reshape(len(ranks), -1)
    mean = np.nanmean(flat, axis=1)
    se = np.nanstd(flat, axis=1, ddof=1) / np.sqrt(np.maximum(
        np.sum(~np.isnan(flat), axis=1), 1))
    return mean, se, p_cv_target


def mc_curve(S, fitter, ranks, p, n_reps, seed=0, n_jobs=N_JOBS):
    """Monte Carlo single-mask CV at observed_fraction = p with n_reps."""
    n = S.shape[0]
    rng = check_random_state(seed)
    cv = np.full((len(ranks), n_reps), np.nan)
    for rep in range(n_reps):
        val_mask = mask_missing_entries(S, float(p), rng,
                                          missing_values=np.nan)
        jobs = [(ri, val_mask, r) for ri, r in enumerate(ranks)]
        scores = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_fit_score)(S, fitter, r, vm, vm, seed)
            for (_, vm, r) in jobs)
        for (ri, _, _), s in zip(jobs, scores):
            cv[ri, rep] = s
    mean = np.nanmean(cv, axis=1)
    if n_reps > 1:
        se = np.nanstd(cv, axis=1, ddof=1) / np.sqrt(np.maximum(
            np.sum(~np.isnan(cv), axis=1), 1))
    else:
        se = np.zeros_like(mean)
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


# ============================================================
# Per-cell × per-seed run
# ============================================================
def run_seed(label, gen, group, seed):
    out_path = os.path.join(PKL_DIR, f"{label}_seed{seed}.pkl")
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            return pickle.load(f)

    log(f"  seed={seed}: generating + spectral pass + recipe K...")
    t0 = time.time()
    S, true_r = gen(seed)
    n = S.shape[0]
    spec = spectral_pass(S, B=20, smooth_window=10)
    spec["k_cut"] = spec["k_smooth"]
    recK = recipe_K(spec, delta=DELTA, k_cv=5,
                    p_floor=P_FLOOR, p_floor_mode="adaptive",
                    M_min=M_MIN)
    p_star = float(recK["p_star"])
    p_cv = float(recK["p_cv"])
    k_cut = int(spec["k_smooth"])
    rmax = min(n // 4, max(40, k_cut * 2 + 10, (true_r if true_r else 0) * 2 + 10))
    ranks = list(range(1, rmax + 1))
    log(f"    n={n}, true_r={true_r}, k_cut={k_cut}, p*={p_star:.3f}, p_cv={p_cv:.3f}, ranks=[1..{rmax}]")

    # Generate fresh test draw with shifted seed
    S_fresh, _ = gen(seed + 1000)

    # True generalization MSE
    log(f"    computing true gen MSE on fresh draw...")
    t_true = time.time()
    true_curve = true_gen_curve(S, S_fresh, fit_symmnmf_admm, ranks, seed,
                                 n_jobs=N_JOBS)
    log(f"      done ({time.time()-t_true:.1f}s)")

    cell_seed = dict(
        label=label, group=group, seed=seed, n=n, true_r=true_r,
        k_cut=k_cut, p_star=p_star, p_cv=p_cv, ranks=ranks,
        true_curve=true_curve, protocols={},
    )

    # PlainKfold across k
    for k_inner in K_VALUES:
        train_frac = (k_inner - 1) / k_inner
        leakage_violation = train_frac > p_star
        t = time.time()
        m, se = plain_kfold_curve(S, fit_symmnmf_admm, ranks, k_inner,
                                     n_reps=1, seed=seed, n_jobs=N_JOBS)
        am = select_argmin(m, ranks); a1se = select_1se(m, se, ranks)
        elap = time.time() - t
        cell_seed["protocols"][f"PlainK{k_inner}"] = dict(
            mean=m, se=se, argmin=am, argmin_1se=a1se,
            train_frac=float(train_frac), leakage=leakage_violation,
            time=elap, p=None, k_inner=k_inner)
        log(f"    PlainK{k_inner} (train={train_frac:.2f}, leak={leakage_violation}): "
            f"argmin={am}, 1SE={a1se}  ({elap:.1f}s)")

    # PreKfold across k (drops to NaN if p_cv > 0.99)
    for k_inner in K_VALUES:
        t = time.time()
        m, se, p_cv_target = prekfold_curve(S, fit_symmnmf_admm, ranks,
                                                k_inner, p_star, n_reps=1,
                                                seed=seed, n_jobs=N_JOBS)
        am = select_argmin(m, ranks); a1se = select_1se(m, se, ranks)
        elap = time.time() - t
        cell_seed["protocols"][f"PreK{k_inner}"] = dict(
            mean=m, se=se, argmin=am, argmin_1se=a1se,
            train_frac=float(p_star), p_cv=float(p_cv_target),
            time=elap, k_inner=k_inner)
        log(f"    PreK{k_inner} (p_cv={p_cv_target:.3f}, train={p_star:.2f}): "
            f"argmin={am}, 1SE={a1se}  ({elap:.1f}s)")

    # MC-pstar across n_reps
    for n_reps in N_REPS_MC_LIST:
        t = time.time()
        m, se = mc_curve(S, fit_symmnmf_admm, ranks, p_star, n_reps,
                            seed=seed, n_jobs=N_JOBS)
        am = select_argmin(m, ranks); a1se = select_1se(m, se, ranks) if n_reps > 1 else am
        elap = time.time() - t
        cell_seed["protocols"][f"MCpstar_N{n_reps}"] = dict(
            mean=m, se=se, argmin=am, argmin_1se=a1se,
            train_frac=float(p_star), n_reps=n_reps, time=elap, p=p_star)
        log(f"    MCpstar_N{n_reps} (train={p_star:.2f}): "
            f"argmin={am}, 1SE={a1se}  ({elap:.1f}s)")

    # MC-matchedK: training fraction = (k-1)/k, single rep — confounder control
    for k_inner in K_VALUES:
        train_frac = (k_inner - 1) / k_inner
        t = time.time()
        m, se = mc_curve(S, fit_symmnmf_admm, ranks, train_frac,
                            n_reps=k_inner, seed=seed, n_jobs=N_JOBS)
        am = select_argmin(m, ranks); a1se = select_1se(m, se, ranks)
        elap = time.time() - t
        cell_seed["protocols"][f"MCmatchK{k_inner}"] = dict(
            mean=m, se=se, argmin=am, argmin_1se=a1se,
            train_frac=float(train_frac), n_reps=k_inner, time=elap,
            p=float(train_frac))
        log(f"    MCmatchK{k_inner} (train={train_frac:.2f}, reps={k_inner}): "
            f"argmin={am}, 1SE={a1se}  ({elap:.1f}s)")

    cell_seed["seed_total_time"] = time.time() - t0
    with open(out_path, "wb") as f:
        pickle.dump(cell_seed, f)
    log(f"  seed={seed} done ({time.time()-t0:.1f}s)")
    return cell_seed


# ============================================================
# Main loop
# ============================================================
log(f"=== Leakage falsification — start {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
log(f"Cells: {len(CELLS)}, n_seeds: {N_SEEDS}")
log(f"Protocols per (cell, seed): "
    f"{len(K_VALUES)} PlainK + {len(K_VALUES)} PreK + "
    f"{len(N_REPS_MC_LIST)} MCpstar + {len(K_VALUES)} MCmatchK = "
    f"{len(K_VALUES)*3 + len(N_REPS_MC_LIST)} settings")
log(f"Output: {OUT_DIR}")

t_global = time.time()
for ci, (label, gen, group) in enumerate(CELLS):
    log(f"\n{'='*72}")
    log(f"[{ci+1}/{len(CELLS)}] CELL {label} ({group})  "
        f"(elapsed: {(time.time()-t_global)/60:.1f}min)")
    log(f"{'='*72}")
    for seed in range(N_SEEDS):
        run_seed(label, gen, group, seed)

log(f"\n=== ALL DONE ===  total {(time.time()-t_global)/60:.1f}min")
LOG.close()
