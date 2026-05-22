"""
Primary experiment: is Recipe K's p^* the correct minimum-sufficient
sampling probability? — per EXPERIMENT_B_PSTAR_round_math.md and
EXPERIMENT_B_PSTAR_round_stats.md.

Setup (BBP-matched, NOT SymmNMF-matched, per round-math):
  Generator: planted Gaussian spikes + Wigner noise:
    S = U Λ U^T + σ E
    U = QR(N(0, I_n×r))                — Haar columns
    Λ = diag(λ_1, ..., λ_r)             — known supercritical eigvals
    E = symmetrise(N(0, I_n×n))         — Wigner with var=1
    σ chosen so spike λ's are well above 2σ√n (BBP threshold)

  Fitter: SoftImpute (convex, unique global optimum) — primary
          PPCA-EM as cross-fitter check
          (NO SymmNMF — its non-convex ADMM confounds the test, see §7.5)

Three falsification tests:

  Test 1: Sharp-transition test (statistician primary)
    Sweep p ∈ [0.10, 0.99] (19 points). Measure SD_seed[r̂(p)].
    Below p* recovery is unstable; above stabilizes.
    Falsifies if no transition near Recipe K's p^* (within ±0.05).

  Test 2: Asymptotic BBP convergence (mathematician primary)
    Vary n ∈ {200, 400, 800, 1600}. Test whether Recipe K's
    p^*_raw(n) → 1/λ_{k*}^2 at rate O(n^{-1/2}).
    Falsifies if no convergence or wrong rate.

  Test 3: Cross-fitter consistency (cheap pre-check)
    p^* is computed from S's spectrum, not the fitter. Recipe K p^*
    should be IDENTICAL across fitters on the same S.
    (This is technically a sanity check on the implementation —
    Recipe K does not depend on fitter, so this should be trivially
    True. We verify it anyway.)

n_seeds=20 paired within (cell, p, n).
n_reps=10 per single-mask CV at each p.

Output: results/experiment_B_v3_pstar_correctness/
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

from sklearn.utils import check_random_state
from joblib import Parallel, delayed

from symmnmf.cross_validation import mask_missing_entries
from _common import spectral_pass, recipe_K
from _representations import fit_softimpute, fit_ppca_em

OUT_DIR = os.path.join(ROOT, "output", "pstar_gap", "pstar_correctness")
PKL_DIR = os.path.join(OUT_DIR, "raw")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PKL_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUT_DIR, "run.log")
LOG = open(LOG_PATH, "a", buffering=1)
def log(msg):
    print(msg, flush=True); LOG.write(msg + "\n")

# ============================================================
# Configuration
# ============================================================
N_SEEDS_PER_N = {400: 10, 800: 5, 1600: 3}
P_SWEEP = [0.10, 0.20, 0.30, 0.35, 0.38, 0.40, 0.42, 0.45, 0.50, 0.55, 0.60, 0.70, 0.85]
N_GRID = [400, 800, 1600]
N_REPS = 5
DELTA = 0.10
P_FLOOR = 0.5
P_FLOOR_MODE = "adaptive"
M_MIN = 2000
N_JOBS = 95
USE_PPCA = False

LAMBDA_C_FACTORS = [1.6, 1.8, 2.0, 2.4, 3.0]


def sparse_rank_grid(rmax):
    """Dense at low ranks (1..5 around true_r=5), step-5 above (10..rmax)."""
    base = list(range(1, 6))
    sparse = list(range(10, rmax + 1, 5))
    return base + sparse


def make_bbp_planted(n, r, lambdas, sigma, seed):
    """Planted spikes + Wigner noise.
    S = U diag(lambdas) U^T + sigma * symmetrise(N(0, I)).
    Returns (S, true_r, lambdas).
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, r))
    U, _ = np.linalg.qr(U)                                    # Haar
    sig = U @ np.diag(lambdas) @ U.T
    G = rng.standard_normal((n, n))
    E = sigma * (G + G.T) / np.sqrt(2)
    return sig + E, r, np.array(lambdas)


def bbp_threshold(lambdas, sigma, n):
    """Bare BBP threshold for spike-k detection: lambda_k * sqrt(p) > sigma * sqrt(n)
    => p_k_BBP = (sigma^2 * n) / lambda_k^2
    For our normalisation, the relevant threshold is per-spike."""
    # Effective spike eigenvalue after Wigner noise: lambda_k vs noise edge 2 sigma sqrt(n)
    # BKYY: spike pops out of bulk iff lambda_k > 2 sigma sqrt(n)
    # But we want "minimum sampling probability" for masked-S BKYY:
    # p_k_BBP = max(0, 4 sigma^2 n / lambda_k^2)
    edge = 2.0 * sigma * np.sqrt(n)
    p_bbp = np.minimum(1.0, (4 * sigma**2 * n) / (lambdas ** 2))
    return p_bbp, edge


# ============================================================
# CV scoring
# ============================================================
def score_off_diag(S, S_hat, mask):
    n = S.shape[0]
    diag = np.eye(n, dtype=bool)
    sm = mask & ~diag
    if sm.sum() == 0:
        return float("nan")
    return float(np.mean((S_hat[sm] - S[sm]) ** 2))


def _fit_score(S, fitter, r, holdout_mask, V_mask, seed):
    try:
        S_hat = fitter(S, r, holdout_mask, seed=seed)
    except Exception:
        return float("nan")
    if not np.all(np.isfinite(S_hat)):
        return float("nan")
    return score_off_diag(S, S_hat, V_mask)


def mc_curve(S, fitter, ranks, p, n_reps, seed=0, n_jobs=N_JOBS):
    n = S.shape[0]
    rng = check_random_state(seed)
    val_masks = [mask_missing_entries(S, float(p), rng, missing_values=np.nan)
                   for _ in range(n_reps)]
    jobs = [(ri, rep, val_masks[rep], r)
              for rep in range(n_reps) for ri, r in enumerate(ranks)]
    scores = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_fit_score)(S, fitter, r, vm, vm, seed)
        for (_, _, vm, r) in jobs)
    cv = np.full((len(ranks), n_reps), np.nan)
    for (ri, rep, _, _), s in zip(jobs, scores):
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
# Configuration: BBP-matched cell with λ ∝ √n (constant per-spike difficulty).
# λ_k = c_k · 2σ√n  →  p_BBP_k = 1/c_k², constant across n.
# c = [1.4, 1.6, 1.8, 2.2, 2.8]  →  p_BBP = [0.510, 0.391, 0.309, 0.207, 0.128].
# Smallest spike sits at p_BBP_min = 0.510 — exactly where the floor mattered.
# ============================================================
R_TRUE = 5
SIGMA = 1.0


def lambdas_for_n(n, sigma=SIGMA, c_factors=LAMBDA_C_FACTORS):
    edge = 2.0 * sigma * np.sqrt(n)
    return [float(c * edge) for c in c_factors]


# ============================================================
# Per-seed run
# ============================================================
def run_seed(n, seed):
    """For one (n, seed): generate S, run sweep over p with both fitters."""
    out_path = os.path.join(PKL_DIR, f"n{n}_seed{seed}.pkl")
    if os.path.exists(out_path):
        return

    log(f"  n={n}, seed={seed}: generating planted spikes...")
    t0 = time.time()
    LAMBDAS = lambdas_for_n(n)
    S, true_r, lambdas = make_bbp_planted(n, R_TRUE, LAMBDAS, SIGMA, seed)
    p_bbp_per_spike, noise_edge = bbp_threshold(lambdas, SIGMA, n)
    p_bbp_min = float(p_bbp_per_spike[-1])
    log(f"    n={n}: noise_edge={noise_edge:.1f}, "
        f"lambda_min={lambdas[-1]:.1f}, p_BBP_min={p_bbp_min:.4f}")

    spec = spectral_pass(S, B=20, smooth_window=10)
    spec["k_cut"] = spec["k_smooth"]
    recK = recipe_K(spec, delta=DELTA, k_cv=5, p_floor=P_FLOOR,
                       p_floor_mode=P_FLOOR_MODE, M_min=M_MIN)
    p_star_recipe = float(recK["p_star"])
    p_star_raw = float(recK["p_star_raw"])
    p_cv_recipe = float(recK["p_cv"])
    p_floor_used = float(recK["p_floor"])
    sigma_hat = recK.get("sigma_hat")
    k_cut = int(spec["k_smooth"])
    log(f"    Recipe K (mode={P_FLOOR_MODE}): k_cut={k_cut}, "
        f"p_star_raw={p_star_raw:.4f}, p_floor={p_floor_used:.4f}, "
        f"p_star={p_star_recipe:.4f}, p_cv={p_cv_recipe:.4f}, "
        f"sigma_hat={sigma_hat:.4f}")

    rmax = min(n // 4, max(40, k_cut * 2 + 10, true_r * 2 + 20))
    ranks = sparse_rank_grid(rmax)

    cell_seed = dict(
        n=n, seed=seed, true_r=true_r, lambdas=lambdas.tolist(), sigma=SIGMA,
        noise_edge=float(noise_edge), p_bbp_min=p_bbp_min,
        p_bbp_per_spike=p_bbp_per_spike.tolist(),
        k_cut=k_cut, p_star_recipe=p_star_recipe,
        p_star_raw=p_star_raw, p_cv_recipe=p_cv_recipe,
        p_floor_used=p_floor_used, p_floor_mode=P_FLOOR_MODE,
        sigma_hat=float(sigma_hat) if sigma_hat is not None else None,
        ranks=ranks, sweeps={"softimp": {}},
    )

    for p in P_SWEEP:
        t = time.time()
        m, se = mc_curve(S, fit_softimpute, ranks, p, n_reps=N_REPS,
                            seed=seed, n_jobs=N_JOBS)
        am = select_argmin(m, ranks); a1se = select_1se(m, se, ranks)
        cell_seed["sweeps"]["softimp"][p] = dict(
            mean=m, se=se, argmin=am, argmin_1se=a1se,
            time=time.time() - t)
        log(f"    softimp p={p:.2f}: argmin={am}, 1SE={a1se}  ({time.time()-t:.1f}s)")

    if USE_PPCA:
        cell_seed["sweeps"]["ppca"] = {}
        for p in P_SWEEP:
            t = time.time()
            m, se = mc_curve(S, fit_ppca_em, ranks, p, n_reps=N_REPS,
                                seed=seed, n_jobs=N_JOBS)
            am = select_argmin(m, ranks); a1se = select_1se(m, se, ranks)
            cell_seed["sweeps"]["ppca"][p] = dict(
                mean=m, se=se, argmin=am, argmin_1se=a1se,
                time=time.time() - t)
            log(f"    ppca p={p:.2f}: argmin={am}, 1SE={a1se}  ({time.time()-t:.1f}s)")

    cell_seed["seed_total_time"] = time.time() - t0
    with open(out_path, "wb") as f:
        pickle.dump(cell_seed, f)
    log(f"  n={n}, seed={seed}: done ({(time.time()-t0)/60:.1f}min)")


# ============================================================
# Main
# ============================================================
log(f"=== p^* correctness experiment — start {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
log(f"Generator: planted spikes with λ_k = c_k · 2σ√n, c={LAMBDA_C_FACTORS}, σ={SIGMA}")
log(f"  → p_BBP_k = 1/c_k² = {[round(1/c**2, 4) for c in LAMBDA_C_FACTORS]} (constant in n)")
log(f"Fitters: SoftImpute (primary{', PPCA-EM' if USE_PPCA else ''})")
log(f"P sweep: {P_SWEEP}  ({len(P_SWEEP)} points, dense around p_BBP_min={1/LAMBDA_C_FACTORS[0]**2:.4f})")
log(f"Floor mode: {P_FLOOR_MODE}")
log(f"N grid: {N_GRID}  (asymptotic test)")
log(f"Seeds per n: {N_SEEDS_PER_N}")
log(f"Reps per (p, seed): {N_REPS}")
log(f"Output: {OUT_DIR}")

t_global = time.time()
for n in N_GRID:
    n_seeds = N_SEEDS_PER_N.get(n, 20)
    log(f"\n{'='*72}")
    log(f"N = {n}  (n_seeds={n_seeds}, elapsed: {(time.time()-t_global)/60:.1f}min)")
    log(f"{'='*72}")
    for seed in range(n_seeds):
        run_seed(n, seed)

log(f"\n=== ALL DONE === total {(time.time()-t_global)/60:.1f}min")
LOG.close()
