"""
CLIP_RBF rmax=150 extended diagnostic — analog of the HD2 censoring diagnostic.

If argmin scales with rmax (60 → ~120) → censoring confirmed.
If argmin stabilizes at some r < 150 → CLIP has a real high rank.
"""
import os, sys, time, pickle, warnings
import numpy as np
from scipy.spatial.distance import cdist

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
from _representations import fit_softimpute

OUT_DIR = os.path.join(ROOT, "output", "pstar_gap", "pstar_gap_C_clip_extended")
PKL_DIR = os.path.join(OUT_DIR, "raw")
os.makedirs(PKL_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "clip_extended.log")
LOG = open(LOG_PATH, "w", buffering=1)
def log(msg):
    print(msg, flush=True); LOG.write(msg + "\n")

N_JOBS = 95
N_REPS = 5
DELTA = 0.10
P_FLOOR = 0.5
P_FLOOR_MODE = "adaptive"
M_MIN = 2000
P_SWEEP = [0.30, 0.50, 0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.93, 0.95]

# Coarser grid out to 150
RANKS_EXT = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150]


def load_clip():
    Df = np.load(os.path.join(DATA_DIR, "model_features", "clip_vit_b_32.npy"))
    dist = cdist(Df, Df, "euclidean")
    bw = float(np.median(dist))
    return np.exp(-(dist / bw) ** 2)


def score_off_diag(S, S_hat, mask):
    n = S.shape[0]
    diag = np.eye(n, dtype=bool)
    sm = mask & ~diag
    if sm.sum() == 0:
        return float("nan")
    return float(np.mean((S_hat[sm] - S[sm]) ** 2))


def _fit_score(S, fitter, r, vm, seed):
    try:
        S_hat = fitter(S, r, vm, seed=seed)
    except Exception:
        return float("nan")
    if not np.all(np.isfinite(S_hat)):
        return float("nan")
    return score_off_diag(S, S_hat, vm)


def mc_curve(S, ranks, p, n_reps, seed=0, n_jobs=N_JOBS):
    rng = check_random_state(seed)
    val_masks = [mask_missing_entries(S, float(p), rng, missing_values=np.nan)
                   for _ in range(n_reps)]
    jobs = [(ri, rep, val_masks[rep], r)
              for rep in range(n_reps) for ri, r in enumerate(ranks)]
    scores = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_fit_score)(S, fit_softimpute, r, vm, seed) for (_, _, vm, r) in jobs)
    cv = np.full((len(ranks), n_reps), np.nan)
    for (ri, rep, _, _), s in zip(jobs, scores):
        cv[ri, rep] = s
    mean = np.nanmean(cv, axis=1)
    se = np.nanstd(cv, axis=1, ddof=1) / np.sqrt(np.maximum(np.sum(~np.isnan(cv), axis=1), 1)) if n_reps > 1 else np.zeros_like(mean)
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


log(f"=== CLIP_RBF rmax=150 extended diag — start {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
log(f"ranks: {RANKS_EXT} ({len(RANKS_EXT)} pts)")
t_global = time.time()

S = load_clip()
n = S.shape[0]
log(f"CLIP_RBF shape={S.shape}, tr(S)={np.trace(S):.3f}")

spec = spectral_pass(S, B=20, smooth_window=10)
spec["k_cut"] = spec["k_smooth"]
recK = recipe_K(spec, delta=DELTA, k_cv=5, p_floor=P_FLOOR,
                   p_floor_mode=P_FLOOR_MODE, M_min=M_MIN)
log(f"Recipe K: k_cut={spec['k_smooth']}, p*_raw={recK['p_star_raw']:.4f}, "
    f"p_floor={recK['p_floor']:.4f}, p*={recK['p_star']:.4f}, p_cv={recK['p_cv']:.4f}, "
    f"σ̂={recK['sigma_hat']:.4f}")

cell = dict(
    label="CLIP_RBF_ext150", n=n,
    k_cut=int(spec["k_smooth"]), p_star_recipe=float(recK["p_star"]),
    p_star_raw=float(recK["p_star_raw"]), p_cv_recipe=float(recK["p_cv"]),
    p_floor_used=float(recK["p_floor"]), p_floor_mode=P_FLOOR_MODE,
    sigma_hat=float(recK["sigma_hat"]) if recK.get("sigma_hat") else None,
    ranks=RANKS_EXT, p_sweep=P_SWEEP, sweep={},
)

t_total = time.time()
for p in P_SWEEP:
    t = time.time()
    m, se = mc_curve(S, RANKS_EXT, p, n_reps=N_REPS, seed=0)
    am = select_argmin(m, RANKS_EXT); a1se = select_1se(m, se, RANKS_EXT)
    cell["sweep"][p] = dict(mean=m, se=se, argmin=am, argmin_1se=a1se, time=time.time()-t)
    log(f"  p={p:.2f}: argmin={am}, 1SE={a1se}  ({time.time()-t:.1f}s)")

cell["total_time"] = time.time() - t_total
out_path = os.path.join(PKL_DIR, "CLIP_RBF_ext150.pkl")
with open(out_path, "wb") as f:
    pickle.dump(cell, f)
log(f"saved {out_path}  (total {(time.time()-t_total)/60:.1f}min)")

# Decision summary
log("\n--- Decision summary ---")
log(f"rmax_old=60: argmin=60 for all p (censoring)")
log(f"rmax_new=150: argmins:")
for p in P_SWEEP:
    log(f"  p={p:.2f}: argmin={cell['sweep'][p]['argmin']}")
log("If argmins lift toward ~120 → censoring confirmed (HD2-class).")
log("If argmins stabilize at interior r < 150 → CLIP has real high rank.")
LOG.close()
