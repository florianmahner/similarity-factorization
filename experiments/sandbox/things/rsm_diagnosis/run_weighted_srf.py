"""Test weighted SRF: continuous observation-count weights in V-update."""
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Import the local modified SRF
sys.path.insert(0, str(Path(__file__).parent))
from srf_weighted import SRF

from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "things" / "triplets_47"
KAPPA_RANKS = PROJECT_ROOT / "experiments" / "analyses" / "things_behavior" / "lowdata" / "srf" / "outputs" / "kappa_alpha0" / "ranks.json"
N = 1854


def acc(w, trips):
    ei, ej, ek = w[trips[:, 0]], w[trips[:, 1]], w[trips[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def main():
    val = np.loadtxt(DATA_DIR / "validationset.txt").astype(int)
    train = np.loadtxt(DATA_DIR / "train_90.txt").astype(int)
    rank = json.load(open(KAPPA_RANKS))["100"]
    log.info(f"Rank: {rank}")

    # Build count RSM (alpha=0)
    ii, jj, kk = train[:, 0], train[:, 1], train[:, 2]
    counts = np.zeros((N, N))
    shown = np.zeros((N, N))
    np.add.at(counts, (ii, jj), 1); np.add.at(counts, (jj, ii), 1)
    for a, b in [(ii, jj), (ii, kk), (jj, kk)]:
        np.add.at(shown, (a, b), 1); np.add.at(shown, (b, a), 1)

    s = np.divide(counts, shown, out=np.full((N, N), np.nan), where=shown > 0)
    np.fill_diagonal(s, 1.0)

    # VICE baseline
    vice_dir = PROJECT_ROOT / "experiments" / "analyses" / "things_behavior" / "lowdata" / "vice" / "outputs" / "models"
    vice_accs = []
    for seed in range(10):
        d = np.load(
            vice_dir / f"vice_100pct_part0_seed{seed}" / "variational" / "4.12mio" / "sslab" / "90" / "256" / "1.0" / str(seed) / "params" / "parameters.npz",
            allow_pickle=True,
        )
        w = np.maximum(d["pruned_q_mu"], 0)
        vice_accs.append(acc(w, val))
    log.info(f"VICE: {np.mean(vice_accs):.4f} +/- {np.std(vice_accs):.4f}")

    # Build weight variants
    median_shown = np.median(shown[shown > 0])
    weights_obs = shown / median_shown
    weights_obs[shown == 0] = 0.0
    np.fill_diagonal(weights_obs, weights_obs.max())

    p = np.divide(counts, shown, out=np.full((N, N), 0.5), where=shown > 0)
    np.fill_diagonal(p, 1.0)
    fisher = shown / np.maximum(p * (1 - p), 0.01)
    fisher[shown == 0] = 0.0
    fisher = fisher / np.median(fisher[fisher > 0])
    np.fill_diagonal(fisher, fisher.max())

    def _fit(name, weight_matrix):
        model = SRF(rank=rank, random_state=42, max_outer=500, max_inner=30, tol=1e-4, verbose=0)
        if weight_matrix is not None:
            model._input_weights = weight_matrix
        w = model.fit_transform(s)
        return name, acc(w, val)

    from joblib import Parallel, delayed
    conditions = [
        ("unweighted", None),
        ("obs x0.5", weights_obs * 0.5),
        ("obs x1.0", weights_obs * 1.0),
        ("obs x2.0", weights_obs * 2.0),
        ("obs x5.0", weights_obs * 5.0),
        ("fisher x0.5", fisher * 0.5),
        ("fisher x1.0", fisher * 1.0),
        ("fisher x2.0", fisher * 2.0),
    ]

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(_fit)(name, w) for name, w in conditions
    )

    for name, a in results:
        log.info(f"  {name:15s}: {a:.4f}")

    log.info(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
