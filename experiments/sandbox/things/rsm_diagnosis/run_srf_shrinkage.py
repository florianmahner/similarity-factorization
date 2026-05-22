"""Post-hoc shrinkage and consensus for SRF embeddings.

Option 2: Shrink dimensions by cross-seed reliability.
Option 1: Consensus from many seeds (50 fits).
Compare with VICE (pruned_q_mu) on validationset.txt.
"""
import logging
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from pysrf import SRF
from scipy.optimize import linear_sum_assignment

from src.utils import get_output_dir
from src.utils.helpers import compute_similarity_matrix_from_triplets

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "things" / "triplets_47"
KAPPA_DIR = PROJECT_ROOT / "experiments" / "analyses" / "things_behavior" / "lowdata" / "srf" / "outputs" / "kappa_alpha0"
N = 1854


def triplet_acc(w, trips):
    ei, ej, ek = w[trips[:, 0]], w[trips[:, 1]], w[trips[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def align_to_reference(ref, w):
    """Align w's columns to ref using Hungarian matching on correlations."""
    corr = np.corrcoef(ref.T, w.T)[:ref.shape[1], ref.shape[1]:]
    row_ind, col_ind = linear_sum_assignment(-np.abs(corr))
    w_aligned = w[:, col_ind]
    for d in range(w_aligned.shape[1]):
        if corr[row_ind[d], col_ind[d]] < 0:
            w_aligned[:, d] *= -1  # shouldn't happen for non-negative
    return w_aligned


def _fit_one(s, rank, seed):
    model = SRF(rank=rank, random_state=seed, max_outer=1000, max_inner=30, tol=1e-4, verbose=0)
    return model.fit_transform(s)


def main():
    log.info("Loading data...")
    val = np.loadtxt(DATA_DIR / "validationset.txt").astype(int)

    # Build RSM from train_90
    train = np.loadtxt(DATA_DIR / "train_90.txt").astype(int)
    s = compute_similarity_matrix_from_triplets(N, train, alpha=1.0)
    s[np.isnan(s)] = 0.5

    # =========================================================================
    # VICE baseline (pruned_q_mu)
    # =========================================================================
    vice_dir = PROJECT_ROOT / "experiments" / "analyses" / "things_behavior" / "lowdata" / "vice" / "outputs" / "models"
    vice_accs = []
    for seed in range(10):
        d = np.load(
            vice_dir / f"vice_100pct_part0_seed{seed}" / "variational" / "4.12mio" / "sslab" / "90" / "256" / "1.0" / str(seed) / "params" / "parameters.npz",
            allow_pickle=True,
        )
        w = np.maximum(d["pruned_q_mu"], 0)
        vice_accs.append(triplet_acc(w, val))
    log.info(f"VICE (pruned_q_mu): {np.mean(vice_accs):.4f} +/- {np.std(vice_accs):.4f}")

    # =========================================================================
    # Load kappa rank from results
    # =========================================================================
    import json
    with open(KAPPA_DIR / "ranks.json") as f:
        kappa_ranks = json.load(f)
    rank = kappa_ranks["100"]
    log.info(f"Kappa rank at 100%: {rank}")

    # =========================================================================
    # Fit 50 SRF seeds in parallel
    # =========================================================================
    n_seeds = 50
    log.info(f"\nFitting {n_seeds} SRF seeds at rank={rank}...")
    embeddings = Parallel(n_jobs=-1, verbose=10)(
        delayed(_fit_one)(s, rank, seed) for seed in range(n_seeds)
    )

    # Single seed baselines
    single_accs = [triplet_acc(w, val) for w in embeddings[:10]]
    log.info(f"\nSingle SRF (10 seeds): {np.mean(single_accs):.4f} +/- {np.std(single_accs):.4f}")

    # =========================================================================
    # Option 2: Cross-seed reliability shrinkage
    # =========================================================================
    log.info("\n" + "=" * 60)
    log.info("OPTION 2: Reliability shrinkage")
    log.info("=" * 60)

    # Align all embeddings to seed 0
    ref = embeddings[0]
    aligned = [ref] + [align_to_reference(ref, w) for w in embeddings[1:]]

    stack = np.stack(aligned)  # (n_seeds, n_objects, rank)

    # Per-dimension reliability: correlation of dimension across seeds
    # For each dimension, compute mean pairwise correlation across seeds
    for n_used in [10, 20, 50]:
        sub = stack[:n_used]
        mean_emb = sub.mean(axis=0)  # (n_objects, rank)

        # Reliability per dimension: average correlation of that dim across seed pairs
        reliability = np.zeros(rank)
        for d in range(rank):
            corrs = []
            for i in range(n_used):
                for j in range(i + 1, min(i + 5, n_used)):  # sample pairs for speed
                    r = np.corrcoef(sub[i, :, d], sub[j, :, d])[0, 1]
                    if not np.isnan(r):
                        corrs.append(r)
            reliability[d] = np.mean(corrs) if corrs else 0

        log.info(f"\n  {n_used} seeds: reliability range [{reliability.min():.3f}, {reliability.max():.3f}], "
                 f"mean={reliability.mean():.3f}")

        # Shrink: multiply each dim by its reliability
        for power in [0.5, 1.0, 2.0]:
            w_shrunk = mean_emb * (np.maximum(reliability, 0) ** power)[None, :]
            acc = triplet_acc(w_shrunk, val)
            log.info(f"    shrink^{power:.1f}: acc={acc:.4f}")

        # Also try: just use the mean without shrinkage
        acc_mean = triplet_acc(mean_emb, val)
        log.info(f"    mean (no shrink): acc={acc_mean:.4f}")

        # Soft threshold
        for lam in [0.001, 0.005, 0.01, 0.05]:
            w_thresh = np.maximum(mean_emb - lam, 0)
            acc = triplet_acc(w_thresh, val)
            log.info(f"    soft_thresh={lam}: acc={acc:.4f}")

    # =========================================================================
    # Option 1: Consensus (select best representative)
    # =========================================================================
    log.info("\n" + "=" * 60)
    log.info("OPTION 1: Consensus (best representative)")
    log.info("=" * 60)

    # Find the embedding most correlated with the mean
    mean_rsm = np.mean([w @ w.T for w in aligned[:50]], axis=0)
    best_corr = -1
    best_idx = 0
    for i, w in enumerate(aligned[:50]):
        rsm_i = w @ w.T
        mask = ~np.eye(N, dtype=bool)
        r = np.corrcoef(rsm_i[mask], mean_rsm[mask])[0, 1]
        if r > best_corr:
            best_corr = r
            best_idx = i

    acc_best = triplet_acc(aligned[best_idx], val)
    log.info(f"  Best representative (seed {best_idx}): acc={acc_best:.4f}")

    # Use mean RSM -> re-factorize
    log.info("  Re-factorizing mean RSM...")
    model = SRF(rank=rank, random_state=42, max_outer=1000, max_inner=30, tol=1e-4, verbose=0)
    w_refactored = model.fit_transform(mean_rsm)
    acc_refactored = triplet_acc(w_refactored, val)
    log.info(f"  Re-factorized mean RSM: acc={acc_refactored:.4f}")

    log.info(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
