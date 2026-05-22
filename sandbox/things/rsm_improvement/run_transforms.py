"""Test RSM transforms before SRF factorization.

Theory: The count RSM S[i,j] = P(choose i,j | shown) is related to true
similarity s_ij through the softmax: P = E_k[exp(s_ij) / sum(exp)].

The logit transform approximately inverts this:
    logit(S[i,j]) ≈ s_ij - E_k[logsumexp(s_ik, s_jk)]

The second term is a context-dependent bias. If roughly constant across
pairs, then logit(S) + const ≈ true similarity matrix.

We test:
  1. Raw proportions (baseline)
  2. Logit transform + shift to non-negative
  3. Pre-denoise (PSD rank truncation) then SRF
  4. Logit + denoise + shift then SRF

For each, we sweep SRF rank and evaluate triplet prediction accuracy.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysrf import SRF
from scipy.special import logit

from src.utils import get_output_dir
from src.utils.helpers import compute_similarity_matrix_from_triplets
from src.utils.io import load_spose_embedding, load_triplets

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
DATA_DIR = Path("data/things")
N_ITEMS = 1854
RANKS = [24, 35, 50, 66]
SEEDS = [0, 1, 2]


def compute_triplet_accuracy(embedding, triplets):
    idx = triplets.astype(int)
    ei, ej, ek = embedding[idx[:, 0]], embedding[idx[:, 1]], embedding[idx[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def psd_denoise(s, rank):
    """Project matrix to rank-K PSD cone."""
    vals, vecs = np.linalg.eigh(s)
    keep = vals > 0
    vals_pos = vals[keep]
    vecs_pos = vecs[:, keep]
    if rank < len(vals_pos):
        idx = np.argsort(vals_pos)[::-1][:rank]
        vals_pos = vals_pos[idx]
        vecs_pos = vecs_pos[:, idx]
    return vecs_pos @ np.diag(vals_pos) @ vecs_pos.T


def make_rsm_raw(triplets, alpha=0.0):
    """Raw count RSM."""
    return compute_similarity_matrix_from_triplets(N_ITEMS, triplets, alpha=alpha)


def make_rsm_logit(triplets, alpha=1.0, eps=0.01):
    """Logit-transformed RSM. Uses alpha=1 to avoid exact 0/1."""
    s = compute_similarity_matrix_from_triplets(N_ITEMS, triplets, alpha=alpha)
    mask = np.isnan(s)
    diag_idx = np.arange(N_ITEMS)

    s_clip = np.clip(s, eps, 1 - eps)
    t = logit(s_clip)
    t[mask] = np.nan
    t_shift = t - np.nanmin(t)
    t_shift[diag_idx, diag_idx] = np.nanmax(t_shift)
    return t_shift


def make_rsm_denoised(triplets, denoise_rank=35, alpha=0.0):
    """Spectrally denoised RSM."""
    s = compute_similarity_matrix_from_triplets(N_ITEMS, triplets, alpha=alpha)
    mask = np.isnan(s)
    s_filled = np.where(mask, 0.5, s)
    s_clean = psd_denoise(s_filled, denoise_rank)
    s_clean = np.clip(s_clean, 0, None)
    s_clean[mask] = np.nan
    np.fill_diagonal(s_clean, np.nanmax(s_clean))
    return s_clean


def make_rsm_logit_denoised(triplets, denoise_rank=35, alpha=1.0, eps=0.01):
    """Logit transform, then spectral denoise in logit space."""
    s = compute_similarity_matrix_from_triplets(N_ITEMS, triplets, alpha=alpha)
    mask = np.isnan(s)
    diag_idx = np.arange(N_ITEMS)

    s_clip = np.clip(s, eps, 1 - eps)
    t = logit(s_clip)
    t[mask] = 0.0
    t[diag_idx, diag_idx] = 0.0

    vals, vecs = np.linalg.eigh(t)
    idx = np.argsort(np.abs(vals))[::-1][:denoise_rank]
    t_clean = vecs[:, idx] @ np.diag(vals[idx]) @ vecs[:, idx].T

    t_shift = t_clean - t_clean.min()
    t_shift[mask] = np.nan
    t_shift[diag_idx, diag_idx] = np.nanmax(t_shift)
    return t_shift


def _run_one(rsm, triplets_val, rank, seed):
    model = SRF(
        rank=rank, random_state=seed,
        max_outer=200, max_inner=50,
        tol=1e-4, verbose=0,
    )
    embedding = model.fit_transform(rsm)
    acc = compute_triplet_accuracy(embedding, triplets_val)
    return acc


def evaluate_transform(name, rsm, triplets_val):
    log.info("Evaluating: %s (shape=%s, NaN=%.1f%%)",
             name, rsm.shape, 100 * np.isnan(rsm).mean())

    tasks = [
        (rsm, triplets_val, rank, seed)
        for rank in RANKS
        for seed in SEEDS
    ]
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_run_one)(r, t, k, s) for r, t, k, s in tasks
    )

    records = []
    idx = 0
    for rank in RANKS:
        for seed in SEEDS:
            records.append({
                "transform": name,
                "rank": rank,
                "seed": seed,
                "val_acc": results[idx],
            })
            idx += 1

    summary = pd.DataFrame(records).groupby("rank")["val_acc"].mean()
    for rank, acc in summary.items():
        log.info("  k=%3d: %.4f", rank, acc)

    return records


def main():
    train_triplets, val_triplets = load_triplets(DATA_DIR)
    spose = load_spose_embedding(DATA_DIR, num_dims=66)
    log.info("SPoSE baseline: %.4f", compute_triplet_accuracy(spose, val_triplets))
    log.info("")

    all_records = []

    rsm_raw = make_rsm_raw(train_triplets, alpha=0.0)
    all_records += evaluate_transform("raw_alpha0", rsm_raw, val_triplets)

    rsm_logit = make_rsm_logit(train_triplets, alpha=1.0)
    all_records += evaluate_transform("logit_alpha1", rsm_logit, val_triplets)

    rsm_denoised = make_rsm_denoised(train_triplets, denoise_rank=35, alpha=0.0)
    all_records += evaluate_transform("denoised_k35", rsm_denoised, val_triplets)

    rsm_logit_den = make_rsm_logit_denoised(train_triplets, denoise_rank=35, alpha=1.0)
    all_records += evaluate_transform("logit_denoised_k35", rsm_logit_den, val_triplets)

    rsm_denoised50 = make_rsm_denoised(train_triplets, denoise_rank=50, alpha=0.0)
    all_records += evaluate_transform("denoised_k50", rsm_denoised50, val_triplets)

    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_DIR / "transform_results.csv", index=False)
    log.info("\nSaved results to %s", OUTPUT_DIR / "transform_results.csv")


if __name__ == "__main__":
    main()
