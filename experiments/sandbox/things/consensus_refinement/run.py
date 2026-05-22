"""Compare consensus strategies for symmetric NMF on VGG16.

Strategies:
  1. select       -- most central run (current default)
  2. best_recon   -- run with lowest reconstruction error
  3. median_raw   -- element-wise median (breaks WW^T ≈ S)
  4. nnls_refine  -- existing refine mode: median + row-wise NNLS vs mean(WW^T)
  5. bsum_refine_s -- median + BSUM refinement targeting original S
  6. bsum_refine_avg -- median + BSUM refinement targeting mean(WW^T)

The key distinction: NNLS row-wise vs BSUM element-wise, and target S vs mean(WW^T).
"""

import logging

import numpy as np
from pysrf import SRF
from pysrf.consensus import AlignedConsensus, EnsembleEmbedding
from pysrf.model import _update_w_impl
from scipy.optimize import nnls
from sklearn.pipeline import Pipeline

from similarity import build_similarity
from src.utils import get_output_dir

log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()

RANK = 46
N_RUNS = 20
REFINE_OUTER = 5
REFINE_INNER = 50


def _reliability(aligned: np.ndarray) -> np.ndarray:
    """Per-dimension reliability from aligned embeddings (Fisher-z averaged)."""
    n_runs, _, rank = aligned.shape
    rel = np.zeros(rank)
    for d in range(rank):
        pairs = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                r = np.corrcoef(aligned[i, :, d], aligned[j, :, d])[0, 1]
                if np.isfinite(r):
                    pairs.append(r)
        if pairs:
            z = np.arctanh(np.clip(pairs, -0.999, 0.999))
            rel[d] = float(np.tanh(np.mean(z)))
    return rel


def _r2_upper(s: np.ndarray, w: np.ndarray) -> float:
    """R^2 on upper triangle (excluding diagonal)."""
    triu = np.triu_indices(s.shape[0], k=1)
    s_upper = s[triu]
    recon_upper = (w @ w.T)[triu]
    ss_res = np.sum((s_upper - recon_upper) ** 2)
    ss_tot = np.sum((s_upper - s_upper.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def _sparsity(w: np.ndarray) -> float:
    return float((w == 0).mean())


def _corr_to_runs(w: np.ndarray, aligned: np.ndarray) -> float:
    """Mean correlation between w and each aligned run (flattened)."""
    corrs = []
    for i in range(aligned.shape[0]):
        r = np.corrcoef(w.ravel(), aligned[i].ravel())[0, 1]
        corrs.append(r)
    return float(np.mean(corrs))


def _nnls_refine(w_init: np.ndarray, s_target: np.ndarray, n_iter: int = 20) -> np.ndarray:
    """Row-wise NNLS refinement (existing approach in pysrf)."""
    w = np.maximum(w_init.copy(), 0)
    n = w.shape[0]
    for _ in range(n_iter):
        for i in range(n):
            w[i, :], _ = nnls(w, s_target[i, :])
    return w


def _bsum_refine(w_init: np.ndarray, s_target: np.ndarray,
                 n_outer: int = 5, n_inner: int = 50) -> np.ndarray:
    """BSUM refinement -- the actual symmetric NMF solver."""
    w = np.maximum(w_init.copy(), 0).astype(np.float64)
    s_target = np.asarray(s_target, dtype=np.float64)
    for _ in range(n_outer):
        w = _update_w_impl(s_target, w, max_iter=n_inner, tol=1e-4)
    return w


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    log.info("Loading VGG16 similarity matrix...")
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "name": "vgg16", "type": "feature_file",
        "path": "data/features/vgg16", "features_file": "vgg16_features.npy",
        "similarity_fn": "linear",
    })
    s = build_similarity(cfg)
    s = s / np.max(s)
    log.info(f"S: {s.shape}, mean={s.mean():.4f}")

    log.info(f"Fitting {N_RUNS} SRF runs at rank={RANK}...")
    pipeline = Pipeline([
        ("ensemble", EnsembleEmbedding(
            SRF(rank=RANK, random_state=42),
            n_runs=N_RUNS, random_state=42, n_jobs=-1,
        )),
        ("consensus", AlignedConsensus(rank=RANK, aggregation="select")),
    ])
    pipeline.fit(s)

    consensus = pipeline.named_steps["consensus"]
    aligned = consensus.aligned_embeddings_  # (n_runs, n, k)
    log.info(f"Aligned: {aligned.shape}")

    reliability = _reliability(aligned)
    log.info(f"Reliability: mean={reliability.mean():.3f}, "
             f"min={reliability.min():.3f}, max={reliability.max():.3f}")

    # Precompute shared quantities
    w_median = np.median(aligned, axis=0)
    s_avg = np.mean([aligned[i] @ aligned[i].T for i in range(N_RUNS)], axis=0)

    strategies = {}

    # 1. Select (most central)
    w = aligned[consensus.selected_run_idx_]
    strategies["select"] = w

    # 2. Best reconstruction
    r2s = [_r2_upper(s, aligned[i]) for i in range(N_RUNS)]
    strategies["best_recon"] = aligned[np.argmax(r2s)]

    # 3. Median raw (no refinement)
    strategies["median_raw"] = w_median

    # 4. NNLS refine: median -> row-wise NNLS targeting mean(WW^T)
    log.info("Running NNLS refine (median -> mean(WW^T))...")
    strategies["nnls_refine"] = _nnls_refine(w_median, s_avg, n_iter=20)

    # 5. BSUM refine: median -> BSUM targeting original S
    log.info("Running BSUM refine (median -> S)...")
    strategies["bsum_S"] = _bsum_refine(w_median, s, REFINE_OUTER, REFINE_INNER)

    # 6. BSUM refine: median -> BSUM targeting mean(WW^T)
    log.info("Running BSUM refine (median -> mean(WW^T))...")
    strategies["bsum_avg"] = _bsum_refine(w_median, s_avg, REFINE_OUTER, REFINE_INNER)

    # Report
    log.info(f"\n{'Strategy':<20} {'R2(S)':>8} {'R2(avg)':>8} {'Sparse':>8} {'Corr2Runs':>10}")
    for name, w in strategies.items():
        r2_s = _r2_upper(s, w)
        r2_avg = _r2_upper(s_avg, w)
        sp = _sparsity(w)
        corr = _corr_to_runs(w, aligned)
        log.info(f"{name:<20} {r2_s:>8.4f} {r2_avg:>8.4f} {sp:>8.4f} {corr:>10.4f}")

    np.savez(OUTPUT_DIR / "embeddings.npz",
             **strategies, aligned=aligned, reliability=reliability)
    log.info(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
