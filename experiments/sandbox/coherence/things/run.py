"""Coherence-based rank detection on THINGS behavioral across triplet percentages.

Compares three rank selection methods from src/coherence:
  A) Null-calibrated activation counting
  B) Kappa changepoint
  C) Cluster consensus (first-cluster boundary)

Runs at 5%, 10%, 20%, 50%, 100% of triplets to show k* increases with data.
Parameter sweep on alpha_tau at 100% only.

Saves full diagnostics (coherence arrays, kappa, activation profiles) for later
deep-dive analysis.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.coherence import compute_coherence
from src.coherence.analysis import (
    baseline_correct,
    estimate_kappa,
    kappa_changepoint,
    per_component_activation,
)
from src.coherence.cluster import cluster_consensus_across_p
from src.utils import get_output_dir
from src.utils.helpers import compute_similarity_matrix_from_triplets
from src.utils.io import load_triplets

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

OUTPUT_DIR = get_output_dir()
DATA_DIR = Path("data/things")
N_OBJECTS = 1854

PERCENTAGES = [5, 10, 20, 50, 100]
K_LIST = list(range(1, 121))
P_LIST = np.linspace(0.05, 0.95, 25)
B = 100
B_NULL = 30

# Lean sweep: only alpha_tau varies at 100%
ALPHA_TAUS = [0.90, 0.95, 0.99]
DEFAULT_ALPHA_TAU = 0.95


def _load_similarity(pct: int, partition: int = 0) -> tuple[np.ndarray, int]:
    """Load triplets at given percentage and build similarity matrix."""
    if pct == 100:
        triplets, _ = load_triplets(DATA_DIR)
    else:
        path = DATA_DIR / "partitions" / f"{pct}pct_part{partition}" / "train_90.txt"
        triplets = np.loadtxt(path).astype(int)

    s = compute_similarity_matrix_from_triplets(N_OBJECTS, triplets, alpha=1.0)
    return s, len(triplets)


def _extract_k_activation(result: dict, alpha_tau: float) -> dict:
    """Method A: null-calibrated activation counting."""
    diag = result["diagnostics"]
    tau_kp = diag["tau_kp"]
    x_ci_lo = diag["x_ci_lo"]
    k_list = result["k_list"]
    p_list = result["p"]
    k_count = len(k_list)

    if tau_kp is None:
        return {"k_star": np.nan, "method": "activation"}

    act = per_component_activation(x_ci_lo, tau_kp, p_list, k_count)
    act_p = act["activation_p"]
    valid = ~np.isnan(act_p)

    if np.any(valid):
        k_star = int(k_list[np.where(valid)[0][-1]])
    else:
        k_star = 0

    return {
        "k_star": k_star,
        "method": "activation",
        "n_activated": int(valid.sum()),
        "activation_p": act_p.tolist(),
    }


def _extract_k_kappa(result: dict) -> dict:
    """Method B: kappa changepoint."""
    diag = result["diagnostics"]
    x_median = diag["x_median"]
    k_list = result["k_list"]
    p_list = result["p"]

    kappa, kappa_info = estimate_kappa(x_median, p_list, hi_band_quantile=0.85)
    k_cut, cp_info = kappa_changepoint(kappa, k_list)

    return {
        "k_star": k_cut,
        "method": "kappa",
        "kappa": kappa.tolist(),
        "kappa_info": kappa_info,
    }


def _extract_k_cluster(result: dict) -> dict:
    """Method C: cluster consensus -- k* = max k in first (signal) cluster."""
    k_list = result["k_list"]

    cc = cluster_consensus_across_p(
        result,
        use_baseline_correction=False,
        aggregation="median",
        metric="spearman",
        sim_threshold=0.85,
        k_window=1,
        min_prefix_points=5,
        persistence_level=0.8,
        require_consecutive=3,
        make_plots=False,
    )

    ref_clusters = cc["ref_clusters"]
    if len(ref_clusters) > 0:
        first_cluster = ref_clusters[0]
        k_star = int(k_list[first_cluster[-1]])
        n_clusters = len(ref_clusters)
    else:
        k_star = 0
        n_clusters = 0

    return {
        "k_star": k_star,
        "method": "cluster",
        "n_clusters": n_clusters,
        "cluster_sizes": [len(c) for c in ref_clusters],
    }


def _run_one(pct: int, alpha_tau: float) -> dict:
    """Run coherence and extract k* via all three methods for one config."""
    log.info(f"--- pct={pct}%, alpha_tau={alpha_tau} ---")

    t0 = time.time()
    s, n_triplets = _load_similarity(pct)
    n_nan = int(np.isnan(s).sum())
    obs_frac = 1.0 - n_nan / s.size
    log.info(f"  S: {s.shape}, triplets={n_triplets}, NaN={n_nan}, obs={obs_frac:.4f}")

    t1 = time.time()
    result = compute_coherence(
        s,
        k_list=K_LIST,
        p_list=P_LIST,
        b=B,
        b_null=B_NULL,
        alpha_tau=alpha_tau,
        ci_level=0.95,
        compute_null=True,
        use_baseline_correction=False,
        show_progress=True,
        n_jobs=-1,
        random_state=42,
    )
    t_coherence = time.time() - t1
    log.info(f"  compute_coherence: {t_coherence:.1f}s")

    act = _extract_k_activation(result, alpha_tau)
    kap = _extract_k_kappa(result)
    clu = _extract_k_cluster(result)

    log.info(f"  k* activation={act['k_star']}, kappa={kap['k_star']}, cluster={clu['k_star']}")

    # Save full diagnostics for later deep-dive
    tag = f"pct{pct}_tau{alpha_tau:.2f}"
    diag_path = OUTPUT_DIR / "diagnostics" / tag
    diag_path.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        diag_path / "coherence.npz",
        Iproj_boot=result["Iproj_boot"],
        Iproj_mean=result["Iproj_mean"],
        evals_ref=result["evals_ref"],
        k_list=result["k_list"],
        p_list=result["p"],
        x_mean=result["diagnostics"]["x_mean"],
        x_median=result["diagnostics"]["x_median"],
        x_ci_lo=result["diagnostics"]["x_ci_lo"],
        x_ci_hi=result["diagnostics"]["x_ci_hi"],
        tau_kp=result["diagnostics"]["tau_kp"],
        activation_p=np.array(act.get("activation_p", [])),
        kappa=np.array(kap.get("kappa", [])),
    )

    return {
        "pct": pct,
        "n_triplets": n_triplets,
        "n_nan": n_nan,
        "obs_frac": obs_frac,
        "alpha_tau": alpha_tau,
        "k_activation": act["k_star"],
        "k_kappa": kap["k_star"],
        "k_cluster": clu["k_star"],
        "n_activated": act.get("n_activated", 0),
        "n_clusters": clu.get("n_clusters", 0),
        "cluster_sizes": str(clu.get("cluster_sizes", [])),
        "runtime_coherence_sec": round(t_coherence, 1),
        "runtime_total_sec": round(time.time() - t0, 1),
    }


def main():
    records = []

    # 1. Parameter sweep at 100% (lean: only alpha_tau)
    for alpha_tau in ALPHA_TAUS:
        rec = _run_one(100, alpha_tau)
        records.append(rec)

    # 2. All other percentages at default alpha_tau
    for pct in PERCENTAGES:
        if pct == 100:
            continue
        rec = _run_one(pct, DEFAULT_ALPHA_TAU)
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "rank_detection.csv", index=False)
    log.info(f"\nResults saved to {OUTPUT_DIR / 'rank_detection.csv'}")

    # Print summary table
    print("\n" + "=" * 80)
    print("RANK DETECTION SUMMARY")
    print("=" * 80)
    cols = ["pct", "alpha_tau", "k_activation", "k_kappa", "k_cluster", "n_nan", "runtime_coherence_sec"]
    print(df[cols].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
