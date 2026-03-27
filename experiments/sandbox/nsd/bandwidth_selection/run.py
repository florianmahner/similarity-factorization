"""RBF bandwidth selection for NSD subject 1.

Runs select_rbf_bandwidth on NSD fMRI betas to find optimal alpha* that
maximizes H(stability, R^2) at the kappa-estimated rank per bandwidth.
"""

import logging

import numpy as np

from datasets import load_dataset
from src.tools.bandwidth import select_rbf_bandwidth
from src.utils import get_output_dir

log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()

SUBJECT_ID = 1
ALPHA_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    log.info(f"Loading NSD subject {SUBJECT_ID} features...")
    ds = load_dataset(
        "nsd",
        subject_id=SUBJECT_ID,
        root="/LOCAL/LABSHARE/natural-scenes-dataset",
        roi_name="nsdgeneral",
        space="func1pt8mm",
        zscore_betas=True,
    )
    features = ds.data
    log.info(f"Features shape: {features.shape}")

    result = select_rbf_bandwidth(
        features,
        alpha_grid=ALPHA_GRID,
        n_runs=5,
        k_max=100,
        n_jobs=-1,
        random_state=42,
    )

    df = result["results"]
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)

    # Save per-dim reliability arrays
    per_dim = {}
    for _, row in df.iterrows():
        per_dim[str(row["alpha"])] = row["reliability_per_dim"]
    np.savez(OUTPUT_DIR / "stability_per_dim.npz", **per_dim)

    log.info(f"\nSaved results to {OUTPUT_DIR / 'results.csv'}")
    log.info(f"\n=== Results ===")
    display_cols = ["alpha", "k_star", "stability_mean", "r2_mean", "h_mean", "sim_mean"]
    log.info(df[display_cols].to_string(index=False))
    log.info(f"\nalpha* = {result['alpha_star']}, k* = {result['k_star']}, "
             f"H = {result['h_star']:.3f}")


if __name__ == "__main__":
    main()
