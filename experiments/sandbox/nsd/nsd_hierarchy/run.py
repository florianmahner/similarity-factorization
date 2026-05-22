"""
NSD Visual Hierarchy: SRF factorization across ROIs.

Explore how representational structure changes along visual hierarchy.
- Early visual: V1, V2, V3, V4 (from prf-visualrois or Kastner2015)
- Category-selective: floc-faces, floc-bodies, floc-places, floc-words

Output structure designed for easy translation to stable experiment.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import json
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from datasets.nsd_utils import (
    get_available_rois,
    get_roi,
    load_nsd_betas,
    NSD_DIR_IRIS,
)
from tools.metrics import gaussian_kernel_similarity
from pysrf import SRF

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def check_roi_structure(subject_id: int = 1) -> dict:
    """Check what ROIs are available and their structure."""
    rois = get_available_rois(subject_id)

    roi_info = {}
    for roi_name in rois:
        if roi_name.startswith(('lh.', 'rh.')):
            continue  # Skip hemisphere-specific for now

        roi_mask = get_roi(subject_id, roi_name)
        unique_vals = np.unique(roi_mask[roi_mask > 0])

        roi_info[roi_name] = {
            "n_unique_values": len(unique_vals),
            "unique_values": unique_vals.tolist(),
            "total_voxels": int((roi_mask > 0).sum()),
        }

    return roi_info


def get_visual_hierarchy_rois(subject_id: int = 1) -> dict[str, np.ndarray]:
    """Get ROI masks for visual hierarchy analysis."""
    rois = {}

    # Check Kastner2015 for retinotopic areas
    kastner = get_roi(subject_id, "Kastner2015")
    # Kastner2015 labels: 1=V1, 2=V2, 3=V3, 4=hV4, 5=VO1, 6=VO2, 7=LO1, 8=LO2, etc.
    kastner_labels = {
        "V1": 1, "V2": 2, "V3": 3, "hV4": 4,
        "VO1": 5, "VO2": 6, "LO1": 7, "LO2": 8,
    }

    for name, label in kastner_labels.items():
        mask = kastner == label
        if mask.sum() > 50:  # At least 50 voxels
            rois[name] = mask
            log.info(f"  {name}: {mask.sum()} voxels")

    # Category-selective ROIs (fLoc localizer)
    # floc-faces values: 1=OFA, 2=FFA-1, 3=FFA-2, 5=aTL-faces
    # floc-places values: 1=OPA, 2=PPA, 3=RSC
    floc_rois = ["floc-faces", "floc-bodies", "floc-places", "floc-words"]
    for roi_name in floc_rois:
        try:
            roi_mask = get_roi(subject_id, roi_name)
            mask = roi_mask > 0
            if mask.sum() > 50:
                rois[roi_name] = mask
                # Show sub-region breakdown
                unique_vals = np.unique(roi_mask[roi_mask > 0])
                breakdown = ", ".join([f"v{int(v)}:{(roi_mask==v).sum()}" for v in unique_vals])
                log.info(f"  {roi_name}: {mask.sum()} voxels ({breakdown})")
        except Exception as e:
            log.warning(f"  {roi_name}: not available ({e})")

    return rois


def compute_rsm(betas: np.ndarray, metric: str = "gaussian") -> np.ndarray:
    """Compute RSM using specified similarity metric.

    Args:
        betas: (n_stimuli, n_voxels) array
        metric: "gaussian" (RBF kernel) or "shifted_cosine" (cosine shifted to [0,1])
    """
    if metric == "gaussian":
        rsm = gaussian_kernel_similarity(betas, betas, sigma=None)
    elif metric == "shifted_cosine":
        from sklearn.metrics.pairwise import cosine_similarity
        rsm = (cosine_similarity(betas) + 1) / 2  # Shift from [-1,1] to [0,1]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    np.fill_diagonal(rsm, 1.0)
    return rsm


def run_srf_on_roi(
    betas: np.ndarray,
    rank: int = 5,
    seed: int = 42,
    metric: str = "gaussian",
) -> dict:
    """Run SRF on betas from a single ROI."""
    rsm = compute_rsm(betas, metric=metric)

    model = SRF(
        rank=rank,
        rho=1.0,
        max_outer=200,
        max_inner=30,
        tol=1e-4,
        verbose=1,
        random_state=seed,
    )
    embedding = model.fit_transform(rsm)
    reconstruction = embedding @ embedding.T

    recon_error = np.sqrt(np.mean((rsm - reconstruction) ** 2))

    return {
        "embedding": embedding,
        "rsm": rsm,
        "reconstruction": reconstruction,
        "recon_rmse": recon_error,
        "n_voxels": betas.shape[1],
        "n_stimuli": betas.shape[0],
    }


def analyze_embedding_structure(embedding: np.ndarray) -> dict:
    """Analyze the structure of an SRF embedding."""
    # Sparsity: fraction of near-zero values
    sparsity = (np.abs(embedding) < 0.01).mean()

    # Effective dimensionality via entropy
    norms = np.linalg.norm(embedding, axis=0)
    norms = norms / norms.sum()
    entropy = -np.sum(norms * np.log(norms + 1e-10))
    eff_dim = np.exp(entropy)

    # Dimension dominance: how much does each dimension contribute
    dim_variance = np.var(embedding, axis=0)
    dim_contribution = dim_variance / dim_variance.sum()

    return {
        "sparsity": float(sparsity),
        "effective_dim": float(eff_dim),
        "dim_contributions": dim_contribution.tolist(),
        "max_dim_contribution": float(dim_contribution.max()),
    }


def main(
    n_stimuli: int | None = None,
    metric: str = "gaussian",
    output_dir: Path | None = None,
    rank: int = 5,
    rois: list[str] | None = None,
):
    """
    Args:
        n_stimuli: Limit number of stimuli for testing. None = use all.
        metric: Similarity metric ("gaussian" or "shifted_cosine")
        output_dir: Output directory. Default: outputs/<metric>/
        rank: Number of SRF dimensions
        rois: List of ROI names to analyze. None = all ROIs.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "outputs" / metric
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_id = 1

    log.info("=" * 60)
    log.info(f"NSD Visual Hierarchy: SRF Analysis (metric={metric}, rank={rank})")
    log.info("=" * 60)

    # Step 1: Check ROI structure
    log.info("\n[1] Checking ROI structure...")
    roi_info = check_roi_structure(subject_id)

    with open(output_dir / "roi_info.json", "w") as f:
        json.dump(roi_info, f, indent=2)

    log.info(f"Found {len(roi_info)} bilateral ROIs")

    # Step 2: Get visual hierarchy ROIs
    log.info("\n[2] Getting visual hierarchy ROIs...")
    hierarchy_rois = get_visual_hierarchy_rois(subject_id)

    # Filter ROIs if specified, and add any requested ROIs not in hierarchy
    if rois is not None:
        # Keep only requested ROIs from hierarchy
        hierarchy_rois = {k: v for k, v in hierarchy_rois.items() if k in rois}
        # Add any requested ROIs not already in hierarchy (e.g., nsdgeneral)
        for roi_name in rois:
            if roi_name not in hierarchy_rois:
                try:
                    roi_mask = get_roi(subject_id, roi_name)
                    mask = roi_mask > 0
                    if mask.sum() > 50:
                        hierarchy_rois[roi_name] = mask
                        log.info(f"  {roi_name}: {mask.sum()} voxels (added from request)")
                except Exception as e:
                    log.warning(f"  {roi_name}: not available ({e})")
        log.info(f"Filtered to ROIs: {list(hierarchy_rois.keys())}")

    log.info(f"Using {len(hierarchy_rois)} ROIs for analysis")

    # Step 3: Load betas for each ROI (sequential - IO bound)
    log.info("\n[3] Loading betas for each ROI...")

    roi_betas = {}
    trials_saved = None
    for roi_name, roi_mask in hierarchy_rois.items():
        log.info(f"  Loading {roi_name}...")
        betas, trials = load_nsd_betas(
            subject_id,
            voxel_indices=roi_mask,
            zscore_betas=True,
            max_workers=8,
        )
        # Subsample stimuli if requested
        if n_stimuli is not None and betas.shape[0] > n_stimuli:
            betas = betas[:n_stimuli]
            trials = trials[:n_stimuli]
        roi_betas[roi_name] = betas
        if trials_saved is None:
            trials_saved = trials
        log.info(f"    Shape: {betas.shape}")

    # Save trial indices for plotting
    np.save(output_dir / "trials.npy", trials_saved)

    # Save human-readable mapping: embedding index → NSD stimulus ID
    trial_mapping = {
        "description": "Maps embedding row index to NSD stimulus ID (indexes into imgBrick)",
        "hdf5_path": "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5",
        "n_stimuli": len(trials_saved),
        "index_to_stimulus_id": {int(i): int(t) for i, t in enumerate(trials_saved)},
    }
    with open(output_dir / "trial_mapping.json", "w") as f:
        json.dump(trial_mapping, f, indent=2)

    log.info(f"  Saved trial indices: {len(trials_saved)}")

    # Step 4: Run SRF across ROIs
    log.info(f"\n[4] Running SRF (metric={metric}, rank={rank})...")

    def process_roi(roi_name: str, betas: np.ndarray) -> dict:
        srf_result = run_srf_on_roi(betas, rank=rank, metric=metric)
        structure = analyze_embedding_structure(srf_result["embedding"])
        return {
            "roi_name": roi_name,
            "srf_result": srf_result,
            "structure": structure,
        }

    # Run sequentially if few ROIs (shows SRF progress), parallel otherwise
    if len(roi_betas) <= 2:
        parallel_results = [process_roi(name, betas) for name, betas in roi_betas.items()]
    else:
        parallel_results = Parallel(n_jobs=-1, verbose=10)(
            delayed(process_roi)(roi_name, betas)
            for roi_name, betas in roi_betas.items()
        )

    # Collect results
    results = []
    embeddings = {}
    rsms = {}

    for res in parallel_results:
        roi_name = res["roi_name"]
        srf_result = res["srf_result"]
        structure = res["structure"]

        results.append({
            "roi": roi_name,
            "n_voxels": srf_result["n_voxels"],
            "n_stimuli": srf_result["n_stimuli"],
            "recon_rmse": srf_result["recon_rmse"],
            **structure,
        })

        embeddings[roi_name] = srf_result["embedding"]
        rsms[roi_name] = srf_result["rsm"]

        log.info(f"  {roi_name}: RMSE={srf_result['recon_rmse']:.4f}, sparsity={structure['sparsity']:.3f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "roi_results.csv", index=False)

    # Save embeddings per ROI
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)
    for roi_name, emb in embeddings.items():
        np.save(embeddings_dir / f"{roi_name}.npy", emb)

    # Save RSMs per ROI (subsampled for size)
    rsm_dir = output_dir / "rsms"
    rsm_dir.mkdir(exist_ok=True)
    for roi_name, rsm in rsms.items():
        np.save(rsm_dir / f"{roi_name}.npy", rsm)

    # Step 5: Compare embeddings across ROIs
    log.info("\n[5] Comparing embedding structure across ROIs...")

    roi_names = list(embeddings.keys())
    n_rois = len(roi_names)

    # RSM-RSM correlations (second-order similarity)
    rsm_corrs = np.zeros((n_rois, n_rois))
    for i, roi_i in enumerate(roi_names):
        for j, roi_j in enumerate(roi_names):
            rsm_i = rsms[roi_i][np.triu_indices_from(rsms[roi_i], k=1)]
            rsm_j = rsms[roi_j][np.triu_indices_from(rsms[roi_j], k=1)]
            rsm_corrs[i, j], _ = spearmanr(rsm_i, rsm_j)

    rsm_corr_df = pd.DataFrame(rsm_corrs, index=roi_names, columns=roi_names)
    rsm_corr_df.to_csv(output_dir / "rsm_correlations.csv")

    log.info("\nRSM correlations between ROIs:")
    log.info(rsm_corr_df.round(3).to_string())

    # Summary
    log.info("\n" + "=" * 60)
    log.info("Summary:")
    log.info(f"  Subject: {subject_id}")
    log.info(f"  ROIs analyzed: {len(hierarchy_rois)}")
    log.info(f"  Rank: {rank}")
    log.info(f"\nResults saved to: {output_dir}")

    # Save metadata for reproducibility
    metadata = {
        "subject_id": subject_id,
        "rank": rank,
        "metric": metric,
        "rois_analyzed": roi_names,
        "n_stimuli": results[0]["n_stimuli"] if results else None,
        "n_stimuli_limit": n_stimuli,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-stimuli", type=int, default=None,
                        help="Limit number of stimuli (for testing)")
    parser.add_argument("--metric", type=str, default="gaussian",
                        choices=["gaussian", "shifted_cosine"],
                        help="Similarity metric")
    parser.add_argument("--rank", type=int, default=5,
                        help="Number of SRF dimensions")
    parser.add_argument("--rois", type=str, nargs="+", default=None,
                        help="ROIs to analyze (e.g., --rois floc-faces V1)")
    args = parser.parse_args()
    main(n_stimuli=args.n_stimuli, metric=args.metric, rank=args.rank, rois=args.rois)
