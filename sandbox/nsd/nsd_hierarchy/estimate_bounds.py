"""
Estimate sampling bounds for NSD subjects.

Run with: poetry run python estimate_bounds.py --subjects 1 2 3 4 5 6 7 8
Or via SLURM: ./scripts/submit sandbox/nsd_hierarchy/estimate_bounds.py -s
"""
from pathlib import Path
import argparse
import json
import logging
import numpy as np
from joblib import Parallel, delayed

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from datasets.nsd_utils import get_roi, load_nsd_betas, get_available_subjects, NSD_DIR_IRIS
from tools.metrics import gaussian_kernel_similarity
from pysrf.bounds import estimate_sampling_bounds_ultra

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)


def compute_rsm_gaussian(betas: np.ndarray) -> np.ndarray:
    """Compute RSM using gaussian kernel with median heuristic."""
    return gaussian_kernel_similarity(betas, betas, sigma=None)


def estimate_bounds_for_subject(
    subject_id: int,
    roi_name: str = "nsdgeneral",
    n_jobs: int = 8,
    random_state: int = 42,
) -> dict:
    """Estimate sampling bounds for a single NSD subject."""
    log.info(f"Processing subject {subject_id}...")

    # Load ROI mask
    roi_mask = get_roi(subject_id, roi_name)
    n_voxels = (roi_mask > 0).sum()
    log.info(f"  ROI {roi_name}: {n_voxels} voxels")

    # Load betas
    log.info(f"  Loading betas...")
    betas, trials = load_nsd_betas(
        subject_id,
        voxel_indices=roi_mask > 0,
        zscore_betas=True,
        max_workers=n_jobs,
    )
    n_stimuli = betas.shape[0]
    log.info(f"  Betas shape: {betas.shape}")

    # Compute RSM with gaussian kernel
    log.info(f"  Computing gaussian kernel RSM...")
    rsm = compute_rsm_gaussian(betas)
    log.info(f"  RSM shape: {rsm.shape}, range: [{rsm.min():.4f}, {rsm.max():.4f}]")

    # Estimate bounds
    log.info(f"  Estimating bounds...")
    pmin, pmax, _ = estimate_sampling_bounds_ultra(
        rsm,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=False,
    )

    result = {
        "subject_id": subject_id,
        "roi_name": roi_name,
        "n_stimuli": n_stimuli,
        "n_voxels": int(n_voxels),
        "pmin": float(pmin),
        "pmax": float(pmax),
        "p_mean": float((pmin + pmax) / 2),
    }

    log.info(f"  Subject {subject_id}: p_min={pmin:.4f}, p_max={pmax:.4f}, mean={result['p_mean']:.4f}")
    return result


def main(
    subjects: list[int] | None = None,
    roi_name: str = "nsdgeneral",
    output_dir: Path | None = None,
    n_jobs: int = 8,
    parallel_subjects: bool = False,
):
    if output_dir is None:
        output_dir = Path(__file__).parent / "outputs" / "bounds"
    output_dir.mkdir(parents=True, exist_ok=True)

    if subjects is None:
        subjects = get_available_subjects(NSD_DIR_IRIS)

    log.info("=" * 60)
    log.info(f"NSD Sampling Bounds Estimation")
    log.info(f"  Subjects: {subjects}")
    log.info(f"  ROI: {roi_name}")
    log.info(f"  Output: {output_dir}")
    log.info("=" * 60)

    results = []

    if parallel_subjects and len(subjects) > 1:
        # Run subjects in parallel (use fewer jobs per subject)
        jobs_per_subject = max(1, n_jobs // len(subjects))
        results = Parallel(n_jobs=len(subjects), verbose=10)(
            delayed(estimate_bounds_for_subject)(s, roi_name, jobs_per_subject)
            for s in subjects
        )
    else:
        # Run subjects sequentially
        for subject_id in subjects:
            result = estimate_bounds_for_subject(subject_id, roi_name, n_jobs)
            results.append(result)

            # Save intermediate result
            with open(output_dir / f"bounds_subject{subject_id}.json", "w") as f:
                json.dump(result, f, indent=2)

    # Save combined results
    combined = {
        "roi_name": roi_name,
        "n_subjects": len(subjects),
        "subjects": results,
        "summary": {
            "pmin_mean": float(np.mean([r["pmin"] for r in results])),
            "pmin_std": float(np.std([r["pmin"] for r in results])),
            "pmax_mean": float(np.mean([r["pmax"] for r in results])),
            "pmax_std": float(np.std([r["pmax"] for r in results])),
            "p_mean_mean": float(np.mean([r["p_mean"] for r in results])),
        },
    }

    with open(output_dir / "bounds_all_subjects.json", "w") as f:
        json.dump(combined, f, indent=2)

    log.info("\n" + "=" * 60)
    log.info("Summary:")
    log.info(f"  p_min: {combined['summary']['pmin_mean']:.4f} +/- {combined['summary']['pmin_std']:.4f}")
    log.info(f"  p_max: {combined['summary']['pmax_mean']:.4f} +/- {combined['summary']['pmax_std']:.4f}")
    log.info(f"  p_mean: {combined['summary']['p_mean_mean']:.4f}")
    log.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=None,
                        help="Subject IDs to process (default: all 8)")
    parser.add_argument("--roi", type=str, default="nsdgeneral",
                        help="ROI name (default: nsdgeneral)")
    parser.add_argument("--n-jobs", type=int, default=8,
                        help="Number of parallel jobs")
    parser.add_argument("--parallel-subjects", action="store_true",
                        help="Run subjects in parallel (use with caution - high memory)")
    args = parser.parse_args()

    main(
        subjects=args.subjects,
        roi_name=args.roi,
        n_jobs=args.n_jobs,
        parallel_subjects=args.parallel_subjects,
    )
