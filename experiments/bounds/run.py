#!/usr/bin/env python
"""Compute sampling bounds for embedding generation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from datasets import load_dataset
from config import get_dataset_path
from pysrf.bounds import estimate_sampling_bounds_fast
from tools.rsa import compute_similarity


SMALL_DATASETS = {"mur92", "cichy118", "peterson-animals", "peterson-various"}
LARGE_DATASETS = {"nsd", "things-monkey-22k", "vit", "things-ooo", "word-association"}


def build_similarity(dataset_slug: str, subject_id: int | None) -> np.ndarray:
    if dataset_slug == "nsd":
        ds = load_dataset(
            "nsd", subject_id=subject_id, roi_name="streams", space="func1pt8mm"
        )
        return compute_similarity(ds.data, ds.data, "gaussian_kernel")

    if dataset_slug == "things-monkey-22k":
        ds = load_dataset(
            "things-monkey-22k", min_reliab=0.3, monkey_type="F", roi="it"
        )
        if hasattr(ds, "rsm"):
            return ds.rsm
        return compute_similarity(ds.data, ds.data, "gaussian_kernel")

    if dataset_slug == "vit":
        ds = np.load(
            "/SSD/projects/deepsim/raw/features/dataset/openai/ViT-L-14/visual/features.npy"
        )
        import pandas as pd

        image_info = pd.read_csv("/SSD/projects/deepsim/raw/features/image_info.csv")
        # find where filename contains _plus
        plus_indices = image_info[image_info["filename"].str.contains("_plus")].index

        ds = ds[plus_indices, :]
        rsm = compute_similarity(ds, ds, "gaussian_kernel")
        return rsm

    if dataset_slug == "things-ooo":
        from utils.io import load_triplets
        from analysis.things.common import compute_similarity_matrix_from_triplets

        train_triplets, validation_triplets = load_triplets(
            Path(get_dataset_path("things-ooo")), number="4.7mio"
        )
        sim = compute_similarity_matrix_from_triplets(1854, train_triplets)
        # replace all nans with zeros using np.nan_to_num, since this does not work well with bound estimation
        sim = np.nan_to_num(sim, nan=0.0)
        return sim

    if dataset_slug == "word-association":
        """Load PPI network with NaN for non-edges."""
        data_dir = Path("/LOCAL/fmahner/similarity-factorization/data/ppi")

        adjacency = np.load(data_dir / "string_adjacency_weighted_nan.npy")
        proteins = np.loadtxt(data_dir / "string_proteins.txt", dtype=str)

        max_val = np.nanmax(adjacency)

        if max_val > 1.0:
            adjacency = adjacency / max_val

        return adjacency, proteins

    ds = load_dataset(dataset_slug)
    if hasattr(ds, "rsm"):
        return ds.rsm
    return compute_similarity(ds.data, ds.data, "gaussian_kernel")


def compute_bounds(
    dataset_slug: str,
    output_dir: Path,
    subject_id: int | None = None,
    n_jobs: int = -1,
    random_state: int = 0,
) -> None:

    if subject_id is not None:
        print(f"subject {subject_id}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    bounds_file = output_dir / "bounds.json"

    if bounds_file.exists():
        print(f"bounds file already exists: {bounds_file}", flush=True)
        print("delete it manually to recompute", flush=True)
        return

    similarity = build_similarity(dataset_slug, subject_id)
    start_time = datetime.now()

    print(
        f"Bounds for {dataset_slug} with similarity matrix shape: {similarity.shape}",
        flush=True,
    )

    pmin, pmax, _ = estimate_sampling_bounds_fast(
        similarity, random_state=random_state, n_jobs=n_jobs, verbose=False
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"bounds computation took {elapsed:.1f}s", flush=True)

    mean_sampling_fraction = 0.5 * (pmin + pmax)

    bounds_data = {
        "dataset": dataset_slug,
        "subject_id": subject_id,
        "pmin": float(pmin),
        "pmax": float(pmax),
        "mean_sampling_fraction": float(mean_sampling_fraction),
        "matrix_shape": list(similarity.shape),
        "computed_at": datetime.now().isoformat(),
        "random_state": random_state,
        "computation_time_seconds": elapsed,
    }

    with open(bounds_file, "w") as f:
        json.dump(bounds_data, f, indent=2)

    print(f"saved bounds to {bounds_file}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="compute sampling bounds")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(SMALL_DATASETS | LARGE_DATASETS),
        help="dataset to analyze",
    )
    parser.add_argument(
        "--subject_id", type=int, default=None, help="subject id for nsd dataset"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None, help="output directory for results"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=-1, help="number of parallel jobs"
    )
    parser.add_argument("--random_state", type=int, default=0, help="random seed")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    if args.output_dir is None:
        args.output_dir = script_dir / "outputs"

    args.output_dir = args.output_dir.expanduser().resolve()

    if args.dataset == "nsd":
        if args.subject_id is None:
            raise ValueError("subject_id required for nsd dataset")
        dataset_output = args.output_dir / "nsd" / f"subj{args.subject_id:02d}"
    else:
        dataset_output = args.output_dir / args.dataset

    compute_bounds(
        args.dataset,
        dataset_output,
        args.subject_id,
        args.n_jobs,
        args.random_state,
    )


if __name__ == "__main__":
    main()
