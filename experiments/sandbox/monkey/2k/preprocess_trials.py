#!/usr/bin/env python3
"""Preprocess MonkeyF and MonkeyN_new to match MonkeyN original format.

Output format (same as MonkeyN original):
- data_it: (n_trials, 256) for IT channels
- zi_list.csv: class name per trial
"""

import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

ROI_SLICES = {
    "N": {"v1": (0, 512), "v4": (512, 768), "it": (768, 1024)},
    "F": {"v1": (0, 512), "it": (512, 832), "v4": (832, 1024)},
}
TIME_WINDOWS = {"v1": (25, 125), "v4": (50, 150), "it": (75, 175)}
BASELINE_WINDOW = (-50, 0)  # Match original MonkeyN preprocessing


def _load_batch(mat_path: str, start: int, end: int):
    with h5py.File(mat_path, "r") as f:
        return np.array(f["ALLMUA"][:, start:end, :])


def load_and_preprocess(mat_path: Path, monkey: str, stim_path: Path, n_jobs: int = 128):
    """Load raw data in parallel, time-average, baseline-subtract per ROI."""

    batch_size = 100

    with h5py.File(mat_path, "r") as f:
        tb = np.array(f["tb"]).flatten()
        allmat = np.array(f["ALLMAT"])
        n_trials = f["ALLMUA"].shape[1]
        stim_ids = allmat[1].astype(np.int32)

    # Precompute time indices
    time_indices = {}
    for roi in ["v1", "v4", "it"]:
        t_start, t_end = TIME_WINDOWS[roi]
        baseline_idx = np.where((tb >= BASELINE_WINDOW[0]) & (tb < BASELINE_WINDOW[1]))[0]
        response_idx = np.where((tb > t_start) & (tb <= t_end))[0]
        time_indices[roi] = (baseline_idx, response_idx)

    # Create batch ranges
    batches = [(i, min(i + batch_size, n_trials)) for i in range(0, n_trials, batch_size)]
    print(f"  Loading {len(batches)} batches with {n_jobs} workers...")

    # Parallel load
    chunks = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_load_batch)(str(mat_path), start, end) for start, end in batches
    )

    # Process chunks and build results
    results = {f"data_{roi}": np.zeros((n_trials, ROI_SLICES[monkey][roi][1] - ROI_SLICES[monkey][roi][0]), dtype=np.float32)
               for roi in ["v1", "v4", "it"]}

    print("  Processing batches...")
    for (start, end), chunk in zip(batches, chunks):
        for roi in ["v1", "v4", "it"]:
            ch_start, ch_end = ROI_SLICES[monkey][roi]
            baseline_idx, response_idx = time_indices[roi]

            baseline = chunk[baseline_idx[0]:baseline_idx[-1]+1, :, ch_start:ch_end]
            response = chunk[response_idx[0]:response_idx[-1]+1, :, ch_start:ch_end]

            baseline_avg = np.nanmean(baseline, axis=0)
            response_avg = np.nanmean(response, axis=0)
            results[f"data_{roi}"][start:end] = (response_avg - baseline_avg).astype(np.float32)

    for roi in ["v1", "v4", "it"]:
        data = results[f"data_{roi}"]
        print(f"  {roi.upper()}: {data.shape}, range: [{data.min():.2f}, {data.max():.2f}]")

    # Load stimulus class names
    with h5py.File(stim_path, "r") as f:
        n_stim = f["test_imgs/class"].shape[0]
        classes = []
        for i in range(n_stim):
            ref = f["test_imgs/class"][i, 0]
            classes.append("".join(chr(int(c)) for c in f[ref][()].flatten() if c < 128))

    # Map trial stim_ids to class names
    zi_list = [classes[sid - 1] for sid in stim_ids]

    return results, zi_list, stim_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monkey", type=str, choices=["F", "N_new"], required=True)
    args = parser.parse_args()

    raw_dir = Path(__file__).parent.parent.parent / "data" / "things-monkey" / "THINGS-2k"
    out_dir = raw_dir / f"monkey{args.monkey.replace('_', '')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.monkey == "F":
        mat_path = raw_dir / "monkeyF" / "THINGS2_MUA_trials.mat"
        monkey_code = "F"
    else:
        mat_path = raw_dir / "monkeyN_new" / "THINGS2_new_MUA_trials.mat"
        monkey_code = "N"

    stim_path = raw_dir / "things2_imgs.mat"

    print(f"\n=== Preprocessing Monkey {args.monkey} ===")
    print(f"Input: {mat_path}")
    print(f"Output: {out_dir}")

    results, zi_list, stim_ids = load_and_preprocess(mat_path, monkey_code, stim_path)

    # Save in same format as MonkeyN original
    mat_out = out_dir / "THINGS_normMUA_raw.mat"
    print(f"\nSaving to {mat_out}...")
    with h5py.File(mat_out, "w") as f:
        for key, data in results.items():
            f.create_dataset(key, data=data, compression="gzip")

    zi_out = out_dir / "zi_list.csv"
    pd.Series(zi_list).to_csv(zi_out, index=False, header=False)
    print(f"Saved zi_list: {len(zi_list)} trials")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Trials: {len(zi_list)}")
    print(f"Unique classes: {len(set(zi_list))}")
    for roi in ["v1", "v4", "it"]:
        print(f"  {roi.upper()}: {results[f'data_{roi}'].shape}")

    print("\nDone!")


if __name__ == "__main__":
    main()
