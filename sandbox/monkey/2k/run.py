#!/usr/bin/env python3
"""Preprocess THINGS-2k monkey MUA data to organized trial format.

Output structure:
    data/things-monkey/2k/processed/
    ├── monkeyN_session1/
    │   ├── things.npz      # THINGS objects
    │   └── faces.npz       # monkey + human faces
    ├── monkeyN_session2/
    │   └── ...
    └── monkeyF/
        └── ...

Each NPZ contains:
    - it, v1, v4: (n_stimuli, max_reps, n_channels) per ROI
    - stimuli: (n_stimuli,) stimulus names
    - n_reps: (n_stimuli,) repetitions per stimulus
"""

import argparse
import h5py
import numpy as np
from pathlib import Path

ROI_SLICES = {
    "N": {"v1": (0, 512), "v4": (512, 768), "it": (768, 1024)},
    "F": {"v1": (0, 512), "it": (512, 832), "v4": (832, 1024)},
}
TIME_WINDOWS = {"v1": (25, 125), "v4": (50, 150), "it": (75, 175)}
BASELINE_WINDOW = (-100, 0)


def load_stim_classes(stim_path: Path) -> list[str]:
    """Load stimulus class names from things2_imgs.mat."""
    with h5py.File(stim_path, "r") as f:
        n = f["test_imgs/class"].shape[0]
        classes = []
        for i in range(n):
            ref = f["test_imgs/class"][i, 0]
            name = "".join(chr(int(c)) for c in f[ref][()].flatten() if c < 128)
            classes.append(name)
    return classes


def classify_stimulus(name: str) -> str:
    """Classify stimulus as 'things' or 'face'.

    Faces are: monkey01-73, person01-73
    THINGS includes 'monkey' (the animal, not a face)
    """
    nl = name.lower()
    # Monkey faces: monkey01, monkey02, ... (but NOT 'monkey' alone)
    if nl.startswith("monkey") and len(nl) > 6:
        return "face"
    # Human faces: person01, person02, ... (including typo person08jpg)
    elif nl.startswith("person"):
        return "face"
    return "things"


def time_average_roi(allmua: np.ndarray, tb: np.ndarray, roi: str, monkey: str) -> np.ndarray:
    """Extract ROI and time-average with baseline subtraction."""
    ch_start, ch_end = ROI_SLICES[monkey][roi]
    t_start, t_end = TIME_WINDOWS[roi]

    baseline_idx = np.where((tb >= BASELINE_WINDOW[0]) & (tb < BASELINE_WINDOW[1]))[0]
    response_idx = np.where((tb > t_start) & (tb <= t_end))[0]

    baseline = allmua[baseline_idx][:, :, ch_start:ch_end]
    response = allmua[response_idx][:, :, ch_start:ch_end]

    baseline_avg = np.nanmean(baseline, axis=0)
    response_avg = np.nanmean(response, axis=0)

    return (response_avg - baseline_avg).astype(np.float32)


def organize_by_stimulus(
    data: dict[str, np.ndarray], stim_ids: np.ndarray, classes: list[str]
) -> tuple[dict, dict]:
    """Organize trial data by stimulus, split into things vs faces."""
    unique_stim = np.unique(stim_ids)

    things_idx, faces_idx = [], []
    things_names, faces_names = [], []

    for stim_id in unique_stim:
        name = classes[stim_id - 1]  # 1-indexed
        cat = classify_stimulus(name)
        if cat == "things":
            things_idx.append(stim_id)
            things_names.append(name)
        else:
            faces_idx.append(stim_id)
            faces_names.append(name)

    def build_output(stim_list: list, names: list) -> dict:
        if len(stim_list) == 0:
            return None

        n_stim = len(stim_list)
        max_reps = max(np.sum(stim_ids == s) for s in stim_list)
        n_reps = np.zeros(n_stim, dtype=np.int32)

        out = {"stimuli": np.array(names), "n_reps": n_reps}

        for roi, roi_data in data.items():
            n_ch = roi_data.shape[1]
            arr = np.full((n_stim, max_reps, n_ch), np.nan, dtype=np.float32)

            for i, stim_id in enumerate(stim_list):
                trials = roi_data[stim_ids == stim_id]
                n_reps[i] = len(trials)
                arr[i, :n_reps[i], :] = trials

            out[roi] = arr

        return out

    things_out = build_output(things_idx, things_names)
    faces_out = build_output(faces_idx, faces_names)

    return things_out, faces_out


def process_time_resolved(mat_path: Path, stim_path: Path, monkey: str) -> tuple[dict, dict]:
    """Process time-resolved data to organized trial format."""
    print(f"Loading {mat_path.name}...")
    classes = load_stim_classes(stim_path)

    with h5py.File(mat_path, "r") as f:
        tb = np.array(f["tb"]).flatten()
        stim_ids = np.array(f["ALLMAT"][1]).astype(np.int32)
        allmua = np.array(f["ALLMUA"])

    print(f"  Shape: {allmua.shape}, {len(np.unique(stim_ids))} stimuli")

    data = {}
    for roi in ["it", "v1", "v4"]:
        print(f"  Processing {roi.upper()}...")
        data[roi] = time_average_roi(allmua, tb, roi, monkey)

    return organize_by_stimulus(data, stim_ids, classes)


def process_time_averaged(mat_path: Path, zi_path: Path, monkey: str) -> tuple[dict, dict]:
    """Process already time-averaged data (monkeyN_session1)."""
    print(f"Loading {mat_path.name}...")

    import pandas as pd
    labels = pd.read_csv(zi_path, header=None)[0].values

    with h5py.File(mat_path, "r") as f:
        keys = list(f.keys())
        print(f"  Available keys: {keys}")

        data = {}
        for roi in ["it", "v1", "v4"]:
            key = f"data_{roi}"
            if key in f:
                data[roi] = np.array(f[key]).astype(np.float32)
                print(f"  {roi.upper()}: {data[roi].shape}")

    unique_labels = np.unique(labels)
    n_stim = len(unique_labels)
    print(f"  {n_stim} unique stimuli, {len(labels)} trials")

    things_idx, faces_idx = [], []
    things_names, faces_names = [], []

    for i, name in enumerate(unique_labels):
        cat = classify_stimulus(name)
        if cat == "things":
            things_idx.append(i)
            things_names.append(name)
        else:
            faces_idx.append(i)
            faces_names.append(name)

    def build_output(indices: list, names: list) -> dict:
        if len(indices) == 0:
            return None

        stim_subset = unique_labels[indices]
        max_reps = max(np.sum(labels == s) for s in stim_subset)
        n_reps = np.zeros(len(indices), dtype=np.int32)

        out = {"stimuli": np.array(names), "n_reps": n_reps}

        for roi, roi_data in data.items():
            n_ch = roi_data.shape[1]
            arr = np.full((len(indices), max_reps, n_ch), np.nan, dtype=np.float32)

            for i, stim in enumerate(stim_subset):
                trials = roi_data[labels == stim]
                n_reps[i] = len(trials)
                arr[i, :n_reps[i], :] = trials

            out[roi] = arr

        return out

    things_out = build_output(things_idx, things_names)
    faces_out = build_output(faces_idx, faces_names)

    return things_out, faces_out


def save_output(data: dict, path: Path):
    """Save organized data to NPZ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)

    # Summary
    n_stim = len(data["stimuli"])
    rois = [k for k in data.keys() if k in ["it", "v1", "v4"]]
    shapes = {roi: data[roi].shape for roi in rois}
    print(f"  Saved {path.name}: {n_stim} stimuli, {shapes}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess THINGS-2k monkey data")
    parser.add_argument(
        "--monkey",
        type=str,
        choices=["N1", "N2", "F", "all"],
        default="all",
        help="Which recording to process (N1=session1, N2=session2, F=monkeyF)",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent.parent.parent / "data" / "things-monkey" / "2k"
    out_dir = data_dir / "processed"
    stim_path = data_dir / "things2_imgs.mat"

    recordings = {
        "N1": ("monkeyN_session1", "N", False),  # (folder, monkey_id, is_time_resolved)
        "N2": ("monkeyN_session2", "N", True),
        "F": ("monkeyF", "F", True),
    }

    to_process = list(recordings.keys()) if args.monkey == "all" else [args.monkey]

    for rec_id in to_process:
        folder, monkey, is_time_resolved = recordings[rec_id]
        rec_dir = data_dir / folder
        out_rec_dir = out_dir / folder

        print(f"\n{'='*60}")
        print(f"Processing {folder}")
        print(f"{'='*60}")

        if is_time_resolved:
            mat_path = rec_dir / "THINGS_normMUA_time_resolved.mat"
            things, faces = process_time_resolved(mat_path, stim_path, monkey)
        else:
            mat_path = rec_dir / "THINGS_normMUA_raw.mat"
            zi_path = rec_dir / "zi_list.csv"
            things, faces = process_time_averaged(mat_path, zi_path, monkey)

        if things is not None:
            save_output(things, out_rec_dir / "things.npz")
        if faces is not None:
            save_output(faces, out_rec_dir / "faces.npz")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
