"""Step 1: Load time-averaged data (or compute from time-resolved if needed).

Usage:
    poetry run python experiments/preprocessing/monkey_2k/step1_time_average.py --recording N1
    poetry run python experiments/preprocessing/monkey_2k/step1_time_average.py --recording N2
    poetry run python experiments/preprocessing/monkey_2k/step1_time_average.py --recording F
"""

import argparse
import logging

import numpy as np

from src.datasets.monkey import (
    DATA_DIR,
    RECORDINGS,
    get_intermediate_path,
    load_stimulus_classes,
    load_time_averaged,
    load_time_resolved,
    load_zi_list,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", required=True, choices=["N1", "N2", "F"])
    parser.add_argument("--roi", default="it", choices=["it", "v1", "v4"])
    parser.add_argument("--force_recompute", action="store_true", help="Recompute from time-resolved even if averaged exists")
    args = parser.parse_args()

    rec = RECORDINGS[args.recording]
    rec_dir = DATA_DIR / rec["folder"]

    time_averaged_path = rec_dir / "THINGS_normMUA_raw.mat"
    time_resolved_path = rec_dir / "THINGS_normMUA_time_resolved.mat"

    # Prefer time-averaged if it exists
    if time_averaged_path.exists() and not args.force_recompute:
        log.info(f"Loading {args.recording} {args.roi.upper()} (time-averaged)...")
        data = load_time_averaged(time_averaged_path, args.roi)
        stim_labels = load_zi_list(args.recording)

    elif time_resolved_path.exists():
        log.info(f"Loading {args.recording} {args.roi.upper()} (computing from time-resolved)...")
        data, stim_ids = load_time_resolved(time_resolved_path, rec["monkey"], args.roi)
        classes, _ = load_stimulus_classes()
        stim_labels = np.array([classes[sid - 1] for sid in stim_ids])

    else:
        raise FileNotFoundError(f"No data found in {rec_dir}")

    # Remove NaN trials
    valid = ~np.isnan(data).any(axis=1)
    if valid.sum() < len(valid):
        log.info(f"  Removing {(~valid).sum()} trials with NaN")
        data, stim_labels = data[valid], stim_labels[valid]

    log.info(f"  Shape: {data.shape}, stimuli: {len(np.unique(stim_labels))}")

    # Save
    out_path = get_intermediate_path(args.recording, args.roi)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, data=data, stim_labels=stim_labels)
    log.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
