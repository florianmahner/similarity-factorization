"""Step 3: Average per stimulus, filter channels, compute RSM.

Usage:
    poetry run python experiments/preprocessing/monkey_2k/step3_average_stimuli.py --recording N1
    poetry run python experiments/preprocessing/monkey_2k/step3_average_stimuli.py --recording N_combined
    poetry run python experiments/preprocessing/monkey_2k/step3_average_stimuli.py --recording F
"""

import argparse
import logging

import numpy as np

from src.datasets.monkey import (
    build_trial_tensor,
    combine_tensors,
    get_output_path,
    get_reliability_path,
    get_things_classes,
    load_intermediate,
)
from src.tools.metrics import gaussian_kernel_similarity

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", required=True, choices=["N1", "N2", "F", "N_combined"])
    parser.add_argument("--roi", default="it", choices=["it", "v1", "v4"])
    parser.add_argument("--min_reliability", type=float, default=0.3)
    args = parser.parse_args()

    things = get_things_classes()

    # Load reliability and create channel mask
    reliability = np.load(get_reliability_path(args.recording, args.roi))
    channel_mask = reliability > args.min_reliability
    log.info(f"Channels > {args.min_reliability}: {channel_mask.sum()}/{len(reliability)}")

    # Build trial tensor
    if args.recording == "N_combined":
        data1, labels1 = load_intermediate("N1", args.roi)
        data2, labels2 = load_intermediate("N2", args.roi)
        t1, n1 = build_trial_tensor(data1, labels1, things)
        t2, n2 = build_trial_tensor(data2, labels2, things)
        tensor, n_reps = combine_tensors(t1, n1, t2, n2)
    else:
        data, labels = load_intermediate(args.recording, args.roi)
        tensor, n_reps = build_trial_tensor(data, labels, things)

    # Average across repetitions
    data_avg = np.array([tensor[i, :n_reps[i], :].mean(axis=0) for i in range(len(things))])

    # Filter channels and compute RSM
    data_filtered = data_avg[:, channel_mask].astype(np.float32)
    log.info(f"Data shape: {data_filtered.shape}")

    log.info("Computing RSM (Gaussian kernel, median heuristic)...")
    rsm = gaussian_kernel_similarity(data_filtered, data_filtered)

    # Save
    out_path = get_output_path(args.recording, args.roi)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        data=data_filtered,
        rsm=rsm,
        stimuli=things,
        reliability=reliability[channel_mask],
        channel_mask=channel_mask,
    )
    log.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
