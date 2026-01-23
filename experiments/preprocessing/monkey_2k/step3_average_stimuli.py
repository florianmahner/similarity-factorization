"""Step 3: Average per stimulus, filter channels, compute RSM.

Usage:
    poetry run python experiments/preprocessing/monkey_2k/step3_average_stimuli.py --recording N1
    poetry run python experiments/preprocessing/monkey_2k/step3_average_stimuli.py --recording N_combined
    poetry run python experiments/preprocessing/monkey_2k/step3_average_stimuli.py --recording N_concat
    poetry run python experiments/preprocessing/monkey_2k/step3_average_stimuli.py --recording F

N_concat: Compute reliability separately for N1 and N2, filter each by cutoff,
          then concatenate channels horizontally -> (1854, n_N1 + n_N2).
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
    parser.add_argument(
        "--recording",
        required=True,
        choices=["N1", "N2", "F", "N_combined", "N_concat"],
    )
    parser.add_argument("--roi", default="it", choices=["it", "v1", "v4"])
    parser.add_argument("--min_reliability", type=float, default=0.3)
    args = parser.parse_args()

    things = get_things_classes()

    if args.recording == "N_concat":
        # Concatenate channels from N1 and N2, filtered separately by reliability
        rel1 = np.load(get_reliability_path("N1", args.roi))
        rel2 = np.load(get_reliability_path("N2", args.roi))
        mask1 = rel1 > args.min_reliability
        mask2 = rel2 > args.min_reliability
        log.info(f"N1 channels > {args.min_reliability}: {mask1.sum()}/{len(rel1)}")
        log.info(f"N2 channels > {args.min_reliability}: {mask2.sum()}/{len(rel2)}")

        # Load and process each session
        data1, labels1 = load_intermediate("N1", args.roi)
        data2, labels2 = load_intermediate("N2", args.roi)
        t1, n1 = build_trial_tensor(data1, labels1, things)
        t2, n2 = build_trial_tensor(data2, labels2, things)

        # Average per stimulus for each session
        avg1 = np.array([t1[i, :n1[i], :].mean(axis=0) for i in range(len(things))])
        avg2 = np.array([t2[i, :n2[i], :].mean(axis=0) for i in range(len(things))])

        # Filter and concatenate
        data_filtered = np.hstack([avg1[:, mask1], avg2[:, mask2]]).astype(np.float32)
        reliability = np.concatenate([rel1[mask1], rel2[mask2]])
        channel_mask = np.concatenate([mask1, mask2])
        log.info(f"Data shape: {data_filtered.shape} (N1: {mask1.sum()}, N2: {mask2.sum()})")

    else:
        # Standard processing for single recording or N_combined
        reliability = np.load(get_reliability_path(args.recording, args.roi))
        channel_mask = reliability > args.min_reliability
        log.info(f"Channels > {args.min_reliability}: {channel_mask.sum()}/{len(reliability)}")

        if args.recording == "N_combined":
            data1, labels1 = load_intermediate("N1", args.roi)
            data2, labels2 = load_intermediate("N2", args.roi)
            t1, n1 = build_trial_tensor(data1, labels1, things)
            t2, n2 = build_trial_tensor(data2, labels2, things)
            tensor, n_reps = combine_tensors(t1, n1, t2, n2)
        else:
            data, labels = load_intermediate(args.recording, args.roi)
            tensor, n_reps = build_trial_tensor(data, labels, things)

        data_avg = np.array([tensor[i, :n_reps[i], :].mean(axis=0) for i in range(len(things))])
        data_filtered = data_avg[:, channel_mask].astype(np.float32)
        reliability = reliability[channel_mask]
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
        reliability=reliability,
        channel_mask=channel_mask,
    )
    log.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
