"""Step 2: Compute split-half reliability.

Usage:
    poetry run python experiments/preprocessing/monkey_2k/step2_reliability.py --recording N1
    poetry run python experiments/preprocessing/monkey_2k/step2_reliability.py --recording N_combined
    poetry run python experiments/preprocessing/monkey_2k/step2_reliability.py --recording F
"""

import argparse
import logging

import numpy as np

from src.datasets.monkey import (
    build_trial_tensor,
    combine_tensors,
    compute_reliability,
    get_reliability_path,
    get_things_classes,
    load_intermediate,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", required=True, choices=["N1", "N2", "F", "N_combined"])
    parser.add_argument("--roi", default="it", choices=["it", "v1", "v4"])
    parser.add_argument("--n_splits", type=int, default=1000)
    args = parser.parse_args()

    things = get_things_classes()
    log.info(f"Target: {len(things)} THINGS stimuli")

    # Build trial tensor
    if args.recording == "N_combined":
        log.info("Loading N1 + N2...")
        data1, labels1 = load_intermediate("N1", args.roi)
        data2, labels2 = load_intermediate("N2", args.roi)

        t1, n1 = build_trial_tensor(data1, labels1, things)
        t2, n2 = build_trial_tensor(data2, labels2, things)
        tensor, n_reps = combine_tensors(t1, n1, t2, n2)

        log.info(f"  N1: reps {n1.min()}-{n1.max()}, N2: reps {n2.min()}-{n2.max()}")
    else:
        log.info(f"Loading {args.recording}...")
        data, labels = load_intermediate(args.recording, args.roi)
        tensor, n_reps = build_trial_tensor(data, labels, things)

    log.info(f"  Tensor: {tensor.shape}, reps {n_reps.min()}-{n_reps.max()}")

    # Compute reliability
    log.info(f"Computing reliability ({args.n_splits} splits)...")
    reliability = compute_reliability(tensor, n_reps, args.n_splits)

    log.info(f"  Median: {np.median(reliability):.3f}")
    log.info(f"  Channels > 0.3: {np.sum(reliability > 0.3)}/{len(reliability)}")

    # Save
    out_path = get_reliability_path(args.recording, args.roi)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, reliability)
    log.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
