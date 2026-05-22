#!/usr/bin/env python3
"""Decode annotations from V1/V4 cross-species CCA spaces.

Runs computation and saves results. Use viz_decoding.py to visualize.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
from pathlib import Path

_here = Path(__file__).resolve()
for up in (1, 2, 3, 4):
    cand = _here.parents[up - 1]
    if (cand / 'config').exists():
        sys.path.append(str(cand))
        break

from config.paths import config
from symbol.decode import run_region


STATE_DIR = Path(config.root_dir) / 'symbol' / 'results'
REGION_STATES = {
    'v1': STATE_DIR / 'v1_cca_state.pkl',
    'v4': STATE_DIR / 'v4_cca_state.pkl',
}


def main():
    for region, path in REGION_STATES.items():
        if not path.exists():
            print(f'Warning: Missing CCA state for {region}: {path}')
            print(f'  Run build_v1v4_cca.py first to generate V1/V4 CCA states.')
            continue
        run_region(path, region)

    print("\nAll regions complete. Use viz_decoding.py to visualize results.")


if __name__ == '__main__':
    main()
