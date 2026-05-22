#!/usr/bin/env python3
import os, sys
from pathlib import Path

_here = os.path.dirname(__file__)
for _u in (1,2,3,4):
    _cand = os.path.abspath(os.path.join(_here, *(['..']*_u)))
    if os.path.exists(os.path.join(_cand, 'config')):
        sys.path.append(_cand) if _cand not in sys.path else None
        break

from config.paths import config
from model.feature.core.vis_props_build import main as compute_vis_props

def extract_visual_features():
    """Extract and save visual properties to vis_props.tsv."""
    compute_vis_props()

if __name__ == '__main__':
    extract_visual_features()