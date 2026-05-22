#!/usr/bin/env python3
"""
Shared helpers for SymNMF utilities (guard, gallery, viz).
"""

import os
import sys
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config.paths import config

FAMILIES = ['all', 'human', 'monkey']


def load_admm_results():
    res_fp = Path(config.results_dir) / 'admm_results' / 'admm_all_results.pkl'
    if not res_fp.exists():
        raise FileNotFoundError(f"ADMM results not found at {res_fp}")
    with open(res_fp, 'rb') as f:
        data = pickle.load(f)
    return data['results'], data['all_stims']


def load_cca_state():
    fp = Path(config.cache_file)
    if not fp.exists():
        raise FileNotFoundError(f"CCA state not found at {fp}")
    with open(fp, 'rb') as f:
        return pickle.load(f)


def pick_rank(family, result):
    custom = config.hyperparameters.get('admm', {}).get('custom_ranks', {})
    target = custom.get(family, result['best_rank'])
    ranks = list(result['components'].keys())
    if target in ranks:
        return target
    return result['best_rank']


def derive_categories(stims):
    cats = []
    for s in stims:
        cats.append('_'.join(s.split('_')[:-1]) if '_' in s else s)
    return cats


def build_img_paths(stims):
    cats = derive_categories(stims)
    img_root = Path(config.data_dir) / 'things' / 'images'
    return [img_root / cat / f"{stim}.jpg" for stim, cat in zip(stims, cats)]


def guard_views_for_family(family, cca_state):
    if family == 'all':
        comps = cca_state['cca_all']['components']
        human = [k for k in comps if k.lower().startswith('human')]
        monkey = [k for k in comps if k.lower().startswith('monkey')]
    else:
        comps = {
            **cca_state['cca_hum']['components'],
            **cca_state['cca_mon']['components'],
        }
        human = list(cca_state['cca_hum']['components'].keys())
        monkey = list(cca_state['cca_mon']['components'].keys())
    return comps, human, monkey


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def guard_csv_path(family: str) -> Path:
    return Path(__file__).resolve().parent / 'results' / 'guard' / f"guard_{family}.csv"


def load_guard_table(family: str) -> pd.DataFrame:
    path = guard_csv_path(family)
    if not path.exists():
        raise FileNotFoundError(f"Guard CSV missing for {family}: {path}")
    return pd.read_csv(path)
