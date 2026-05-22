import os, sys
import pandas as pd
import numpy as np
# Project imports (robust sys.path)
_here = os.path.dirname(__file__)
for _u in (1,2,3,4):
    _cand = os.path.abspath(os.path.join(_here, *(['..']*_u)))
    if os.path.exists(os.path.join(_cand, 'config')):
        sys.path.append(_cand) if _cand not in sys.path else None
        break
from config.paths import config
import pickle

def load_mri(sub, roi=None, min_splithalf=None):
    """Load MRI data; optional ROI filter and split-half reliability threshold."""
    paths = config.get_mri_paths(sub)
    
    # Base matrices and metadata
    data = pd.read_hdf(paths['betas']).drop(columns='voxel_id', errors='ignore').T.astype('float32')
    stim_info = pd.read_csv(paths['stiminfo'])
    voxmeta = pd.read_csv(paths['voxmeta'])
    
    # Build voxel mask: reliability → ROI (if provided)
    mask = np.ones(len(voxmeta), dtype=bool)
    if min_splithalf is not None:
        mask &= voxmeta['splithalf_corrected'] >= min_splithalf
    if roi is not None:
        if isinstance(roi, str):
            roi = [roi]
        mask &= voxmeta[roi].any(axis=1)
    data = data.loc[:, mask].values
    col = 'exemplar' if 'exemplar' in stim_info.columns else 'exemplar_name'
    stims = stim_info[col].astype(str).tolist()
    assert len(stims) == data.shape[0], "rows != #stims"
    return data, stims, mask

def get_channel_mask(monkey, roi=None):
    if roi is None:
        return slice(None)
        
    if monkey == 'N': # https://gin.g-node.org/paolo_papale/TVSD/src/master/_code/norm_MUA.m
        masks = {
            'v1': slice(0, 512),
            'v4': slice(512, 768),
            'it': slice(768, 1024)
        }
    else:  # monkey F
        masks = {
            'v1': slice(0, 512),
            'it': slice(512, 832),
            'v4': slice(832, 1024)
        }
    return masks[roi.lower()]

def load_macq(monkey, min_reliab=None, roi=None):
    paths = config.get_macq_paths(monkey)
    
    monkey_data = np.load(paths['data'], allow_pickle=True).item()
    data = monkey_data['train_MUA'].astype('float32').T
    reliab = monkey_data['reliab'].mean(axis=1)
    
    # ROI selection and track kept channel indices
    roi_mask = get_channel_mask(monkey, roi)
    if isinstance(roi_mask, slice):
        inc_chans = np.arange(data.shape[1])[roi_mask]
    else:
        inc_chans = np.where(roi_mask)[0]
    
    data = data[:, roi_mask]
    reliab = reliab[roi_mask]

    # Reliability threshold (if provided)
    if min_reliab is not None:
        reliab_mask = reliab >= min_reliab
        data = data[:, reliab_mask]
        inc_chans = inc_chans[reliab_mask]

    stim_info = pd.read_csv(paths['stiminfo'])
    stims = stim_info['exemplar'].tolist()
    assert len(stims) == data.shape[0], "rows != #stims"
    return data, stims, inc_chans

def load_macq_tr(monkey, time_window=None, min_reliab=None, roi=None):
    paths_tr = config.get_macq_tr_paths(monkey)
    paths_avg = config.get_macq_paths(monkey)

    with open(paths_tr['data'], 'rb') as f:
        data_dict = pickle.load(f)
    
    data = data_dict['mua'].astype('float32')
    timepoints = data_dict['timepoints']
    stim_info = pd.read_csv(paths_tr['stiminfo'])

    monkey_data_avg = np.load(paths_avg['data'], allow_pickle=True).item()
    reliab = monkey_data_avg['reliab'].mean(axis=1)

    # ROI selection
    roi_mask = get_channel_mask(monkey, roi)
    data = data[:, :, roi_mask]
    reliab = reliab[roi_mask]  # Also subset reliability scores

    # Reliability threshold (if provided)
    if min_reliab is not None:
        reliab_mask = reliab >= min_reliab
        data = data[:, :, reliab_mask]
    
    if time_window is not None:
        start_ms, end_ms = time_window
        time_mask = (timepoints >= start_ms) & (timepoints <= end_ms)
        data = data[time_mask]
        timepoints = timepoints[time_mask]
    
    stims = stim_info['exemplar'].tolist()
    assert len(stims) == data.shape[1], "trials != #stims"
        
    return data, stims, timepoints

def match_datasets(named_datasets: dict):
    """Intersect and align datasets on shared stimulus labels.

    Accepts dict of name -> (data, stims, ...). Extra items are ignored.
    Returns dict name -> aligned data, and the ordered list of shared stims.
    """
    # Shared items across datasets
    stims_sets = [set(map(str, v[1])) for v in named_datasets.values()]
    common_stims = sorted(set.intersection(*stims_sets))
    
    assert len(common_stims) > 0, "No common items found"
    
    # Index and align each dataset to common order
    matched = {}
    indices_by_dataset = {}
    for name, vals in named_datasets.items():
        data, stims = vals[0], list(map(str, vals[1]))
        indices = [stims.index(s) for s in common_stims]
        matched[name] = data[indices]
        indices_by_dataset[name] = indices
    
    # Exclude any items with NaNs in any dataset
    valid_rows = np.all([~np.isnan(data).any(axis=1) for data in matched.values()], axis=0)
    common_stims = [s for i, s in enumerate(common_stims) if valid_rows[i]]
    matched = {name: data[valid_rows] for name, data in matched.items()}
    
    # Verify matching
    lengths = {name: data.shape[0] for name, data in matched.items()}
    assert len(set(lengths.values())) == 1, f"Mismatched lengths: {lengths}"
    assert all(len(data) == len(common_stims) for data in matched.values()), "Data length doesn't match stimuli"
    
    print(f"Found {len(common_stims)} common items")
    print("First 5:", common_stims[:5])
    
    return matched, common_stims
