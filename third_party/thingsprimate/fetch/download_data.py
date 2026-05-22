#!/usr/bin/env python3
"""
Download THINGS dataset components in order of size/priority.
Run with: python download_things_data.py
"""

import os
import sys
import urllib.request
import zipfile
import requests
import numpy as np
import pandas as pd
import h5py
import pickle
import shutil
from tqdm import tqdm
from datetime import datetime

# Project imports (robust sys.path)
_here = os.path.dirname(__file__)
for _u in (1,2,3,4):
    _cand = os.path.abspath(os.path.join(_here, *(['..']*_u)))
    if os.path.exists(os.path.join(_cand, 'config')):
        sys.path.append(_cand) if _cand not in sys.path else None
        break

from config.paths import config

def log_msg(msg):
   """Print timestamped message"""
   print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def fetch_zip(url, dest, pwd=None, skip_if_exists=True, flatten=None):
   """Download and extract zip file"""
   os.makedirs(dest, exist_ok=True)
   if skip_if_exists and os.path.isdir(dest) and os.listdir(dest):
       log_msg(f"Skip (exists): {dest}")
       return
   
   tmp = os.path.join(str(dest), "_tmp.zip")
   log_msg(f"Downloading to {dest}...")
   
   try:
       r = requests.get(url, stream=True)
       r.raise_for_status()
       total_size = int(r.headers.get('content-length', 0))
       
       with open(tmp, "wb") as f:
           with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest)) as pbar:
               for chunk in r.iter_content(8192):
                   f.write(chunk)
                   pbar.update(len(chunk))
       
       log_msg(f"Extracting {tmp}...")
       with zipfile.ZipFile(tmp, "r") as z:
           if pwd:
               z.extractall(dest, pwd=pwd.encode())
           else:
               z.extractall(dest)
       
       if flatten:
           src = os.path.join(dest, flatten)
           if os.path.isdir(src):
               for n in os.listdir(src):
                   shutil.move(os.path.join(src, n), os.path.join(dest, n))
               os.rmdir(src)
   finally:
       if os.path.exists(tmp):
           os.remove(tmp)
   
   log_msg(f"Completed: {dest}")

def fetch_file(url, dest):
   """Download single file"""
   os.makedirs(os.path.dirname(dest), exist_ok=True)
   if os.path.exists(dest):
       log_msg(f"Skip (exists): {dest}")
       return
   
   log_msg(f"Downloading {os.path.basename(dest)}...")
   r = requests.get(url)
   r.raise_for_status()
   with open(dest, "wb") as f:
       f.write(r.content)
   log_msg(f"Saved: {dest}")

def main():
   log_msg("="*60)
   log_msg("THINGS DATA DOWNLOAD SCRIPT")
   log_msg("="*60)
   
   # Create all directories
   log_msg("\n1. Creating directories...")
   os.makedirs(config.data_dir, exist_ok=True)
   os.makedirs(config.things_dir, exist_ok=True)
   os.makedirs(config.macq_dir, exist_ok=True)
   os.makedirs(config.timeavg_dir, exist_ok=True)
   os.makedirs(config.timeres_dir, exist_ok=True)
   os.makedirs(config.mri_dir, exist_ok=True)
   log_msg("Directories created")
   
   # =========================================================================
   # PHASE 1: LIGHTWEIGHT DATA AND IMAGES (things/)
   # =========================================================================
   log_msg("\n" + "="*60)
   log_msg("PHASE 1: LIGHTWEIGHT THINGS DATA AND IMAGES")
   log_msg("="*60)
   
   # 1.1 Property and metadata TSV files
   log_msg("\n1.1 Downloading property and metadata files...")
   things_property_path = os.path.join(config.things_dir, 'things_property.tsv')
   things_metadata_path = os.path.join(config.things_dir, 'things_metadata.tsv')
   
   if not os.path.exists(things_property_path):
       urllib.request.urlretrieve('https://osf.io/download/7cz69/', things_property_path)
       log_msg("Downloaded things_property.tsv")
   
   if not os.path.exists(things_metadata_path):
       urllib.request.urlretrieve('https://osf.io/download/um6a9/', things_metadata_path)
       log_msg("Downloaded things_metadata.tsv")
   
   # Process properties
   log_msg("Processing THINGS properties...")
   properties_df = pd.read_csv(things_property_path, sep='\t')
   metadata_df = pd.read_csv(things_metadata_path, sep='\t')
   
   mean_property_columns = [
       c for c in properties_df.columns 
       if c.startswith('property_') and c.endswith('_mean')
   ]
   properties_simple = properties_df[['uniqueID'] + mean_property_columns].copy()
   properties_simple.columns = [
       col.replace('property_', '').replace('_mean', '') 
       if col != 'uniqueID' else col
       for col in properties_simple.columns
   ]
   
   properties_simple['uniqueID'] = properties_simple['uniqueID'].astype(str)
   metadata_df['Word'] = metadata_df['Word'].astype(str)
   
   categories_simple = metadata_df[['Word', 'All Bottom-up Categories']]
   things_properties = (
       properties_simple
       .merge(categories_simple, left_on='uniqueID', right_on='Word', how='left')
       .drop(columns=['Word'])
       .rename(columns={'All Bottom-up Categories': 'categories'})
   )
   
   things_properties.to_csv(
       os.path.join(config.things_dir, 'things_properties.csv'),
       index=False
   )
   log_msg(f"Saved processed properties ({len(things_properties)} items)")
   
   # 1.2 Categories TSV
   log_msg("\n1.2 Downloading categories TSV...")
   fetch_file(
       "https://raw.githubusercontent.com/ViCCo-Group/dimension_encoding/master/data/Categories_final_20200131_fixedUniqueID.tsv",
       os.path.join(config.things_dir, "Categories_final_20200131_fixedUniqueID.tsv")
   )
   
   # 1.3 Level folders (metadata)
   log_msg("\n1.3 Downloading level folders...")
   level_downloads = [
       ("https://files.osf.io/v1/resources/jum2f/providers/osfstorage/66d068fc9e146696878c987e/?zip=", "01_image-level"),
       ("https://files.osf.io/v1/resources/jum2f/providers/osfstorage/66d068c4f6f9282a0989c1e1/?zip=", "02_object-level"),
       ("https://files.osf.io/v1/resources/jum2f/providers/osfstorage/66d06922c27bea2c64180627/?zip=", "03_category-level"),
   ]
   
   for url, subfolder in level_downloads:
       dest_path = os.path.join(config.things_dir, subfolder)
       fetch_zip(url, dest_path, skip_if_exists=True)
   
   # 1.4 Behavioral embeddings
   log_msg("\n1.4 Downloading behavioral embeddings...")
   fetch_zip(
       "https://files.osf.io/v1/resources/f5rn6/providers/osfstorage/62d6b2b5f66a943a782311a1/?zip=",
       os.path.join(config.things_dir, "behav_embed", "data")
   )
   fetch_zip(
       "https://files.osf.io/v1/resources/f5rn6/providers/osfstorage/62d6b2edf66a943a71230223/?zip=",
       os.path.join(config.things_dir, "behav_embed", "variables")
   )
   
   # 1.5 THINGS images (encrypted)
   log_msg("\n1.5 Downloading THINGS images (this may take a while)...")
   fetch_zip(
       "https://osf.io/download/rdxy2/",
       os.path.join(config.things_dir, "images"),
       pwd="things4all",
       flatten="object_images"
   )
   
   # =========================================================================
   # PHASE 2: TIME-AVERAGED MACAQUE DATA
   # =========================================================================
   log_msg("\n" + "="*60)
   log_msg("PHASE 2: TIME-AVERAGED MACAQUE DATA")
   log_msg("="*60)
   
   get_macq_paths = config.get_macq_paths
   
   macaque_urls = {
       'F': {
           'data_url': 'https://gin.g-node.org/paolo_papale/TVSD/raw/master/monkeyF/THINGS_normMUA.mat',
           'img_url': 'https://gin.g-node.org/paolo_papale/TVSD/raw/master/monkeyF/_logs/things_imgs.mat'
       },
       'N': {
           'data_url': 'https://gin.g-node.org/paolo_papale/TVSD/raw/master/monkeyN/THINGS_normMUA.mat',
           'img_url': 'https://gin.g-node.org/paolo_papale/TVSD/raw/master/monkeyN/_logs/things_imgs.mat'
       }
   }
   
   data_keys = [
       'SNR', 'SNR_max', 'lats', 'reliab', 'oracle',
       'train_MUA', 'test_MUA', 'test_MUA_reps', 'tb'
   ]
   img_keys = ['class', 'things_path']
   
   for monkey, urls in macaque_urls.items():
       log_msg(f"\nProcessing Monkey {monkey}...")
       
       data_mat = os.path.join(config.timeavg_dir, f'monkey{monkey}_THINGS_normMUA.mat')
       img_mat = os.path.join(config.timeavg_dir, f'monkey{monkey}_things_imgs.mat')
       
       # Download if needed
       if not os.path.exists(data_mat):
           log_msg(f"Downloading neural data for monkey {monkey}...")
           r = requests.get(urls['data_url'])
           with open(data_mat, 'wb') as f:
               f.write(r.content)
       
       if not os.path.exists(img_mat):
           log_msg(f"Downloading image data for monkey {monkey}...")
           r = requests.get(urls['img_url'])
           with open(img_mat, 'wb') as f:
               f.write(r.content)
       
       # Process data
       log_msg(f"Processing data for monkey {monkey}...")
       image_data = {}
       with h5py.File(img_mat, 'r') as f_img:
           for split in ['train_imgs', 'test_imgs']:
               prefix = split.split('_')[0]
               n_items = f_img[f'{split}/class'].shape[0]
               for key in img_keys:
                   ds = f_img[f'{split}/{key}']
                   strs = []
                   for i in range(n_items):
                       ref = ds[i, 0]
                       arr = f_img[ref][()]
                       codes = arr.flatten().astype(int)
                       strs.append(''.join(chr(c) for c in codes if c < 128))
                   image_data[f'{prefix}_{key}'] = strs
       
       with h5py.File(data_mat, 'r') as f_data:
           neural_data = {k: np.array(f_data[k]).T for k in data_keys}
       
       neural_data.update(image_data)
       np.save(get_macq_paths(monkey)['data'], neural_data)
       
       # Clean up
       os.remove(data_mat)
       os.remove(img_mat)
       
       # Generate stiminfo CSV
       paths = get_macq_paths(monkey)
       data = np.load(paths['data'], allow_pickle=True).item()
       cats = data['train_class']
       exs = [
           os.path.splitext(os.path.basename(p.replace('\\', '/')))[0]
           for p in data['train_things_path']
       ]
       stiminfo_df = pd.DataFrame({'category': cats, 'exemplar': exs})
       stiminfo_df.to_csv(paths['stiminfo'], index=False)
       log_msg(f"Monkey {monkey}: {len(exs)} stimuli processed")
   
   # =========================================================================
   # PHASE 3: fMRI DATA
   # =========================================================================
   log_msg("\n" + "="*60)
   log_msg("PHASE 3: fMRI DATA")
   log_msg("="*60)
   
   mri_masks_zip = os.path.join(config.mri_dir, "brainmasks.zip")
   mri_betas_zip = os.path.join(config.mri_dir, "betas_csv.zip")
   
   # Brain masks
   log_msg("\n3.1 Downloading brain masks...")
   if not os.path.exists(os.path.join(config.mri_dir, 'brainmasks')):
       resp = requests.get("https://plus.figshare.com/ndownloader/files/36682242", stream=True)
       total_size = int(resp.headers.get('content-length', 0))
       
       with open(mri_masks_zip, 'wb') as f:
           with tqdm(total=total_size, unit='B', unit_scale=True, desc="Brain masks") as pbar:
               for chunk in resp.iter_content(8192):
                   f.write(chunk)
                   pbar.update(len(chunk))
       
       with zipfile.ZipFile(mri_masks_zip, 'r') as z:
           z.extractall(config.mri_dir)
       os.remove(mri_masks_zip)
       log_msg("Brain masks extracted")
   
   # Beta CSVs
   log_msg("\n3.2 Downloading beta CSVs...")
   betas_dir = os.path.join(config.mri_dir, 'betas_csv')
   if not os.path.exists(betas_dir):
       resp = requests.get("https://plus.figshare.com/ndownloader/files/43635873", stream=True)
       total_size = int(resp.headers.get('content-length', 0))
       
       with open(mri_betas_zip, 'wb') as f:
           with tqdm(total=total_size, unit='B', unit_scale=True, desc="Beta CSVs") as pbar:
               for chunk in resp.iter_content(8192):
                   f.write(chunk)
                   pbar.update(len(chunk))
       
       with zipfile.ZipFile(mri_betas_zip, 'r') as z:
           z.extractall(config.mri_dir)
       os.remove(mri_betas_zip)
       log_msg("Beta CSVs extracted")
   
   # Generate fMRI stimulus metadata
   log_msg("\n3.3 Processing fMRI stimulus metadata...")
   get_mri_paths = config.get_mri_paths
   for subject in ['01', '02', '03']:
       paths = get_mri_paths(subject)
       meta_csv_path = os.path.join(betas_dir, f"sub-{subject}_StimulusMetadata.csv")
       metadata_df = pd.read_csv(meta_csv_path)
       
       categories = []
       exemplars = []
       for stim in metadata_df['stimulus']:
           exemplar = os.path.splitext(stim)[0]
           category = exemplar.rsplit('_', 1)[0]
           exemplars.append(exemplar)
           categories.append(category)
       
       stiminfo_df = pd.DataFrame({'category': categories, 'exemplar': exemplars})
       stiminfo_df.to_csv(paths['stiminfo'], index=False)
       log_msg(f"Subject {subject}: {len(exemplars)} stimuli processed")
   
   # =========================================================================
   # PHASE 4: TIME-RESOLVED MACAQUE DATA (LARGEST)
   # =========================================================================
   log_msg("\n" + "="*60)
   log_msg("PHASE 4: TIME-RESOLVED MACAQUE DATA (LARGEST)")
   log_msg("="*60)
   
   time_res_dir = config.timeres_dir
   get_macq_tr_paths = config.get_macq_tr_paths
   
   monkey_trials_info = {
       'F': 'https://gin.g-node.org/paolo_papale/TVSD/raw/master/monkeyF/THINGS_MUA_trials.mat',
       'N': 'https://gin.g-node.org/paolo_papale/TVSD/raw/master/monkeyN/THINGS_MUA_trials.mat'
   }
   
   for monkey, url in monkey_trials_info.items():
       log_msg(f"\nProcessing time-resolved data for Monkey {monkey}...")
       
       paths = get_macq_tr_paths(monkey)
       np_path = paths['data']
       mat_path = os.path.join(time_res_dir, f'monkey{monkey}_THINGS_MUA_trials.mat')
       
       # Download if needed
       if not os.path.exists(mat_path) and not os.path.exists(np_path):
           log_msg(f"Downloading time-resolved data for monkey {monkey} (this will take a while)...")
           resp = requests.get(url, stream=True)
           total = int(resp.headers.get('content-length', 0))
           
           with open(mat_path, 'wb') as f_mat:
               with tqdm(total=total, unit='B', unit_scale=True, desc=f"Monkey {monkey} trials") as pbar:
                   for chunk in resp.iter_content(8192):
                       f_mat.write(chunk)
                       pbar.update(len(chunk))
       
       # Process if needed
       if not os.path.exists(np_path):
           log_msg(f"Processing trial data for monkey {monkey}...")
           with h5py.File(mat_path, 'r') as f_mat:
               allmua = np.array(f_mat['ALLMUA'])
               allmat = np.array(f_mat['ALLMAT'])
               timebase = np.array(f_mat['tb']).flatten()
           
           valid = allmat[1, :].astype(int) > 0
           allmua = allmua[:, valid, :]
           allmat = allmat[:, valid]
           
           data_dict = {'mua': allmua, 'trial_info': allmat, 'timepoints': timebase}
           with open(np_path, 'wb') as f_npy:
               pickle.dump(data_dict, f_npy, protocol=4)
           
           # Clean up large .mat file
           if os.path.exists(mat_path):
               os.remove(mat_path)
               log_msg(f"Cleaned up {mat_path}")
       
       # Generate stimulus info
       log_msg(f"Generating stimulus info for monkey {monkey}...")
       with open(np_path, 'rb') as f_npy:
           trial_data = pickle.load(f_npy)
       
       trial_info = trial_data['trial_info'][1, :].astype(int)
       valid_idxs = trial_info > 0
       stim_ids = trial_info[valid_idxs] - 1
       
       static_paths = get_macq_paths(monkey)
       static_data = np.load(static_paths['data'], allow_pickle=True).item()
       categories = [static_data['train_class'][i] for i in stim_ids]
       exemplars = [
           os.path.splitext(os.path.basename(static_data['train_things_path'][i]))[0]
           for i in stim_ids
       ]
       
       stiminfo_df = pd.DataFrame({
           'trial_idx': np.where(valid_idxs)[0],
           'stim_id': trial_data['trial_info'][1, valid_idxs],
           'category': categories,
           'exemplar': exemplars
       })
       stiminfo_df.to_csv(paths['stiminfo'], index=False)
       log_msg(f"Monkey {monkey}: {len(exemplars)} trials processed")
   
   # =========================================================================
   # COMPLETION
   # =========================================================================
   log_msg("\n" + "="*60)
   log_msg("ALL DOWNLOADS COMPLETE")
   log_msg("="*60)
   log_msg(f"Data saved to: {config.data_dir}")

if __name__ == "__main__":
   try:
       main()
   except KeyboardInterrupt:
       log_msg("\n\nDownload interrupted by user")
       sys.exit(1)
   except Exception as e:
       log_msg(f"\n\nError occurred: {e}")
       raise
