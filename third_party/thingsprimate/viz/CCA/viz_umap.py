import os, sys, pickle
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from umap import UMAP
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# Project imports
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Remove plotting dependencies
from config.paths import config

def load_cca_spaces():
    """Load 3 CCA spaces from cached results"""
    state_fp = config.cache_file
    if not os.path.exists(state_fp):
        raise FileNotFoundError(f"CCA state file not found: {state_fp}")

    print(f"Loading CCA results from {state_fp}")
    with open(state_fp, 'rb') as f:
        s = pickle.load(f)

    # Extract components for each space
    cca_all = s['cca_all']
    cca_hum = s['cca_hum']
    cca_mon = s['cca_mon']

    # Get mean components across views
    comps_all = np.mean([cca_all['components'][v] for v in cca_all['components'].keys()], axis=0)
    comps_hum = np.mean([cca_hum['components'][v] for v in cca_hum['components'].keys()], axis=0)
    comps_mon = np.mean([cca_mon['components'][v] for v in cca_mon['components'].keys()], axis=0)

    stims = s['all_stims']
    cats = ['_'.join(sv.split('_')[:-1]) if '_' in sv else sv for sv in stims]

    return {
        'cross-species': comps_all,
        'human': comps_hum,
        'monkey': comps_mon
    }, stims, cats

def load_residual_spaces():
    """Load residualized monkey and human spaces"""
    resid_fp = config.results_dir / 'residualized' / 'ccaw_residual.pkl'
    if not os.path.exists(resid_fp):
        print(f"Residual file not found: {resid_fp}")
        return {}, [], []

    print(f"Loading residual results from {resid_fp}")
    with open(resid_fp, 'rb') as f:
        s = pickle.load(f)

    # Use the actual residualized spaces (not NMF)
    comps_mon_resid = s['monkey']['resid']
    comps_hum_resid = s['human']['resid']

    stims = s['stims']
    cats = ['_'.join(sv.split('_')[:-1]) if '_' in sv else sv for sv in stims]

    return {
        'human': comps_hum_resid,
        'monkey': comps_mon_resid
    }, stims, cats

def run_umap(components):
    """Run UMAP on components"""
    components = np.asarray(components, dtype=np.float32)

    reducer = UMAP(
        n_neighbors=min(15, components.shape[0]-1),
        min_dist=0.1,
        random_state=42
    )
    return reducer.fit_transform(components)

def load_image_cache(stims, cats):
    """Load all images once into memory cache"""
    print(f"Loading {len(stims)} images into cache...")
    img_base = config.things_dir / 'images'
    cache = {}

    for i, stim in enumerate(stims):
        try:
            img_path = img_base / cats[i] / f'{stim}.jpg'
            if img_path.exists():
                img = Image.open(img_path).convert('RGB')
                img = img.resize((40, 40), Image.LANCZOS)
                cache[i] = np.array(img)
        except Exception:
            continue

        if (i + 1) % 1000 == 0:
            print(f"  Cached {i + 1}/{len(stims)} images")

    print(f"Cached {len(cache)} images successfully")
    return cache

def plot_umap_scatter(embedding, stims, cats, title, save_path, img_cache):
    """Create 2D scatter plot with cached image thumbnails"""
    plt.figure(figsize=(24, 20))

    print(f"Creating scatter plot with {len(stims)} image thumbnails...")

    # Set up the plot limits first
    x_range = embedding[:, 0].max() - embedding[:, 0].min()
    y_range = embedding[:, 1].max() - embedding[:, 1].min()

    plt.xlim(embedding[:, 0].min() - 0.1*x_range, embedding[:, 0].max() + 0.1*x_range)
    plt.ylim(embedding[:, 1].min() - 0.1*y_range, embedding[:, 1].max() + 0.1*y_range)

    # Add each cached image as a thumbnail at its UMAP coordinates
    for i, (x, y) in enumerate(embedding):
        if i not in img_cache:
            continue

        # Create smaller thumbnail at coordinates
        imagebox = OffsetImage(img_cache[i], zoom=0.4)  # ~16px effective size
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0)
        plt.gca().add_artist(ab)

        if (i + 1) % 1000 == 0:
            print(f"  Added {i + 1}/{len(stims)} thumbnails")

    plt.title(f'UMAP Visualization: {title}', fontsize=16, pad=20)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)

    # Clean up axes
    plt.xticks([])
    plt.yticks([])

    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {save_path}")

def main():
    # Load CCA spaces
    spaces, stims, cats = load_cca_spaces()

    print(f"Loaded {len(stims)} stimuli")
    for name, comps in spaces.items():
        print(f"{name}: {comps.shape}")

    # Cache all images once
    img_cache = load_image_cache(stims, cats)

    # Create output directory
    out_dir = config.fig_dir / 'umap'
    os.makedirs(out_dir, exist_ok=True)

    # Run UMAP on each space
    for name, components in spaces.items():
        print(f"\nProcessing {name} space...")
        embedding = run_umap(components)

        save_path = out_dir / f'umap_{name.replace("-", "_")}.pdf'
        plot_umap_scatter(embedding, stims, cats, name, save_path, img_cache)

    print(f"\nUMAP visualizations saved in {out_dir}")

    # Load and process residualized spaces
    resid_spaces, resid_stims, resid_cats = load_residual_spaces()

    if resid_spaces:
        print(f"\nLoaded {len(resid_stims)} residual stimuli")
        for name, comps in resid_spaces.items():
            print(f"residual {name}: {comps.shape}")

        # Use same image cache if stimuli match, otherwise create new cache
        if resid_stims == stims:
            resid_cache = img_cache
            print("Reusing existing image cache for residuals")
        else:
            resid_cache = load_image_cache(resid_stims, resid_cats)

        # Create residual output directory
        resid_dir = out_dir / 'resid'
        os.makedirs(resid_dir, exist_ok=True)

        # Run UMAP on each residualized space
        for name, components in resid_spaces.items():
            print(f"\nProcessing residual {name} space...")
            embedding = run_umap(components)

            save_path = resid_dir / f'umap_{name}.pdf'
            plot_umap_scatter(embedding, resid_stims, resid_cats, f"residual {name}", save_path, resid_cache)

        print(f"\nResidual UMAP visualizations saved in {resid_dir}")
    else:
        print("\nNo residual spaces found - skipping residual analysis")

if __name__ == '__main__':
    main()