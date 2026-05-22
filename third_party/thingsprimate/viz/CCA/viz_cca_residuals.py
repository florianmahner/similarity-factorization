"""
Visualize NMF components

Loads results from model/CCA/CCA_residuals.py:
- For each species (human, monkey):
  - W (N x 20) non-negative membership matrix
  - For each component k, plot a positive-only image grid of top-weighted stimuli

Image-grid code is adapted from viz/ADMM/viz_admm.py to keep a consistent look.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from PIL import Image
from joblib import Parallel, delayed
import sys

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from config.paths import config


# ---- Minimal helpers (adapted from viz_admm.py) ----
MOSAIC_BORDER_WIDTH = 4.0


def _square_and_resize(path, target):
    try:
        im = Image.open(path).convert('RGB')
        w, h = im.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((target, target), Image.BILINEAR)
        # add light border
        arr = np.array(im)
        thick = max(2, target // 60)
        border_color = (64, 64, 64)
        arr[:thick, :] = border_color
        arr[-thick:, :] = border_color
        arr[:, :thick] = border_color
        arr[:, -thick:] = border_color
        return Image.fromarray(arr)
    except Exception:
        blank = np.full((target, target, 3), 128, np.uint8)
        return Image.fromarray(blank)

def _square_and_resize_noborder(path, target):
    try:
        im = Image.open(path).convert('RGB')
        w, h = im.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((target, target), Image.BILINEAR)
        return im
    except Exception:
        blank = np.full((target, target, 3), 200, np.uint8)
        return Image.fromarray(blank)


def _build_mosaic(paths, rows, cols, px):
    blank = np.full((px, px, 3), 128, np.uint8)
    tiles = [_square_and_resize(p, px) if p is not None else Image.fromarray(blank) for p in paths]
    tiles += [Image.fromarray(blank)] * (rows * cols - len(tiles))
    tiles = [np.asarray(t) for t in tiles]
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    return grid


def _border(ax, color='black', lw=3):
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor='none', edgecolor=color, linewidth=lw, clip_on=False))


def _plot_distribution_positive_only(ax, vec, order, ipos, inv, pos_cand=None):
    # Simple ranking band to mirror ADMM look
    xs = inv[ipos]
    ax.scatter(xs, np.zeros_like(xs), s=6, c='#444444', alpha=0.8)
    ax.set_xticks([]); ax.set_yticks([])


def plot_comp_imgs_positive_only(comp_idx, W, stims, cats, img_dir, title=None, rows=4, cols=6, img_px=180, dpi=300, subsample=True, sub_n=120):
    vec = W[:, comp_idx]
    n_cells = rows * cols
    order = np.argsort(vec)

    # Positive-only: take top subset then random sample
    if subsample:
        n_total = len(vec)
        n_cand = min(sub_n, max(n_total // 2, n_cells))
        pos_cand = order[-n_cand:]
        ipos = np.random.choice(pos_cand, size=min(n_cells, len(pos_cand)), replace=False)
    else:
        pos_cand = order[-n_cells:]
        ipos = pos_cand

    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))

    # Figure with only positive grid + small distribution row
    fw = cols * img_px / dpi
    fh = (rows + 0.5) * img_px / dpi
    fig = plt.figure(figsize=(fw, fh), dpi=dpi, constrained_layout=False)
    gs = GridSpec(2, 1, height_ratios=[rows, 0.5], hspace=0, figure=fig)

    # Positive grid
    pos_paths = [os.path.join(img_dir, cats[i], f"{stims[i]}.jpg") for i in ipos]
    pos_mosaic = _build_mosaic(pos_paths, rows, cols, img_px)
    ax_top = fig.add_subplot(gs[0])
    ax_top.imshow(pos_mosaic, interpolation='nearest', aspect='auto')
    ax_top.axis('off')
    _border(ax_top, color=config.plotting.get('positive_color', '#8B0000'), lw=MOSAIC_BORDER_WIDTH)

    # Distribution axis
    ax_mid = fig.add_subplot(gs[1])
    ax_mid.set_xlim(-0.5, pos_mosaic.shape[1] - 0.5)
    ax_mid.margins(x=0, y=0)
    _plot_distribution_positive_only(ax_mid, vec, order, ipos, inv, pos_cand)
    ax_mid.set_yticks([])
    ax_mid.set_xticks([])

    if title:
        fig.suptitle(title, fontsize=14, y=0.99)
    return fig


def _save_fig(fig, species, comp_idx=None, extra=None):
    if fig is None:
        return
    fig_dir = os.path.join(config.fig_dir, 'cca_residuals', species, 'components')
    os.makedirs(fig_dir, exist_ok=True)
    fmt = config.plotting.get('savefig_format', 'pdf')
    if comp_idx is not None:
        fname = f'comp_{comp_idx+1}.{fmt}'
    elif extra:
        fname = f'{extra}.{fmt}'
    else:
        fname = f'figure.{fmt}'
    fp = os.path.join(fig_dir, fname)
    dpi = 600 if extra == 'images' else 300
    fig.savefig(fp, bbox_inches='tight', dpi=dpi)
    print(f"Saved {species} component figure to {fp}")
    plt.close(fig)


if __name__ == "__main__":
    # Load residual results
    res_fp = os.path.join(config.results_dir, 'residualized', 'ccaw_residual.pkl')
    if not os.path.exists(res_fp):
        raise FileNotFoundError(f"Residuals file not found: {res_fp}")
    with open(res_fp, 'rb') as f:
        R = pickle.load(f)

    stims = R['stims']
    cats = ['_'.join(s.split('_')[:-1]) if '_' in s else s for s in stims]
    img_dir = os.path.join(config.data_dir, 'things', 'images')

    jobs = int(R.get('params', {}).get('n_jobs', config.analysis.get('n_jobs', 1)))
    rows, cols, img_px, dpi = 4, 6, 180, 300

    def render_species(species_key, species_label):
        W = R[species_key]['nmf']['W']
        rank = R[species_key]['nmf']['rank']
        # Generate components 1..rank
        tasks = list(range(rank))
        def _render(ci):
            title = f"{species_label.upper()} NMF Component {ci+1}"
            fig = plot_comp_imgs_positive_only(ci, W, stims, cats, img_dir, title=title,
                                               rows=rows, cols=cols, img_px=img_px, dpi=dpi,
                                               subsample=True, sub_n=120)
            _save_fig(fig, species_label, comp_idx=ci)
        if jobs and jobs != 1:
            Parallel(n_jobs=jobs)(delayed(_render)(ci) for ci in tasks)
        else:
            for ci in tasks:
                _render(ci)

    # Skip individual per-component figures; replaced by giant canvas below
    # render_species('human', 'human')
    # render_species('monkey', 'monkey')

    # ----- Giant canvas: top images for ALL NMF components -----
    def render_all_components_canvas(img_px=120, per_row=24, label_h=18, sep_rows=2):
        Wm = R['monkey']['nmf']['W']; rm = int(R['monkey']['nmf']['rank'])
        Wh = R['human']['nmf']['W'];  rh = int(R['human']['nmf']['rank'])
        rm = min(rm, 20); rh = min(rh, 20)

        blocks = []
        lbl_pos = []  # (y_px, text)

        def top_paths(W, ci, n):
            v = W[:, ci]
            idx = np.argsort(v)[-n:][::-1]
            return [os.path.join(img_dir, cats[i], f"{stims[i]}.jpg") for i in idx]

        # Monkey rows (1..20)
        cum_h = 0
        for ci in range(rm):
            paths = top_paths(Wm, ci, per_row)
            row = _build_mosaic(paths, 1, per_row, img_px)
            w = row.shape[1]
            strip = np.full((label_h, w, 3), 255, np.uint8)
            # label ABOVE row: strip then row
            blocks.append(strip)
            blocks.append(row)
            # label y at center of strip region
            y = cum_h + (label_h // 2)
            lbl_pos.append((y, f"Macaque Unique C{ci+1}"))
            cum_h += label_h + img_px

        # Separator: two empty rows worth of whitespace between species
        if blocks:
            w = blocks[0].shape[1]
            sep = np.full((sep_rows*img_px, w, 3), 255, np.uint8)
            blocks.append(sep)
            cum_h += sep.shape[0]

        # Human rows (21..40)
        for ci in range(rh):
            paths = top_paths(Wh, ci, per_row)
            row = _build_mosaic(paths, 1, per_row, img_px)
            w = row.shape[1]
            strip = np.full((label_h, w, 3), 255, np.uint8)
            blocks.append(strip)
            blocks.append(row)
            y = cum_h + (label_h // 2)
            lbl_pos.append((y, f"Human Unique C{ci+1}"))
            cum_h += label_h + img_px

        big = np.vstack(blocks)

        # Plot as single image and overlay compact labels on the left
        h, w, _ = big.shape
        fig_w = w / 200.0
        fig_h = h / 200.0
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(big, interpolation='nearest', aspect='auto')
        ax.axis('off')

        # Titles below each row (left aligned on white strip)
        for y, lab in lbl_pos:
            ax.text(6, y, lab, va='center', ha='left', color='black', fontsize=8)

        out_dir = os.path.join(config.fig_dir, 'residualized')
        os.makedirs(out_dir, exist_ok=True)
        fmt = config.plotting.get('savefig_format', 'pdf')
        fp_all = os.path.join(out_dir, f'components_all.{fmt}')
        fig.savefig(fp_all, bbox_inches='tight', dpi=300)
        print(f"Saved ALL components canvas to {fp_all}")
        plt.close(fig)

    # render_all_components_canvas(img_px=120, per_row=24)

    # ----- Image-wise MDS (2D) of residuals -----
    import warnings
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    def _coords_mds(R, seed=42):
        # Metric MDS (no PCA shortcut)
        X = StandardScaler().fit_transform(R.astype(np.float32, copy=False))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MDS(n_components=2, metric=True, n_init=1, max_iter=300, random_state=seed, dissimilarity='euclidean')
            P = m.fit_transform(X)
        return P

    def _img(path, px, noborder=False):
        return _square_and_resize_noborder(path, px) if noborder else _square_and_resize(path, px)

    def _get_matrix(key):
        if key == 'shared':
            return np.asarray(R['shared']['scores'])
        return np.asarray(R[key]['resid'])

    def plot_mds(species_key, species_label, n_thumbs=2000, thumb_px=120, dot_sz=3, thumb_zoom=1.0, fig_size=(7.5, 7.0), border=False, dpi=600):
        Y = _get_matrix(species_key)
        P = _coords_mds(Y)
        n = len(stims)
        # Subsample thumbs uniformly at random
        rng = np.random.default_rng(42)
        k = min(n_thumbs, n)
        idx_th = rng.choice(np.arange(n), size=k, replace=False)
        mask = np.ones(n, bool); mask[idx_th] = False

        fig = plt.figure(figsize=fig_size, dpi=dpi)
        ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])

        # Plot rest as dots
        ax.scatter(P[mask,0], P[mask,1], s=dot_sz, c='#999999', alpha=0.6, linewidths=0)

        # Overlay thumbnails
        for i in idx_th:
            ip = os.path.join(img_dir, cats[i], f"{stims[i]}.jpg")
            try:
                im = _img(ip, thumb_px, noborder=not border)
                oi = OffsetImage(im, zoom=float(thumb_zoom))
                ab = AnnotationBbox(oi, (P[i,0], P[i,1]), frameon=False, pad=0.0)
                ax.add_artist(ab)
            except Exception:
                pass

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{species_label.upper()} Residuals: MDS (N={n}, thumbs={k})", fontsize=12)
        # Save
        fig_dir2 = os.path.join(config.fig_dir, 'residualized')
        os.makedirs(fig_dir2, exist_ok=True)
        fmt = config.plotting.get('savefig_format', 'pdf')
        fp = os.path.join(fig_dir2, f'mds_{species_label}.{fmt}')
        fig.savefig(fp, bbox_inches='tight', dpi=dpi)
        print(f"Saved MDS figure to {fp}")
        plt.close(fig)

    # Render MDS for all spaces (metric), much higher DPI, larger dots, thumbnails smaller
    # plot_mds('human', 'human', n_thumbs=2000, thumb_px=40, dot_sz=30, thumb_zoom=0.5, fig_size=(15, 13.5), border=False, dpi=1200)
    # plot_mds('monkey', 'monkey', n_thumbs=2000, thumb_px=40, dot_sz=30, thumb_zoom=0.5, fig_size=(15, 13.5), border=False, dpi=1200)
    plot_mds('shared', 'cross',  n_thumbs=2000, thumb_px=40, dot_sz=30, thumb_zoom=0.5, fig_size=(15, 13.5), border=False, dpi=1200)

    # ----- t-SNE and UMAP embeddings -----
    from sklearn.manifold import TSNE
    try:
        from umap import UMAP
        _has_umap = True
    except Exception:
        _has_umap = False

    def _coords_tsne(R, seed=42, pca_n=50, perplexity=30.0):
        X = StandardScaler().fit_transform(R.astype(np.float32, copy=False))
        k = int(min(pca_n, X.shape[1]))
        if k and k > 0:
            X = PCA(n_components=k, svd_solver='randomized', random_state=seed).fit_transform(X)
        ts = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto', n_iter=1000, random_state=seed)
        return ts.fit_transform(X)

    def _coords_umap(R, seed=42, n_neighbors=15, min_dist=0.1, pca_n=50):
        X = StandardScaler().fit_transform(R.astype(np.float32, copy=False))
        k = int(min(pca_n, X.shape[1]))
        if k and k > 0:
            X = PCA(n_components=k, svd_solver='randomized', random_state=seed).fit_transform(X)
        um = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
        return um.fit_transform(X)

    def plot_embed(method, species_key, species_label, n_thumbs=2000, thumb_px=100, dot_sz=30, thumb_zoom=0.5, fig_size=(15,13.5), dpi=1200):
        Y = _get_matrix(species_key)
        if method == 'tsne':
            P = _coords_tsne(Y)
        elif method == 'umap' and _has_umap:
            P = _coords_umap(Y)
        elif method == 'umap' and not _has_umap:
            print('UMAP not available; skipping')
            return
        else:
            return

        n = len(stims)
        rng = np.random.default_rng(42)
        k = min(n_thumbs, n)
        idx_th = rng.choice(np.arange(n), size=k, replace=False)
        mask = np.ones(n, bool); mask[idx_th] = False

        fig = plt.figure(figsize=fig_size, dpi=dpi)
        ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
        ax.scatter(P[mask,0], P[mask,1], s=dot_sz, c='#999999', alpha=0.6, linewidths=0)
        for i in idx_th:
            ip = os.path.join(img_dir, cats[i], f"{stims[i]}.jpg")
            try:
                im = _square_and_resize_noborder(ip, thumb_px)
                oi = OffsetImage(im, zoom=float(thumb_zoom))
                ab = AnnotationBbox(oi, (P[i,0], P[i,1]), frameon=False, pad=0.0)
                ax.add_artist(ab)
            except Exception:
                pass
        ax.set_xticks([]); ax.set_yticks([])
        out_dir = os.path.join(config.fig_dir, 'residualized')
        os.makedirs(out_dir, exist_ok=True)
        fmt = config.plotting.get('savefig_format', 'pdf')
        fp = os.path.join(out_dir, f'{method}_{species_label}.{fmt}')
        fig.savefig(fp, bbox_inches='tight', dpi=dpi)
        print(f'Saved {method.upper()} figure to {fp}')
        plt.close(fig)

    # TSNE and UMAP for human, monkey, and cross-species
    # plot_embed('tsne', 'human', 'human')
    # plot_embed('tsne', 'monkey', 'monkey')
    # plot_embed('tsne', 'shared', 'cross')
    # plot_embed('umap', 'human', 'human')
    # plot_embed('umap', 'monkey', 'monkey')
    # plot_embed('umap', 'shared', 'cross')
