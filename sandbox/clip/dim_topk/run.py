"""Visualize CLIP ViT-L/14 consensus dimensions: top-k highest-loading images per dim.

Uses the already-fit AlignedConsensus embedding (rank=45) from the dimensionality
chain — does NOT refit. Each row of the output figure is one dimension, showing
the TOP_K objects ranked by their loading W[:, d].

Run:
    poetry run python sandbox/clip/dim_topk/run.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.utils import get_output_dir

ROOT = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization")
OUT = get_output_dir()
CONSENSUS_DIR = ROOT / "experiments/datasets/consensus/outputs/clip_vit_l14"
CLIP_NPZ = ROOT / "data/features/clip_vit_l14/things_plus_clip_vit_l14.npz"
SSD_ROOT = Path("/data/labshare/_stachelschwein/SSD")

TOP_K = 12
THUMB = 84
PAD = 3
ROW_LABEL_W = 80
TITLE_H = 30


def ssd_path(p: str) -> Path:
    p = str(p)
    if p.startswith("/SSD/"):
        return SSD_ROOT / p.removeprefix("/SSD/")
    return Path(p)


def load_font():
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(cand).exists():
            return ImageFont.truetype(cand, 10), ImageFont.truetype(cand, 13)
    return ImageFont.load_default(), ImageFont.load_default()


def make_grid(w: np.ndarray, paths: np.ndarray, categories: np.ndarray,
              reliability: np.ndarray, out_path: Path) -> None:
    n_dims = w.shape[1]
    # Order dims by RELIABILITY descending — most stable dims first
    order = np.argsort(reliability)[::-1]

    width = ROW_LABEL_W + TOP_K * THUMB + (TOP_K + 1) * PAD
    row_h = THUMB + 2 * PAD
    height = TITLE_H + n_dims * row_h + PAD
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font, title_font = load_font()

    draw.text(
        (PAD, 8),
        f"CLIP ViT-L/14 + RBF (sigma=0.4*med) consensus rank={n_dims}  —  "
        f"top-{TOP_K} highest-loading per dim, ordered by reliability",
        fill=(20, 20, 20), font=title_font,
    )

    for r, d in enumerate(order):
        y0 = TITLE_H + r * row_h
        loadings = w[:, d]
        top_idx = np.argsort(loadings)[::-1][:TOP_K]
        nnz = int((loadings > 0).sum())
        rel = float(reliability[d])
        draw.text(
            (4, y0 + 6),
            f"d{int(d) + 1}\nrel={rel:.2f}\nnnz={nnz}\nmax={loadings.max():.2f}",
            fill=(20, 20, 20), font=font,
        )
        for c, idx in enumerate(top_idx):
            x = ROW_LABEL_W + PAD + c * (THUMB + PAD)
            y = y0 + PAD
            try:
                img = Image.open(ssd_path(paths[int(idx)])).convert("RGB")
                img.thumbnail((THUMB, THUMB))
                canvas.paste(img, (x + (THUMB - img.width) // 2, y + (THUMB - img.height) // 2))
            except Exception as e:
                draw.rectangle([x, y, x + THUMB, y + THUMB], outline=(150, 150, 150))
                draw.text((x + 4, y + 4), "miss", fill=(120, 120, 120), font=font)
            cat = str(categories[int(idx)])[:13]
            draw.rectangle([x, y + THUMB - 13, x + THUMB, y + THUMB - 1], fill=(255, 255, 255))
            draw.text((x + 2, y + THUMB - 12), cat, fill=(20, 20, 20), font=font)

    canvas.save(out_path)
    print(f"saved {out_path}  ({width}x{height} px)", flush=True)


def main() -> None:
    embedding = np.load(CONSENSUS_DIR / "embedding.npy")
    reliability = np.load(CONSENSUS_DIR / "cv_reliability.npy")
    summary = json.loads((CONSENSUS_DIR / "summary.json").read_text())
    print(f"loaded consensus rank={summary['rank']}  shape={embedding.shape}  "
          f"rel mean={summary['cv_reliability_mean']:.3f}  min={summary['cv_reliability_min']:.3f}")

    npz = np.load(CLIP_NPZ, allow_pickle=True)
    paths = npz["plus_image_paths"]
    categories = npz["categories"]
    print(f"CLIP metadata: paths={len(paths)} categories={len(categories)}")
    assert len(paths) == embedding.shape[0]

    out_path = OUT / f"clip_dim_topk_rank{embedding.shape[1]}_k{TOP_K}.png"
    make_grid(embedding, paths, categories, reliability, out_path)


if __name__ == "__main__":
    main()
