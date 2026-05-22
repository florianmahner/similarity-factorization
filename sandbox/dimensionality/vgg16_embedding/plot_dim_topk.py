"""Top-k highest-loading images per dimension of the VGG16 Gram embedding.

For each dimension d of the centered-kernel PCA embedding Z (n x N_DIMS),
sort objects by Z[:, d] descending and show the top TOP_K images. One row
per dimension.

Edit the constants below to control how much is plotted.

Run:
    poetry run python sandbox/dimensionality/vgg16_embedding/plot_dim_topk.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "outputs" / "vgg16_200d"
EMBED_PATH = OUT / "embedding_top200.npy"
META_PATH = ROOT / "data/features/vgg16/metadata.csv"
SSD_ROOT = Path("/data/labshare/_stachelschwein/SSD")

N_TOP_DIMS = 30   # rows; bump to 200 for the full embedding
TOP_K = 12        # images per dim
THUMB = 90        # px per thumbnail
PAD = 4
ROW_LABEL_W = 56
TITLE_H = 28


def ssd_path(p: str) -> Path:
    if p.startswith("/SSD/"):
        return SSD_ROOT / p.removeprefix("/SSD/")
    return Path(p)


def load_font():
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, 11), ImageFont.truetype(candidate, 13)
    return ImageFont.load_default(), ImageFont.load_default()


def main() -> None:
    z = np.load(EMBED_PATH)
    meta = pd.read_csv(META_PATH)
    assert z.shape[0] == len(meta), f"dim mismatch: Z {z.shape} vs meta {len(meta)}"
    n, total_dims = z.shape
    n_dims = min(N_TOP_DIMS, total_dims)

    width = ROW_LABEL_W + TOP_K * THUMB + (TOP_K + 1) * PAD
    row_h = THUMB + 2 * PAD
    height = TITLE_H + n_dims * row_h + PAD
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font, title_font = load_font()

    draw.text(
        (PAD, 6),
        f"VGG16 200-d Gram embedding — top-{TOP_K} positive-loading images per dim "
        f"(showing dims 1..{n_dims})",
        fill=(20, 20, 20),
        font=title_font,
    )

    for r in range(n_dims):
        y0 = TITLE_H + r * row_h
        loadings = z[:, r]
        order = np.argsort(loadings)[::-1][:TOP_K]

        draw.text(
            (4, y0 + THUMB // 2 - 6),
            f"dim {r + 1}",
            fill=(20, 20, 20),
            font=font,
        )

        for c, idx in enumerate(order):
            x = ROW_LABEL_W + PAD + c * (THUMB + PAD)
            y = y0 + PAD
            try:
                img = Image.open(ssd_path(str(meta.loc[int(idx), "path"]))).convert("RGB")
                img.thumbnail((THUMB, THUMB))
                canvas.paste(img, (x + (THUMB - img.width) // 2, y + (THUMB - img.height) // 2))
            except Exception:
                draw.rectangle([x, y, x + THUMB, y + THUMB], outline=(150, 150, 150))
                draw.text((x + 4, y + 4), "missing", fill=(120, 120, 120), font=font)
            # category label (below thumbnail, overlaid on white strip if needed)
            cat = str(meta.loc[int(idx), "category"])[:14]
            draw.text((x + 2, y + THUMB - 13), cat, fill=(20, 20, 20), font=font)

    out_path = OUT / f"dim_topk_top{n_dims}_k{TOP_K}.png"
    canvas.save(out_path)
    print(f"saved {out_path}  ({width}x{height} px)")


if __name__ == "__main__":
    main()
