"""Visualize CLIP ViT-L/14 RBF SRF embedding at rank=50.

Fits a single SRF run on the cached CLIP RBF similarity at rank=50, then
shows the top-k highest-W loading images per dim as a tall image grid.

Run:
    poetry run python sandbox/dimensionality/clip_rbf_cv/plot_srf_dim_topk.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pysrf import SRF

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "outputs" / "srf_rank50"
OUT.mkdir(parents=True, exist_ok=True)

S_PATH = Path(__file__).resolve().parent / "outputs" / "clip_rbf.npy"
NPZ_PATH = ROOT / "data/features/clip_vit_l14/things_plus_clip_vit_l14.npz"
PLUS_DIR = Path("/data/labshare/_stachelschwein/SSD/datasets/things/plus")

RANK = 50
SEED = 0
SRF_KWARGS = dict(rho=3.0, max_outer=50, max_inner=30, tol=0.0, verbose=1)

TOP_K = 12
THUMB = 90
PAD = 3
ROW_LABEL_W = 70
TITLE_H = 30


def load_font():
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(cand).exists():
            return ImageFont.truetype(cand, 10), ImageFont.truetype(cand, 13)
    return ImageFont.load_default(), ImageFont.load_default()


def main() -> None:
    s = np.load(S_PATH).astype(np.float64)
    npz = np.load(NPZ_PATH, allow_pickle=True)
    categories = np.asarray(npz["categories"])
    assert len(categories) == s.shape[0] == 1854

    bounds = (float(s.min()), float(s.max()))
    print(f"[{time.strftime('%H:%M:%S')}] fit SRF rank={RANK} on n={s.shape[0]} bounds={bounds}", flush=True)
    t0 = time.time()
    est = SRF(rank=RANK, missing_values=np.nan, bounds=bounds,
              random_state=SEED, **SRF_KWARGS)
    est.fit(s)
    elapsed = time.time() - t0
    w = est.w_
    print(f"[{time.strftime('%H:%M:%S')}] fit done in {elapsed:.1f}s  W shape={w.shape}  "
          f"max={w.max():.4f}  sparsity_zero={(w == 0).mean():.4f}", flush=True)
    np.save(OUT / f"W_rank{RANK}.npy", w)

    # order dims by column mass (most "expressed" first)
    dim_mass = w.sum(axis=0)
    order = np.argsort(dim_mass)[::-1]

    width = ROW_LABEL_W + TOP_K * THUMB + (TOP_K + 1) * PAD
    row_h = THUMB + 2 * PAD
    height = TITLE_H + RANK * row_h + PAD
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font, title_font = load_font()
    draw.text(
        (PAD, 8),
        f"CLIP ViT-L/14 RBF SRF rank={RANK} — top-{TOP_K} highest-W per dim "
        f"(rows ordered by column mass)",
        fill=(20, 20, 20), font=title_font,
    )

    for r, d in enumerate(order):
        y0 = TITLE_H + r * row_h
        loadings = w[:, d]
        top_idx = np.argsort(loadings)[::-1][:TOP_K]
        nnz = int((loadings > 0).sum())
        draw.text(
            (4, y0 + THUMB // 2 - 18),
            f"dim {r + 1}\n(#{int(d) + 1})\nnnz={nnz}\nmax={loadings.max():.2f}",
            fill=(20, 20, 20), font=font,
        )
        for c, idx in enumerate(top_idx):
            x = ROW_LABEL_W + PAD + c * (THUMB + PAD)
            y = y0 + PAD
            img_path = PLUS_DIR / f"{categories[int(idx)]}.jpg"
            try:
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((THUMB, THUMB))
                canvas.paste(img, (x + (THUMB - img.width) // 2, y + (THUMB - img.height) // 2))
            except Exception:
                draw.rectangle([x, y, x + THUMB, y + THUMB], outline=(150, 150, 150))
                draw.text((x + 4, y + 4), "missing", fill=(120, 120, 120), font=font)
            cat = str(categories[int(idx)])[:13]
            draw.rectangle([x, y + THUMB - 13, x + THUMB, y + THUMB - 1], fill=(255, 255, 255))
            draw.text((x + 2, y + THUMB - 12), cat, fill=(20, 20, 20), font=font)

    grid_path = OUT / f"dim_topk_rank{RANK}_k{TOP_K}.png"
    canvas.save(grid_path)
    print(f"saved grid: {grid_path}  ({width}x{height} px)", flush=True)

    # CSV with top-20 categories per dim
    import pandas as pd
    rows = []
    for d in range(w.shape[1]):
        top = np.argsort(w[:, d])[::-1][:20]
        rows.append({
            "dim": d + 1,
            "col_mass": float(w[:, d].sum()),
            "nnz": int((w[:, d] > 0).sum()),
            "max_loading": float(w[:, d].max()),
            "top20_categories": ",".join(str(categories[int(i)]) for i in top),
        })
    pd.DataFrame(rows).to_csv(OUT / f"dim_top20_categories_rank{RANK}.csv", index=False)

    (OUT / "summary.json").write_text(json.dumps({
        "kernel": "CLIP ViT-L/14 RBF (median-heuristic sigma)",
        "rank": RANK,
        "seed": SEED,
        "srf_kwargs": SRF_KWARGS,
        "fit_seconds": round(elapsed, 1),
        "n": int(s.shape[0]),
        "W_shape": list(w.shape),
        "W_sparsity_zero": float((w == 0).mean()),
    }, indent=2) + "\n")
    print(f"summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
