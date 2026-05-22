"""SRF embedding (rank=200) of vgg16 — top-k highest-loading images per dim.

Fits SRF on the full vgg16 similarity matrix at rank=200 (single run, default
srf_kwargs), then for each dim d sorts objects by W[:, d] descending and
shows the top TOP_K thumbnails in a single grid image.

This is the *true* symmetric-NMF embedding (S ~ W W^T, W >= 0), not a PCA-style
Gram eigendecomposition.

Run:
    poetry run python sandbox/dimensionality/vgg16_embedding/plot_srf_dim_topk.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from pysrf import SRF

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "outputs" / "vgg16_srf_200d"
OUT.mkdir(parents=True, exist_ok=True)

K_PATH = ROOT / "experiments/datasets/dimensionality/outputs/cache/vgg16.npy"
META_PATH = ROOT / "data/features/vgg16/metadata.csv"
SSD_ROOT = Path("/data/labshare/_stachelschwein/SSD")

RANK = 200
SEED = 0
SRF_KWARGS = dict(rho=3.0, max_outer=50, max_inner=30, tol=0.0, verbose=1)

N_TOP_DIMS = RANK   # plot all 200 dims; lower for a smaller figure
TOP_K = 12
THUMB = 84
PAD = 3
ROW_LABEL_W = 64
TITLE_H = 30


def ssd_path(p: str) -> Path:
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


def fit_srf(s: np.ndarray) -> np.ndarray:
    bounds = (float(s.min()), float(s.max()))
    print(f"[{time.strftime('%H:%M:%S')}] fitting SRF rank={RANK} on n={s.shape[0]}  "
          f"bounds={bounds}  kwargs={SRF_KWARGS}", flush=True)
    t0 = time.time()
    est = SRF(rank=RANK, missing_values=np.nan, bounds=bounds,
              random_state=SEED, **SRF_KWARGS)
    est.fit(s.astype(np.float64))
    elapsed = time.time() - t0
    w = est.w_  # shape (n, rank), W >= 0
    print(f"[{time.strftime('%H:%M:%S')}] fit done in {elapsed:.1f}s   W shape={w.shape}  "
          f"W min={w.min():.4f}  max={w.max():.4f}  sparsity={(w == 0).mean():.4f}", flush=True)
    return w, elapsed


def make_grid(w: np.ndarray, meta: pd.DataFrame, out_path: Path) -> None:
    n_dims = min(N_TOP_DIMS, w.shape[1])
    # Order dims by total mass (column sum) so dim 1 is the most "expressed"
    dim_mass = w.sum(axis=0)
    order = np.argsort(dim_mass)[::-1][:n_dims]

    width = ROW_LABEL_W + TOP_K * THUMB + (TOP_K + 1) * PAD
    row_h = THUMB + 2 * PAD
    height = TITLE_H + n_dims * row_h + PAD
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font, title_font = load_font()

    draw.text(
        (PAD, 8),
        f"VGG16 SRF embedding rank={RANK} (W>=0) — top-{TOP_K} highest-loading per dim "
        f"(ordered by column mass)",
        fill=(20, 20, 20), font=title_font,
    )

    for r, d in enumerate(order):
        y0 = TITLE_H + r * row_h
        loadings = w[:, d]
        top_idx = np.argsort(loadings)[::-1][:TOP_K]
        nnz = int((loadings > 0).sum())
        draw.text(
            (4, y0 + THUMB // 2 - 14),
            f"dim {r + 1}\n(orig #{int(d) + 1})\nnnz={nnz}\nmax={loadings.max():.2f}",
            fill=(20, 20, 20), font=font,
        )
        for c, idx in enumerate(top_idx):
            x = ROW_LABEL_W + PAD + c * (THUMB + PAD)
            y = y0 + PAD
            try:
                img = Image.open(ssd_path(str(meta.loc[int(idx), "path"]))).convert("RGB")
                img.thumbnail((THUMB, THUMB))
                canvas.paste(img, (x + (THUMB - img.width) // 2, y + (THUMB - img.height) // 2))
            except Exception:
                draw.rectangle([x, y, x + THUMB, y + THUMB], outline=(150, 150, 150))
                draw.text((x + 4, y + 4), "missing", fill=(120, 120, 120), font=font)
            cat = str(meta.loc[int(idx), "category"])[:13]
            # tiny white strip behind text for readability
            draw.rectangle([x, y + THUMB - 13, x + THUMB, y + THUMB - 1], fill=(255, 255, 255, 200))
            draw.text((x + 2, y + THUMB - 12), cat, fill=(20, 20, 20), font=font)

    canvas.save(out_path)
    print(f"saved {out_path}  ({width}x{height} px)", flush=True)


def main() -> None:
    s = np.load(K_PATH).astype(np.float64)
    meta = pd.read_csv(META_PATH)
    assert len(meta) == s.shape[0]

    w, elapsed = fit_srf(s)
    np.save(OUT / f"W_rank{RANK}.npy", w)

    make_grid(w, meta, OUT / f"dim_topk_rank{RANK}_k{TOP_K}.png")

    # also dump a per-dim CSV with top-20 categories for searchability
    rows = []
    for d in range(w.shape[1]):
        top = np.argsort(w[:, d])[::-1][:20]
        rows.append({
            "dim": d + 1,
            "col_mass": float(w[:, d].sum()),
            "nnz": int((w[:, d] > 0).sum()),
            "max_loading": float(w[:, d].max()),
            "top20_categories": ",".join(str(meta.loc[int(i), "category"]) for i in top),
        })
    pd.DataFrame(rows).to_csv(OUT / f"dim_top20_categories_rank{RANK}.csv", index=False)

    (OUT / "summary.json").write_text(json.dumps({
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
