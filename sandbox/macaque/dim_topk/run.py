"""Quick macaque SRF embedding + top-k image visualization.

Single SRF fit on the things_macaque22k_tight similarity (sigma=0.4*median):
  rank = 89  (from new coherence estimate)
  max_outer = 20  (NOT fully converged — quick look only)
  8 BLAS threads

Then per-dim top-12 image grid using THINGS core images. Used to eyeball
whether rank=89 looks coherent or whether dims are visually mush.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from datasets.monkey import load_macaque
from pysrf import SRF
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()
SIM_PATH = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/datasets/dimensionality/outputs/cache/things_macaque22k_tight.npy")
FILENAMES_PATH = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/sandbox/things/things_monkey_22k_embedding/outputs/22k_gaussian/filenames.txt")
SSD_ROOT = Path("/data/labshare/_stachelschwein/SSD/datasets/things/core")

RANK = 89
MAX_OUTER = 20
SEED = 0
TOP_K = 12
THUMB = 84
PAD = 3
ROW_LABEL_W = 80
TITLE_H = 30


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def category_from_filename(filename: str) -> str:
    return Path(filename).name.rsplit("_", 1)[0]


def build_paths_from_filenames(stimuli: list[str]) -> list[Path]:
    if not FILENAMES_PATH.exists():
        raise FileNotFoundError(f"Exact THINGS filename list not found: {FILENAMES_PATH}")

    filenames = [line.strip() for line in FILENAMES_PATH.read_text().splitlines() if line.strip()]
    if len(filenames) != len(stimuli):
        raise ValueError(f"filename count {len(filenames)} does not match macaque rows {len(stimuli)}")

    mismatches = [
        (i, filename, stimulus)
        for i, (filename, stimulus) in enumerate(zip(filenames, stimuli, strict=True))
        if category_from_filename(filename) != stimulus
    ]
    if mismatches:
        preview = ", ".join(f"row {i}: {filename} vs {stimulus}" for i, filename, stimulus in mismatches[:5])
        raise ValueError(f"filename list is not aligned with macaque stimuli ({preview})")

    paths = [SSD_ROOT / category_from_filename(filename) / filename for filename in filenames]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} exact THINGS image files are missing, first paths:\n{preview}")

    return paths


def load_font():
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(cand).exists():
            return ImageFont.truetype(cand, 10), ImageFont.truetype(cand, 13)
    return ImageFont.load_default(), ImageFont.load_default()


def make_grid(w: np.ndarray, paths: list[Path], categories: list[str],
              out_path: Path) -> None:
    n_dims = w.shape[1]
    # Order dims by total column mass so most "expressed" dims come first.
    col_mass = w.sum(axis=0)
    order = np.argsort(col_mass)[::-1]

    width = ROW_LABEL_W + TOP_K * THUMB + (TOP_K + 1) * PAD
    row_h = THUMB + 2 * PAD
    height = TITLE_H + n_dims * row_h + PAD
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font, title_font = load_font()

    draw.text(
        (PAD, 8),
        f"macaque IT (n=22248) - SRF rank={n_dims}, max_outer={MAX_OUTER} (NOT converged) - "
        f"top-{TOP_K} highest-loading per dim",
        fill=(20, 20, 20), font=title_font,
    )

    for r, d in enumerate(order):
        y0 = TITLE_H + r * row_h
        loadings = w[:, d]
        top_idx = np.argsort(loadings)[::-1][:TOP_K]
        nnz = int((loadings > 1e-8).sum())
        draw.text(
            (4, y0 + 6),
            f"d{int(d)+1}\nmass={col_mass[d]:.1f}\nnnz={nnz}\nmax={loadings.max():.2f}",
            fill=(20, 20, 20), font=font,
        )
        for c, idx in enumerate(top_idx):
            x = ROW_LABEL_W + PAD + c * (THUMB + PAD)
            y = y0 + PAD
            p = paths[int(idx)]
            try:
                img = Image.open(p).convert("RGB")
                img.thumbnail((THUMB, THUMB))
                canvas.paste(img, (x + (THUMB - img.width) // 2, y + (THUMB - img.height) // 2))
            except Exception:
                draw.rectangle([x, y, x + THUMB, y + THUMB], outline=(150, 150, 150))
                draw.text((x + 4, y + 4), "miss", fill=(120, 120, 120), font=font)
            cat = str(categories[int(idx)])[:13]
            draw.rectangle([x, y + THUMB - 13, x + THUMB, y + THUMB - 1], fill=(255, 255, 255))
            draw.text((x + 2, y + THUMB - 12), cat, fill=(20, 20, 20), font=font)

    canvas.save(out_path)
    log(f"saved {out_path}  ({width}x{height} px)")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"loading similarity from {SIM_PATH}")
    sim = np.load(SIM_PATH).astype(np.float64)
    log(f"  n={sim.shape[0]}  diag mean={np.diag(sim).mean():.4f}  off-diag range=[{sim[~np.eye(sim.shape[0], dtype=bool)].min():.4f}, {sim[~np.eye(sim.shape[0], dtype=bool)].max():.4f}]")

    log(f"loading stimuli metadata...")
    _, stimuli, _ = load_macaque(
        "22k", "F",
        root="/data/labshare/_stachelschwein/SSD/datasets/things/macaque",
        roi="it", min_reliab=0.3, average_exemplars=False,
    )
    assert len(stimuli) == sim.shape[0]
    paths = build_paths_from_filenames(list(stimuli))
    log(f"  stimuli={len(stimuli)} unique cats={len(set(stimuli))}")
    log(f"  loaded exact THINGS filenames from {FILENAMES_PATH} images={len(paths)} missing=0")
    # spot-check first 3 paths exist
    for p in paths[:3]:
        log(f"  sample path: {p}  exists={p.exists()}")

    log(f"fitting SRF rank={RANK} max_outer={MAX_OUTER} (BLAS threads from OMP env)")
    t0 = time.time()
    est = SRF(
        rank=RANK, random_state=SEED, missing_values=np.nan,
        rho=3.0, max_inner=30, max_outer=MAX_OUTER, tol=0.0, check_input=False,
        verbose=1,
    )
    est.fit(sim)
    w = est.w_
    elapsed = time.time() - t0
    log(f"fit done in {elapsed:.1f}s   W shape={w.shape}   sparsity (zeros)={(w == 0).mean():.4f}   "
        f"min={w.min():.4f}  max={w.max():.4f}")

    np.save(OUTPUT_DIR / f"W_rank{RANK}_outer{MAX_OUTER}.npy", w)
    make_grid(w, paths, list(stimuli), OUTPUT_DIR / f"dim_topk_rank{RANK}_outer{MAX_OUTER}.png")
    log("done")


if __name__ == "__main__":
    main()
