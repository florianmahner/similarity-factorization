"""THINGS 5% low data - bounds with missing values."""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pysrf import SRF
from pysrf.bounds import estimate_sampling_bounds_ultra
from pysrf.cross_validation import cross_val_score
from src.utils.io import load_triplets
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()

N_OBJECTS = 1854

def build_similarity_no_smoothing(triplets, n):
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))
    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            if a != b:
                shown[a, b] += 1
                shown[b, a] += 1
        if i != j:
            counts[i, j] += 1
            counts[j, i] += 1
    similarity = np.full((n, n), np.nan)
    observed = shown > 0
    similarity[observed] = counts[observed] / shown[observed]
    np.fill_diagonal(similarity, 1.0)
    return similarity

def impute_similarity(S, rank, n_seeds=3):
    reconstructions = []
    for seed in range(n_seeds):
        print(f"  Impute seed {seed+1}/{n_seeds}")
        model = SRF(rank=rank, missing_values=np.nan, random_state=seed, max_outer=50, tol=1e-4)
        model.fit(S)
        reconstructions.append(model.reconstruct())
    return np.clip(np.mean(reconstructions, axis=0), np.nanmin(S), np.nanmax(S))

if __name__ == "__main__":
    print("THINGS 5%")
    train_triplets, _ = load_triplets(Path("data/things"), number="4.7mio")
    rng = np.random.default_rng(42)
    triplets = train_triplets[rng.choice(len(train_triplets), int(len(train_triplets)*0.05), replace=False)]
    print(f"Triplets: {len(triplets):,}")
    
    S = build_similarity_no_smoothing(triplets, N_OBJECTS)
    missing_frac = np.isnan(S).sum() / (N_OBJECTS * N_OBJECTS)
    print(f"Missing: {missing_frac*100:.1f}%")
    
    print("Imputing...")
    S_imp = impute_similarity(S, rank=10, n_seeds=3)
    
    print("Bounds...")
    pmin, pmax, _ = estimate_sampling_bounds_ultra(S_imp, random_state=42)
    sf = (pmin + pmax) / 2
    print(f"pmin={pmin:.3f}, pmax={pmax:.3f}, sf={sf:.3f}")
    
    ranks = [2, 5, 10, 15, 20, 30]
    print(f"CV ranks={ranks}")
    cv = cross_val_score(S, param_grid={"rank": ranks}, n_repeats=3, sampling_fraction=sf, 
                         random_state=42, n_jobs=-1, verbose=1, missing_values=np.nan)
    
    print(f"\nOPTIMAL RANK: {cv.best_params_['rank']}")
    scores = cv.cv_results_.groupby("rank")["score"].mean()
    for r in ranks:
        print(f"  rank {r}: {scores[r]:.4f}")
    
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump({"optimal_rank": int(cv.best_params_["rank"]), "pmin": float(pmin), 
                   "pmax": float(pmax), "sf": float(sf), "missing_frac": float(missing_frac)}, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}")
