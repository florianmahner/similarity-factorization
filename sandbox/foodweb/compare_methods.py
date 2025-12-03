#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare SRF factors to community detection results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

output_dir = Path("experiments/development/foodweb/outputs")
srf_run = sorted(output_dir.rglob("*/"))[-1]
print(f"Loading SRF results from: {srf_run}")
srf_species = pd.read_csv(srf_run / "species_with_factors.csv")

validation_df = pd.read_csv(output_dir / "structure_validation.csv")
srf_species["louvain_community"] = validation_df["louvain_community"].values

n_factors = 15

fig, axes = plt.subplots(3, 2, figsize=(14, 16))

ax = axes[0, 0]
for comm in range(3):
    comm_mask = srf_species["louvain_community"] == comm
    log_mass = np.log10(srf_species.loc[comm_mask, "body_mass_g"].replace(0, np.nan).dropna())
    ax.hist(log_mass, bins=20, alpha=0.6, label=f"Community {comm} (n={comm_mask.sum()})")
ax.set_xlabel("log10 body mass (g)")
ax.set_ylabel("Count")
ax.set_title("Louvain communities by body mass")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 1]
factor_mass_corr = []
for f in range(n_factors):
    valid_idx = srf_species["body_mass_g"] > 0
    rho, _ = spearmanr(
        srf_species.loc[valid_idx, f"factor_{f}"],
        np.log10(srf_species.loc[valid_idx, "body_mass_g"])
    )
    factor_mass_corr.append(rho)
colors = ['red' if abs(r) > 0.2 else 'gray' for r in factor_mass_corr]
ax.bar(range(n_factors), factor_mass_corr, color=colors, alpha=0.7)
ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel("SRF factor")
ax.set_ylabel("Spearman correlation")
ax.set_title("SRF factors vs log body mass")
ax.grid(alpha=0.3, axis='y')

ax = axes[1, 0]
comm_factor_overlap = np.zeros((3, n_factors))
for comm in range(3):
    comm_mask = srf_species["louvain_community"] == comm
    for f in range(n_factors):
        comm_factor_overlap[comm, f] = srf_species.loc[comm_mask, f"factor_{f}"].sum()
im = ax.imshow(comm_factor_overlap, cmap='YlOrRd', aspect='auto')
ax.set_xlabel("SRF factor")
ax.set_ylabel("Louvain community")
ax.set_title("Community-factor overlap (sum of loadings)")
ax.set_yticks(range(3))
ax.set_yticklabels([f"C{i} (n={int((srf_species['louvain_community']==i).sum())})" for i in range(3)])
plt.colorbar(im, ax=ax, label="Sum loading")

ax = axes[1, 1]
for f in [0, 4, 8, 14]:
    loadings = srf_species[f"factor_{f}"]
    valid_mass = srf_species["body_mass_g"] > 0
    ax.scatter(
        np.log10(srf_species.loc[valid_mass, "body_mass_g"]),
        loadings[valid_mass],
        alpha=0.5,
        s=20,
        label=f"F{f}"
    )
ax.set_xlabel("log10 body mass (g)")
ax.set_ylabel("Factor loading")
ax.set_title("Selected factors vs body mass")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[2, 0]
for comm in range(3):
    comm_mask = srf_species["louvain_community"] == comm
    trophic = srf_species.loc[comm_mask, "trophic_level"]
    ax.hist(trophic, bins=20, alpha=0.6, label=f"Community {comm}")
ax.set_xlabel("Trophic level")
ax.set_ylabel("Count")
ax.set_title("Trophic level by community")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[2, 1]
factor_sparsity = []
for f in range(n_factors):
    nonzero = (srf_species[f"factor_{f}"] > 0.01).sum()
    factor_sparsity.append(nonzero)
ax.bar(range(n_factors), factor_sparsity, color='steelblue', alpha=0.7)
ax.axhline(163, color='red', linestyle='--', linewidth=1, label='All species')
ax.set_xlabel("SRF factor")
ax.set_ylabel("Species with loading > 0.01")
ax.set_title("Factor sparsity (k=15, normalized)")
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plot_path = output_dir / "method_comparison.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved: {plot_path}")

print("\n=== METHOD COMPARISON SUMMARY ===")
print(f"SRF: k={n_factors}, AUC=0.929, AP=0.920")
print(f"Louvain: k=3, Q=0.343")
print(f"\nFactor-body mass correlations:")
for f, rho in enumerate(factor_mass_corr):
    if abs(rho) > 0.2:
        print(f"  Factor {f:2d}: ρ = {rho:+.3f}")

