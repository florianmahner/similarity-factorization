#!/usr/bin/env python3
"""
Create summary plots for the ablation study.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()

# Data from experiments
data_ablation = {
    "Dataset": ["1.47M (5%)", "1.47M (10%)", "1.47M (20%)", "1.47M (30%)",
                "1.47M (50%)", "1.47M (70%)", "1.47M (100%)", "4.7M (100%)"],
    "Triplets": [65705, 131410, 262821, 394232, 657053, 919874, 1314107, 4120663],
    "Coverage": [10.9, 20.8, 37.8, 51.8, 72.5, 86.0, 97.6, 100.0],
    "SRF": [43.54, 48.68, 54.30, 57.39, 60.48, 61.98, 63.31, 63.97],
    "SPoSE": [65.38, 65.38, 65.38, 65.38, 65.38, 65.38, 65.38, 64.12],
}

df = pd.DataFrame(data_ablation)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Accuracy vs Triplets
ax = axes[0, 0]
ax.plot(df["Triplets"] / 1e6, df["SRF"], 'o-', label="SRF", markersize=8)
ax.axhline(y=65.38, color='green', linestyle='--', alpha=0.7, label="SPoSE (1.47M)")
ax.axhline(y=64.12, color='blue', linestyle='--', alpha=0.7, label="SPoSE (4.7M)")
ax.axhline(y=66.67, color='gray', linestyle=':', alpha=0.7, label="Noise ceiling")
ax.set_xlabel("Training triplets (millions)")
ax.set_ylabel("Validation accuracy (%)")
ax.set_title("Accuracy vs Training Data Size")
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# Plot 2: Accuracy vs Coverage
ax = axes[0, 1]
ax.plot(df["Coverage"], df["SRF"], 'o-', label="SRF", markersize=8, color='orange')
ax.axhline(y=65.38, color='green', linestyle='--', alpha=0.7, label="SPoSE")
ax.axhline(y=66.67, color='gray', linestyle=':', alpha=0.7, label="Noise ceiling")
ax.set_xlabel("Pair coverage (%)")
ax.set_ylabel("Validation accuracy (%)")
ax.set_title("Accuracy vs RSM Coverage")
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# Plot 3: Gap to SPoSE
ax = axes[1, 0]
gaps = [65.38 - srf for srf in df["SRF"][:-1]]  # Exclude 4.7M
gaps.append(64.12 - 63.97)  # 4.7M gap
ax.bar(range(len(df)), gaps, alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(df)))
ax.set_xticklabels([f"{c:.0f}%" for c in df["Coverage"]], rotation=45)
ax.set_xlabel("RSM Coverage")
ax.set_ylabel("Gap to SPoSE (%)")
ax.set_title("Performance Gap: SPoSE - SRF")
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: NaN impact summary
ax = axes[1, 1]
nan_data = {
    "Metric": ["Total pairs", "Observed", "NaN (unobserved)", "Val triplets\nwith NaN"],
    "Count": [1717731, 1676263, 41468, 42730],
    "Percentage": [100, 97.6, 2.4, 29.3],
}
colors = ['steelblue', 'green', 'red', 'orange']
bars = ax.bar(nan_data["Metric"], nan_data["Percentage"], color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel("Percentage (%)")
ax.set_title("NaN Impact on 1.47M Dataset")
ax.set_ylim(0, 105)
for bar, count in zip(bars, nan_data["Count"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{count:,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "summary.pdf", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved summary.pdf")

# Create a text summary
summary = """
========================================
TRIPLET ABLATION STUDY SUMMARY
========================================

DATASET COMPARISON
------------------
              1.47M Dataset    4.7M Dataset
Triplets:     1,314,107        4,120,663
Coverage:     97.6%            100.0%
NaN pairs:    41,468 (2.4%)    ~8 (0.0%)
SRF acc:      63.31%           63.97%
SPoSE acc:    65.38%           64.12%
Gap:          -2.07%           -0.15%

KEY FINDINGS
------------
1. Coverage matters: 97.6% → 100% gives +0.66% accuracy
2. NaN fill strategy: minimal impact (~0.2% spread)
3. Optimal fill value: 0.5 (or row mean)
4. 29% of val triplets contain NaN pairs
5. Gap to SPoSE is fundamental, not due to NaN handling

RECOMMENDATIONS
---------------
- Use 0.5 as default NaN fill (current approach is fine)
- More data helps more than better NaN handling
- To beat SPoSE: need fundamentally different RSM construction
"""
print(summary)

with open(OUTPUT_DIR / "summary.txt", "w") as f:
    f.write(summary)
print("Saved summary.txt")
