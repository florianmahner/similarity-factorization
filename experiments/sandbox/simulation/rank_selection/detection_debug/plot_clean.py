"""Create clean rank detection scatter plot (matching screenshot style)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

csv_path = Path("/LOCAL/fmahner/similarity-factorization/outputs/experiments/simulation/rank_detection/rank_detection_results.csv")
output_dir = Path.cwd() / "outputs"
output_dir.mkdir(exist_ok=True)

df = pd.read_csv(csv_path)

rng = np.random.default_rng(42)
jitter = 0.8
df["x"] = df["true_rank"] + rng.uniform(-jitter, jitter, len(df))
df["y"] = df["selected_rank"] + rng.uniform(-jitter, jitter, len(df))

fig, ax = plt.subplots(figsize=(4.5, 4))

snr_values = sorted(df["snr"].unique())
markers = ["o", "s", "^", "D"]
colors = sns.color_palette("tab10", n_colors=len(snr_values))

for i, snr in enumerate(snr_values):
    subset = df[df["snr"] == snr]
    ax.scatter(
        subset["x"], subset["y"],
        s=40, alpha=0.7,
        marker=markers[i],
        color=colors[i],
        label=f"{snr}",
        edgecolors="white",
        linewidths=0.4,
    )

min_r, max_r = df["true_rank"].min(), df["true_rank"].max()
ax.plot([min_r-1, max_r+1], [min_r-1, max_r+1], "--", color="gray", lw=1.2, zorder=0)

ax.set_xlabel("True rank", fontsize=12)
ax.set_ylabel("Selected rank", fontsize=12)
ax.set_xlim(min_r - 2, max_r + 2)
ax.set_ylim(min_r - 2, max_r + 2)

ax.legend(title="SNR", loc="upper left", frameon=False, fontsize=10, title_fontsize=11)
sns.despine()

plt.tight_layout()
fig.savefig(output_dir / "rank_scatter_clean.pdf", dpi=300, bbox_inches="tight")
print(f"Saved: {output_dir / 'rank_scatter_clean.pdf'}")

# Print summary
print("\nSummary:")
for tr in sorted(df["true_rank"].unique()):
    sub = df[df["true_rank"] == tr]
    mae = sub["abs_error"].mean()
    med = sub["selected_rank"].median()
    print(f"  k={tr:2d}: median_selected={med:.0f}, MAE={mae:.1f}")
