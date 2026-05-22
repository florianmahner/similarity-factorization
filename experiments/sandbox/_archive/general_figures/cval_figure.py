"""Generate publication-quality train/validation schematic for rank selection."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.figure_theme import CMAP, GRAY, apply_theme, despine

apply_theme()

# Generate smooth curves - smaller rank range for subtle overfitting
x = np.linspace(0, 50, 500)

# Train error: monotonically decreasing
train = 0.70 * np.exp(-0.10 * x) + 0.08

# Validation error: U-shaped with subtle upturn
gap = 0.02 + 0.0015 * x
val_upturn = 0.0003 * np.maximum(0, x - 18) ** 2
validation = train + gap + val_upturn

# Find optimal rank
optimal_idx = np.argmin(validation)
optimal_rank = x[optimal_idx]
optimal_val = validation[optimal_idx]

# Create figure
fig, ax = plt.subplots(figsize=(2.6, 2.0))

# Plot curves
ax.plot(x, train, color=GRAY["dark"], linewidth=1.8, label="Train", zorder=3)
ax.plot(x, validation, color=CMAP[0], linewidth=1.8, label="Validation", zorder=3)

# Mark optimal point
ax.axvline(optimal_rank, color=GRAY["light"], linestyle="--", linewidth=0.8, zorder=1)
ax.scatter(
    [optimal_rank],
    [optimal_val],
    color=CMAP[0],
    s=35,
    zorder=4,
    edgecolor="white",
    linewidth=1.0,
)

# Labels
ax.set_xlabel("Rank", fontsize=10)
ax.set_ylabel("MSE", fontsize=10)

# Clean axis
ax.set_xlim(0, 50)
ax.set_ylim(0, None)
ax.set_xticks([])
ax.set_yticks([])

# Optimal rank label inside plot near the point
ax.text(
    optimal_rank + 2,
    optimal_val,
    "k*",
    ha="left",
    va="center",
    fontsize=10,
    style="italic",
)

despine(ax)

# Legend
ax.legend(
    loc="upper right",
    frameon=False,
    fontsize=8,
    handlelength=1.2,
    labelspacing=0.3,
)

plt.tight_layout()

# Save as SVG
output_path = Path(__file__).parent / "outputs" / "cval.svg"
output_path.parent.mkdir(exist_ok=True)
fig.savefig(output_path, format="svg", bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

print(f"Saved: {output_path}")
