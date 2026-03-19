"""Plot RSA vs SRF comparison from existing results."""

from pathlib import Path

import pandas as pd

from src.colors import ROSE, TEAL, CYAN, GRAY_DARK
from src.utils.figure_theme import create_figure, despine, save_figure


def main():
    # Load results
    results_dir = Path(__file__).parent / "outputs/260109/105354"
    df = pd.read_csv(results_dir / "results.csv")

    # --- PLOT 1: Direct RSA vs SRF_Struct comparison (one panel per factor) ---
    factors = df["factor"].unique()
    fig, axes = create_figure("full_width", ncols=len(factors))

    for idx, factor in enumerate(factors):
        ax = axes[idx]
        sub = df[df["factor"] == factor]

        for method, color, marker in [
            ("RSA", ROSE, "o"),
            ("SRF_Struct", TEAL, "s"),
        ]:
            data = sub[sub["method"] == method].sort_values("snr")
            ax.plot(
                data["snr"],
                data["p_value"],
                f"{marker}-",
                color=color,
                label=method.replace("_", " "),
                markersize=6,
                linewidth=1.5,
            )

        ax.axhline(0.05, color=GRAY_DARK, linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("SNR")
        ax.set_ylabel("P-value" if idx == 0 else "")
        ax.set_title(factor.capitalize())
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(-0.02, 0.52)
        ax.set_yscale("linear")
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right", frameon=False)
        despine(ax)

    save_figure(fig, results_dir / "rsa_vs_srf_by_factor.pdf")

    # --- PLOT 2: All three methods, averaged across factors ---
    fig, ax = create_figure("single")

    pval_avg = df.groupby(["snr", "method"])["p_value"].mean().reset_index()

    for method, color, marker in [
        ("RSA", ROSE, "o"),
        ("SRF_Struct", TEAL, "s"),
        ("SRF_Decod", CYAN, "^"),
    ]:
        data = pval_avg[pval_avg["method"] == method].sort_values("snr")
        ax.plot(
            data["snr"],
            data["p_value"],
            f"{marker}-",
            color=color,
            label=method.replace("_", " "),
            markersize=6,
            linewidth=1.5,
        )

    ax.axhline(0.05, color=GRAY_DARK, linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("SNR")
    ax.set_ylabel("P-value (mean across factors)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.02, 0.52)
    ax.legend(loc="upper right", frameon=False)
    despine(ax)

    save_figure(fig, results_dir / "rsa_vs_srf_averaged.pdf")

    # --- PLOT 3: Log-scale p-values for better resolution at low values ---
    fig, axes = create_figure("full_width", ncols=len(factors))

    for idx, factor in enumerate(factors):
        ax = axes[idx]
        sub = df[df["factor"] == factor]

        for method, color, marker in [
            ("RSA", ROSE, "o"),
            ("SRF_Struct", TEAL, "s"),
        ]:
            data = sub[sub["method"] == method].sort_values("snr")
            ax.plot(
                data["snr"],
                data["p_value"],
                f"{marker}-",
                color=color,
                label=method.replace("_", " "),
                markersize=6,
                linewidth=1.5,
            )

        ax.axhline(0.05, color=GRAY_DARK, linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("SNR")
        ax.set_ylabel("P-value" if idx == 0 else "")
        ax.set_title(factor.capitalize())
        ax.set_yscale("log")
        ax.set_ylim(5e-5, 2)
        ax.set_xlim(-0.02, 0.52)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower left", frameon=False)
        despine(ax)

    save_figure(fig, results_dir / "rsa_vs_srf_by_factor_log.pdf")

    print(f"Saved plots to {results_dir}")


if __name__ == "__main__":
    main()
