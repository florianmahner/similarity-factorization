import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


class AdaptiveSampler:
    """Principled sampling strategies for gradient saturation detection."""

    def __init__(self, eval_func: callable, x_range: tuple = (0.01, 1.0)):
        """
        Parameters:
        -----------
        eval_func : callable
            Function that takes observed_fraction and returns best_rank
        x_range : tuple
            (min_fraction, max_fraction) to sample over
        """
        self.eval_func = eval_func
        self.x_min, self.x_max = x_range
        self.x_samples = []
        self.y_samples = []

    def geometric_progression_sampling(self, n_points: int = 15) -> np.ndarray:
        """
        Sample more densely at low fractions where rapid change typically occurs.
        Uses geometric progression: dense at start, sparse at end.
        """
        # Create geometric progression from 0 to 1
        t = np.linspace(0, 1, n_points)
        # Apply power function to concentrate samples at low values
        power = 2.5  # Higher power = more concentration at low values
        x_normalized = t**power

        # Scale to actual range
        x_samples = self.x_min + x_normalized * (self.x_max - self.x_min)
        return x_samples

    def log_linear_sampling(
        self, n_points: int = 15, transition_point: float = 0.3
    ) -> np.ndarray:
        """
        Logarithmic spacing for low fractions, linear for high fractions.
        """
        n_log = int(n_points * 0.6)  # 60% of points in log region
        n_lin = n_points - n_log

        # Logarithmic part (low fractions)
        log_points = np.logspace(
            np.log10(self.x_min), np.log10(transition_point), n_log
        )

        # Linear part (high fractions)
        lin_points = np.linspace(transition_point, self.x_max, n_lin + 1)[
            1:
        ]  # Exclude overlap

        return np.concatenate([log_points, lin_points])

    def adaptive_sequential_sampling(
        self,
        initial_points: int = 8,
        max_points: int = 20,
        gradient_threshold: float = 0.1,
    ) -> tuple:
        """
        Start with few points, add more where gradient is high.
        This is the most principled approach for saturation detection.
        """
        # Initial coarse sampling
        x_initial = self.geometric_progression_sampling(initial_points)
        y_initial = np.array([self.eval_func(x) for x in x_initial])

        self.x_samples = list(x_initial)
        self.y_samples = list(y_initial)

        for iteration in range(max_points - initial_points):
            if len(self.x_samples) < 5:  # Need minimum points for gradient
                break

            # Compute gradient at current sample points
            x_sorted, y_sorted = self._sort_samples()
            gradient = self._compute_gradient(x_sorted, y_sorted)

            # Find region with highest gradient magnitude
            high_gradient_idx = np.argmax(np.abs(gradient))

            # Check if gradient is above threshold
            if np.abs(gradient[high_gradient_idx]) < gradient_threshold:
                print(
                    f"Convergence reached at {len(self.x_samples)} points (gradient < {gradient_threshold})"
                )
                break

            # Add new point in high-gradient region
            if high_gradient_idx < len(x_sorted) - 1:
                new_x = (
                    x_sorted[high_gradient_idx] + x_sorted[high_gradient_idx + 1]
                ) / 2
            else:
                # Handle edge case
                new_x = (
                    x_sorted[high_gradient_idx]
                    + (self.x_max - x_sorted[high_gradient_idx]) / 2
                )

            # Ensure new point is in valid range and not too close to existing points
            if (
                self.x_min <= new_x <= self.x_max
                and min(np.abs(np.array(self.x_samples) - new_x)) > 0.01
            ):

                new_y = self.eval_func(new_x)
                self.x_samples.append(new_x)
                self.y_samples.append(new_y)
                print(
                    f"Added point {len(self.x_samples)}: x={new_x:.3f}, y={new_y:.3f}, gradient={gradient[high_gradient_idx]:.4f}"
                )

        x_final, y_final = self._sort_samples()
        return x_final, y_final

    def curvature_guided_sampling(self, n_points: int = 15) -> np.ndarray:
        """
        Sample based on expected curvature: more points where second derivative is high.
        """
        # Start with coarse sampling
        x_coarse = np.linspace(self.x_min, self.x_max, 8)
        y_coarse = np.array([self.eval_func(x) for x in x_coarse])

        # Compute second derivative (curvature)
        if len(x_coarse) >= 5:
            second_deriv = savgol_filter(
                y_coarse, window_length=5, polyorder=3, deriv=2
            )
            curvature = np.abs(second_deriv)

            # Create sampling density proportional to curvature
            density = curvature / np.sum(curvature)

            # Generate points based on density
            cumulative_density = np.cumsum(density)
            cumulative_density = cumulative_density / cumulative_density[-1]

            # Interpolate to get sample positions
            uniform_samples = np.linspace(0, 1, n_points)
            interp_func = interp1d(
                cumulative_density,
                x_coarse,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            x_samples = interp_func(uniform_samples)

            # Ensure bounds
            x_samples = np.clip(x_samples, self.x_min, self.x_max)
            return np.unique(x_samples)  # Remove duplicates

        # Fallback to geometric if too few points
        return self.geometric_progression_sampling(n_points)

    def _sort_samples(self) -> tuple:
        """Sort samples by x values."""
        sorted_indices = np.argsort(self.x_samples)
        return (
            np.array(self.x_samples)[sorted_indices],
            np.array(self.y_samples)[sorted_indices],
        )

    def _compute_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute smoothed gradient."""
        if len(x) >= 5:
            return savgol_filter(
                y,
                window_length=min(5, len(x)),
                polyorder=min(3, len(x) - 1),
                deriv=1,
                delta=np.mean(np.diff(x)),
                mode="interp",
            )
        else:
            return np.gradient(y, x)


def compare_sampling_strategies(
    eval_func: callable, true_x: np.ndarray, true_y: np.ndarray
):
    """Compare different sampling strategies."""

    sampler = AdaptiveSampler(eval_func)
    strategies = {
        "Uniform (current)": np.linspace(0.01, 1.0, 15),
        "Geometric": sampler.geometric_progression_sampling(15),
        "Log-linear": sampler.log_linear_sampling(15),
        "Curvature-guided": sampler.curvature_guided_sampling(15),
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, (name, x_samples) in enumerate(strategies.items()):
        ax = axes[i]

        # Sample at these points
        y_samples = np.array([eval_func(x) for x in x_samples])

        # Plot true function
        ax.plot(true_x, true_y, "k-", alpha=0.3, linewidth=2, label="True function")

        # Plot samples
        ax.scatter(
            x_samples,
            y_samples,
            color="red",
            s=60,
            alpha=0.8,
            label=f"{name} ({len(x_samples)} points)",
            zorder=5,
        )

        # Compute and plot interpolated function
        if len(x_samples) >= 3:
            from scipy.interpolate import interp1d

            interp_func = interp1d(
                x_samples,
                y_samples,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            x_interp = np.linspace(0.01, 1.0, 200)
            y_interp = interp_func(x_interp)
            ax.plot(
                x_interp, y_interp, "--", alpha=0.7, linewidth=2, label="Interpolated"
            )

        ax.set_xlabel("Observed fraction")
        ax.set_ylabel("Best rank")
        ax.set_title(f"{name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add efficiency metric
        mse = np.mean(
            (
                interp1d(
                    x_samples,
                    y_samples,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )(true_x)
                - true_y
            )
            ** 2
        )
        ax.text(
            0.02,
            0.98,
            f"MSE: {mse:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

    plt.tight_layout()
    return fig


def demonstrate_adaptive_sampling():
    """Demonstrate adaptive sampling with a realistic saturation curve."""

    # Create realistic saturation curve
    def saturation_curve(x):
        """Realistic rank vs observed fraction curve with saturation."""
        # Sharp drop initially, then saturation
        return 25 * np.exp(-5 * x) + 15 + 2 * np.random.normal(0, 0.1)

    # True function for comparison
    x_true = np.linspace(0.01, 1.0, 100)
    np.random.seed(42)  # For reproducible noise
    y_true = np.array([saturation_curve(x) for x in x_true])

    # Test adaptive sampling
    sampler = AdaptiveSampler(saturation_curve)

    print("=== ADAPTIVE SEQUENTIAL SAMPLING ===")
    x_adaptive, y_adaptive = sampler.adaptive_sequential_sampling(
        initial_points=6, max_points=15, gradient_threshold=0.5
    )

    # Compare strategies
    print(f"\nFinal adaptive sampling used {len(x_adaptive)} points")
    fig = compare_sampling_strategies(saturation_curve, x_true, y_true)

    # Show adaptive sampling process
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot adaptive sampling result
    ax1.plot(x_true, y_true, "k-", alpha=0.3, linewidth=2, label="True function")
    ax1.scatter(
        x_adaptive,
        y_adaptive,
        color="red",
        s=80,
        alpha=0.9,
        label=f"Adaptive sampling ({len(x_adaptive)} points)",
        zorder=5,
    )

    # Add sample order numbers
    for i, (x, y) in enumerate(zip(x_adaptive, y_adaptive)):
        ax1.annotate(
            f"{i+1}",
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )

    ax1.set_xlabel("Observed fraction")
    ax1.set_ylabel("Best rank")
    ax1.set_title("Adaptive sequential sampling result")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot gradient to show why points were chosen
    if len(x_adaptive) >= 5:
        gradient = savgol_filter(
            y_adaptive,
            window_length=5,
            polyorder=3,
            deriv=1,
            delta=np.mean(np.diff(x_adaptive)),
            mode="interp",
        )
        ax2.plot(
            x_adaptive, gradient, "o-", color="orange", linewidth=2, label="Gradient"
        )
        ax2.axhline(y=0.5, color="red", linestyle="--", label="Convergence threshold")
        ax2.set_xlabel("Observed fraction")
        ax2.set_ylabel("Gradient magnitude")
        ax2.set_title("Gradient at sample points")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, fig2, x_adaptive


if __name__ == "__main__":
    # Demonstrate the approach
    fig1, fig2, x_samples = demonstrate_adaptive_sampling()

    print(f"\n=== RECOMMENDED SAMPLING STRATEGY ===")
    print(f"Adaptive sampling used only {len(x_samples)} points vs 50 uniform points")
    print(f"Sample points: {[f'{x:.3f}' for x in x_samples]}")
    print(f"\nFor your cross-validation script, replace:")
    print(f"  default=list(np.linspace(0.01, 1.0, 50).round(3))")
    print(f"With:")
    print(f"  default={[round(x, 3) for x in x_samples]}")

    # save the figure
    from pathlib import Path

    path = Path("./results/adaptive_sampling")
    path.mkdir(parents=True, exist_ok=True)
    fig1.savefig(path / "adaptive_sampling.png")
    fig2.savefig(path / "adaptive_sampling_gradient.png")
