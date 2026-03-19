"""Quick test: does signed correlation fix 2-level LOO alignment?"""

import itertools as it
import numpy as np
from pysrf import SRF
from src.tools.rsa import loo_alignment, global_alignment
from src.utils.helpers import add_positive_noise_with_snr

def create_factorial(n_levels_per_factor):
    """Create factorial with given levels per factor."""
    levels = {f"F{i}": [f"l{j}" for j in range(n)] for i, n in enumerate(n_levels_per_factor)}
    items = list(it.product(*levels.values()))
    features = []
    for i, (factor, lvls) in enumerate(levels.items()):
        idx = [lvls.index(item[i]) for item in items]
        features.append(np.eye(len(lvls))[idx])
    return np.hstack(features)

def test_alignment(x, name, n_repeats=10):
    """Test if alignment correctly matches columns."""
    print(f"\n{name}: {x.shape[0]} items, {x.shape[1]} cols")

    loo_correct = []
    global_correct = []

    for seed in range(n_repeats):
        x_noisy = add_positive_noise_with_snr(x, ratio=1.0, rng=seed)
        s = x_noisy @ x_noisy.T
        w = SRF(rank=x.shape[1], verbose=False, tol=0.0, random_state=seed).fit_transform(s)

        # Check alignment by correlation
        w_loo = loo_alignment(w, x)
        w_global = global_alignment(w, x)

        # Correlation of each aligned dim with corresponding ground truth col
        loo_corrs = [np.corrcoef(w_loo[:, i], x[:, i])[0, 1] for i in range(x.shape[1])]
        global_corrs = [np.corrcoef(w_global[:, i], x[:, i])[0, 1] for i in range(x.shape[1])]

        loo_correct.append(np.mean([c > 0.5 for c in loo_corrs]))
        global_correct.append(np.mean([c > 0.5 for c in global_corrs]))

    print(f"  LOO correct:    {np.mean(loo_correct)*100:.1f}%")
    print(f"  Global correct: {np.mean(global_correct)*100:.1f}%")

def main():
    print("=" * 50)
    print("Testing signed correlation alignment")
    print("=" * 50)

    # 2-level factors (was broken with |cor|)
    x_2level = create_factorial([2, 2, 2, 2])  # 16 items, 8 cols
    test_alignment(x_2level, "4 factors × 2 levels")

    # 3-level factors (always worked)
    x_3level = create_factorial([3, 3, 3])  # 27 items, 9 cols
    test_alignment(x_3level, "3 factors × 3 levels")

if __name__ == "__main__":
    main()
