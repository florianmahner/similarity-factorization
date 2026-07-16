# Similarity-Based Representation Factorization (SRF)

## Overview

This repository contains code to reproduce the results of our paper. SRF recovers low-dimensional, non-negative, interpretable embeddings directly from similarity data measured in minds, brains, and machines.

The method itself — the model, cross-validation, rank estimation, and consensus embeddings — is provided as a separate package, [`pysrf`](https://github.com/florianmahner/pysrf) ([documentation](https://florianmahner.github.io/pysrf/)), vendored here under `third_party/pysrf/`. This repository builds the similarity matrices, runs the analyses, and assembles the figures for the paper. A self-contained demo that runs in seconds is in [`demo/`](demo/) — start there.

## 1. System requirements

**Operating system.** Linux or macOS (tested on macOS 27, Apple M1 Max, 32 GB RAM). No platform-specific code is used; Windows should work but is untested.

**Python.** ≥3.10 for the demo and the core `pysrf` package; 3.12 for the full reproduction environment (tested with 3.12.13; the project pins `>=3.12,<3.14`).

**Dependencies** (tested versions): `pysrf` 0.1.0, `numpy` 2.2.6, `scipy` 1.17.1, `scikit-learn` 1.8.0 (also verified with 1.7.2; **1.7 is the minimum** — earlier versions fail the model's parameter validation), plus `pandas`, `joblib`, and `tqdm`. Reproducing every figure additionally uses `matplotlib`, `seaborn`, `statsmodels`, `networkx`, `torch`/`torchvision`, `transformers`, and `hydra-core`; all are declared in [`pyproject.toml`](pyproject.toml) and pinned in [`poetry.lock`](poetry.lock).

**Hardware.** A normal laptop CPU suffices; no non-standard hardware is required. SRF uses multi-threaded BLAS (set `OMP_NUM_THREADS` to control threading), and an optional Cython extension — compiled automatically when `pysrf` is installed with a C compiler present, with a pure-Python fallback otherwise — gives a further 10–50× speedup on large matrices. A GPU is optional and only used for the deep-network feature-extraction preprocessing (CLIP/DINOv3), not for SRF itself.

## 2. Installation guide

### Demo only (≈1–2 minutes)

Needs only `pip`. From the repository root:

```bash
python -m pip install -r demo/requirements-demo.txt
python -m pip install ./third_party/pysrf
```

### Full reproduction (≈5–15 minutes)

Needs Python 3.12 and [Poetry](https://python-poetry.org/); install time is dominated by the large dependencies (`torch`, `tensorflow`, `transformers`).

```bash
curl -sSL https://install.python-poetry.org | python3 -   # install Poetry if needed
git clone https://github.com/florianmahner/similarity-factorization.git
cd similarity-factorization
poetry install                                            # also installs pysrf from third_party/
```

## 3. Demo

The demo fits SRF to a simulated `100 × 100` similarity matrix built from a known sparse, non-negative embedding with 6 latent dimensions ([`demo/README.md`](demo/README.md) has details). It estimates the number of dimensions, fits the model, scores recovery against the ground truth, and refits with 40% of the entries hidden. From the repository root:

```bash
python demo/run_demo.py    # or: poetry run python demo/run_demo.py
```

Expected output (runs in ≈3 s on a normal laptop):

```
[1/4] estimating rank (bootstrap eigenspace coherence)...
      estimated rank k* = 6 (true rank = 6)
[2/4] fitting SRF at rank 6...
      converged in 2 iterations
[3/4] scoring...
      similarity reconstruction r  = 1.000
      mean dimension recovery r    = 1.000
[4/4] refitting with 40% of entries hidden (missing-data mode)...
      fraction hidden              = 0.40
      held-out reconstruction r    = 1.000

DEMO PASSED
```

The estimated rank should equal 6 and all correlations should be near 1.0; the last digit may vary across platforms and BLAS libraries.

## 4. Instructions for use

### Running SRF on your own data

Given any symmetric similarity matrix `S` (an `n × n` NumPy array, with missing entries marked as `NaN`):

```python
import numpy as np
from pysrf import SRF, estimate_rank

S = np.load("my_similarity.npy")

rank = estimate_rank(S, k_max=30)["k_star"]  # 1. estimate the number of dimensions
model = SRF(rank=rank, random_state=0)
W = model.fit_transform(S)                   # 2. non-negative embedding (items x dimensions)
S_hat = model.reconstruct()                  # 3. reconstruction W @ W.T
```

Each row of `W` gives an item's loadings on the recovered dimensions; near-zero loadings mean a dimension is irrelevant to that item. SRF handles missing entries directly (no imputation) and supports cross-validated rank selection and stable consensus embeddings — see the [`pysrf` documentation](https://florianmahner.github.io/pysrf/) for the full API (`cross_val_score`, `GridSearchCV`, `EnsembleFit`, `estimate_sampling_bounds`, …).

### Reproducing the paper

The analyses that produce the paper's quantitative results and figures live in [`experiments/`](experiments), configured with [Hydra](https://hydra.cc/); the [experiments README](experiments/README.md) lists the commands for every analysis group and the scripts that assemble each figure.

Download the preprocessed similarity matrices and consensus embeddings (hosted on OSF) into `data/`:

```bash
make data
```

The raw source datasets (THINGS images, NSD, macaque recordings, …) are obtained from their original providers; see the data-availability statement in the paper and set their local paths in `configs/paths/local.yaml`.

Example — build a consensus embedding for the `mur92` dataset:

```bash
poetry run python -m experiments.datasets.dimensionality.run mode=all only=[mur92]  # 1. select rank by cross-validation
poetry run python -m experiments.datasets.consensus.run dataset=mur92               # 2. fit the consensus embedding
poetry run python -m experiments.datasets.visualize.run dataset=mur92               # 3. visualize top items per dimension
```

## License

All software in this repository, including the vendored `pysrf` package ([`third_party/pysrf/`](third_party/pysrf)), is released under the [BSD 3-Clause License](LICENSE), a permissive license approved by the [Open Source Initiative](https://opensource.org/license/bsd-3-clause). The raw source datasets are not covered by this license and are subject to the terms of their original providers (see the data-availability statement in the paper).

## Contact

For questions or issues, open a [GitHub issue](https://github.com/florianmahner/similarity-factorization/issues) or contact Florian Mahner (<florian.mahner@gmail.com>).
