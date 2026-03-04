<p align="center">
  <img src="assets/logo.png" alt="pysrf" width="200">
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-4B8BBE?style=flat-square&logo=python&logoColor=FFD43B" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-A78BFA?style=flat-square" alt="License: MIT"></a>
  <a href="https://florianmahner.github.io/pysrf/"><img src="https://img.shields.io/badge/docs-online-06B6D4?style=flat-square&logo=readthedocs&logoColor=white" alt="Documentation"></a>
  <a href="https://github.com/florianmahner/pysrf/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/florianmahner/pysrf/ci.yml?style=flat-square&logo=githubactions&logoColor=white&label=CI" alt="CI"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-261230?style=flat-square&logo=ruff&logoColor=D7FF64" alt="ruff"></a>
  <a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-F472B6?style=flat-square&logo=pre-commit&logoColor=white" alt="pre-commit"></a>
</p>

---

Sparse non-negative factorization of similarity matrices: $S \approx WW^\top$.
Handles missing data, estimates dimensionality via cross-validation, and
produces stable consensus embeddings.

<p align="center">
  <img src="assets/factorization.svg" alt="S ≈ W × Wᵀ factorization diagram" width="800">
</p>

## Install

```bash
git clone https://github.com/florianmahner/pysrf.git && cd pysrf && ./setup.sh
```

## Usage

```python
from pysrf import SRF

model = SRF(rank=10, random_state=42)
w = model.fit_transform(s)
```

## Docs

Full guide at **[florianmahner.github.io/pysrf](https://florianmahner.github.io/pysrf/)**.

## Reference

Mahner, F. P.\*, Lam, K. C.\*, & Hebart, M. N. (2025). Interpretable
dimensions from sparse representational similarities. *In preparation*.
