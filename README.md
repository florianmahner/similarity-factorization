# Similarity-Based Representation Factorization (SRF)

## Overview

This repository contains code to reproduce the results of our paper. SRF recovers low-dimensional, non-negative, interpretable embeddings directly from similarity data measured in minds, brains, and machines.

The method itself — the model, cross-validation, rank estimation, and consensus embeddings — is provided as a separate package, [`pysrf`](https://github.com/florianmahner/pysrf) ([documentation](https://florianmahner.github.io/pysrf/)). This repository builds the similarity matrices, runs the analyses, and assembles the figures for the paper.

## Installation

### 1. Install Poetry

This project requires **Python 3.12** and [Poetry](https://python-poetry.org/) for dependency management.

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Clone the repository

```bash
git clone https://github.com/florianmahner/similarity-factorization.git
cd similarity-factorization
```

### 3. Install dependencies

```bash
poetry install
```

This also installs `pysrf` from `third_party/pysrf/`.

## Main Experiments

### Downloading Data

The preprocessed similarity matrices and consensus embeddings are hosted on OSF. Download them into `data/` by running:

```bash
make data
```

The raw source datasets (THINGS images, NSD, macaque recordings, ...) are obtained from their original providers; see the data-availability statement in the paper and set their local paths in `configs/paths/local.yaml`.

### Running Experiments

Experiments are configured with [Hydra](https://hydra.cc/) and run as modules. For example, to build a consensus embedding for the `mur92` dataset:

```bash
poetry run python -m experiments.datasets.consensus.run dataset=mur92
```

A detailed guide is available in the [experiments README](experiments/README.md).

## Contact

For questions or issues, open a [GitHub issue](https://github.com/florianmahner/similarity-factorization/issues) or contact Florian Mahner (<florian.mahner@gmail.com>).
