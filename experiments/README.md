# Experiments

Reproduction code for the paper. Each task builds a similarity matrix, fits SRF, and writes results or figures. The factorization algorithm itself is the separate [`pysrf`](https://florianmahner.github.io/pysrf/) package; everything here calls it.

The analyses in the paper are organized into the following groups:

- **Datasets**: for each dataset, build a similarity matrix, select its dimensionality by cross-validation, and fit a stable consensus embedding.
- **Simulations**: validate factor recovery, missing-data imputation, and rank selection on synthetic data.
- **THINGS behavior**: fit and evaluate embeddings from human odd-one-out triplets.
- **RSA comparison**: compare the statistical power of RSA against SRF for hypothesis testing.
- **SWOW**: predict behavioral word properties from the word-association embedding.
- **Figures**: assemble the paper figures from the analysis outputs.

Each task is a folder with a `run.py` and a `config.yaml`, configured with [Hydra](https://hydra.cc/) and run as a module. Shared defaults and dataset definitions live in [`configs`](../configs); outputs are written to `outputs/experiments/<experiment_name>/<task>/`. Override any config key inline (e.g. `dataset=nsd seed=0`). The commands below use the default settings from the paper. Feature-extraction scripts (DNN features, macaque pipeline) are under `preprocessing/`.

### Datasets

Run the stages in order; each reads the previous stage's output. For feature datasets (CLIP, DINOv3, NSD, macaque), select the kernel bandwidth first under `datasets/bandwidth_selection/`.

```bash
# 1. select dimensionality by cross-validation
poetry run python -m experiments.datasets.dimensionality.run mode=all only=[mur92]
# 2. fit the consensus embedding at the selected rank
poetry run python -m experiments.datasets.consensus.run dataset=mur92
# 3. visualize the top items per dimension
poetry run python -m experiments.datasets.visualize.run dataset=mur92
```

Datasets available via `dataset=`: `mur92`, `cichy118`, `nsd`, `swow`, `things_behavior`, `things_monkey_22k`, `peterson` (+ `_animals`/`_various`), `clip_vit_l14`, `dinov3`, `vgg16`, `vit`.

### Simulations

```bash
poetry run python -m experiments.analyses.simulation.rank_detection.run     # rank selection vs. baselines
poetry run python -m experiments.analyses.simulation.imputation.run         # missing-data imputation
poetry run python -m experiments.analyses.simulation.interpretability.run   # factor recovery and stability
```

### THINGS behavior

```bash
poetry run python -m experiments.analyses.things_behavior.similarity_48.run       # predict the 48-object similarity matrix
poetry run python -m experiments.analyses.things_behavior.pairwise.run            # dimension-wise recovery
poetry run python -m experiments.analyses.things_behavior.triplet_prediction.run  # SRF vs. SPoSE / VICE triplet accuracy
poetry run python -m experiments.analyses.things_behavior.rank_sweep.run          # triplet accuracy vs. rank
poetry run python -m experiments.analyses.things_behavior.reliability.run         # split-half dimension reliability
poetry run python -m experiments.analyses.things_behavior.lowdata.comparison.run  # SRF vs. VICE in the low-data regime
```

### RSA comparison

```bash
poetry run python -m experiments.analyses.rsa.factorial.run   # power on a factorial design
poetry run python -m experiments.analyses.rsa.spose.run       # SPoSE dimension recovery
```

### SWOW

```bash
poetry run python -m experiments.analyses.swow.run            # predict Glasgow Norms from the SWOW embedding
```

### Figures

Each figure script reads the outputs of the analyses above and writes a PDF. Run the corresponding analyses first.

```bash
poetry run python -m experiments.figures.plot_simulation.plot         # simulations
poetry run python -m experiments.figures.plot_datasets.plot           # datasets overview
poetry run python -m experiments.figures.plot_embeddings.plot         # observed vs. reconstructed similarity
poetry run python -m experiments.figures.plot_things.plot             # THINGS behavior
poetry run python -m experiments.figures.plot_rsa_comparison.plot     # RSA vs. SRF power
poetry run python -m experiments.figures.plot_dimensionality_cv.plot  # cross-validated rank selection
poetry run python -m experiments.figures.plot_supplementary.plot      # supplementary figures
```
