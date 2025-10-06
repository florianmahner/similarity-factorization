# embedding generation pipeline

consensus embedding generation using similarity representation factorization with the updated pysrf API.

## overview

the pipeline:
1. loads similarity matrices from datasets
2. estimates optimal sampling bounds using random matrix theory
3. performs cross-validation to select optimal rank
4. fits ensemble embeddings with multiple stable runs (default: 50)
5. clusters consensus embeddings to obtain final k-dimensional representation

uses the new pysrf pipeline API with `EnsembleEmbedding` and `ClusterEmbedding`.

## datasets

configured datasets in `/ptmp/fmahner/`:
- `mur92` - 92 object similarity dataset
- `cichy118` - 118 object similarity dataset  
- `peterson-animals` - peterson animals dataset
- `peterson-various` - peterson various objects dataset
- `nsd` - natural scenes dataset (8 subjects, fmri data)
- `things-monkey-22k` - things monkey neural recordings
- `vit` - vision transformer features

## rank grids

- small datasets (mur92, cichy118, peterson-*): ranks 1-30 (step 1)
- large datasets (nsd, monkey, vit): ranks 5-150 (step 5)

## usage

### debug mode (interactive partition, fast)

```bash
cd experiments/embedding_generation
./submit_debug.sh [dataset_name]
```

parameters:
- 4 cpus
- 8gb memory
- 30min time limit
- 2 cv repeats
- 5 stable runs

### production mode (full analysis)

```bash
cd experiments/embedding_generation
./submit_all.sh  # submits all configured datasets
```

parameters:
- 16-32 cpus (dataset dependent)
- 32-400gb memory (nsd requires 400gb)
- 6-24h time limit (dataset dependent)
- 5 cv repeats
- 50 stable runs

### manual submission

```bash
cd /u/fmahner/similarity-factorization
source setup_env.sh

poetry run python experiments/embedding_generation/run.py \
    --dataset peterson-animals \
    --n_jobs 4 \
    --random_state 0 \
    --n_cv_repeats 5 \
    --n_stable_runs 50
```

for nsd with specific subject:
```bash
poetry run python experiments/embedding_generation/run.py \
    --dataset nsd \
    --subject_id 1 \
    --n_jobs 32 \
    --random_state 0 \
    --n_cv_repeats 5 \
    --n_stable_runs 50
```

## outputs

results are saved to `experiments/embedding_generation/outputs/{dataset}/`:

```
{dataset}/
├── cv_results.joblib           # cross-validation results
├── clustering_results.joblib   # clustering metrics for k selection
├── stacked_embeddings.npy      # stacked stable embeddings (n, rank * n_runs)
├── consensus_embedding.npy     # final clustered embedding (n, k)
├── summary.json                # metadata {optimal_rank, optimal_clusters, p_min, p_max, etc}
└── analysis_summary.png        # cv and clustering plots
```

for nsd: `outputs/nsd/subj{01-08}/`

## logs

slurm logs are saved to `logs/` with timestamps in filenames:
```
logs/
├── dataset1_YYYYMMDD_HHMMSS_jobid.out
├── dataset1_YYYYMMDD_HHMMSS_jobid.err
├── dataset2_YYYYMMDD_HHMMSS_jobid.out
└── dataset2_YYYYMMDD_HHMMSS_jobid.err
```

## monitoring

check job status:
```bash
squeue -u $USER
```

view logs:
```bash
tail -f logs/*.out
tail -f logs/{dataset}_*.err
```

cancel all jobs:
```bash
scancel -u $USER
```

## architecture

- **main script**: `./run.py` (in `experiments/embedding_generation/`)
- **debug script**: `/u/fmahner/similarity-factorization/scripts/debug_embeddings.py`
- **submission scripts**: `./submit_debug.sh`, `./submit_all.sh`

the pipeline uses:
- `pysrf.cross_val_score()` for rank selection
- `pysrf.consensus.EnsembleEmbedding` for stable ensemble generation
- `pysrf.consensus.ClusterEmbedding` for dimensionality reduction via clustering
- sklearn `Pipeline` to chain operations

## changes from previous version

- updated to use new pysrf consensus API (pipeline-based)
- run from repo root to ensure correct imports
- simplified log structure (flat directory with timestamps in filenames)
- simplified codebase using sklearn pipelines
- explicit pmin/pmax calculation and storage
