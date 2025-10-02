# similarity representation analysis

this pipeline performs similarity-based representation factorization on neural and behavioral datasets using the pysrf package.

## overview

the pipeline:
1. loads similarity matrices from datasets
2. estimates optimal sampling bounds using random matrix theory
3. performs cross-validation to select optimal rank
4. fits multiple (50) stable runs with different initializations to create consensus embeddings
5. clusters the consensus embeddings to obtain final k-dimensional representation

## datasets

configured datasets in `/ptmp/fmahner/`:
- `mur92` - 92 object similarity dataset
- `cichy118` - 118 object similarity dataset  
- `peterson-animals` - peterson animals dataset
- `peterson-various` - peterson various objects dataset
- `nsd` - natural scenes dataset (8 subjects, fmri data)
- `things-monkey-22k` - things monkey neural recordings

## rank grids

small datasets (mur92, cichy118, peterson-*): ranks 1-30 (step 1)
large datasets (nsd, monkey): ranks 5-150 (step 5)

## usage

### debug mode (interactive partition, fast)

```bash
cd experiments/similarity_analysis
./submit_debug.sh [dataset_name]
```

parameters:
- 4 cpus
- 8gb memory
- 30min time limit
- 2 cv repeats
- 5 stable runs

### production mode (full analysis)

single dataset:
```bash
cd experiments/similarity_analysis
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
poetry run python run.py \
    --dataset peterson-animals \
    --n_jobs 4 \
    --random_state 0 \
    --n_cv_repeats 5 \
    --n_stable_runs 50
```

for nsd with specific subject:
```bash
poetry run python run.py \
    --dataset nsd \
    --subject_id 1 \
    --n_jobs 32 \
    --random_state 0 \
    --n_cv_repeats 5 \
    --n_stable_runs 50
```

## outputs

results are saved to `experiments/similarity_analysis/outputs/{dataset}/`:

```
{dataset}/
├── similarity.npy              # input similarity matrix (n, n)
├── sampling_bounds.json        # rmt bounds {p_min, p_max, observed_fraction}
├── cv_results.csv              # cross-validation scores for all ranks
├── grid.joblib                 # full cv object (for inspection)
├── consensus_embeddings.npy    # stacked stable embeddings (n, rank * 50)
├── clustering_results.csv      # clustering metrics for k selection
├── final_embedding.npy         # final clustered embedding (n, k)
└── summary.json                # metadata {optimal_rank, optimal_clusters, etc}
```

for nsd: `outputs/nsd/subj{01-08}/`

## monitoring

check job status:
```bash
squeue -u $USER
```

view logs:
```bash
tail -f logs/{dataset}_*.out
tail -f logs/{dataset}_*.err
```

cancel all jobs:
```bash
scancel -u $USER
```

## generic slurm templates

reusable templates are in `/u/fmahner/similarity-factorization/slurm/templates/`:
- `debug.template.sh` - interactive partition, quick testing
- `production.template.sh` - standard compute nodes, full analysis

these can be used by other experiments by substituting placeholders:
- `{log_dir}` - directory for slurm logs
- `{job_name}` - job identifier
- `{work_dir}` - working directory
- `{cpus}` - number of cpus
- `{mem}` - memory in mb
- `{time}` - time limit (production only)
- `{command}` - command to execute

