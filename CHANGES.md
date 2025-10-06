# Changes Summary - Embedding Generation Pipeline Update

## Completed Tasks

### 1. Updated pysrf Subtree ✓
- Removed old pysrf subtree and added fresh version from `https://github.com/florianmahner/pysrf.git master`
- New pysrf uses pipeline-based consensus API with `EnsembleEmbedding` and `ClusterEmbedding`
- Old API with `fit_stable()` and `cluster_stable()` replaced by sklearn Pipeline pattern

### 2. Refactored Embedding Generation Workflow ✓

#### Updated Main Script: `experiments/embedding_generation/run.py`
- Replaced old stability API with new pipeline-based approach:
  ```python
  pipeline = Pipeline([
      ('ensemble', EnsembleEmbedding(SRF(rank=optimal_rank), n_runs=50)),
      ('cluster', ClusterEmbedding(min_clusters=min_k, max_clusters=max_k))
  ])
  ```
- Explicit `pmin`/`pmax` calculation using `estimate_sampling_bounds_fast()`
- Saves `p_min`, `p_max`, and `observed_fraction` in `summary.json`
- Timestamped log directories for better organization

#### Key Changes in Pipeline:
- **Before**: `cross_val_score()` with `n_stable_runs` and `cluster_stable` params
- **After**: Separate `cross_val_score()` for rank selection, then explicit Pipeline for consensus
- **Stacked embeddings**: Accessed via `pipeline.named_steps['ensemble'].embeddings_`
- **Cluster results**: Accessed via `pipeline.named_steps['cluster'].cluster_results_`
- **Best k**: Accessed via `pipeline.named_steps['cluster'].best_k_`

#### Output Files (unchanged structure):
- `cv_results.joblib` - cross-validation results
- `clustering_results.joblib` - clustering metrics
- `stacked_embeddings.npy` - stacked stable embeddings (n, rank × n_runs)
- `consensus_embedding.npy` - final clustered embedding (n, k)
- `summary.json` - metadata with `p_min`, `p_max`, `observed_fraction` added
- `analysis_summary.png` - cv and clustering plots

### 3. Updated Submission Scripts ✓

#### `submit_debug.sh`
- Creates timestamped log directories: `logs/YYYYMMDD_HHMMSS/`
- Calls `run.py` with reduced parameters (2 cv repeats, 5 stable runs)
- Working directory: `experiments/embedding_generation/`

#### `submit_all.sh`
- Submits all datasets with timestamped logs
- Each dataset gets its own log subdirectory
- Production parameters (5 cv repeats, 50 stable runs)

### 4. Simplified Log Structure ✓
Flat directory with timestamps in filenames:
```
logs/
├── dataset1_YYYYMMDD_HHMMSS_jobid.out
├── dataset1_YYYYMMDD_HHMMSS_jobid.err
├── dataset2_YYYYMMDD_HHMMSS_jobid.out
└── dataset2_YYYYMMDD_HHMMSS_jobid.err
```
No nested subdirectories - much simpler to navigate!

### 5. Updated Documentation ✓
- `experiments/embedding_generation/README.md` fully updated
- Reflects new pipeline API usage
- Documents timestamped log structure
- Includes manual submission examples

### 6. Debug Script ✓
- Created `scripts/debug_embeddings.py` for quick testing
- Uses minimal parameters (2 cv repeats, 3 stable runs)
- Outputs to `experiments/embedding_generation/outputs/debug/`

## Technical Details

### New pysrf API Attributes
- `EnsembleEmbedding.embeddings_` - stacked embeddings from all runs
- `ClusterEmbedding.cluster_results_` - DataFrame with silhouette scores
- `ClusterEmbedding.best_k_` - optimal number of clusters
- `ADMMGridSearchCV.cv_results_` - cross-validation results DataFrame
- `ADMMGridSearchCV.best_params_` - best parameters dict
- `ADMMGridSearchCV.best_score_` - best validation score

### Sampling Bounds
New explicit calculation:
```python
pmin, pmax, _ = estimate_sampling_bounds_fast(
    similarity, random_state=random_state, n_jobs=n_jobs, verbose=True
)
sampling_fraction = 0.5 * (pmin + pmax)
```

These values are now saved in `summary.json` as:
- `"p_min"`: lower bound
- `"p_max"`: upper bound  
- `"observed_fraction"`: sampling fraction used

## Migration Notes

### Old Code Pattern
```python
cv_result = cross_val_score(
    similarity,
    n_stable_runs=50,
    cluster_stable=True,
    ...
)
cv_result.stable_embeddings_
cv_result.clustered_embedding_
cv_result.best_k_
cv_result.cluster_results_
```

### New Code Pattern
```python
cv_result = cross_val_score(
    similarity,
    fit_final_estimator=False,
    ...
)

pipeline = Pipeline([
    ('ensemble', EnsembleEmbedding(SRF(rank=optimal_rank), n_runs=50)),
    ('cluster', ClusterEmbedding(min_clusters=min_k, max_clusters=max_k))
])
pipeline.fit(similarity)
final_embedding = pipeline.transform(similarity)

stacked_embeddings = pipeline.named_steps['ensemble'].embeddings_
cluster_results = pipeline.named_steps['cluster'].cluster_results_
best_k = pipeline.named_steps['cluster'].best_k_
```

## Environment Setup

Python 3.12 is now required. Load environment:
```bash
source /u/fmahner/similarity-factorization/setup_env.sh
poetry env use python3
poetry install
```

## Testing

Run debug script to verify pipeline:
```bash
cd /u/fmahner/similarity-factorization
source setup_env.sh
poetry run python scripts/debug_embeddings.py
```

Or submit debug job:
```bash
cd experiments/embedding_generation
./submit_debug.sh peterson-animals
```

## Files Modified
- ✓ `experiments/embedding_generation/run.py` - complete rewrite
- ✓ `experiments/embedding_generation/submit_debug.sh` - updated paths and logging
- ✓ `experiments/embedding_generation/submit_all.sh` - updated paths and logging
- ✓ `experiments/embedding_generation/README.md` - updated documentation

## Files Created
- ✓ `scripts/debug_embeddings.py` - debug testing script
- ✓ `CHANGES.md` - this file

## Files Removed
- N/A (old `run.py` was overwritten)

## Backward Compatibility

Output file structure is preserved:
- Same file names
- Same `.npy` and `.joblib` formats
- `summary.json` has additional fields but maintains existing ones
- Existing analysis notebooks should work with minimal changes

