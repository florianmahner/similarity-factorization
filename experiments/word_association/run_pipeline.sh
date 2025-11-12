#!/bin/bash

# Word Association Analysis Pipeline
# Uncomment the steps you want to run

embedding_dir="experiments/word_association/outputs/final-100d"
# embedding_dir="experiments/development/word_association/outputs/251105/112132"


# Step 1: Compute SRF embeddings from SWOW data
poetry run python experiments/word_association/run.py --output-dir ${embedding_dir} --rank 100

# Step 2: Compute semantic axes from SRF embeddings
# poetry run python experiments/word_association/compute_semantic_axes.py --embedding-dir ${embedding_dir}

# Step 3: Map GloVe embeddings to SRF space
# poetry run python experiments/word_association/map_glove_to_srf.py --embedding-dir ${embedding_dir} --glove-path data/dolma_300_2024_1.2M.100_combined.txt --test-size 0.1

# Step 4: Project arbitrary words onto semantic axes
# poetry run python experiments/word_association/project_arbitrary_words.py --transform-dir ${embedding_dir} --glove-path data/dolma_300_2024_1.2M.100_combined.txt --words "algorithm,democracy,neuroscience,cryptocurrency,blockchain"

