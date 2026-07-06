.PHONY: data

# Download preprocessed similarity matrices and consensus embeddings from OSF into data/.
data:
	poetry run python scripts/download_data.py
