# Demo

A small, self-contained demonstration of Similarity-Based Representation
Factorization (SRF) on simulated data — no external downloads, and it runs in
a few seconds (≈3 s on an Apple M1 Max) with or without the optional Cython
extension.

## Files

- `demo_similarity.npz` — the demo dataset: a `100 x 100` similarity matrix
  built from a known sparse, non-negative embedding of 100 objects with 6
  latent dimensions (`similarity = embedding @ embedding.T`). Committed with
  the repository; no need to regenerate it.
- `run_demo.py` — runs the full SRF pipeline on the dataset.
- `make_demo_data.py` — regenerates `demo_similarity.npz` (NumPy only; needed
  only to change the simulation).
- `requirements-demo.txt` — the minimal dependencies to run the demo.

## Install and run

From the repository root:

```bash
python -m pip install -r demo/requirements-demo.txt
python -m pip install ./third_party/pysrf
python demo/run_demo.py
```

If the full project is installed with Poetry (`poetry install`), everything is
already present — run `poetry run python demo/run_demo.py` instead.

## Expected output

```
============================================================
SRF demo
============================================================
similarity matrix : (100, 100)
true rank         : 6

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

============================================================
DEMO PASSED
============================================================
```

The four steps demonstrate the core capabilities reported in the paper:

1. **Rank selection** — the number of latent dimensions is estimated from the
   similarity matrix alone (bootstrap eigenspace coherence) and matches the
   ground truth (6).
2. **Factorization** — SRF recovers a non-negative embedding whose product
   reconstructs the similarity matrix.
3. **Recovery** — the recovered dimensions match the ground-truth embedding
   (mean per-dimension correlation ≈ 1.0 after matching columns).
4. **Missing data** — with 40% of the entries hidden, SRF reconstructs the
   held-out similarities it never observed, without imputation.

Correlations may vary in the last digit across platforms and BLAS libraries,
but `k* = 6` and correlations near 1.0 should hold.
