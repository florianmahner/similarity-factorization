# configs/

Hydra configs. `base.yaml` is the global default inherited by every task. It sets `seed: 42`, `n_jobs: -1`, `project_root`, and the output layout: each run writes to `outputs/experiments/<experiment_name>/<task>/` (with a per-job `.status/log`). Task configs set `experiment_name` and `task`; datasets are pulled in with a `dataset=` override.

| Dir | Contents |
|-----|----------|
| `base.yaml` | Global defaults: seed, `n_jobs`, `project_root`, Hydra run/output dir pattern, job logging. |
| `paths/local.yaml` | Data roots: `data_dir`, `output_dir`, `ssd`, `labshare`. Machine-specific. |
| `dataset/*.yaml` | Per-dataset definitions (`name`, `type`, `path`, `rank_range`, ...). |
| `experiment/*/` | Task configs grouped by domain (see below). |
| `hydra/` | Job-logging and launcher (`local`, `slurm`) configs. |

Available datasets (`dataset=<name>`): `cichy118`, `clip_vit_l14`, `clip_vit_l14_sigma0.5`, `dinov3`, `mur92`, `nsd`, `nsd_sigma0.5`, `peterson`, `peterson_animals`, `peterson_various`, `swow`, `things_behavior`, `things_monkey_22k`, `things_monkey_22k_f`, `things_monkey_22k_n`, `things_monkey_22k_tight`, `vgg16`, `vit`.

Experiment domains (`experiment/`): `bounds`, `coherence`, `rsa_comparison`, `simulation`, `swow`, `things_behavior`.

## Run on your machine

Edit `configs/paths/local.yaml` — the `ssd`/`labshare` roots and `project_root` in `base.yaml` are absolute paths tied to one server; change them before running elsewhere.

Override example:

```bash
poetry run python -m experiments.datasets.consensus.run dataset=mur92 seed=0
```
