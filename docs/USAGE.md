# Usage Guide

This project uses a unified experiment runner and a flattened configuration structure powered by Hydra.

## Experiment Runner

The `./experiment` script is the single entry point for all production experiments.

### Basic Commands

```bash
# Run a single task
./experiment <experiment> <task> [overrides]
# Example:
./experiment bounds estimate dataset=mur92
```

### Execution Modes

| Flag | Description |
|------|-------------|
| `--bg` | Runs the command in a detached `screen` session (e.g., `gemini_bounds_estimate`). Prevents duplicate runs. |
| `--sweep` | Runs a parallel parameter sweep using Joblib. |
| `--slurm` | Submits the job(s) to the SLURM cluster using Submitit. |
| `--dry` | Prints the resolved configuration without running the code. |

### Examples

**Run in background (persistent):**
```bash
./experiment bounds estimate dataset=mur92 --bg
```

**Run a parameter sweep:**
```bash
./experiment bounds estimate dataset=mur92,cichy118 --sweep
```

**Submit to SLURM:**
```bash
./experiment bounds estimate dataset=mur92 --slurm
```

## Configuration Structure

The configuration is managed in `configs/` and flattened for simplicity:

*   `configs/config.yaml`: Global defaults (paths, logging, launcher).
*   `configs/paths.yaml`: Standardized system paths.
*   `configs/experiment/`: Experiment-specific settings (e.g., `bounds.yaml`, `ppi.yaml`).
*   `configs/dataset/`: Shared dataset definitions (e.g., `mur92.yaml`).
*   `configs/launcher/`: Execution environments (`local.yaml`, `slurm.yaml`).

### Output Paths

*   **Production:** Defaults to static paths: `experiments/<exp_name>/outputs/<task_name>/`.
*   **Development:** Use local configs in `development/<project>/` to enable timestamped outputs for iteration.
