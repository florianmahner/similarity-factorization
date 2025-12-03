# Project Workflow & Standards

This document outlines the mandatory workflow, file structure, and coding standards for this project. All AI assistants (Gemini, Claude, etc.) must adhere strictly to these rules.

## 1. Core Mandate: Development vs. Production

### **Development (Self-Contained Modules)**
*   **Where:** `development/<project_name>/`
*   **Structure:** Each experiment is a complete, isolated module:
    ```
    development/my_experiment/
    ├── run.py              # Atomic runner (ONE configuration). No internal loops.
    ├── config.yaml         # Minimal config (inherits base via submit script).
    ├── plot.py             # Auto-detected by submit script. Aggregates results.
    └── outputs/            # Auto-generated. Contains timestamped runs.
    ```
*   **Execution:** ALWAYS use the `submit` script from the project root:
    ```bash
    ./scripts/submit development/my_experiment/run.py [overrides...]
    ```
*   **Lifecycle:**
    - Failed/not interesting? → `rm -rf development/my_experiment/`
    - Success? → Refactor and promote to `experiments/`

### **Production (Canonical Experiments)**
*   **Where:** `experiments/<name>/`
*   **Config:** `configs/experiment/<name>.yaml`
*   **Outputs:** Static paths at `experiments/<name>/outputs/<task>/` (reproducible).
*   **Rule:** Stable, versioned experiments. No timestamps.

---

## 2. Execution & CLI

### **Unified Submission Script (`./scripts/submit`)**
The `submit` script is the primary entry point. It handles:
1.  **Directory Context**: Automatically changes directory to the script's folder (`cd development/...`).
2.  **Path Resolution**: Sets global `project_root` so data paths (`${project_root}/data`) resolve correctly.
3.  **Auto-Plotting**: If `plot.py` exists, it runs it after the experiment finishes.
4.  **Auto-Cleanup**: Recursively deletes "empty" run folders (containing only logs) to keep `outputs/` clean.

**Usage Examples:**
```bash
# 1. Fast Debug (Single Run)
./scripts/submit development/ppi/run.py size=200 method=srf

# 2. Parameter Sweep (Multirun) - Auto-plots results!
./scripts/submit development/ppi/run.py -m size=200,500 method=srf,node2vec

# 3. Slurm Submission
./scripts/submit development/ppi/run.py -m size=2000 method=srf hydra/launcher=slurm
```

---

## 3. Configuration Structure (Hydra)

The project uses a **flattened** Hydra configuration strategy.

### **Directory Layout (`configs/`)**
| File/Folder | Purpose |
| :--- | :--- |
| `base.yaml` | **Global Base.** Defines defaults, logging, and `project_root`. |
| `config.yaml` | **Entry Point wrapper.** Redirects to `base` (allows `python run.py` fallback). |
| `paths/local.yaml` | **Path Definitions.** Uses `${project_root}` (hardcoded or injected). |
| `mode/` | **Environment Logic.** `development.yaml` (timestamped) vs `production.yaml`. |
| `launcher/` | **Execution Backends.** `local.yaml` (default) vs `slurm.yaml`. |

### **Development Config Pattern**
Local configs should be minimal and inherit from `/base`.
```yaml
# development/my_exp/config.yaml
defaults:
  - /base
  - _self_

# Task params...
size: 200
method: srf
```

---

## 4. Coding Standards

*   **Atomic Runs:** `run.py` should execute **ONE** configuration. Use Hydra (`-m`) for loops.
*   **Paths:** Use `pathlib.Path`. Never hardcode absolute paths; use `cfg.data_dir`.
*   **Plotting:** `plot.py` should be robust:
    - Accept `--folder` arg (passed by `submit`).
    - Save artifacts (plots/CSVs) **inside** the run folder (`outputs/YYYY-MM-DD/HH-MM-SS/`).
    - Clean up raw JSONs if `--cleanup` is passed.

---

## 5. Plotting Standards

*   **Publication Ready:** Clean, labeled, correctly scaled.
*   **Libraries:** `seaborn` and `matplotlib`.
*   **Location:** Save plots next to the data in the `outputs/` timestamp folder.
*   **Descriptive Filenames:** `comparison_rank30_nodes2000.png`.

---

## 6. Memory Bank (Project Specifics)

*   **Data Dir:** Defined in `configs/paths/local.yaml` using `${project_root}/data`.
*   **Cleanup:** `submit` deletes folders with only `.log` files.
*   **Dashboard:** The project includes a dashboard scanner that watches `outputs/`.
*   **Slurm:** Use `hydra/launcher=slurm` for heavy jobs.
