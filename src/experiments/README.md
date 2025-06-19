# Experiment Abstraction

This module provides a unified abstraction for running experiments in the similarity factorization project. It replaces the previous ad-hoc experiment scripts with a more structured and maintainable approach.

## Overview

The experiment abstraction consists of:

1. **Base Classes**: `Experiment` and `ExperimentConfig` provide the foundation
2. **Registry System**: Automatic registration and discovery of experiments
3. **Unified Interface**: Single script to run any experiment
4. **Flexible Configuration**: Easy parameter customization and overrides

## Key Components

### Experiment Base Class

The `Experiment` abstract base class defines the interface that all experiments must implement:

```python
class Experiment(ABC):
    @abstractmethod
    def setup(self) -> None:
        """Setup experiment (load data, create tasks, etc.)"""
        pass
    
    @abstractmethod
    def run_single_trial(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experimental trial."""
        pass
    
    @abstractmethod
    def get_trials(self) -> List[Dict[str, Any]]:
        """Get list of all experimental trials to run."""
        pass
```

### Experiment Configuration

The `ExperimentConfig` class provides a flexible way to configure experiments:

```python
@dataclass
class ExperimentConfig:
    name: str
    output_dir: str = "results"
    n_jobs: int = -1
    overwrite: bool = False
    params: Dict[str, Any] = field(default_factory=dict)
```

### Registry System

Experiments are automatically registered using the `@register_experiment` decorator:

```python
@register_experiment(name="clustering", description="Clustering benchmark")
class ClusteringExperiment(Experiment):
    # Implementation...
```

## Available Experiments

### 1. Clustering Experiment (`clustering_new.py`)

Compares different clustering methods (SyNMF, KMeans, NMF) across various datasets.

**Key Parameters:**
- `rank`: Factorization rank
- `seeds`: Random seeds for reproducibility
- `datasets`: List of datasets to test
- `max_outer/max_inner`: ADMM iteration limits

**Usage:**
```python
from experiments.clustering_new import create_clustering_config

config = create_clustering_config(
    rank=10,
    seeds=[0, 42, 123],
    datasets=["iris", "wine", "breast_cancer"]
)
```

### 2. SPoSE Experiment (`spose_new.py`)

Reconstruction and comparison experiments using SPoSE embeddings.

**Key Parameters:**
- `experiment_type`: "reconstruction", "low_data", or "both"
- `snr_values`: Signal-to-noise ratios to test
- `data_percentages`: Data sampling percentages
- `embedding_path`: Path to SPoSE embeddings

**Usage:**
```python
from experiments.spose_new import create_spose_config

config = create_spose_config(
    experiment_type="reconstruction",
    snr_values=[0.0, 0.5, 1.0],
    embedding_path="data/misc/spose_embedding_49d.txt"
)
```

### 3. RSA Experiment (`rsa_new.py`)

Hypothesis testing using RSA, NMF, and latent space methods.

**Key Parameters:**
- `snr_values`: Signal-to-noise ratios
- `n_repeats`: Number of experimental repeats
- `n_permutations`: Number of permutations for statistical tests
- `max_objects`: Number of objects to test

**Usage:**
```python
from experiments.rsa_new import create_rsa_config

config = create_rsa_config(
    snr_values=np.linspace(0, 0.5, 11),
    n_repeats=500,
    n_permutations=1000
)
```

## Running Experiments

### Command Line Interface

Use the main script to run experiments:

```bash
# List available experiments
python scripts/run_experiments.py --list

# Run a specific experiment
python scripts/run_experiments.py --experiment clustering

# Run multiple experiments
python scripts/run_experiments.py --experiments clustering rsa

# Run all experiments
python scripts/run_experiments.py --all

# Run with custom parameters
python scripts/run_experiments.py --experiment clustering --param rank=20 --param max_outer=200

# Run with overwrite
python scripts/run_experiments.py --experiment clustering --overwrite
```

### Programmatic Interface

```python
from experiments.base import run_experiment
from experiments.clustering_new import create_clustering_config

# Create configuration
config = create_clustering_config(
    name="my_clustering",
    output_dir="results/my_experiment",
    rank=15,
    seeds=[0, 42]
)

# Run experiment
results = run_experiment("clustering", config)
print(f"Results shape: {results.shape}")
```

## Creating New Experiments

To create a new experiment:

1. **Create the experiment class:**
```python
from experiments.base import Experiment, register_experiment

@register_experiment(name="my_experiment", description="My custom experiment")
class MyExperiment(Experiment):
    def setup(self):
        # Load data, create models, etc.
        pass
    
    def get_trials(self):
        # Return list of trial configurations
        return [{"param1": value1, "param2": value2}, ...]
    
    def run_single_trial(self, trial):
        # Run a single trial and return results
        return {"metric1": value1, "metric2": value2}
```

2. **Create a configuration factory:**
```python
def create_my_experiment_config(**params) -> ExperimentConfig:
    config = ExperimentConfig(
        name="my_experiment",
        output_dir="results/my_experiment"
    )
    
    default_params = {
        "param1": "default_value1",
        "param2": "default_value2"
    }
    
    for key, value in params.items():
        default_params[key] = value
    
    config.params = default_params
    return config
```

3. **Add to the main script:**
```python
# In scripts/run_experiments.py
from experiments.my_experiment import create_my_experiment_config

def create_experiment_configs():
    configs = {}
    # ... other configs ...
    configs["my_experiment"] = create_my_experiment_config()
    return configs
```

## Output Structure

Each experiment creates a structured output:

```
results/
├── experiment_name/
│   ├── results.csv          # Main results
│   └── config.json          # Configuration used
```

The results are automatically saved as CSV files and can be easily loaded for analysis:

```python
import pandas as pd
results = pd.read_csv("results/clustering/results.csv")
```

## Benefits

1. **Consistency**: All experiments follow the same interface
2. **Reproducibility**: Configurations are automatically saved
3. **Parallelization**: Built-in support for parallel execution
4. **Flexibility**: Easy parameter customization and overrides
5. **Maintainability**: Clear separation of concerns
6. **Extensibility**: Easy to add new experiments

## Migration from Old Experiments

The old experiment files (`clustering.py`, `spose.py`, `rsa.py`) are still available but are being replaced by the new abstraction. To migrate:

1. Identify the core functionality in the old experiment
2. Implement the three abstract methods in a new class
3. Create a configuration factory function
4. Register the experiment using the decorator
5. Update any scripts that use the old experiment

## Examples

See `scripts/example_usage.py` for comprehensive examples of how to use the experiment abstraction. 