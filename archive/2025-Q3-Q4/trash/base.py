from typing import Dict, Any, List, Type, Callable, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from pathlib import Path
import json
import pandas as pd
from joblib import Parallel, delayed

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    name: str
    output_dir: str = "results"
    n_jobs: int = -1
    overwrite: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Global experiment registry
_EXPERIMENT_REGISTRY: Dict[str, Type["Experiment"]] = {}


def register_experiment(name: Optional[str] = None, description: Optional[str] = None):
    """
    Decorator to register an experiment class.

    Args:
        name: Name for the experiment (defaults to class name in lowercase)
        description: Description of what the experiment does

    Usage:
        @register_experiment(name="clustering", description="Clustering benchmark")
        class ClusteringExperiment(Experiment):
            ...
    """

    def decorator(cls: Type["Experiment"]) -> Type["Experiment"]:
        experiment_name = name or cls.__name__.lower().replace("experiment", "")

        # Add metadata to the class
        cls._experiment_name = experiment_name
        cls._experiment_description = description or cls.__doc__ or "No description"

        # Register the experiment
        _EXPERIMENT_REGISTRY[experiment_name] = cls

        logger.info(f"Registered experiment: {experiment_name}")
        return cls

    return decorator


def get_experiment(name: str) -> Type["Experiment"]:
    """Get an experiment class by name."""
    if name not in _EXPERIMENT_REGISTRY:
        available = ", ".join(_EXPERIMENT_REGISTRY.keys())
        raise ValueError(f"Unknown experiment '{name}'. Available: {available}")
    return _EXPERIMENT_REGISTRY[name]


def list_experiments() -> Dict[str, Dict[str, Any]]:
    """List all registered experiments with their metadata."""
    return {
        name: {
            "class": cls.__name__,
            "description": cls._experiment_description,
        }
        for name, cls in _EXPERIMENT_REGISTRY.items()
    }


class Experiment(ABC):
    """Base class for different types of experiments."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) / config.name

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

    def run(self) -> pd.DataFrame:
        """Run the full experiment."""
        # Setup
        self.setup()

        # Check if results exist
        results_file = self.output_dir / "results.csv"
        if results_file.exists() and not self.config.overwrite:
            logger.info(f"Loading existing results from {results_file}")
            return pd.read_csv(results_file)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get all trials
        trials = self.get_trials()
        logger.info(f"Running {len(trials)} trials...")

        # Run in parallel
        results = Parallel(n_jobs=self.config.n_jobs, verbose=1)(
            delayed(self.run_single_trial)(trial) for trial in trials
        )

        # Flatten results if needed (some experiments return lists)
        flattened_results = []
        for result in results:
            if isinstance(result, list):
                flattened_results.extend(result)
            elif result is not None:
                flattened_results.append(result)

        # Convert to DataFrame and save
        df = pd.DataFrame(flattened_results)
        df.to_csv(results_file, index=False)

        # Save config and metadata
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        return df
