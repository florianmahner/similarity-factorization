#!/usr/bin/env python3
"""
Example usage of the new Experiment abstraction.

This script demonstrates how to:
1. Create experiment configurations
2. Run individual experiments
3. Run multiple experiments
4. Access experiment results
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.base import ExperimentConfig, list_experiments, run_experiment
from experiments.clustering_new import create_clustering_config
from experiments.spose_new import create_spose_config
from experiments.rsa_new import create_rsa_config


def example_1_basic_usage():
    """Example 1: Basic usage with default configuration."""
    print("Example 1: Basic usage with default configuration")
    print("=" * 50)
    
    # Create a simple clustering experiment config
    config = create_clustering_config(
        name="clustering_demo",
        output_dir="results/demo",
        overwrite=True,
        # Use smaller parameters for faster execution
        seeds=[0, 42],
        datasets=["iris", "wine"],
        max_outer=10,
        max_inner=10
    )
    
    print(f"Running clustering experiment with {len(config.params['seeds'])} seeds and {len(config.params['datasets'])} datasets")
    
    # Run the experiment
    results = run_experiment("clustering", config)
    
    print(f"Experiment completed!")
    print(f"Results shape: {results.shape}")
    print(f"Columns: {list(results.columns)}")
    print(f"First few rows:")
    print(results.head())
    print()


def example_2_custom_parameters():
    """Example 2: Using custom parameters."""
    print("Example 2: Using custom parameters")
    print("=" * 50)
    
    # Create a custom RSA experiment config
    config = create_rsa_config(
        name="rsa_demo",
        output_dir="results/demo",
        overwrite=True,
        # Custom parameters for faster execution
        max_objects=20,
        n_repeats=5,
        n_permutations=50,
        snr_values=[0.0, 0.5]
    )
    
    print(f"Running RSA experiment with {config.params['n_repeats']} repeats and {len(config.params['snr_values'])} SNR values")
    
    # Run the experiment
    results = run_experiment("rsa", config)
    
    print(f"Experiment completed!")
    print(f"Results shape: {results.shape}")
    print(f"Unique methods: {results['Method'].unique()}")
    print(f"SNR values tested: {sorted(results['SNR'].unique())}")
    print()


def example_3_experiment_registry():
    """Example 3: Using the experiment registry."""
    print("Example 3: Using the experiment registry")
    print("=" * 50)
    
    # List all available experiments
    experiments = list_experiments()
    print("Available experiments:")
    for name, info in experiments.items():
        print(f"  {name}: {info['description']}")
    print()


def example_4_manual_config():
    """Example 4: Creating configuration manually."""
    print("Example 4: Creating configuration manually")
    print("=" * 50)
    
    # Create a configuration manually
    config = ExperimentConfig(
        name="manual_demo",
        output_dir="results/demo",
        n_jobs=2,
        overwrite=True
    )
    
    # Set custom parameters
    config.set_param("rank", 5)
    config.set_param("max_outer", 20)
    config.set_param("max_inner", 10)
    config.set_param("seeds", [0, 42])
    config.set_param("datasets", ["iris"])
    
    print(f"Manual config created:")
    print(f"  Name: {config.name}")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Parameters: {config.params}")
    print()


def example_5_batch_processing():
    """Example 5: Processing multiple experiments."""
    print("Example 5: Processing multiple experiments")
    print("=" * 50)
    
    # Create configurations for multiple experiments
    configs = {
        "clustering": create_clustering_config(
            name="clustering_batch",
            output_dir="results/batch",
            overwrite=True,
            seeds=[0],
            datasets=["iris"],
            max_outer=5,
            max_inner=5
        ),
        "rsa": create_rsa_config(
            name="rsa_batch", 
            output_dir="results/batch",
            overwrite=True,
            max_objects=10,
            n_repeats=2,
            n_permutations=20,
            snr_values=[0.0, 0.5]
        )
    }
    
    print(f"Running {len(configs)} experiments in batch...")
    
    # Run experiments one by one (you could also use run_all_experiments)
    results = {}
    for name, config in configs.items():
        print(f"Running {name}...")
        try:
            results[name] = run_experiment(name, config)
            print(f"  ✓ Completed {name}: {results[name].shape[0]} results")
        except Exception as e:
            print(f"  ✗ Failed {name}: {e}")
            results[name] = None
    
    print(f"\nBatch processing completed!")
    print(f"Successful: {[name for name, result in results.items() if result is not None]}")
    print()


if __name__ == "__main__":
    print("Experiment Abstraction Examples")
    print("=" * 60)
    print()
    
    # Run examples
    example_3_experiment_registry()
    example_4_manual_config()
    
    # Uncomment the following lines to run actual experiments
    # (These will take some time to execute)
    
    # example_1_basic_usage()
    # example_2_custom_parameters()
    # example_5_batch_processing()
    
    print("Examples completed!")
    print("\nTo run actual experiments, uncomment the example functions in the script.")
    print("You can also use the main run_experiments.py script:")
    print("  python scripts/run_experiments.py --list")
    print("  python scripts/run_experiments.py --experiment clustering") 