#!/usr/bin/env python3
"""
Main script to run all experiments using the new Experiment abstraction.

This script provides a unified interface to run any registered experiment
with flexible configuration options.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.base import (
    ExperimentConfig, 
    list_experiments, 
    run_experiment, 
    run_all_experiments
)
from experiments.clustering_new import create_clustering_config
from experiments.spose_new import create_spose_config
from experiments.rsa_new import create_rsa_config


def create_experiment_configs() -> Dict[str, ExperimentConfig]:
    """Create default configurations for all experiments."""
    configs = {}
    
    # Clustering experiment
    configs["clustering"] = create_clustering_config(
        name="clustering",
        output_dir="results/benchmarks",
        n_jobs=-1,
        overwrite=False,
        rank=10,
        max_outer=100,
        max_inner=100,
        verbose=False,
        seeds=[0, 42, 123, 456, 789],
        datasets=["iris", "wine", "breast_cancer", "digits", "mnist", "orl"],
        max_mnist_samples=1000
    )
    
    # SPoSE experiment
    configs["spose"] = create_spose_config(
        name="spose",
        output_dir="results/spose",
        n_jobs=-1,
        overwrite=False,
        rank=49,
        max_outer=100,
        max_inner=100,
        verbose=False,
        seeds=[0, 42, 123, 456, 789],
        snr_values=[1.0],
        data_percentages=[1.0],
        experiment_type="reconstruction",
        similarity_measure="cosine",
        embedding_path="data/misc/spose_embedding_49d.txt",
        words_path="data/misc/labels_spose_66d_short.txt",
        things_path="/path/to/things/dataset",  # Update this path
        ground_truth_path="data/misc/ground_truth.mat",
        ground_truth_rsm_key="rsm",
        triplets_path="data/spose_triplets/triplets_large_final_correctednc_correctedorder.csv",
        validation_triplets_path="data/spose_triplets/validationset.txt",
        n_items=1854
    )
    
    # RSA experiment
    configs["rsa"] = create_rsa_config(
        name="rsa",
        output_dir="results/benchmarks",
        n_jobs=-1,
        overwrite=False,
        max_objects=50,
        selected_dims=[3, 5, 8, 12, 14],
        snr_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],  # Reduced for faster testing
        n_repeats=10,  # Reduced for faster testing
        n_permutations=100,  # Reduced for faster testing
        alpha=0.05,
        similarity_metric="linear",
        max_outer=30,
        max_inner=10,
        tolerance=0.0,
        verbose=False,
        n_hypotheses=3,
        hypothesis_type="random"
    )
    
    return configs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run experiments using the new Experiment abstraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available experiments
  python run_experiments.py --list
  
  # Run a specific experiment
  python run_experiments.py --experiment clustering
  
  # Run multiple experiments
  python run_experiments.py --experiments clustering rsa
  
  # Run all experiments
  python run_experiments.py --all
  
  # Run with custom parameters
  python run_experiments.py --experiment clustering --param rank=20 --param max_outer=200
  
  # Run with overwrite
  python run_experiments.py --experiment clustering --overwrite
        """
    )
    
    # Experiment selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List all available experiments")
    group.add_argument("--experiment", type=str, help="Run a specific experiment")
    group.add_argument("--experiments", nargs="+", help="Run multiple experiments")
    group.add_argument("--all", action="store_true", help="Run all experiments")
    
    # General options
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs")
    parser.add_argument("--output-dir", type=str, default="results", help="Base output directory")
    
    # Parameter overrides
    parser.add_argument("--param", nargs="+", help="Override experiment parameters (key=value)")
    
    return parser.parse_args()


def parse_params(param_args):
    """Parse parameter overrides from command line."""
    params = {}
    if param_args:
        for param in param_args:
            if "=" in param:
                key, value = param.split("=", 1)
                # Try to convert to appropriate type
                try:
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string
                params[key] = value
    return params


def main():
    """Main function."""
    args = parse_args()
    
    # Import experiments to register them
    import experiments.clustering_new
    import experiments.spose_new
    import experiments.rsa_new
    
    if args.list:
        # List all available experiments
        experiments = list_experiments()
        print("Available experiments:")
        print("-" * 50)
        for name, info in experiments.items():
            print(f"{name:20} - {info['description']}")
        return
    
    # Get default configurations
    configs = create_experiment_configs()
    
    # Parse parameter overrides
    param_overrides = parse_params(args.param)
    
    # Apply parameter overrides to all configs
    for config in configs.values():
        for key, value in param_overrides.items():
            config.set_param(key, value)
    
    # Apply general options
    for config in configs.values():
        if args.overwrite:
            config.overwrite = True
        if args.n_jobs != -1:
            config.n_jobs = args.n_jobs
        if args.output_dir != "results":
            config.output_dir = args.output_dir
    
    # Determine which experiments to run
    if args.experiment:
        experiments_to_run = [args.experiment]
    elif args.experiments:
        experiments_to_run = args.experiments
    elif args.all:
        experiments_to_run = list(configs.keys())
    else:
        experiments_to_run = []
    
    # Validate experiment names
    available_experiments = list(configs.keys())
    invalid_experiments = [exp for exp in experiments_to_run if exp not in available_experiments]
    if invalid_experiments:
        print(f"Error: Unknown experiments: {invalid_experiments}")
        print(f"Available experiments: {available_experiments}")
        return 1
    
    # Run experiments
    print(f"Running experiments: {experiments_to_run}")
    print("=" * 50)
    
    results = {}
    for exp_name in experiments_to_run:
        if exp_name in configs:
            print(f"\nRunning experiment: {exp_name}")
            print("-" * 30)
            try:
                results[exp_name] = run_experiment(exp_name, configs[exp_name])
                print(f"✓ Completed {exp_name}")
                print(f"  Results shape: {results[exp_name].shape}")
                print(f"  Output: {configs[exp_name].output_dir}/{exp_name}/results.csv")
            except Exception as e:
                print(f"✗ Failed to run {exp_name}: {e}")
                results[exp_name] = None
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    successful = [name for name, result in results.items() if result is not None]
    failed = [name for name, result in results.items() if result is None]
    
    if successful:
        print(f"✓ Successful: {successful}")
    if failed:
        print(f"✗ Failed: {failed}")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main()) 