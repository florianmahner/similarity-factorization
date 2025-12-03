#!/bin/bash
# Node Classification Benchmark - Full Parameter Sweep
#
# Usage:
#   ./scripts/run_node_classification_benchmark.sh [--local|--slurm|--dry-run]
#
# Runs all datasets at full size with multiple ranks.
# Methods are trained with production-quality parameters.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCHMARK_SCRIPT="$PROJECT_ROOT/experiments/ppi/tasks/node_classification_benchmark.py"

MODE="${1:---local}"

# Full sweep parameters
DATASETS="wikipedia,blogcatalog,ppi"
METHODS="srf,deepwalk,node2vec,line,spectral"
RANKS="32,64,128"
SEEDS="42"

# Training parameters (set in benchmark_config.yaml defaults)
# - SRF: max_outer=500
# - DeepWalk/Node2Vec: num_paths=20, path_length=80
# - LINE: epoch=1000, batch_size=512

# Total configs: 3 datasets x 5 methods x 3 ranks = 45 jobs

echo "=============================================="
echo "Node Classification Benchmark"
echo "=============================================="
echo "Datasets: $DATASETS"
echo "Methods:  $METHODS"
echo "Ranks:    $RANKS"
echo "Total:    45 configurations"
echo "Mode:     $MODE"
echo ""
echo "Training parameters (from config):"
echo "  SRF:           max_outer=500"
echo "  DeepWalk/N2V:  num_paths=20, path_length=80"
echo "  LINE:          epoch=1000, batch_size=512"
echo ""

cd "$PROJECT_ROOT"

case "$MODE" in
    --dry-run)
        echo "DRY RUN - Command that would be executed:"
        echo ""
        echo "poetry run python $BENCHMARK_SCRIPT -m \\"
        echo "    dataset=$DATASETS \\"
        echo "    method=$METHODS \\"
        echo "    size=null \\"
        echo "    rank=$RANKS \\"
        echo "    seed=$SEEDS"
        ;;
    --local)
        echo "Running locally (sequential, ~2-3 hours)..."
        echo ""
        poetry run python "$BENCHMARK_SCRIPT" -m \
            dataset=$DATASETS \
            method=$METHODS \
            size=null \
            rank=$RANKS \
            seed=$SEEDS
        ;;
    --slurm)
        echo "Submitting to SLURM (45 parallel jobs)..."
        echo ""
        poetry run python "$BENCHMARK_SCRIPT" -m \
            dataset=$DATASETS \
            method=$METHODS \
            size=null \
            rank=$RANKS \
            seed=$SEEDS \
            hydra/launcher=slurm
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [--local|--slurm|--dry-run]"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Benchmark complete. Run plotting script:"
echo "  poetry run python experiments/ppi/tasks/plot_node_classification.py"
echo "=============================================="
