#!/bin/bash
# Full VICE low-data experiment runner
# Runs training, aggregation, and plotting automatically
# All output captured to log files

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/outputs/experiments/things_behavior/vice_lowdata"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$LOG_DIR"

MAIN_LOG="$LOG_DIR/full_run_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=========================================="
echo "VICE Low-Data Experiment - Full Run"
echo "Started: $(date)"
echo "Project root: $PROJECT_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Log file: $MAIN_LOG"
echo "=========================================="

cd "$PROJECT_ROOT"

# Step 1: Verify partitions exist
echo ""
echo "[$(date +%H:%M:%S)] Step 1: Verifying partitions..."
PARTITION_COUNT=$(ls -d data/things/partitions/*pct_part* 2>/dev/null | wc -l)
echo "Found $PARTITION_COUNT partition directories"

if [ "$PARTITION_COUNT" -lt 37 ]; then
    echo "ERROR: Expected 37 partitions, found $PARTITION_COUNT"
    echo "Run: poetry run python experiments/things_behavior/vice_lowdata.py --prepare"
    exit 1
fi
echo "Partitions OK"

# Step 2: Run training
echo ""
echo "[$(date +%H:%M:%S)] Step 2: Starting training (370 jobs on 4 GPUs)..."
echo "This will take several hours..."

TRAIN_LOG="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"
echo "Training log: $TRAIN_LOG"

poetry run python experiments/things_behavior/vice_lowdata_runner.py 2>&1 | tee "$TRAIN_LOG"

TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "WARNING: Training exited with code $TRAIN_EXIT"
fi

# Step 3: Aggregate results
echo ""
echo "[$(date +%H:%M:%S)] Step 3: Aggregating results..."

AGG_LOG="$LOG_DIR/aggregation_$(date +%Y%m%d_%H%M%S).log"
poetry run python experiments/things_behavior/vice_lowdata.py --aggregate 2>&1 | tee "$AGG_LOG"

# Step 4: Create plots
echo ""
echo "[$(date +%H:%M:%S)] Step 4: Creating plots..."

PLOT_LOG="$LOG_DIR/plotting_$(date +%Y%m%d_%H%M%S).log"
poetry run python experiments/things_behavior/vice_lowdata_plot.py 2>&1 | tee "$PLOT_LOG"

# Final summary
echo ""
echo "=========================================="
echo "VICE Low-Data Experiment - Complete"
echo "Finished: $(date)"
echo "=========================================="
echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "  (no CSV files)"
echo ""
echo "Figures:"
ls -la "$OUTPUT_DIR/figures/"*.pdf 2>/dev/null || echo "  (no PDF files)"
echo ""
echo "Models trained:"
ls -d "$OUTPUT_DIR/models/vice_"* 2>/dev/null | wc -l
echo ""
echo "Log files:"
ls -la "$LOG_DIR/"
