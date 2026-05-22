#!/bin/bash
# Sync selected figures from remote to local
# Run from LOCAL machine: ./sync_figures.sh user@remote:/path/to/similarity-factorization ./figures/

set -e

REMOTE="${1:-fmahner@cluster:/LOCAL/fmahner/similarity-factorization}"
LOCAL_DIR="${2:-./figures}"

mkdir -p "$LOCAL_DIR"

FILES=(
    # Simulation
    "outputs/experiments/simulation/plots/dirichlet_properties.pdf"
    "outputs/experiments/simulation/plots/alpha_factors_0_1.pdf"
    "outputs/experiments/simulation/plots/alpha_factors_1_0.pdf"
    "outputs/experiments/simulation/plots/alpha_factors_10_0.pdf"
    "outputs/experiments/simulation/plots/alpha_factors_100_0.pdf"
    "outputs/experiments/simulation/plots/srf_performance.pdf"

    # RSA comparison
    "outputs/experiments/rsa_comparison/factorial_power.pdf"
    "outputs/experiments/rsa_comparison/spose_power_50.pdf"
    "outputs/experiments/rsa_comparison/spose_factors.pdf"
    "outputs/experiments/rsa_comparison/spose_rsm.pdf"
    "outputs/experiments/rsa_comparison/factorial_factors.pdf"
    "outputs/experiments/rsa_comparison/factorial_rsm.pdf"

    # Things behavior
    "outputs/experiments/things_behavior/plots/things_predicted_48.pdf"
    "outputs/experiments/things_behavior/plots/66/pairwise_reconstruction.pdf"
    "outputs/experiments/things_behavior/plots/66/accuracy_comparison.pdf"
    "outputs/experiments/things_behavior/plots/66/low_data_accuracy.pdf"

    # Word association
    "outputs/experiments/word_association/plots/concreteness_prediction.pdf"
    "outputs/experiments/word_association/plots/size_prediction.pdf"
    "outputs/experiments/word_association/plots/valence_prediction.pdf"
    "outputs/experiments/word_association/plots/animacy_prediction.pdf"
)

for f in "${FILES[@]}"; do
    rsync -avz "$REMOTE/$f" "$LOCAL_DIR/"
done

echo "Synced ${#FILES[@]} files to $LOCAL_DIR"
