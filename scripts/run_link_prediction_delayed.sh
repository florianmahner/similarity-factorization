#!/bin/bash
# Run link_prediction for huri and string after 6 hour delay
# Started: $(date)

cd /LOCAL/fmahner/similarity-factorization

echo "$(date) | Sleeping for 6 hours before starting link prediction..."
sleep 6h

echo "$(date) | Starting link_prediction for huri..."
./scripts/submit experiments/ppi/link_prediction.py link_prediction.dataset=huri
HURI_EXIT=$?
echo "$(date) | HuRI completed with exit code: $HURI_EXIT"

echo "$(date) | Starting link_prediction for string..."
./scripts/submit experiments/ppi/link_prediction.py link_prediction.dataset=string
STRING_EXIT=$?
echo "$(date) | STRING completed with exit code: $STRING_EXIT"

echo "$(date) | All jobs finished. HuRI=$HURI_EXIT, STRING=$STRING_EXIT"
