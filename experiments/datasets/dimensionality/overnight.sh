#!/bin/bash
# Chained overnight run of the dimensionality experiment.
#
# Stage 1: small + medium (mur92, peterson_*, things_behavior, vgg16) -- mode=all
# Stage 2: big tier (nsd_subj01, nsd_subj02, things_macaque22k) -- mode=all
# Stage 3: overview PDF across whatever finished
#
# Each stage's .status/log is copied to .status/log.<stage> after it ends so
# we keep per-stage logs even though the runner reuses .status/.
#
# Launch (detached, survives terminal close):
#   cd <project_root>
#   nohup bash experiments/datasets/dimensionality/overnight.sh \
#     > experiments/datasets/dimensionality/outputs/overnight.log 2>&1 &
#
# Inspect:
#   tail -f experiments/datasets/dimensionality/outputs/overnight.log
#   ./scripts/jobs                                  # live job tracking
#   dash                                            # curses dashboard
set -u

cd "$(dirname "$0")/../../.." || exit 1
STATUS=experiments/datasets/dimensionality/outputs/.status
mkdir -p "$STATUS"

ts() { date +"%Y-%m-%d %H:%M:%S"; }

archive_log() {
    local tag="$1"
    if [ -f "$STATUS/log" ]; then
        cp "$STATUS/log" "$STATUS/log.${tag}"
    fi
}

run_stage() {
    local tag="$1"; shift
    local label="$1"; shift
    echo
    echo "[$(ts)] ================================================================"
    echo "[$(ts)] STAGE ${tag}: ${label}"
    echo "[$(ts)] cmd: ./scripts/submit experiments/datasets/dimensionality/run.py $*"
    echo "[$(ts)] ================================================================"
    ./scripts/submit experiments/datasets/dimensionality/run.py "$@"
    local rc=$?
    archive_log "${tag}"
    echo "[$(ts)] STAGE ${tag} exit=${rc}"

    # Hydra swallows in-process exceptions and still returns 0. Verify that
    # every dataset named in only=[...] actually produced a cross_validation.json.
    local only=""
    for a in "$@"; do
        case "$a" in only=\[*) only="${a#only=[}"; only="${only%]}";; esac
    done
    if [ -n "$only" ]; then
        local missing=""
        local IFS=','
        for d in $only; do
            d="${d// /}"
            if [ ! -s "experiments/datasets/dimensionality/outputs/${d}/cross_validation.json" ]; then
                missing="${missing} ${d}"
            fi
        done
        if [ -n "$missing" ]; then
            echo "[$(ts)] STAGE ${tag} WARNING: no cross_validation.json for:${missing}"
            echo "[$(ts)] STAGE ${tag} grep'ing log for tracebacks:"
            grep -E "Error executing job|Traceback|ConfigKeyError|raise" "${STATUS}/log.${tag}" | head -20 || true
        fi
    fi

    if [ $rc -ne 0 ]; then
        echo "[$(ts)] STAGE ${tag} failed; continuing anyway (downstream stages are resume-safe)."
    fi
    return 0
}

echo "[$(ts)] overnight.sh start"
echo "[$(ts)] cwd=$(pwd)"

run_stage 1_fast "small + medium (mur92, peterson_animals, peterson_various, things_behavior, vgg16)" \
    mode=all cache_similarity=true \
    'only=[mur92,peterson_animals,peterson_various,things_behavior,vgg16]'

run_stage 2_big "large datasets (swow, nsd_subj01, nsd_subj02, things_macaque22k)" \
    mode=all cache_similarity=true \
    'only=[swow,nsd_subj01,nsd_subj02,things_macaque22k]'

run_stage 3_overview "overview PDF across all finished datasets" \
    mode=all cache_similarity=true make_overview=true

echo
echo "[$(ts)] overnight.sh done"
