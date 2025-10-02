#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_usage() {
    echo "usage: $0 [command]"
    echo ""
    echo "commands:"
    echo "  status    - show current job status"
    echo "  detailed  - show detailed job information"
    echo "  cancel    - cancel all user jobs"
    echo "  tail      - tail most recent log file"
    echo ""
}

show_status() {
    echo "current slurm jobs:"
    squeue -u $USER
}

show_detailed() {
    echo "detailed job information:"
    squeue -u $USER -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"
}

cancel_jobs() {
    echo "cancelling all jobs for $USER"
    scancel -u $USER
    echo "done"
}

tail_recent_log() {
    local log_dir="${1:-$SCRIPT_DIR/../experiments/similarity_analysis/logs}"
    
    if [ ! -d "$log_dir" ]; then
        echo "log directory not found: $log_dir"
        return 1
    fi
    
    local latest=$(ls -t "$log_dir"/*.out 2>/dev/null | head -1)
    
    if [ -z "$latest" ]; then
        echo "no log files found in $log_dir"
        return 1
    fi
    
    echo "tailing: $latest"
    echo ""
    tail -f "$latest"
}

case "${1:-status}" in
    status)
        show_status
        ;;
    detailed)
        show_detailed
        ;;
    cancel)
        cancel_jobs
        ;;
    tail)
        tail_recent_log "$2"
        ;;
    *)
        show_usage
        exit 1
        ;;
esac

