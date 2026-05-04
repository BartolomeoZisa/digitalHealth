#!/bin/bash

# Get absolute path to project root (Bart folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_ROOT/logs/raw_models/$TIMESTAMP"

mkdir -p "$LOG_DIR"
echo "📂 Logs being written to: $LOG_DIR"
echo "tail $LOG_DIR/*.log to monitor progress in real-time."

# ========================================================
# Master Process (The Watcher)
# ========================================================
{
    cd "$PROJECT_ROOT"

    # 1. Load and EXPORT environment variables so subshells can see them
    if [ -f .env ]; then
        set -a
        source .env
        set +a
    fi

    # 2. Source notification functions
    source notify.sh

    # 3. Setup Virtual Env
    source venv/bin/activate

    declare -a scripts=(
        "experiments/cnnspectrogram/efficientnet.py"
        "experiments/cnnspectrogram/efficientnetdelta.py"
        "experiments/cnnspectrogram/resnet.py"
    )

    pids=()
    names=()

    # Launch all models
    # Launch all models sequentially
    for script in "${scripts[@]}"; do
        name=$(basename "$script" .py)
        log_file="$LOG_DIR/${name}.log"

        echo "🚀 Starting $name..."
        python -u "$script" > "$log_file" 2>&1

        exit_code=$?

        if [ $exit_code -eq 0 ]; then
            notify_success "$name"
        else
            notify_failure "$name" "$exit_code"
            # optional: stop everything if one fails
            # exit $exit_code
        fi
    done

    # Final Summary
    print_summary

} > "$LOG_DIR/execution.log" 2>&1 & 

MASTER_PID=$!
disown $MASTER_PID

echo "✅ Master Watcher (PID: $MASTER_PID) is independent."
echo "🌐 You can safely logout. Summary will be sent to Discord."