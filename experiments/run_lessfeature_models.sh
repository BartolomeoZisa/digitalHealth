#!/bin/bash

# Get absolute path to project root (Bart folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_ROOT/logs/lessfeature_models/$TIMESTAMP"

mkdir -p "$LOG_DIR"
echo "📂 Logs being written to: $LOG_DIR"

# ========================================================
# Master Process (The Watcher)
# ========================================================
{
    cd "$PROJECT_ROOT"

    # 1. Load and EXPORT environment variables so subshells can see them
    if [ -f .env ]; then
        set -a            # Automatically export all variables defined from here
        source .env
        set +a
    fi

    # 2. Source notification functions
    source notify.sh

    # 3. Setup Virtual Env
    source venv/bin/activate

    declare -a scripts=(
        "experiments/svm/svm_lessfeature_nested.py"
        "experiments/logisticregression/logreg_lessfeature_nested.py"
        "experiments/randomforest/randomforest_lessfeature_nested.py"
        "experiments/xgboost/xgboost_lessfeature_nested.py"
    )

    pids=()
    names=()

    # Launch all models
    for script in "${scripts[@]}"; do
        name=$(basename "$script" .py)
        log_file="$LOG_DIR/${name}.log"

        echo "🚀 Starting $name..."
        python -u "$script" > "$log_file" 2>&1 &
        
        pids+=("$!")
        names+=("$name")
    done

    # Wait for each model and update arrays
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}"
        exit_code=$?

        if [ $exit_code -eq 0 ]; then
            notify_success "${names[$i]}"
        else
            notify_failure "${names[$i]}" "$exit_code"
        fi
    done

    # Final Summary (MUST stay inside these braces)
    print_summary

} > "$LOG_DIR/execution.log" 2>&1 & 

MASTER_PID=$!
disown $MASTER_PID

echo "✅ Master Watcher (PID: $MASTER_PID) is independent."
echo "🌐 You can safely logout. Summary will be sent to Discord."