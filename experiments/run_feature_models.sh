#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/feature_models/$TIMESTAMP"
mkdir -p "$LOG_DIR"

echo "Logs will be stored in: $LOG_DIR"

source venv/bin/activate

declare -a scripts=(
    "experiments/svm/svm_feature_nested.py"
    "experiments/logisticregression/logreg_feature_nested.py"
    "experiments/randomforest/randomforest_feature_nested.py"
    "experiments/xgboost/xgboost_feature_nested.py"
)

# ---- notification helper ----
notify() {
    title="$1"
    message="$2"

    if command -v osascript &> /dev/null; then
        osascript -e "display notification \"$message\" with title \"$title\""
    fi

    if command -v notify-send &> /dev/null; then
        notify-send "$title" "$message"
    fi
}

# ---- launch jobs ----
for script in "${scripts[@]}"; do
    name=$(basename "$script" .py)
    log_file="$LOG_DIR/${name}.log"

    echo "Starting $name..."

    (
        python "$script" > "$log_file" 2>&1
        exit_code=$?

        if [ $exit_code -eq 0 ]; then
            echo "$name SUCCESS" >> "$LOG_DIR/status.log"
            notify "✅ $name done" "Training completed successfully"
        else
            echo "$name FAILED (exit $exit_code)" >> "$LOG_DIR/status.log"
            notify "❌ $name failed" "Exit code: $exit_code (check logs)"
        fi
    ) &

    echo "  -> launched $name in background"
done

echo "All jobs launched. Terminal is free."