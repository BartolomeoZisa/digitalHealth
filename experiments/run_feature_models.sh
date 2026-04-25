#!/bin/bash

# Create logs directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/feature_models_$TIMESTAMP"
mkdir -p "$LOG_DIR"

echo "Logs will be stored in: $LOG_DIR"

# Activate virtual environment (adjust if needed)
source venv/bin/activate

# List of feature model scripts
declare -a scripts=(
    "experiments/svm/svm_feature_nested.py"
    "experiments/logisticregression/logreg_feature_nested.py"
    "experiments/randomforest/randomforest_feature_nested.py"
    "experiments/xgboost/xgboost_feature_nested.py"
)

# Run each script in background
for script in "${scripts[@]}"; do
    name=$(basename "$script" .py)
    log_file="$LOG_DIR/${name}.log"

    echo "Starting $name..."
    nohup python "$script" > "$log_file" 2>&1 &

    echo "  -> PID: $! | Log: $log_file"
done

echo "All jobs started."
echo "Use 'ps aux | grep python' to monitor."
echo "Use 'tail -f $LOG_DIR/<file>.log' to watch logs."