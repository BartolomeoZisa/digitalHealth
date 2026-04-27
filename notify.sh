#!/bin/bash

# Tracking Arrays
MODELS_SUCCESS=()
MODELS_FAILED=()

send_discord() {
    local message="$1"

    if [ -z "$DISCORD_WEBHOOK_URL" ]; then
        echo "⚠️ ERROR: DISCORD_WEBHOOK_URL not found in environment."
        return
    fi

    # Use jq to safely encode the message into JSON
    # This handles newlines, quotes, and backslashes automatically
    json_data=$(jq -n --arg msg "$message" '{"content": $msg}')
    
    status_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d "$json_data" \
        "$DISCORD_WEBHOOK_URL")

    if [[ "$status_code" =~ ^2 ]]; then
        echo "✅ Discord notification sent (HTTP $status_code)."
    else
        echo "❌ Discord notification failed (HTTP $status_code). Payload: $json_data"
    fi
}

notify_success() {
    local model="$1"
    MODELS_SUCCESS+=("$model")
    echo "SUCCESS: $model"
}

notify_failure() {
    local model="$1"
    local code="$2"
    MODELS_FAILED+=("$model (exit $code)")
    echo "FAILURE: $model (exit $code)"
}

print_summary() {
    # Build Discord Message String
    summary="📊 **ML RUN COMPLETE**\n\n"
    
    if [ ${#MODELS_SUCCESS[@]} -gt 0 ]; then
        summary+="✅ **SUCCESS:**\n"
        for m in "${MODELS_SUCCESS[@]}"; do summary+=" - $m\n"; done
    fi

    if [ ${#MODELS_FAILED[@]} -gt 0 ]; then
        summary+="\n❌ **FAILED:**\n"
        for m in "${MODELS_FAILED[@]}"; do summary+=" - $m\n"; done
    fi

    # Print to local log
    echo -e "\n$summary"
    
    # Send to Discord
    send_discord "$summary"
}