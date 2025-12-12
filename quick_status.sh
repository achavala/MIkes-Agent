#!/bin/bash
#
# Quick Training Status Check
# Fast overview of training progress
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE=".training.pid"
LATEST_LOG=$(ls -t training_*.log 2>/dev/null | head -1)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚡ QUICK TRAINING STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if training is running
if [ -f "$PID_FILE" ]; then
    TRAINING_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$TRAINING_PID" ] && ps -p "$TRAINING_PID" > /dev/null 2>&1; then
        CPU=$(ps -p "$TRAINING_PID" -o %cpu= | tr -d ' ')
        MEM=$(ps -p "$TRAINING_PID" -o %mem= | tr -d ' ')
        ETIME=$(ps -p "$TRAINING_PID" -o etime= | tr -d ' ')
        echo "✅ Training: RUNNING"
        echo "   PID: $TRAINING_PID | CPU: ${CPU}% | Memory: ${MEM}% | Runtime: $ETIME"
    else
        echo "❌ Training: NOT RUNNING"
    fi
else
    echo "❌ Training: NOT RUNNING"
fi

echo ""

# Get latest checkpoint
LATEST_CHECKPOINT=$(ls -t models/checkpoints/mike_historical_model_*_steps.zip 2>/dev/null | head -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    # Extract timesteps from filename
    TIMESTEPS=$(echo "$LATEST_CHECKPOINT" | grep -oE '[0-9]+_steps' | grep -oE '[0-9]+')
    TOTAL=5000000
    PROGRESS=$(echo "scale=2; $TIMESTEPS * 100 / $TOTAL" | bc)
    REMAINING=$((TOTAL - TIMESTEPS))
    
    echo "📊 Progress: $PROGRESS%"
    echo "   Timesteps: $TIMESTEPS / $TOTAL"
    echo "   Remaining: $REMAINING"
    
    # Estimate time remaining (if we have runtime)
    if [ -n "$ETIME" ] && [ "$TIMESTEPS" -gt 0 ]; then
        # Parse elapsed time (format: HH:MM:SS or MM:SS)
        if [[ "$ETIME" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
            HOURS=${BASH_REMATCH[1]}
            MINUTES=${BASH_REMATCH[2]}
            SECONDS=${BASH_REMATCH[3]}
            ELAPSED_SEC=$((HOURS * 3600 + MINUTES * 60 + SECONDS))
        elif [[ "$ETIME" =~ ^([0-9]+):([0-9]+)$ ]]; then
            MINUTES=${BASH_REMATCH[1]}
            SECONDS=${BASH_REMATCH[2]}
            ELAPSED_SEC=$((MINUTES * 60 + SECONDS))
        else
            ELAPSED_SEC=0
        fi
        
        if [ "$ELAPSED_SEC" -gt 0 ] && [ "$TIMESTEPS" -gt 0 ]; then
            RATE=$(echo "scale=2; $TIMESTEPS / $ELAPSED_SEC" | bc)
            REMAINING_SEC=$(echo "scale=0; $REMAINING / $RATE" | bc)
            REMAINING_HOURS=$(echo "scale=1; $REMAINING_SEC / 3600" | bc)
            echo "   ⏱️  Estimated remaining: ~${REMAINING_HOURS} hours"
        fi
    fi
else
    echo "📊 Progress: No checkpoints yet"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

