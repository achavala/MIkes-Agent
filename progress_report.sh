#!/bin/bash
#
# Training Progress Report
# Generates a formatted progress report
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE=".training.pid"
LATEST_LOG=$(ls -t training_*.log 2>/dev/null | head -1)

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "📊 TRAINING PROGRESS REPORT"
echo "   Generated: $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Process Status
if [ -f "$PID_FILE" ]; then
    TRAINING_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$TRAINING_PID" ] && ps -p "$TRAINING_PID" > /dev/null 2>&1; then
        CPU=$(ps -p "$TRAINING_PID" -o %cpu= | tr -d ' ')
        MEM=$(ps -p "$TRAINING_PID" -o %mem= | tr -d ' ')
        ETIME=$(ps -p "$TRAINING_PID" -o etime= | tr -d ' ')
        STATUS="✅ RUNNING"
    else
        STATUS="❌ NOT RUNNING"
        CPU="N/A"
        MEM="N/A"
        ETIME="N/A"
    fi
else
    STATUS="❌ NOT RUNNING"
    CPU="N/A"
    MEM="N/A"
    ETIME="N/A"
fi

# Progress Calculation
LATEST_CHECKPOINT=$(ls -t models/checkpoints/mike_historical_model_*_steps.zip 2>/dev/null | head -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    TIMESTEPS=$(echo "$LATEST_CHECKPOINT" | grep -oE '[0-9]+_steps' | grep -oE '[0-9]+')
    TOTAL=5000000
    PROGRESS=$(echo "scale=2; $TIMESTEPS * 100 / $TOTAL" | bc)
    REMAINING=$((TOTAL - TIMESTEPS))
    
    # Time estimate
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
        REMAINING_MINS=$(echo "scale=0; ($REMAINING_SEC % 3600) / 60" | bc)
        ESTIMATE="${REMAINING_HOURS}h ${REMAINING_MINS}m"
    else
        ESTIMATE="Calculating..."
    fi
else
    TIMESTEPS=0
    TOTAL=5000000
    PROGRESS=0.00
    REMAINING=$TOTAL
    ESTIMATE="N/A"
fi

CHECKPOINT_COUNT=$(ls -1 models/checkpoints/mike_historical_model_*_steps.zip 2>/dev/null | wc -l | tr -d ' ')

# Format report
printf "%-25s %s\n" "Status:" "$STATUS"
printf "%-25s %s\n" "CPU Usage:" "${CPU}%"
printf "%-25s %s\n" "Memory Usage:" "${MEM}%"
printf "%-25s %s\n" "Runtime:" "$ETIME"
echo ""
printf "%-25s %s\n" "Progress:" "${PROGRESS}%"
printf "%-25s %s\n" "Timesteps:" "$TIMESTEPS / $TOTAL"
printf "%-25s %s\n" "Remaining:" "$REMAINING timesteps"
printf "%-25s %s\n" "Estimated Time:" "$ESTIMATE"
printf "%-25s %s\n" "Checkpoints Saved:" "$CHECKPOINT_COUNT"
echo ""

# Progress milestones
echo "Progress Milestones:"
MILESTONES=(25 50 75 90 100)
for milestone in "${MILESTONES[@]}"; do
    milestone_ts=$((TOTAL * milestone / 100))
    if [ "$TIMESTEPS" -ge "$milestone_ts" ]; then
        printf "  ✅ %2d%% (%s timesteps)\n" "$milestone" "$milestone_ts"
    else
        remaining_ts=$((milestone_ts - TIMESTEPS))
        printf "  ⏱️  %2d%% (%s timesteps) - %s remaining\n" "$milestone" "$milestone_ts" "$remaining_ts"
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"

