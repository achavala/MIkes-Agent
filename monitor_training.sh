#!/bin/bash
#
# Real-time Training Monitor
# Continuously monitors training progress with auto-refresh
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REFRESH_INTERVAL=${1:-10}  # Default 10 seconds, can be overridden: ./monitor_training.sh 5

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 REAL-TIME TRAINING MONITOR"
echo "   Refresh: Every $REFRESH_INTERVAL seconds | Press Ctrl+C to stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

while true; do
    clear
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 TRAINING STATUS - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Process status
    PID_FILE=".training.pid"
    if [ -f "$PID_FILE" ]; then
        TRAINING_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
        if [ -n "$TRAINING_PID" ] && ps -p "$TRAINING_PID" > /dev/null 2>&1; then
            CPU=$(ps -p "$TRAINING_PID" -o %cpu= | tr -d ' ')
            MEM=$(ps -p "$TRAINING_PID" -o %mem= | tr -d ' ')
            ETIME=$(ps -p "$TRAINING_PID" -o etime= | tr -d ' ')
            echo "✅ Training: RUNNING (PID: $TRAINING_PID)"
            echo "   CPU: ${CPU}% | Memory: ${MEM}% | Runtime: $ETIME"
        else
            echo "❌ Training: NOT RUNNING"
        fi
    else
        echo "❌ Training: NOT RUNNING"
    fi
    
    echo ""
    
    # Progress
    LATEST_CHECKPOINT=$(ls -t models/checkpoints/mike_historical_model_*_steps.zip 2>/dev/null | head -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        TIMESTEPS=$(echo "$LATEST_CHECKPOINT" | grep -oE '[0-9]+_steps' | grep -oE '[0-9]+')
        TOTAL=5000000
        PROGRESS=$(echo "scale=2; $TIMESTEPS * 100 / $TOTAL" | bc)
        REMAINING=$((TOTAL - TIMESTEPS))
        
        # Progress bar
        BAR_LENGTH=50
        FILLED=$(echo "scale=0; $TIMESTEPS * $BAR_LENGTH / $TOTAL" | bc)
        EMPTY=$((BAR_LENGTH - FILLED))
        
        BAR=""
        for ((i=0; i<FILLED; i++)); do BAR="${BAR}█"; done
        for ((i=0; i<EMPTY; i++)); do BAR="${BAR}░"; done
        
        echo "📊 Progress: $PROGRESS%"
        echo "   [$BAR]"
        echo "   Timesteps: $TIMESTEPS / $TOTAL"
        echo "   Remaining: $REMAINING"
        
        # Time estimate
        if [ -n "$ETIME" ] && [ "$TIMESTEPS" -gt 0 ]; then
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
            
            if [ "$ELAPSED_SEC" -gt 0 ]; then
                RATE=$(echo "scale=2; $TIMESTEPS / $ELAPSED_SEC" | bc)
                REMAINING_SEC=$(echo "scale=0; $REMAINING / $RATE" | bc)
                REMAINING_HOURS=$(echo "scale=1; $REMAINING_SEC / 3600" | bc)
                REMAINING_MINS=$(echo "scale=0; ($REMAINING_SEC % 3600) / 60" | bc)
                echo "   ⏱️  Rate: $RATE tps | Remaining: ~${REMAINING_HOURS}h ${REMAINING_MINS}m"
            fi
        fi
    else
        echo "📊 Progress: No checkpoints yet"
    fi
    
    echo ""
    
    # Latest log metrics
    LATEST_LOG=$(ls -t training_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
        echo "📝 Latest Metrics:"
        tail -10 "$LATEST_LOG" | grep -E "total_timesteps|ep_rew_mean|loss|learning_rate" | tail -3 | sed 's/^/   /'
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "   Refreshing in $REFRESH_INTERVAL seconds... (Ctrl+C to stop)"
    
    sleep "$REFRESH_INTERVAL"
done

