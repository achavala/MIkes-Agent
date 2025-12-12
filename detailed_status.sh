#!/bin/bash
#
# Detailed Training Status Check
# Comprehensive analysis of training progress
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE=".training.pid"
LATEST_LOG=$(ls -t training_*.log 2>/dev/null | head -1)
CAFFEINATE_PID_FILE=".caffeinate.pid"

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "📊 DETAILED TRAINING STATUS"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# 1. Process Status
echo "1️⃣ PROCESS STATUS"
echo "───────────────────────────────────────────────────────────────────────────────"
if [ -f "$PID_FILE" ]; then
    TRAINING_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$TRAINING_PID" ] && ps -p "$TRAINING_PID" > /dev/null 2>&1; then
        CPU=$(ps -p "$TRAINING_PID" -o %cpu= | tr -d ' ')
        MEM=$(ps -p "$TRAINING_PID" -o %mem= | tr -d ' ')
        ETIME=$(ps -p "$TRAINING_PID" -o etime= | tr -d ' ')
        VSZ=$(ps -p "$TRAINING_PID" -o vsz= | tr -d ' ')
        RSS=$(ps -p "$TRAINING_PID" -o rss= | tr -d ' ')
        
        echo "✅ Training: RUNNING"
        echo "   PID: $TRAINING_PID"
        echo "   CPU: ${CPU}%"
        echo "   Memory: ${MEM}% (${RSS} KB RSS, ${VSZ} KB VSZ)"
        echo "   Runtime: $ETIME"
    else
        echo "❌ Training: NOT RUNNING (stale PID file)"
    fi
else
    echo "❌ Training: NOT RUNNING (no PID file)"
fi

echo ""

# 2. Sleep Prevention
echo "2️⃣ SLEEP PREVENTION"
echo "───────────────────────────────────────────────────────────────────────────────"
if [ -f "$CAFFEINATE_PID_FILE" ]; then
    CAFFEINATE_PID=$(cat "$CAFFEINATE_PID_FILE" 2>/dev/null || echo "")
    if [ -n "$CAFFEINATE_PID" ] && ps -p "$CAFFEINATE_PID" > /dev/null 2>&1; then
        echo "✅ Sleep Prevention: ACTIVE (PID: $CAFFEINATE_PID)"
    else
        echo "❌ Sleep Prevention: NOT ACTIVE"
    fi
else
    echo "❌ Sleep Prevention: NOT ACTIVE (no PID file)"
fi

# Power status
if pmset -g batt | grep -q "AC Power"; then
    echo "✅ Power: PLUGGED IN (lid-closed safe)"
else
    echo "⚠️  Power: ON BATTERY (lid-closed may stop training)"
fi

echo ""

# 3. Checkpoint Analysis
echo "3️⃣ CHECKPOINT ANALYSIS"
echo "───────────────────────────────────────────────────────────────────────────────"
LATEST_CHECKPOINT=$(ls -t models/checkpoints/mike_historical_model_*_steps.zip 2>/dev/null | head -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    TIMESTEPS=$(echo "$LATEST_CHECKPOINT" | grep -oE '[0-9]+_steps' | grep -oE '[0-9]+')
    TOTAL=5000000
    PROGRESS=$(echo "scale=2; $TIMESTEPS * 100 / $TOTAL" | bc)
    REMAINING=$((TOTAL - TIMESTEPS))
    
    CHECKPOINT_SIZE=$(ls -lh "$LATEST_CHECKPOINT" | awk '{print $5}')
    CHECKPOINT_TIME=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$LATEST_CHECKPOINT" 2>/dev/null || stat -c "%y" "$LATEST_CHECKPOINT" 2>/dev/null | cut -d' ' -f1-2)
    
    CHECKPOINT_COUNT=$(ls -1 models/checkpoints/mike_historical_model_*_steps.zip 2>/dev/null | wc -l | tr -d ' ')
    
    echo "✅ Latest Checkpoint:"
    echo "   File: $(basename $LATEST_CHECKPOINT)"
    echo "   Timesteps: $TIMESTEPS / $TOTAL"
    echo "   Progress: $PROGRESS%"
    echo "   Remaining: $REMAINING timesteps"
    echo "   Size: $CHECKPOINT_SIZE"
    echo "   Time: $CHECKPOINT_TIME"
    echo "   Total Checkpoints: $CHECKPOINT_COUNT"
    
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
            
            echo ""
            echo "   ⏱️  Time Estimates:"
            echo "      Rate: $RATE timesteps/second"
            echo "      Estimated remaining: ${REMAINING_HOURS} hours (${REMAINING_MINS} minutes)"
        fi
    fi
else
    echo "⚠️  No checkpoints found yet"
fi

echo ""

# 4. Log File Analysis
echo "4️⃣ LOG FILE ANALYSIS"
echo "───────────────────────────────────────────────────────────────────────────────"
if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    LOG_SIZE=$(ls -lh "$LATEST_LOG" | awk '{print $5}')
    LOG_LINES=$(wc -l < "$LATEST_LOG" 2>/dev/null || echo "0")
    LOG_TIME=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$LATEST_LOG" 2>/dev/null || stat -c "%y" "$LATEST_LOG" 2>/dev/null | cut -d' ' -f1-2)
    
    echo "✅ Log File: $LATEST_LOG"
    echo "   Size: $LOG_SIZE"
    echo "   Lines: $LOG_LINES"
    echo "   Last Updated: $LOG_TIME"
    echo ""
    echo "   Latest Metrics:"
    tail -15 "$LATEST_LOG" | grep -E "total_timesteps|ep_rew_mean|loss|learning_rate" | tail -5 | sed 's/^/      /'
else
    echo "❌ Log file not found"
fi

echo ""

# 5. Final Model Status
echo "5️⃣ FINAL MODEL STATUS"
echo "───────────────────────────────────────────────────────────────────────────────"
FINAL_MODEL=$(ls -t models/mike_historical_model.zip models/mike_rl_model.zip 2>/dev/null | head -1)
if [ -n "$FINAL_MODEL" ]; then
    MODEL_SIZE=$(ls -lh "$FINAL_MODEL" | awk '{print $5}')
    MODEL_TIME=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$FINAL_MODEL" 2>/dev/null || stat -c "%y" "$FINAL_MODEL" 2>/dev/null | cut -d' ' -f1-2)
    echo "✅ Final Model: $(basename $FINAL_MODEL)"
    echo "   Size: $MODEL_SIZE"
    echo "   Time: $MODEL_TIME"
else
    echo "⚠️  Final model not yet created (training in progress)"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"

