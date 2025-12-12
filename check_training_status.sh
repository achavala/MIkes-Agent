#!/bin/bash
#
# Quick Training Status Check
# Shows if training is running, progress, and health
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE=".training.pid"
LOG_FILE=$(ls -t training_*.log 2>/dev/null | head -1)
CAFFEINATE_PID_FILE=".caffeinate.pid"

echo "═══════════════════════════════════════════════════════════"
echo "📊 TRAINING STATUS CHECK"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check training process
if [ -f "$PID_FILE" ]; then
    TRAINING_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$TRAINING_PID" ] && ps -p "$TRAINING_PID" > /dev/null 2>&1; then
        echo "✅ Training: RUNNING (PID: $TRAINING_PID)"
        
        # Check CPU usage
        CPU=$(ps -p "$TRAINING_PID" -o %cpu= | tr -d ' ')
        MEM=$(ps -p "$TRAINING_PID" -o %mem= | tr -d ' ')
        echo "   CPU: ${CPU}% | Memory: ${MEM}%"
    else
        echo "❌ Training: NOT RUNNING (stale PID file)"
    fi
else
    echo "❌ Training: NOT RUNNING (no PID file)"
fi

echo ""

# Check caffeinate
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

echo ""

# Check power status
if pmset -g batt | grep -q "AC Power"; then
    echo "✅ Power: PLUGGED IN (lid-closed safe)"
else
    echo "⚠️  Power: ON BATTERY (lid-closed may stop training)"
fi

echo ""

# Check latest checkpoint
LATEST_CHECKPOINT=$(ls -t models/mike_rl_model_*.zip 2>/dev/null | head -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    CHECKPOINT_SIZE=$(ls -lh "$LATEST_CHECKPOINT" | awk '{print $5}')
    CHECKPOINT_TIME=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$LATEST_CHECKPOINT" 2>/dev/null || stat -c "%y" "$LATEST_CHECKPOINT" 2>/dev/null | cut -d' ' -f1-2)
    echo "📁 Latest Checkpoint:"
    echo "   File: $(basename $LATEST_CHECKPOINT)"
    echo "   Size: $CHECKPOINT_SIZE"
    echo "   Time: $CHECKPOINT_TIME"
else
    echo "📁 Latest Checkpoint: None yet"
fi

echo ""

# Check log file
if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(ls -lh "$LOG_FILE" | awk '{print $5}')
    LOG_LINES=$(wc -l < "$LOG_FILE" 2>/dev/null || echo "0")
    echo "📝 Log File: $LOG_FILE"
    echo "   Size: $LOG_SIZE | Lines: $LOG_LINES"
    echo ""
    echo "   Last 5 lines:"
    tail -5 "$LOG_FILE" | sed 's/^/   /'
else
    echo "📝 Log File: Not found"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"

