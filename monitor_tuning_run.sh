#!/bin/bash
# Monitor the 100k tuning run and extract diagnostics at key checkpoints

LOG_FILE="logs/training/mike_momentum_model_v2_intraday_tune1_100k.log"
MODEL_NAME="mike_momentum_model_v2_intraday_tune1"

echo "🔍 Monitoring training run: $MODEL_NAME"
echo "📊 Log file: $LOG_FILE"
echo ""

# Check if process is running
if ! pgrep -f "train_historical_model.*$MODEL_NAME" > /dev/null; then
    echo "⚠️  Training process not found. Checking if it completed or failed..."
    if [ -f "$LOG_FILE" ]; then
        echo "✅ Log file exists. Last 20 lines:"
        tail -20 "$LOG_FILE"
    else
        echo "❌ Log file not found. Training may not have started."
    fi
    exit 1
fi

echo "✅ Training process is running"
echo ""

# Extract latest diagnostics
echo "📈 Latest MomentumDiagnostics:"
grep "MomentumDiagnostics" "$LOG_FILE" 2>/dev/null | tail -1

echo ""
echo "📊 Current Training Status:"
tail -30 "$LOG_FILE" 2>/dev/null | grep -E "(time/|train/|rollout/|MomentumDiagnostics)" | tail -15

echo ""
echo "🔢 Checkpoint Summary:"
echo "Step 5k:"
grep -A 5 "MomentumDiagnostics @ step=5,000" "$LOG_FILE" 2>/dev/null | head -6 || echo "  Not reached yet"

echo ""
echo "Step 10k:"
grep -A 5 "MomentumDiagnostics @ step=10,000" "$LOG_FILE" 2>/dev/null | head -6 || echo "  Not reached yet"

echo ""
echo "Step 25k:"
grep -A 5 "MomentumDiagnostics @ step=25,000" "$LOG_FILE" 2>/dev/null | head -6 || echo "  Not reached yet"

echo ""
echo "Step 50k:"
grep -A 5 "MomentumDiagnostics @ step=50,000" "$LOG_FILE" 2>/dev/null | head -6 || echo "  Not reached yet"

echo ""
echo "💡 To watch live: tail -f $LOG_FILE | grep -E '(MomentumDiagnostics|time/|train/)'"





