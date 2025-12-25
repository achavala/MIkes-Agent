#!/bin/bash
# Quick script to extract all MomentumDiagnostics checkpoints from training log

LOG_FILE="logs/training/mike_momentum_model_v2_intraday_full_500k.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    exit 1
fi

echo "📊 EXTRACTING ALL MOMENTUM DIAGNOSTICS"
echo "======================================"
echo ""

# Get all checkpoint steps
STEPS=$(grep "MomentumDiagnostics @" "$LOG_FILE" | grep -o "step=[0-9,]*" | sed 's/step=//' | tr ',' ' ')

if [ -z "$STEPS" ]; then
    echo "⚠️  No diagnostics found yet. Training may still be initializing."
    echo "💡 Check back in a few minutes."
    exit 0
fi

echo "✅ Found checkpoints:"
echo "$STEPS" | tr ' ' '\n' | head -20
echo ""

# Extract latest checkpoint
LATEST=$(echo "$STEPS" | tr ' ' '\n' | tail -1)
echo "📈 Latest checkpoint: step=$LATEST"
echo ""
grep -A 6 "MomentumDiagnostics @ step=$LATEST" "$LOG_FILE" 2>/dev/null || echo "  Not found"

echo ""
echo "======================================"
echo "📋 All Checkpoints Summary:"
echo ""

# Extract key metrics for each checkpoint
for step in 5000 10000 25000 50000 100000 250000 500000; do
    if grep -q "MomentumDiagnostics @ step=$step" "$LOG_FILE"; then
        echo "Step $step:"
        grep -A 6 "MomentumDiagnostics @ step=$step" "$LOG_FILE" | grep -E "(action 0:|BUY rate=|HOLD rate=)" | head -3
        echo ""
    fi
done

echo "💡 For full details, run:"
echo "   grep -A 6 'MomentumDiagnostics @ step=XXX' $LOG_FILE"





