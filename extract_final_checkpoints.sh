#!/bin/bash
# Extract final checkpoints (100k, 250k, 500k) from 500k training log

LOG_FILE="logs/training/mike_momentum_model_v2_intraday_full_500k.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    exit 1
fi

echo "📊 EXTRACTING FINAL CHECKPOINTS"
echo "================================"
echo ""

# Check if training is complete
if grep -q "Training completed\|Saving model" "$LOG_FILE" 2>/dev/null; then
    echo "✅ Training appears to be complete"
    echo ""
fi

# Extract 100k
echo "=== STEP 100,000 ==="
if grep -q "MomentumDiagnostics @ step=100,000" "$LOG_FILE"; then
    grep -A 6 "MomentumDiagnostics @ step=100,000" "$LOG_FILE"
else
    echo "  ⏳ Not reached yet"
fi
echo ""

# Extract 250k
echo "=== STEP 250,000 ==="
if grep -q "MomentumDiagnostics @ step=250,000" "$LOG_FILE"; then
    grep -A 6 "MomentumDiagnostics @ step=250,000" "$LOG_FILE"
else
    echo "  ⏳ Not reached yet"
fi
echo ""

# Extract 500k (final)
echo "=== STEP 500,000 (FINAL) ==="
if grep -q "MomentumDiagnostics @ step=500,000" "$LOG_FILE"; then
    grep -A 6 "MomentumDiagnostics @ step=500,000" "$LOG_FILE"
else
    echo "  ⏳ Not reached yet"
fi
echo ""

# Summary table
echo "================================"
echo "📋 SUMMARY TABLE"
echo "================================"
echo ""
echo "| Step | HOLD % | Combined BUY % | Strong-Setup BUY % |"
echo "|------|--------|----------------|-------------------|"

for step in "5,000" "10,000" "25,000" "50,000" "100,000" "250,000" "500,000"; do
    if grep -q "MomentumDiagnostics @ step=$step" "$LOG_FILE"; then
        # Extract HOLD percentage
        hold=$(grep -A 6 "MomentumDiagnostics @ step=$step" "$LOG_FILE" | grep "action 0:" | grep -o "[0-9.]*%" | head -1)
        # Extract BUY percentages
        buy_call=$(grep -A 6 "MomentumDiagnostics @ step=$step" "$LOG_FILE" | grep "action 1:" | grep -o "[0-9.]*%" | head -1)
        buy_put=$(grep -A 6 "MomentumDiagnostics @ step=$step" "$LOG_FILE" | grep "action 2:" | grep -o "[0-9.]*%" | head -1)
        # Extract strong-setup BUY rate
        strong_buy=$(grep -A 6 "MomentumDiagnostics @ step=$step" "$LOG_FILE" | grep "BUY rate=" | grep -o "[0-9.]*%" | head -1)
        
        if [ -n "$hold" ] && [ -n "$buy_call" ] && [ -n "$buy_put" ] && [ -n "$strong_buy" ]; then
            # Calculate combined BUY
            buy_call_num=$(echo "$buy_call" | sed 's/%//')
            buy_put_num=$(echo "$buy_put" | sed 's/%//')
            combined_buy=$(echo "$buy_call_num + $buy_put_num" | bc)
            combined_buy="${combined_buy}%"
            
            echo "| $step | $hold | $combined_buy | $strong_buy |"
        fi
    fi
done

echo ""
echo "💡 For full details, see: $LOG_FILE"





