#!/bin/bash
# Quick Training Status Checker
# Shows process status, latest logs, and key metrics

echo "🔍 LSTM Training Status Check"
echo "=============================="
echo ""

# Check if process is running
PROCESS=$(ps aux | grep -E "train_historical_model|retrain_lstm" | grep -v grep)
if [ -z "$PROCESS" ]; then
    echo "❌ Training process NOT running"
    echo ""
    echo "Possible reasons:"
    echo "  - Training completed"
    echo "  - Training crashed"
    echo "  - Training hasn't started yet"
    echo ""
    echo "Check logs:"
    echo "  tail -100 training_output.log"
else
    PID=$(echo "$PROCESS" | awk '{print $2}')
    CPU=$(echo "$PROCESS" | awk '{print $3}')
    MEM=$(echo "$PROCESS" | awk '{print $4}')
    echo "✅ Training process IS running"
    echo "   PID: $PID"
    echo "   CPU: ${CPU}%"
    echo "   Memory: ${MEM}%"
    echo ""
fi

# Check log file
if [ -f "training_output.log" ]; then
    LOG_SIZE=$(wc -l < training_output.log)
    echo "📊 Log file: training_output.log ($LOG_SIZE lines)"
    echo ""
    
    # Show last 10 lines
    echo "📝 Latest output (last 10 lines):"
    echo "-----------------------------------"
    tail -10 training_output.log
    echo ""
    
    # Check for key indicators
    echo "🔍 Key Status Indicators:"
    echo "-----------------------------------"
    
    if grep -q "RecurrentPPO available" training_output.log; then
        echo "✅ LSTM is active (RecurrentPPO detected)"
    else
        echo "⚠️  LSTM status unknown (check logs)"
    fi
    
    if grep -q "Training for" training_output.log; then
        echo "✅ Training has started"
    fi
    
    if grep -q "MomentumDiagnostics" training_output.log; then
        LAST_DIAG=$(grep "MomentumDiagnostics" training_output.log | tail -1)
        echo "✅ Diagnostics available: $LAST_DIAG"
    fi
    
    if grep -q "Saving model" training_output.log; then
        LAST_SAVE=$(grep "Saving model" training_output.log | tail -1)
        echo "✅ Checkpoint saved: $LAST_SAVE"
    fi
    
    if grep -q "Error\|Exception\|Traceback" training_output.log; then
        echo "❌ Errors detected in logs!"
        echo "   Check: grep -i error training_output.log"
    fi
    
    # Check training log file
    TRAINING_LOG=$(ls -t logs/training/mike_momentum_model_v3_lstm_*.log 2>/dev/null | head -1)
    if [ -n "$TRAINING_LOG" ]; then
        echo ""
        echo "📁 Training log file: $TRAINING_LOG"
        TRAINING_LOG_SIZE=$(wc -l < "$TRAINING_LOG" 2>/dev/null || echo "0")
        echo "   Lines: $TRAINING_LOG_SIZE"
    fi
    
else
    echo "⚠️  Log file not found: training_output.log"
    echo "   Training may not have started yet"
fi

echo ""
echo "💡 Quick Commands:"
echo "   Watch live: tail -f training_output.log"
echo "   Check diagnostics: grep 'MomentumDiagnostics' training_output.log | tail -5"
echo "   Check errors: grep -i error training_output.log | tail -10"
echo "   Stop training: pkill -f train_historical_model"
