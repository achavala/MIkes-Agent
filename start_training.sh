#!/bin/bash
#
# Start Training Script
# Automatically handles sleep prevention and background training
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
SYMBOLS="SPY,QQQ,SPX"
START_DATE="2002-01-01"
TIMESTEPS=5000000
SAVE_FREQ=100000
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_${TIMESTAMP}.log"
PID_FILE=".training.pid"

echo "=" | tr '=' '═'
echo "🚀 STARTING TRAINING"
echo "=" | tr '=' '═'
echo ""

# Check if training already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  Training is already running (PID: $OLD_PID)"
        echo "   Log file: training_*.log"
        echo ""
        echo "To stop current training:"
        echo "  kill $OLD_PID"
        echo "  rm $PID_FILE"
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

# Check virtual environment
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please create it first: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check data files
echo "📊 Checking data files..."
if [ ! -f "data/historical/enriched/SPY_enriched_2002-01-01_latest.pkl" ]; then
    echo "❌ Enriched data files not found!"
    echo "   Please run data collection first:"
    echo "   python collect_quant_features.py --symbols SPY,QQQ,SPX"
    exit 1
fi

# Prevent sleep
echo "☕ Preventing Mac from sleeping..."
./prevent_sleep.sh start

# Check disk space
echo "💾 Checking disk space..."
AVAILABLE=$(df -h . | tail -1 | awk '{print $4}')
echo "   Available: $AVAILABLE"
echo ""

# Start training
echo "🎯 Starting training..."
echo "   Symbols: $SYMBOLS"
echo "   Timesteps: $TIMESTEPS"
echo "   Save frequency: Every 50,000 steps (hardcoded in script)"
echo "   Log file: $LOG_FILE"
echo ""

# Start training with nohup
# Note: save_freq is hardcoded to 50000 in train_historical_model.py (CheckpointCallback)
nohup python train_historical_model.py \
    --symbols "$SYMBOLS" \
    --start-date "$START_DATE" \
    --timesteps "$TIMESTEPS" \
    --use-greeks \
    --regime-balanced \
    > "$LOG_FILE" 2>&1 &

TRAINING_PID=$!
echo "$TRAINING_PID" > "$PID_FILE"

echo "✅ Training started!"
echo ""
echo "Process ID: $TRAINING_PID"
echo "Log file: $LOG_FILE"
echo "PID file: $PID_FILE"
echo ""
echo "📊 Monitor training:"
echo "   tail -f $LOG_FILE"
echo ""
echo "📁 Check checkpoints:"
echo "   ls -lth models/*.zip | head -5"
echo ""
echo "🛑 Stop training:"
echo "   kill $TRAINING_PID"
echo "   ./prevent_sleep.sh stop"
echo ""
echo "=" | tr '=' '═'

