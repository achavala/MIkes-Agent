#!/bin/bash
#
# Quick Start Training Script
# One-command setup for lid-closed training
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "═══════════════════════════════════════════════════════════"
echo "🚀 QUICK START TRAINING (LID-CLOSED READY)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check if MacBook is on power
echo "🔌 Checking power status..."
if pmset -g batt | grep -q "AC Power"; then
    echo "   ✅ MacBook is plugged into power (required for lid-closed)"
else
    echo "   ⚠️  WARNING: MacBook is on battery!"
    echo "   Training will stop when lid closes if not plugged in."
    echo ""
    read -p "   Continue anyway? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   Exiting. Please plug in power adapter first."
        exit 1
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
DATA_FILES=(
    "data/historical/enriched/SPY_enriched_2002-01-01_latest.pkl"
    "data/historical/enriched/QQQ_enriched_2002-01-01_latest.pkl"
    "data/historical/enriched/SPX_enriched_2002-01-01_latest.pkl"
)

MISSING_FILES=()
for file in "${DATA_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo "❌ Missing data files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "   Please run data collection first:"
    echo "   python collect_quant_features.py --symbols SPY,QQQ,SPX"
    exit 1
fi

echo "   ✅ All data files found"

# Check disk space
echo "💾 Checking disk space..."
AVAILABLE_GB=$(df -h . | tail -1 | awk '{print $4}' | sed 's/[^0-9.]//g')
echo "   Available: $(df -h . | tail -1 | awk '{print $4}')"

# Prevent sleep
echo "☕ Starting sleep prevention (lid-closed compatible)..."
./prevent_sleep.sh start

# Check if training already running
if [ -f ".training.pid" ]; then
    OLD_PID=$(cat .training.pid)
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  Training is already running (PID: $OLD_PID)"
        echo ""
        echo "   To monitor: tail -f training_*.log"
        echo "   To stop: kill $OLD_PID && rm .training.pid"
        exit 1
    else
        rm -f .training.pid
    fi
fi

# Start training
echo "🎯 Starting training..."
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_${TIMESTAMP}.log"

nohup python train_historical_model.py \
    --symbols SPY,QQQ,SPX \
    --start-date 2002-01-01 \
    --timesteps 5000000 \
    --use-greeks \
    --regime-balanced \
    > "$LOG_FILE" 2>&1 &

TRAINING_PID=$!
echo "$TRAINING_PID" > .training.pid

echo "═══════════════════════════════════════════════════════════"
echo "✅ TRAINING STARTED!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📊 Process ID: $TRAINING_PID"
echo "📝 Log file: $LOG_FILE"
echo ""
echo "⏱️  Estimated completion: ~7 days (CPU) or ~1.75 days (GPU)"
echo ""
echo "📋 Next Steps:"
echo "   1. Wait 2-3 minutes"
echo "   2. Check log: tail -20 $LOG_FILE"
echo "   3. If everything looks good, close laptop lid!"
echo "   4. Make sure MacBook stays plugged into power"
echo ""
echo "📊 Monitor training:"
echo "   tail -f $LOG_FILE"
echo ""
echo "🔍 Check status:"
echo "   ./prevent_sleep.sh status"
echo "   ps -p $TRAINING_PID"
echo ""
echo "🛑 Stop training:"
echo "   kill $TRAINING_PID"
echo "   ./prevent_sleep.sh stop"
echo ""
echo "═══════════════════════════════════════════════════════════"

