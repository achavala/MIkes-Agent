#!/bin/bash
# Restart Mike Agent - Stop existing and start fresh
# Usage: ./restart_agent.sh

cd "$(dirname "$0")"

echo "🔄 Restarting Mike Agent..."

# Step 1: Stop any existing agent processes
echo "1️⃣ Stopping existing agent processes..."
pkill -f "mike_agent_live_safe.py" 2>/dev/null
sleep 2

# Verify processes are stopped
if pgrep -f "mike_agent_live_safe.py" > /dev/null; then
    echo "⚠️  Some processes still running, force killing..."
    pkill -9 -f "mike_agent_live_safe.py" 2>/dev/null
    sleep 1
fi

# Step 2: Check dependencies
echo "2️⃣ Checking dependencies..."
MISSING_DEPS=0

if ! python3 -c "import stable_baselines3" 2>/dev/null; then
    echo "   ⚠️  stable-baselines3 not installed"
    MISSING_DEPS=1
fi

if ! python3 -c "import alpaca_trade_api" 2>/dev/null; then
    echo "   ⚠️  alpaca-trade-api not installed"
    MISSING_DEPS=1
fi

if ! python3 -c "import yfinance" 2>/dev/null; then
    echo "   ⚠️  yfinance not installed"
    MISSING_DEPS=1
fi

if [ $MISSING_DEPS -eq 1 ]; then
    echo ""
    echo "❌ ERROR: Missing dependencies!"
    echo "   Install with: pip install stable-baselines3 alpaca-trade-api yfinance"
    exit 1
fi

echo "   ✅ All dependencies installed"

# Step 3: Verify file compiles
echo "3️⃣ Validating code..."
if ! python3 -m py_compile mike_agent_live_safe.py 2>/dev/null; then
    echo "❌ ERROR: mike_agent_live_safe.py has syntax errors!"
    echo "   Please fix errors before restarting."
    exit 1
fi

echo "   ✅ Code compiles successfully"

# Step 4: Create logs directory
mkdir -p logs

# Step 5: Start agent in background
echo "4️⃣ Starting agent..."
nohup python3 mike_agent_live_safe.py > logs/agent_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Get the PID
AGENT_PID=$!
sleep 3

# Step 6: Verify agent started
if ps -p $AGENT_PID > /dev/null; then
    echo "✅ Agent started successfully!"
    echo "   PID: $AGENT_PID"
    echo "   Log file: logs/agent_$(date +%Y%m%d)_*.log"
    echo ""
    echo "📊 Monitor logs with:"
    echo "   tail -f logs/agent_$(date +%Y%m%d)_*.log"
    echo ""
    echo "🛑 To stop agent:"
    echo "   pkill -f mike_agent_live_safe.py"
else
    echo "❌ ERROR: Agent failed to start!"
    echo "   Check logs/agent_$(date +%Y%m%d)_*.log for errors"
    exit 1
fi
