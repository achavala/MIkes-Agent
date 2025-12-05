#!/bin/bash
# Deploy Mike Agent v3 - Comprehensive Version
cd "$(dirname "$0")"

echo "🚀 Deploying Mike Agent v3 - Comprehensive Version"
echo "=================================================="
echo ""
echo "✅ Features:"
echo "  • Fixed -15% Stop-Loss (always)"
echo "  • Trading Symbols: SPY, QQQ, SPX"
echo "  • 5-Tier Take-Profit System"
echo "  • 14 Comprehensive Safeguards"
echo "  • Volatility Regime Engine"
echo "  • Trade Database Logging"
echo ""
echo "📊 Logs will be written to:"
echo "  • mike.log (main log)"
echo "  • mike_error.log (errors)"
echo ""

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Run the agent with warnings suppressed
echo "🚀 Starting agent..."
python3 -W ignore::DeprecationWarning -u mike_agent_live_safe.py > mike.log 2> mike_error.log &

AGENT_PID=$!
echo "✅ Agent started (PID: $AGENT_PID)"
echo ""
echo "📝 To monitor:"
echo "  tail -f mike.log"
echo ""
echo "🛑 To stop:"
echo "  kill $AGENT_PID"
echo "  or: pkill -f mike_agent_live_safe.py"
echo ""
echo "Agent is running in the background..."
