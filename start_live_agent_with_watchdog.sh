#!/bin/bash
# Start live agent with watchdog to ensure it always runs during market hours

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================================="
echo "STARTING LIVE AGENT WITH WATCHDOG"
echo "=================================================================================="
echo ""

# Check if watchdog is already running
if pgrep -f "ensure_live_agent_running.py" > /dev/null; then
    echo "⚠️  Watchdog is already running"
    WATCHDOG_PID=$(pgrep -f "ensure_live_agent_running.py" | head -1)
    echo "   PID: $WATCHDOG_PID"
    echo ""
    read -p "Kill existing watchdog and restart? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kill $WATCHDOG_PID 2>/dev/null || true
        sleep 2
    else
        echo "Keeping existing watchdog running"
        exit 0
    fi
fi

# Check if live agent is already running
if pgrep -f "mike_agent_live_safe.py" > /dev/null; then
    echo "⚠️  Live agent is already running"
    AGENT_PID=$(pgrep -f "mike_agent_live_safe.py" | head -1)
    echo "   PID: $AGENT_PID"
    echo ""
    read -p "Kill existing agent and restart? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kill $AGENT_PID 2>/dev/null || true
        sleep 2
    else
        echo "Keeping existing agent running"
    fi
fi

# Create logs directory
mkdir -p logs

# Start watchdog in background
echo "🚀 Starting watchdog..."
nohup python3 ensure_live_agent_running.py > logs/watchdog.log 2>&1 &
WATCHDOG_PID=$!
echo "   Watchdog PID: $WATCHDOG_PID"
echo "   Logs: logs/watchdog.log"
echo ""

# Wait a moment for watchdog to start
sleep 3

# Check if watchdog started successfully
if ! kill -0 $WATCHDOG_PID 2>/dev/null; then
    echo "❌ Watchdog failed to start. Check logs/watchdog.log"
    echo ""
    echo "Last 20 lines of watchdog.log:"
    tail -20 logs/watchdog.log 2>/dev/null || echo "No log file found"
    exit 1
fi

# Verify watchdog is actually running (not just started)
sleep 1
if ! kill -0 $WATCHDOG_PID 2>/dev/null; then
    echo "❌ Watchdog process died immediately. Check logs/watchdog.log"
    echo ""
    echo "Last 20 lines of watchdog.log:"
    tail -20 logs/watchdog.log 2>/dev/null || echo "No log file found"
    exit 1
fi

echo "✅ Watchdog started successfully"
echo ""
echo "The watchdog will:"
echo "  - Monitor market hours (9:30 AM - 4:00 PM EST, Mon-Fri)"
echo "  - Ensure live agent is running during market hours"
echo "  - Kill any backtest processes during market hours"
echo "  - Start live agent if it's not running during market hours"
echo ""
echo "To stop:"
echo "  pkill -f ensure_live_agent_running.py"
echo "  pkill -f mike_agent_live_safe.py"
echo ""
echo "To view logs:"
echo "  tail -f logs/watchdog.log"
echo "  tail -f logs/live_agent_\$(date +%Y%m%d).log"
echo ""
echo "=================================================================================="

