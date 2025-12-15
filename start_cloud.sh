#!/bin/bash
set -e

# Activate venv if it exists (Railway/Nixpacks specific)
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Default to port 8080 if PORT is not set
PORT="${PORT:-8080}"

echo "=================================================="
echo "🚀 STARTING MIKE AGENT CLOUD DEPLOYMENT"
echo "=================================================="
echo "   PORT: $PORT"
echo "   MODE: ${MODE:-paper (default)}"

# Log the running version (Tag/Commit)
echo "📦 Version Info:"
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "   Commit: $(git rev-parse --short HEAD)"
  echo "   Tag:    $(git describe --tags --exact-match 2>/dev/null || echo 'No exact tag')"
else
  echo "   Git info not available"
fi

echo "--------------------------------------------------"
echo "1️⃣  Starting Dashboard (Background Service)"
# Start Streamlit Dashboard in the background
streamlit run dashboard_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true &
DASHBOARD_PID=$!
echo "   ✅ Dashboard started with PID $DASHBOARD_PID"

echo "--------------------------------------------------"
echo "2️⃣  Starting Trading Agent (Foreground Service)"
# Start Trading Agent (in Live Mode if configured, otherwise Paper)
if [ "$MODE" = "live" ]; then
    echo "🚀 Starting Agent in LIVE mode..."
    # Using python -u for unbuffered output so logs show up immediately in Railway
    python -u mike_agent_live_safe.py --live --key "$ALPACA_KEY" --secret "$ALPACA_SECRET"
else
    echo "🧪 Starting Agent in PAPER mode..."
    python -u mike_agent_live_safe.py --key "$ALPACA_KEY" --secret "$ALPACA_SECRET"
fi

# Wait for any process to exit
wait -n
  
# Exit with status of process that exited first
exit $?
