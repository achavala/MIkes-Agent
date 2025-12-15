#!/bin/bash
# Activate venv if it exists (Railway/Nixpacks specific)
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Default to port 8080 if PORT is not set
PORT="${PORT:-8080}"

echo "🚀 Starting Cloud Deployment on PORT $PORT"

# Log the running version (Tag/Commit)
echo "📦 Version Info:"
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "   Commit: $(git rev-parse --short HEAD)"
  echo "   Tag:    $(git describe --tags --exact-match 2>/dev/null || echo 'No exact tag')"
else
  echo "   Git info not available"
fi

# Start Streamlit Dashboard in the background
streamlit run dashboard_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true &


# Start Trading Agent (in Live Mode if configured, otherwise Paper)
# Use --live flag if MODE env var is set to 'live', otherwise defaults to paper
if [ "$MODE" = "live" ]; then
    echo "🚀 Starting Agent in LIVE mode..."
    python mike_agent_live_safe.py --live --key "$ALPACA_KEY" --secret "$ALPACA_SECRET"
else
    echo "🧪 Starting Agent in PAPER mode..."
    python mike_agent_live_safe.py --key "$ALPACA_KEY" --secret "$ALPACA_SECRET"
fi

