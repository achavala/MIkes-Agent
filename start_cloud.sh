#!/bin/bash
# Cloud startup script for Fly.io
# Runs both the trading agent and Streamlit dashboard

set -e

echo "🚀 Starting Cloud Deployment on PORT ${PORT:-8080}"

# Get Git version info (if available)
GIT_VERSION=$(git describe --tags --exact-match 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "📦 Version Info: Git ${GIT_VERSION}"

# Determine mode from environment variable
MODE=${MODE:-paper}
echo "🧪 Starting Agent in ${MODE^^} mode..."

# Start the trading agent in the background
python mike_agent_live_safe.py &
AGENT_PID=$!

# Wait a moment for agent to initialize
sleep 2

# Start Streamlit dashboard
# Use PORT from Fly.io or default to 8080
export STREAMLIT_SERVER_PORT=${PORT:-8080}
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true

echo "📊 Starting Streamlit dashboard on port ${STREAMLIT_SERVER_PORT}..."
streamlit run dashboard_app.py \
    --server.port=${STREAMLIT_SERVER_PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false &

DASHBOARD_PID=$!

# Function to handle shutdown
cleanup() {
    echo "🛑 Shutting down..."
    kill $AGENT_PID 2>/dev/null || true
    kill $DASHBOARD_PID 2>/dev/null || true
    wait
    exit 0
}

trap cleanup SIGTERM SIGINT

# Wait for both processes
wait
