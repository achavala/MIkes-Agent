#!/bin/bash
# Start Streamlit GUI Dashboard
# Usage: ./start_gui.sh

cd "$(dirname "$0")"

# Kill any existing Streamlit processes
echo "🔄 Stopping any existing Streamlit processes..."
lsof -ti:8501 2>/dev/null | xargs kill -9 2>/dev/null
sleep 2

# Check if app.py has syntax errors
echo "🔍 Checking app.py syntax..."
if ! python3 -m py_compile app.py 2>/dev/null; then
    echo "❌ app.py has syntax errors. Please fix them first."
    exit 1
fi

# Create logs directory
mkdir -p logs

# Start Streamlit in background
echo "🚀 Starting Streamlit dashboard on http://localhost:8501..."
nohup python3 -m streamlit run app.py \
    --server.port=8501 \
    --server.address=localhost \
    --server.headless=true \
    > logs/streamlit.log 2>&1 &

# Wait a moment for it to start
sleep 5

# Check if it's running
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 | grep -q "200"; then
    echo "✅ Streamlit dashboard is running!"
    echo "📊 Open your browser to: http://localhost:8501"
    echo "📝 Logs: tail -f logs/streamlit.log"
else
    echo "❌ Streamlit failed to start. Check logs/streamlit.log for errors."
    tail -20 logs/streamlit.log
    exit 1
fi

