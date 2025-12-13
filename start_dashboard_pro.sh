#!/bin/bash
# Start Professional Dashboard
# Resilient startup script that works across reboots

cd "$(dirname "$0")"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 STARTING MIKE AGENT PROFESSIONAL DASHBOARD"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create necessary directories
mkdir -p dashboard_data
mkdir -p logs

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "❌ Error: streamlit not installed"
    echo "   Installing streamlit and dependencies..."
    pip install streamlit plotly pandas numpy
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

# Check if dashboard_app.py exists
if [ ! -f "dashboard_app.py" ]; then
    echo "❌ Error: dashboard_app.py not found"
    exit 1
fi

# Kill any existing Streamlit processes on port 8501
echo "🧹 Cleaning up existing Streamlit processes..."
lsof -ti:8501 2>/dev/null | xargs kill -9 2>/dev/null
sleep 1
echo "✅ Cleanup complete"
echo ""

# Validate dashboard_app.py syntax
echo "🔍 Validating dashboard_app.py..."
python3 -c "import ast; ast.parse(open('dashboard_app.py').read())" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Error: dashboard_app.py has syntax errors"
    exit 1
fi
echo "✅ dashboard_app.py syntax valid"
echo ""

# Start Streamlit
echo "🌐 Starting Professional Dashboard..."
echo "   URL: http://localhost:8501"
echo "   Port: 8501"
echo ""
echo "   Press Ctrl+C to stop"
echo ""

streamlit run dashboard_app.py --server.port 8501 --server.headless true

