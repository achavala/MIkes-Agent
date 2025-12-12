#!/bin/bash
# Start Streamlit Dashboard for Mike Agent

cd "$(dirname "$0")"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 STARTING MIKE AGENT DASHBOARD"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No virtual environment found, using system Python"
fi

# Kill any existing Streamlit processes on our ports
echo "🧹 Cleaning up existing Streamlit processes..."
lsof -ti:8501 -ti:8502 -ti:8503 2>/dev/null | xargs kill -9 2>/dev/null
sleep 1
echo "✅ Cleanup complete"
echo ""

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found"
    exit 1
fi

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "❌ Error: streamlit not installed"
    echo "   Installing streamlit..."
    pip install streamlit
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install streamlit"
        exit 1
    fi
fi

# Validate app.py syntax
echo "🔍 Validating app.py..."
python3 -c "import ast; ast.parse(open('app.py').read())" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Error: app.py has syntax errors"
    exit 1
fi
echo "✅ app.py syntax valid"
echo ""

# Start Streamlit
echo "🌐 Starting Streamlit dashboard..."
echo "   URL: http://localhost:8501"
echo "   Port: 8501"
echo ""
echo "   Press Ctrl+C to stop"
echo ""

streamlit run app.py --server.port 8501
