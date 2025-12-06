#!/bin/bash
# Quick start script for paper trading
# Usage: ./start_paper_trading.sh

echo "============================================================"
echo "Mike Agent v3 - Paper Trading Quick Start"
echo "============================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Create one with: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
echo "🔍 Checking dependencies..."
if ! python -c "import alpaca_trade_api" 2>/dev/null; then
    echo "⚠️  Warning: alpaca-trade-api not found"
    echo "   Install with: pip install -r requirements.txt"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Test Alpaca connection
echo ""
echo "🔐 Testing Alpaca connection..."
python test_alpaca_connection.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Connection test failed!"
    echo "   Please check your API keys in config.py"
    echo "   Then run: python test_alpaca_connection.py"
    exit 1
fi

echo ""
echo "============================================================"
echo "✅ Ready to Start Paper Trading!"
echo "============================================================"
echo ""
echo "Options:"
echo ""
echo "1. Start Agent (Paper Trading):"
echo "   python mike_agent_live_safe.py"
echo ""
echo "2. Start Dashboard (Monitoring):"
echo "   streamlit run app.py"
echo ""
echo "3. Start Both (in separate terminals):"
echo "   Terminal 1: python mike_agent_live_safe.py"
echo "   Terminal 2: streamlit run app.py"
echo ""
echo "Press Ctrl+C in agent terminal to stop"
echo ""
read -p "Start agent now? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Starting Mike Agent..."
    echo "   Press Ctrl+C to stop"
    echo ""
    python mike_agent_live_safe.py
else
    echo ""
    echo "📋 To start later, run:"
    echo "   python mike_agent_live_safe.py"
    echo ""
fi


