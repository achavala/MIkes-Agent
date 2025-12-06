#!/bin/bash
# Quick Start Script for Live Trading

echo "=========================================="
echo "MIKE AGENT v3 - LIVE TRADING START"
echo "=========================================="
echo ""

# Check if venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Activating virtual environment..."
    source venv/bin/activate
fi

# Check config
echo "📋 Checking configuration..."
if grep -q "paper-api.alpaca.markets" config.py 2>/dev/null; then
    echo "⚠️  WARNING: Still using PAPER API URL!"
    echo "   Edit config.py and change to: https://api.alpaca.markets"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if database exists
if [ ! -f "trades_database.db" ]; then
    echo "📊 Creating trade database..."
    python -c "from trade_database import TradeDatabase; TradeDatabase()"
fi

# Start agent
echo "🚀 Starting agent..."
echo ""
python mike_agent_live_safe.py
