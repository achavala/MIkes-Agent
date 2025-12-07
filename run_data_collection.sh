#!/bin/bash
# Script to run historical data collection with proper environment setup

cd /Users/chavala/Mike-agent-project

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found!"
    echo "   Please create one first: python3 -m venv venv"
    exit 1
fi

# Install yfinance if not available
python3 -c "import yfinance" 2>/dev/null || {
    echo "📦 Installing yfinance..."
    pip install yfinance -q
    echo "✅ yfinance installed"
}

# Install other dependencies
python3 -c "import pandas" 2>/dev/null || pip install pandas -q
python3 -c "import numpy" 2>/dev/null || pip install numpy -q

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Starting Historical Data Collection"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run the collection script with provided arguments or defaults
if [ $# -eq 0 ]; then
    # Default: collect daily data for SPY, QQQ from 2002
    python3 collect_historical_data.py \
        --symbols SPY,QQQ \
        --start-date 2002-01-01 \
        --interval 1d
else
    # Use provided arguments
    python3 collect_historical_data.py "$@"
fi

