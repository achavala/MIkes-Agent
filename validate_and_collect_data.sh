#!/bin/bash
# Validate environment and collect historical data

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📥 HISTORICAL DATA COLLECTION - VALIDATION & SETUP"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✅ Found virtual environment"
    echo "   Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  No virtual environment found"
    echo "   Using system Python"
fi

echo ""
echo "🔍 Checking dependencies..."
python3 -c "import yfinance; print('✅ yfinance')" 2>/dev/null || echo "❌ yfinance - NOT FOUND"
python3 -c "import pandas; print('✅ pandas')" 2>/dev/null || echo "❌ pandas - NOT FOUND"
python3 -c "import numpy; print('✅ numpy')" 2>/dev/null || echo "❌ numpy - NOT FOUND"
python3 -c "from historical_training_system import HistoricalDataCollector; print('✅ historical_training_system')" 2>/dev/null || echo "❌ historical_training_system - NOT FOUND"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Starting Data Collection..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⚠️  This will take 8-24 hours for 20+ years of data"
echo "   You can stop and resume anytime (data is cached)"
echo ""
echo "Starting in 5 seconds... (Ctrl+C to cancel)"
sleep 5

# Run data collection
python3 collect_historical_data.py --symbols SPY,QQQ --start-date 2002-01-01

