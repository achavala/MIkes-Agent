#!/bin/bash
# 5-Day Backtest Test Run Script
# Quick validation run with behavioral profile

cd "$(dirname "$0")"

echo "======================================================================"
echo "  5-DAY BACKTEST TEST RUN"
echo "======================================================================"
echo ""

# Activate virtual environment if it exists
if [ -d "venv_validation" ]; then
    source venv_validation/bin/activate
    echo "✅ Virtual environment activated"
elif [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No virtual environment found. Using system Python."
fi

echo ""
echo "🚀 Running 5-day backtest..."
echo ""

# Run the Python script
python3 run_5day_test.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ Test run completed successfully!"
    echo "   Check logs/5day_test/ for detailed logs and results."
else
    echo ""
    echo "❌ Test run failed with exit code: $exit_code"
fi

exit $exit_code
