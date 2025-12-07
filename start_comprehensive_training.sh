#!/bin/bash

# 🚀 COMPREHENSIVE TRAINING START SCRIPT
# Trains RL model on 23 years of historical data (2002-2025)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎓 COMPREHENSIVE RL TRAINING PIPELINE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This will:"
echo "  1. Collect historical data (SPY, QQQ, SPX since 2002)"
echo "  2. Prepare training dates (balanced by regime)"
echo "  3. Train RL model on all market conditions"
echo "  4. Save trained model"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✅ Virtual environment found. Activating..."
    source venv/bin/activate
else
    echo "⚠️  Virtual environment not found. Using system Python."
fi

# Check Python version
echo "🐍 Python version:"
python3 --version

# Check dependencies
echo ""
echo "📦 Checking dependencies..."
python3 -c "import pandas, numpy, yfinance, stable_baselines3, gymnasium" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Please install requirements:"
    echo "   pip install -r requirements.txt"
    exit 1
fi
echo "✅ All dependencies installed"

# Create directories
mkdir -p training_data
mkdir -p trained_models
mkdir -p logs

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 STARTING COMPREHENSIVE TRAINING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start training
python3 comprehensive_training_pipeline.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ TRAINING COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Trained model saved to: trained_models/"
echo ""
echo "Next steps:"
echo "  1. Validate model on test data"
echo "  2. Integrate into live trading"
echo "  3. Start with paper trading"
echo ""

