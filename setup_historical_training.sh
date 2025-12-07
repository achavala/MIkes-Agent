#!/bin/bash
# Setup script for historical training - installs dependencies and validates environment

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 SETUP: Historical Training Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✅ Found virtual environment"
    echo "   Activating..."
    source venv/bin/activate
    echo "   ✅ Activated"
else
    echo "⚠️  No virtual environment found"
    read -p "   Create one? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        echo "   ✅ Created and activated"
    else
        echo "   Using system Python"
    fi
fi

echo ""
echo "📦 Installing/updating dependencies..."
echo ""

# Install core dependencies for historical training
pip install --upgrade pip 2>/dev/null || true

echo "   Installing yfinance..."
pip install yfinance -q

echo "   Installing pandas..."
pip install pandas -q

echo "   Installing numpy..."
pip install numpy -q

echo "   Installing scipy (for Black-Scholes)..."
pip install scipy -q

echo "   Installing stable-baselines3 (for RL)..."
pip install stable-baselines3[extra] -q

echo "   Installing gymnasium (for RL environment)..."
pip install gymnasium -q

echo ""
echo "✅ Dependencies installed"
echo ""
echo "🔍 Validating imports..."
echo ""

python3 << EOF
import sys
errors = []

try:
    import yfinance
    print("✅ yfinance")
except ImportError as e:
    print(f"❌ yfinance - {e}")
    errors.append("yfinance")

try:
    import pandas
    print("✅ pandas")
except ImportError as e:
    print(f"❌ pandas - {e}")
    errors.append("pandas")

try:
    import numpy
    print("✅ numpy")
except ImportError as e:
    print(f"❌ numpy - {e}")
    errors.append("numpy")

try:
    import scipy
    print("✅ scipy")
except ImportError as e:
    print(f"❌ scipy - {e}")
    errors.append("scipy")

try:
    from historical_training_system import HistoricalDataCollector
    print("✅ historical_training_system")
except ImportError as e:
    print(f"❌ historical_training_system - {e}")
    errors.append("historical_training_system")

if errors:
    print(f"\n⚠️  Missing dependencies: {', '.join(errors)}")
    sys.exit(1)
else:
    print("\n✅ All dependencies available!")
    sys.exit(0)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ SETUP COMPLETE - Ready for data collection!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Next steps:"
    echo "  1. Run: python collect_historical_data.py --symbols SPY,QQQ --start-date 2002-01-01"
    echo "  2. Wait 8-24 hours for data collection"
    echo "  3. Then train: python train_historical_model.py ..."
    echo ""
else
    echo ""
    echo "⚠️  Setup incomplete - please fix errors above"
fi

