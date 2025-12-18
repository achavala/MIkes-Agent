#!/bin/bash
# 🚀 TRAIN MODEL WITH ALL 23 FEATURES
# This script trains a new model with EMA, MACD, VWAP, RSI, and all technical indicators

set -e  # Exit on error

echo "======================================================================"
echo "🚀 TRAINING MODEL WITH ALL 23 FEATURES"
echo "======================================================================"
echo ""
echo "Features included:"
echo "  • OHLCV (5)"
echo "  • VIX + VIX Delta (2)"
echo "  • Technical Indicators (11): EMA, VWAP, RSI, MACD, ATR, Candle patterns, Pullback, Breakout, Trend, Momentum"
echo "  • Greeks (4): Delta, Gamma, Theta, Vega"
echo ""
echo "Symbols: SPY, QQQ, IWM (all with 23 features)"
echo "Data Source: Massive API (1-minute bars, last 2 years)"
echo "Training: 5M timesteps, regime-balanced"
echo ""

# Load .env file if it exists
if [ -f .env ]; then
    echo "✅ Loading .env file..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check for required API keys
if [ -z "$MASSIVE_API_KEY" ] && [ -z "$POLYGON_API_KEY" ]; then
    echo "⚠️  WARNING: MASSIVE_API_KEY or POLYGON_API_KEY not set"
    echo "   Training will use enriched data (if available) or yfinance"
fi

# Detect Python command (python3 on macOS, python on some Linux)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ ERROR: Python not found. Please install Python 3.8+"
    exit 1
fi

echo "🎓 Starting training..."
echo "Using Python: $PYTHON_CMD"
echo ""

$PYTHON_CMD train_historical_model.py \
  --symbols SPY,QQQ,IWM \
  --start-date 2020-01-01 \
  --end-date 2025-12-17 \
  --timesteps 5000000 \
  --model-name mike_23feature_model \
  --use-greeks \
  --human-momentum \
  --regime-balanced \
  --data-source massive \
  --intraday-days 730 \
  --learning-rate 3e-5 \
  --ent-coef 0.08 \
  --gamma 0.92

echo ""
echo "======================================================================"
echo "✅ TRAINING COMPLETE!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Verify model: ls -lh models/mike_23feature_model.zip"
echo "  2. Update MODEL_PATH in mike_agent_live_safe.py"
echo "  3. Deploy: fly deploy --app mike-agent-project"
echo ""

