#!/bin/bash
# 🚀 RESUME TRAINING SCRIPT
# Continues training from the interrupted model (mike_23feature_model.zip)

set -e  # Exit on error

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ ERROR: Python not found. Please install Python 3.8+"
    exit 1
fi

echo "======================================================================"
echo "🔄 RESUMING TRAINING (From ~2.5M steps)"
echo "======================================================================"
echo "Loading model: models/mike_23feature_model.zip"
echo "Target: 2,500,000 additional steps (to reach ~5M total)"
echo ""

# Load .env file if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Run training (loading the existing model)
# Note: We use --model-path to load existing weights
# We set timesteps to 2500000 (remaining steps)

$PYTHON_CMD train_historical_model.py \
  --symbols SPY,QQQ,IWM \
  --start-date 2020-01-01 \
  --end-date 2025-12-17 \
  --timesteps 2500000 \
  --model-name mike_23feature_model_final \
  --load-model models/mike_23feature_model.zip \
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
echo "✅ TRAINING COMPLETED (Total ~5M steps)"
echo "======================================================================"
echo "Final model: models/mike_23feature_model_final.zip"
echo ""
echo "Next steps:"
echo "  1. Rename final model: mv models/mike_23feature_model_final.zip models/mike_23feature_model.zip"
echo "  2. Deploy as usual"





