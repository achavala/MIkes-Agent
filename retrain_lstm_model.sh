#!/bin/bash
# Retrain Model with LSTM Backbone
# This script retrains the RL model using RecurrentPPO (LSTM + action masking)

set -e  # Exit on error

echo "🔥 Starting LSTM Model Retraining..."
echo ""

# Check if RecurrentPPO is available
python3 -c "
try:
    from sb3_contrib import RecurrentPPO
    print('✅ RecurrentPPO is available')
except ImportError:
    print('❌ RecurrentPPO not available')
    print('   Install with: pip install sb3-contrib')
    exit(1)
" || exit 1

# Check if MASSIVE_API_KEY is set
if [ -z "$MASSIVE_API_KEY" ]; then
    echo "⚠️  MASSIVE_API_KEY not set in environment"
    echo "   Checking .env file..."
    if [ -f .env ]; then
        source .env
        if [ -z "$MASSIVE_API_KEY" ]; then
            echo "❌ MASSIVE_API_KEY not found in .env"
            echo "   Add MASSIVE_API_KEY=your_key to .env file"
            exit 1
        else
            echo "✅ MASSIVE_API_KEY loaded from .env"
        fi
    else
        echo "❌ .env file not found"
        echo "   Create .env file with: MASSIVE_API_KEY=your_key"
        exit 1
    fi
else
    echo "✅ MASSIVE_API_KEY is set"
fi

# Model name
MODEL_NAME="mike_momentum_model_v3_lstm"
TIMESTEPS=500000

echo ""
echo "📊 Training Configuration:"
echo "   Model Name: $MODEL_NAME"
echo "   Timesteps: $TIMESTEPS"
echo "   Architecture: RecurrentPPO (LSTM + Action Masking)"
echo "   Observation: 20×23 (human-momentum features)"
echo "   Symbols: SPY, QQQ, SPX"
echo "   Data Source: Massive/Polygon (1-minute bars)"
echo ""

# Create logs directory
mkdir -p logs/training

# Start training
echo "🚀 Starting training..."
echo "   This will take several hours (can run in background)"
echo "   Progress will be saved to: logs/training/${MODEL_NAME}_${TIMESTEPS}.log"
echo ""

python3 -u train_historical_model.py \
  --symbols SPY,QQQ,SPX \
  --timesteps $TIMESTEPS \
  --model-name $MODEL_NAME \
  --human-momentum \
  --data-source massive \
  --intraday-days 60 \
  --learning-rate 3e-5 \
  --ent-coef 0.08 \
  --gamma 0.92 \
  --n-steps 512 \
  2>&1 | tee "logs/training/${MODEL_NAME}_${TIMESTEPS}.log"

echo ""
echo "✅ Training complete!"
echo "   Model saved to: models/${MODEL_NAME}_${TIMESTEPS}.zip"
echo ""
echo "📋 Next Steps:"
echo "   1. Validate model: python3 validate_model.py --model models/${MODEL_NAME}_${TIMESTEPS}.zip"
echo "   2. Update MODEL_PATH in mike_agent_live_safe.py to use new model"
echo "   3. Test live agent with new LSTM model"





