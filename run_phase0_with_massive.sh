#!/bin/bash
# Phase 0 Backtest Runner with .env loading
# This script loads .env file and runs the Phase 0 backtest

cd "$(dirname "$0")"

# Load .env file if it exists
if [ -f .env ]; then
    echo "📋 Loading .env file..."
    export $(grep -v '^#' .env | xargs)
    echo "✅ Environment variables loaded from .env"
else
    echo "⚠️  No .env file found"
fi

# Verify MASSIVE_API_KEY is set
if [ -z "$MASSIVE_API_KEY" ] && [ -z "$POLYGON_API_KEY" ]; then
    echo "⚠️  MASSIVE_API_KEY not set. Will use Alpaca API only."
else
    echo "✅ MASSIVE_API_KEY is set (will be used as Priority 2)"
fi

echo ""
echo "🚀 Starting Phase 0 backtest..."
echo ""

# Run the backtest
python3 phase0_backtest/run_phase0.py


