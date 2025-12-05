#!/bin/bash
echo "🔍 MIKE AGENT LIVE READINESS CHECK"
echo "=================================="
echo ""

# Check 1: RL Model
echo "1. RL Model:"
if [ -f "mike_rl_agent.zip" ] || [ -f "mike_rl_agent_v3.zip" ]; then
    echo "   ✅ RL model file found"
else
    echo "   ❌ RL model file missing"
fi

# Check 2: Config file
echo ""
echo "2. Configuration:"
if [ -f "config.py" ]; then
    echo "   ✅ config.py exists"
    if grep -q "ALPACA_KEY" config.py && ! grep -q "YOUR_PAPER_KEY" config.py; then
        echo "   ✅ API keys configured"
    else
        echo "   ⚠️  API keys may need configuration"
    fi
    if grep -q "paper-api.alpaca.markets" config.py; then
        echo "   ⚠️  Currently set to PAPER trading mode"
    fi
else
    echo "   ❌ config.py missing"
fi

# Check 3: Manual confirmation
echo ""
echo "3. Manual Intervention:"
if grep -q "input.*YES" mike_agent_live_safe.py; then
    echo "   ⚠️  Manual confirmation required (line 1690)"
    echo "      → Will block automated deployment"
else
    echo "   ✅ No manual confirmation required"
fi

# Check 4: Safeguards
echo ""
echo "4. Safeguards:"
if grep -q "FIXED_STOP_LOSS" mike_agent_live_safe.py; then
    echo "   ✅ Fixed stop-loss configured"
fi
if grep -q "MAX_CONCURRENT = 10" mike_agent_live_safe.py; then
    echo "   ✅ Max positions limit: 10"
fi
if grep -q "DAILY_LOSS_LIMIT" mike_agent_live_safe.py; then
    echo "   ✅ Daily loss limit active"
fi

# Check 5: Dependencies
echo ""
echo "5. Dependencies:"
python3 -c "import alpaca_trade_api; print('   ✅ alpaca-trade-api')" 2>/dev/null || echo "   ❌ alpaca-trade-api missing"
python3 -c "import stable_baselines3; print('   ✅ stable-baselines3')" 2>/dev/null || echo "   ❌ stable-baselines3 missing"
python3 -c "import yfinance; print('   ✅ yfinance')" 2>/dev/null || echo "   ❌ yfinance missing"

echo ""
echo "=================================="
