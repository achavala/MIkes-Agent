# Setup Instructions - Mike Agent v3 RL + Live Trading

## 🚀 Quick Setup (2 Minutes)

### Option 1: Use Setup Script (Recommended)

```bash
cd /Users/chavala/Mike-agent-project
./setup_rl_live.sh
```

### Option 2: Manual Setup

Run these commands **exactly as written** (quotes fix zsh bracket issues):

```bash
cd /Users/chavala/Mike-agent-project
source venv/bin/activate

# Install RL + trading dependencies (90 seconds)
pip install "stable-baselines3[extra]" gym==0.26.2 gymnasium alpaca-trade-api yfinance pandas-ta

# Optional: TensorFlow for Apple Silicon
pip install tensorflow-macos tensorflow-metal
```

## ✅ After Installation

### Step 1: Train the RL Model (2-5 minutes first time)

```bash
python mike_rl_agent.py --train
```

This will:
- Download SPY data (Nov 3 - Dec 1, 2025)
- Train PPO agent for 100,000 timesteps
- Save model to `mike_rl_agent.zip`

### Step 2: Launch Dashboard

```bash
./run.sh
```

Or directly:
```bash
streamlit run app.py
```

### Step 3: (Optional) Start Live Trading

```bash
# Make sure you've configured Alpaca keys in config.py first!
python mike_agent_live_alpaca.py
```

## 📦 What Gets Installed

- **stable-baselines3[extra]** - RL algorithms (PPO)
- **gym==0.26.2** - OpenAI Gym (environment)
- **gymnasium** - Modern Gym replacement
- **alpaca-trade-api** - Live trading API
- **yfinance** - Market data
- **pandas-ta** - Technical analysis
- **tensorflow-macos** (optional) - TensorFlow for Apple Silicon
- **tensorflow-metal** (optional) - Metal acceleration

## ⚠️ Important Notes

1. **Quotes are required** - `"stable-baselines3[extra]"` must be quoted for zsh
2. **Takes ~90 seconds** - Be patient during installation
3. **Train model first** - Must train before backtesting/live trading
4. **Alpaca keys needed** - For live trading, configure in `config.py`

## 🐛 Troubleshooting

### "Command not found: pip"
```bash
source venv/bin/activate
```

### "zsh: no matches found"
- Use quotes: `"stable-baselines3[extra]"`
- Or escape brackets: `stable-baselines3\[extra\]`

### "Module not found"
- Make sure venv is activated
- Re-run installation commands

### "Model not found"
- Train model first: `python mike_rl_agent.py --train`

## ✅ Verification

After setup, verify everything works:

```bash
python -c "from stable_baselines3 import PPO; print('✓ RL OK')"
python -c "import alpaca_trade_api; print('✓ Alpaca OK')"
python -c "from mike_rl_agent import MikeTradingEnv; print('✓ Agent OK')"
```

## 🎯 You're Ready!

Once setup completes:
1. ✅ Train model: `python mike_rl_agent.py --train`
2. ✅ Launch dashboard: `./run.sh`
3. ✅ Start trading: `python mike_agent_live_alpaca.py`

**You're 2 minutes from glory!** 🚀

