# Quick Start - Live Trading

## 🚀 3 Steps to Go Live

### Step 1: Get Alpaca Paper Keys (Free)

1. Go to: https://app.alpaca.markets/paper/dashboard
2. Sign up (free)
3. Go to API Keys section
4. Copy your API Key ID and Secret Key

### Step 2: Configure Keys

Edit `config.py`:
```python
ALPACA_KEY = "PKxxxxxxxxxxxxxxxxxx"
ALPACA_SECRET = "SKxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Step 3: Run

```bash
# Make sure model is trained first
python mike_rl_agent.py --train

# Start live trading (paper mode)
python mike_agent_live_alpaca.py
```

## ✅ That's It!

The agent will:
- Connect to Alpaca
- Load the RL model
- Start trading SPY 0DTE options
- Print every decision in real-time

## 📊 What You'll See

```
============================================================
MIKE AGENT v3 – RL EDITION – LIVE WITH ALPACA
============================================================
Mode: PAPER TRADING
Symbols: SPY, QQQ
Risk per trade: 7%
============================================================

✓ Connected to Alpaca (PAPER)
  Account Status: ACTIVE
  Buying Power: $100,000.00
Loading RL model from mike_rl_agent.zip...
✓ Model loaded successfully

🚀 Agent is now trading...
Press Ctrl+C to stop

[14:30:15] SPY: $450.25 | Action: 1 | Status: FLAT
  → Submitting BUY CALL order: 5x SPY241201C00450000 @ strike $450.00
  ✓ Order submitted: abc123-def456-ghi789

[14:31:20] SPY: $450.50 | Action: 0 | Status: CALL 5x SPY241201C00450000
```

## ⚠️ Important

1. **Paper trade first** - Test for at least 1 week
2. **Monitor closely** - Watch the first few trades
3. **Start small** - Use minimum capital initially
4. **Understand risks** - Options can lose 100%

## 🔄 Switch to Live

When ready (after 1+ week paper trading):

```bash
python mike_agent_live_alpaca.py --live
```

**WARNING**: This uses REAL MONEY!

## 🛑 Stop Trading

Press `Ctrl+C` to stop the agent safely.

---

**You're ready to print money.** 🚀

