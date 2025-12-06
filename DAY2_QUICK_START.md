# 🚀 Day 2: Quick Start - Paper Trading

**Fast-track guide to get paper trading started in 5 minutes**

---

## ⚡ **5-MINUTE QUICK START**

### Step 1: Verify Alpaca Account (2 minutes)

1. Go to: https://app.alpaca.markets/paper/dashboard
2. Sign up (free) if you don't have an account
3. Get your API keys:
   - Navigate to **API Keys** section
   - Copy your **API Key ID** (starts with `PK...`)
   - Copy your **Secret Key** (starts with `SK...`)

### Step 2: Update config.py (1 minute)

```bash
# Edit config.py
nano config.py  # or use your preferred editor
```

Update these lines:
```python
ALPACA_KEY = 'YOUR_PAPER_API_KEY_ID'      # Replace with your PK...
ALPACA_SECRET = 'YOUR_PAPER_SECRET_KEY'   # Replace with your SK...
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'
```

### Step 3: Test Connection (1 minute)

```bash
# Activate virtual environment
source venv/bin/activate

# Test connection
python test_alpaca_connection.py
```

✅ If successful, you'll see account details.  
❌ If failed, check your API keys.

### Step 4: Start Paper Trading (1 minute)

**Option A: Use Quick Start Script**
```bash
./start_paper_trading.sh
```

**Option B: Manual Start**
```bash
source venv/bin/activate
python mike_agent_live_safe.py
```

### Step 5: Monitor (Ongoing)

**Start Dashboard (in new terminal):**
```bash
streamlit run app.py
```

Open in browser: http://localhost:8501

**Or check Alpaca Dashboard:**
https://app.alpaca.markets/paper/dashboard

---

## ✅ **VERIFICATION CHECKLIST**

Before starting, verify:

- [ ] Alpaca account created
- [ ] API keys copied
- [ ] config.py updated with keys
- [ ] Connection test passes
- [ ] Virtual environment activated
- [ ] Dependencies installed

---

## 📊 **WHAT TO EXPECT**

When you start the agent, you should see:

```
============================================================
MIKE AGENT v3 – RL EDITION – LIVE WITH ALPACA + 10X RISK SAFEGUARDS
============================================================
Mode: PAPER TRADING
Symbols: SPY, QQQ
Risk per trade: 7%
============================================================

✓ Connected to Alpaca (PAPER)
  Account Status: ACTIVE
  Equity: $100,000.00
  Buying Power: $100,000.00

[INFO] Agent started with full protection
[INFO] MAX POSITION SIZE: $25,000.00 (25% of $100,000.00 equity)
[INFO] 13/13 SAFEGUARDS: ACTIVE

🚀 Agent is now trading...
Press Ctrl+C to stop
```

---

## 📋 **NEXT STEPS**

1. **Let agent run** - It will monitor the market
2. **Check dashboard** - Watch for trades
3. **Review logs** - Monitor activity
4. **Track performance** - Use monitoring checklist

**Full setup guide:** See `DAY2_PAPER_TRADING_SETUP.md`  
**Monitoring guide:** See `MONITORING_CHECKLIST.md`

---

## 🆘 **TROUBLESHOOTING**

**Connection fails?**
- Verify API keys are correct
- Check internet connection
- Make sure you're using PAPER keys (not live)

**Agent won't start?**
- Check dependencies: `pip install -r requirements.txt`
- Verify virtual environment is activated
- Check error messages in output

**No trades executing?**
- Market might be closed (hours: 9:30 AM - 4:00 PM EST)
- All risk filters might be blocking trades
- Check logs for details

---

## 📚 **MORE INFORMATION**

- **Full Setup Guide:** `DAY2_PAPER_TRADING_SETUP.md`
- **Monitoring Checklist:** `MONITORING_CHECKLIST.md`
- **Project Status:** `PENDING_AND_NEXT_STEPS.md`

---

**You're ready to start paper trading! 🚀**


