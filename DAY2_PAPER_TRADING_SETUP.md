# 📊 Day 2: Paper Trading Setup & Monitoring

**Date:** December 4, 2025  
**Goal:** Configure Alpaca paper account and begin monitoring/validation

---

## ✅ **PRE-FLIGHT CHECKLIST**

Before starting paper trading, verify:

- [ ] Git cleanup completed (Day 1) ✅
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Virtual environment activated
- [ ] RL model trained (if using RL version) OR using safe version
- [ ] `config.py` exists and is configured

---

## 🔧 **STEP 1: ALPACA PAPER ACCOUNT SETUP**

### 1.1 Create/Verify Alpaca Account

1. **Go to Alpaca Paper Trading Dashboard:**
   - URL: https://app.alpaca.markets/paper/dashboard
   - Sign up for free if you don't have an account

2. **Verify Account Status:**
   - Account should be **ACTIVE**
   - Paper account comes with $100,000 virtual capital

### 1.2 Get API Keys

1. **Navigate to API Keys:**
   - In Alpaca dashboard: **Your Account** → **API Keys**
   - Or direct link: https://app.alpaca.markets/paper/dashboard/overview

2. **Generate/Copy Keys:**
   - **API Key ID** (starts with `PK...`)
   - **Secret Key** (starts with `SK...`)
   - ⚠️ **Keep these secret!** Never commit to git.

### 1.3 Enable Options Trading

1. **Verify Options Access:**
   - Paper accounts typically have options enabled by default
   - Check: **Your Account** → **Settings** → **Trading Preferences**

2. **Note:**
   - Paper trading uses simulated market data
   - Orders execute instantly at simulated prices
   - Perfect for testing without risk

---

## 🔐 **STEP 2: CONFIGURE API KEYS**

### 2.1 Update config.py

**Current Status:** `config.py` exists but may need verification

**Check your current keys:**

```bash
# View config (don't show keys publicly)
grep "ALPACA_KEY" config.py
```

**Update if needed:**

```python
# Edit config.py
ALPACA_KEY = 'YOUR_PAPER_API_KEY_ID'      # Replace with your PK...
ALPACA_SECRET = 'YOUR_PAPER_SECRET_KEY'   # Replace with your SK...
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'  # Paper trading URL
MODE = 'paper'  # Ensure this is 'paper' for testing
```

### 2.2 Verify Keys Are Gitignored

✅ Already configured in `.gitignore`:
```
config.py
```

**Important:** Never commit `config.py` to git with real keys!

### 2.3 Test Connection (Optional)

You can test the connection before running the full agent:

```python
# Quick connection test (create test_connection.py)
import config
import alpaca_trade_api as tradeapi

api = tradeapi.REST(
    config.ALPACA_KEY,
    config.ALPACA_SECRET,
    config.ALPACA_BASE_URL,
    api_version='v2'
)

account = api.get_account()
print(f"✓ Connected to Alpaca")
print(f"  Status: {account.status}")
print(f"  Equity: ${float(account.equity):,.2f}")
print(f"  Buying Power: ${float(account.buying_power):,.2f}")
```

Run: `python test_connection.py`

---

## 🚀 **STEP 3: START PAPER TRADING**

### 3.1 Choose Your Agent Version

**Option A: Safe Version (Recommended for Day 2)**
- File: `mike_agent_live_safe.py`
- Features: 13 safeguards, volatility regimes, comprehensive risk management
- Best for: First-time paper trading validation

**Option B: RL Version**
- File: `mike_agent_live_alpaca.py`
- Features: Reinforcement learning model integration
- Requires: Trained model (`mike_rl_agent.zip`)

### 3.2 Start the Agent

**Using Safe Version (Recommended):**

```bash
# Activate virtual environment
source venv/bin/activate

# Start agent (paper mode by default)
python mike_agent_live_safe.py
```

**Expected Output:**
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
[INFO] STOP-LOSSES ACTIVE: Hard -30% | Normal -20%
[INFO] 13/13 SAFEGUARDS: ACTIVE

🚀 Agent is now trading...
Press Ctrl+C to stop

[09:30:15] SPY: $450.25 | Action: 0 | Equity: $100,000.00 | Status: FLAT
```

### 3.3 Running in Background (Optional)

**Using `nohup` (keeps running after terminal closes):**

```bash
nohup python mike_agent_live_safe.py > agent_output.log 2>&1 &
```

**Using `screen` or `tmux`:**

```bash
# Install screen if needed
brew install screen  # macOS

# Start screen session
screen -S mike_agent

# Run agent
python mike_agent_live_safe.py

# Detach: Ctrl+A, then D
# Reattach: screen -r mike_agent
```

---

## 📊 **STEP 4: MONITORING SETUP**

### 4.1 Dashboard Monitoring

**Start Streamlit Dashboard:**

```bash
# In a new terminal
source venv/bin/activate
streamlit run app.py
```

Dashboard opens at: **http://localhost:8501**

**Dashboard Features:**
- ✅ Real-time market status
- ✅ Current positions (from Alpaca)
- ✅ Trading history
- ✅ Portfolio summary
- ✅ Activity log
- ✅ Auto-refresh every 8 seconds

### 4.2 Alpaca Dashboard

**Monitor on Alpaca Website:**
- URL: https://app.alpaca.markets/paper/dashboard/overview
- View: Positions, Orders, Account Value, P&L

**What to Watch:**
- ✅ Orders executing correctly
- ✅ Position sizes match expectations
- ✅ Stop-losses triggering properly
- ✅ Take-profits executing
- ✅ No unexpected errors

### 4.3 Log Files

**Agent Logs:**

```bash
# Daily log files
logs/mike_agent_safe_YYYYMMDD.log

# Main log
mike.log

# Error log
mike_error.log
```

**Monitor Logs:**

```bash
# Watch log in real-time
tail -f logs/mike_agent_safe_$(date +%Y%m%d).log

# Or watch main log
tail -f mike.log
```

**What to Look For:**
- ✅ Successful order submissions
- ✅ Position updates
- ✅ Stop-loss triggers
- ✅ Take-profit executions
- ⚠️ Any error messages
- ⚠️ Risk limit warnings

### 4.4 CSV Trade History

**Trade Data:**

```bash
# Trade history CSV
mike_agent_trades.csv
mike_agent_trades_YYYYMMDD_HHMMSS.csv
```

**View Trades:**

```bash
# View recent trades
tail -20 mike_agent_trades.csv

# Or open in spreadsheet
open mike_agent_trades.csv  # macOS
```

---

## ✅ **STEP 5: VALIDATION CHECKLIST**

### Day 1 Validation (First Session)

**Connection:**
- [ ] Agent connects to Alpaca successfully
- [ ] Account status shows ACTIVE
- [ ] Buying power is correct ($100k paper)

**Trading:**
- [ ] Agent scans market correctly
- [ ] Orders submit without errors
- [ ] Positions appear in Alpaca dashboard
- [ ] Position sizes are reasonable (within limits)

**Risk Management:**
- [ ] Daily loss limit active (-15%)
- [ ] Max position size enforced (25% equity)
- [ ] Stop-losses working
- [ ] Time-of-day filter active (no trades after 2:30 PM)

**Monitoring:**
- [ ] Dashboard shows live data
- [ ] Log files updating correctly
- [ ] No error messages in logs
- [ ] Trade CSV updating

### Week 1 Validation (Daily Checks)

**Daily Monitoring:**
- [ ] Agent runs without crashing
- [ ] All safeguards functioning
- [ ] Trades executing as expected
- [ ] No unexpected behavior

**Performance Metrics to Track:**
- Daily P&L
- Win rate
- Average win/loss
- Max drawdown
- Position sizing accuracy
- Stop-loss execution rate
- Take-profit execution rate

**Document Observations:**
- Winning trade patterns
- Losing trade patterns
- Market conditions during trades
- Any edge cases encountered

---

## 📋 **STEP 6: DAILY MONITORING ROUTINE**

### Morning Routine (Before Market Open)

1. **Check Agent Status:**
   ```bash
   # Verify agent is running
   ps aux | grep mike_agent_live_safe
   
   # Or check logs
   tail -20 logs/mike_agent_safe_$(date +%Y%m%d).log
   ```

2. **Review Previous Day:**
   - Check Alpaca dashboard for final positions
   - Review trade history CSV
   - Note any issues from logs

3. **Start Dashboard:**
   ```bash
   streamlit run app.py
   ```

### During Market Hours

1. **Periodic Checks (Every 1-2 Hours):**
   - Monitor dashboard for new trades
   - Check Alpaca dashboard for positions
   - Review log files for errors
   - Verify risk limits not hit

2. **Key Times to Watch:**
   - **9:30 AM EST** - Market open (first trades)
   - **2:30 PM EST** - Trading window closes (no new entries)
   - **4:00 PM EST** - Market close

### End of Day Routine

1. **Review Day's Performance:**
   - Total P&L for the day
   - Number of trades
   - Win/loss ratio
   - Any significant events

2. **Check All Positions:**
   - Verify no unexpected open positions
   - Check stop-losses executed
   - Review take-profit executions

3. **Document:**
   - Log daily P&L
   - Note any issues
   - Record observations

4. **Prepare for Next Day:**
   - Agent should auto-restart next day
   - Or restart manually if needed

---

## 🎯 **STEP 7: MONITORING CHECKLIST**

### Real-Time Monitoring

**Dashboard (streamlit):**
- [ ] Market status indicator
- [ ] Current positions table
- [ ] Trading history
- [ ] Portfolio summary
- [ ] Activity log

**Alpaca Dashboard:**
- [ ] Positions tab
- [ ] Orders tab
- [ ] Account value
- [ ] Daily P&L

**Log Files:**
- [ ] No error messages
- [ ] Order submissions successful
- [ ] Position updates occurring
- [ ] Risk warnings (if any)

### Performance Tracking

**Daily Metrics:**
- Starting equity
- Ending equity
- Daily P&L ($ and %)
- Number of trades
- Win rate
- Average win/loss
- Max position size used

**Weekly Summary:**
- Total return for week
- Win rate
- Max drawdown
- Best/worst trade
- Issues encountered

---

## ⚠️ **STEP 8: TROUBLESHOOTING**

### Common Issues

**1. Connection Errors:**
```
Error: Failed to connect to Alpaca
```
**Solution:**
- Verify API keys in `config.py`
- Check internet connection
- Verify Alpaca service status

**2. Model Not Found:**
```
Error: mike_rl_agent.zip not found
```
**Solution:**
- Use `mike_agent_live_safe.py` (doesn't require RL model)
- Or train model first: `python mike_rl_agent.py --train`

**3. No Trades Executing:**
**Possible Reasons:**
- Market closed (check hours: 9:30 AM - 4:00 PM EST)
- All risk filters blocking trades
- No valid signals from strategy

**4. Position Sizes Too Large/Small:**
- Check risk percentage in config
- Verify equity calculation
- Review position sizing logic in logs

**5. Agent Crashes:**
- Check error log: `mike_error.log`
- Review last entries in daily log
- Restart agent and monitor

### Emergency Stops

**Stop Agent Immediately:**
```bash
# Find process
ps aux | grep mike_agent_live_safe

# Kill process
kill <PID>

# Or use Ctrl+C if running in foreground
```

**Manual Position Close (if needed):**
- Use Alpaca dashboard to close positions
- Or use Alpaca API directly

---

## 📈 **STEP 9: SUCCESS CRITERIA**

### Week 1 Goals

**Technical:**
- ✅ Agent runs 24/7 without crashes
- ✅ All safeguards functioning correctly
- ✅ Orders execute as expected
- ✅ No critical errors

**Performance:**
- Track daily P&L
- Monitor win rate
- Verify risk limits working
- Document all trades

### Week 2-4 Goals

**Validation:**
- Consistent performance metrics
- Understand trade patterns
- Verify all edge cases handled
- Build confidence in system

**Decision Point:**
After 2-4 weeks, assess:
- Is performance acceptable?
- Are all safeguards working?
- Any issues that need fixing?
- Ready for live trading? (NOT YET - more validation needed)

---

## 🔄 **STEP 10: NEXT STEPS AFTER DAY 2**

### Immediate (Today):
1. ✅ Configure Alpaca account
2. ✅ Update config.py with API keys
3. ✅ Start agent in paper mode
4. ✅ Set up monitoring
5. ✅ Run first validation checks

### This Week:
1. Monitor agent daily
2. Track all metrics
3. Document observations
4. Fix any issues that arise

### Next Week:
1. Continue daily monitoring
2. Analyze performance patterns
3. Review and optimize
4. Extended validation period

---

## 📝 **QUICK REFERENCE**

### Start Agent
```bash
source venv/bin/activate
python mike_agent_live_safe.py
```

### Start Dashboard
```bash
streamlit run app.py
```

### Monitor Logs
```bash
tail -f logs/mike_agent_safe_$(date +%Y%m%d).log
```

### Check Agent Status
```bash
ps aux | grep mike_agent
```

### Alpaca Dashboard
https://app.alpaca.markets/paper/dashboard

---

## ✅ **DAY 2 COMPLETE WHEN:**

- [ ] Alpaca account created/verified
- [ ] API keys configured in config.py
- [ ] Agent starts successfully
- [ ] Dashboard running and showing data
- [ ] First validation checklist completed
- [ ] Monitoring routine established

**Status:** 🚀 **READY TO START PAPER TRADING!**

---

*Next: Begin daily monitoring and validation for 1-4 weeks*


