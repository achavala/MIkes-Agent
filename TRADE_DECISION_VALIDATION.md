# 🔍 Trade Decision Validation Guide

## 📊 How Trades Are Picked or Rejected

### Decision Flow (Step-by-Step)

```
1. MARKET DATA COLLECTION
   ↓
2. RL MODEL ANALYSIS
   → Outputs: Action (0=HOLD, 1=BUY CALL, 2=BUY PUT) + Confidence
   ↓
3. SAFEGUARD CHECKS (10 Layers)
   → Checks: Risk limits, cooldowns, position limits, etc.
   ↓
4. DECISION
   ├─ BLOCKED → ⛔ Telegram alert + Log reason
   └─ ALLOWED → ✅ Trade executed + Telegram alert
```

---

## 🤖 Step 1: RL Model Decision

The RL model analyzes market data and outputs:

- **Action 0 (HOLD)**: No trade signal
- **Action 1 (BUY CALL)**: Bullish signal
- **Action 2 (BUY PUT)**: Bearish signal
- **Confidence**: 0.0 to 1.0 (strength of signal)

**What to look for in logs:**
```
🔍 SPY RL Probs: ['0.45', '0.35', '0.20'] | Action=1 | Strength=0.65
```

---

## 🛡️ Step 2: Safeguard Checks (10 Layers)

### Safeguard 1: Daily Loss Limit (-15%)
- **Blocks if**: Daily PnL ≤ -15%
- **Action**: Closes all positions, shuts down
- **Log**: `🚨 SAFEGUARD 1 TRIGGERED: Daily loss limit hit`

### Safeguard 1.5: Hard Dollar Loss Limit (-$500)
- **Blocks if**: Daily loss > $500
- **Action**: Closes all positions, shuts down
- **Log**: `🚨 SAFEGUARD 1.5 TRIGGERED: Hard daily loss limit`

### Safeguard 2: Max Drawdown (-20%)
- **Blocks if**: Drawdown ≤ -20%
- **Action**: Closes all positions, shuts down
- **Log**: `🚨 SAFEGUARD 2 TRIGGERED: Max drawdown breached`

### Safeguard 3: VIX Kill Switch (VIX > 28)
- **Blocks if**: VIX > 28 (crash mode)
- **Log**: `⛔ BLOCKED: VIX 29.5 > 28 (crash mode)`

### Safeguard 4: Time Filter (After 3:50 PM EST)
- **Blocks if**: After cutoff time (if enabled)
- **Log**: `⛔ BLOCKED: After 15:50 EST (time filter)`

### Safeguard 5: Max Concurrent Positions (3)
- **Blocks if**: Already have 3 open positions
- **Log**: `⛔ BLOCKED: Max concurrent positions (3) reached | Current: 3/3`

### Safeguard 6: Max Daily Trades (20)
- **Blocks if**: Daily trades ≥ 20
- **Log**: `⛔ BLOCKED: Max daily trades (20) reached`

### Safeguard 7: Order Size Sanity Check
- **Blocks if**: Notional > $10,000 limit
- **Log**: `⛔ BLOCKED: Notional $12,000 > $10,000 limit`

### Safeguard 8: Max Position Size (Regime-Adjusted)
- **Blocks if**: Position would exceed regime limit (15-30% of equity)
- **Log**: `⛔ BLOCKED: Position would exceed 25% limit (NORMAL_VOL regime)`

### Safeguard 8.5: Max Trades Per Symbol (100)
- **Blocks if**: Symbol trade count ≥ 100
- **Log**: `⛔ BLOCKED: Max trades per symbol (100) reached for SPY`

### Safeguard 8.6: Global Trade Cooldown (5 seconds)
- **Blocks if**: Less than 5 seconds since last trade
- **Log**: `⛔ BLOCKED: Global trade cooldown active | 3s < 5s`

### Safeguard 8.7: Stop-Loss Cooldown (3 minutes)
- **Blocks if**: Re-entry within 3 minutes after stop-loss
- **Log**: `⛔ BLOCKED: Stop-loss cooldown active for SPY | 2 minute(s) remaining`

### Safeguard 8.8: Per-Symbol Cooldown (10 seconds)
- **Blocks if**: Re-entry on same symbol within 10 seconds
- **Log**: `⛔ BLOCKED: Per-symbol cooldown active for SPY | 5s remaining`

### Safeguard 8.9: Trailing-Stop Cooldown (60 seconds)
- **Blocks if**: Re-entry within 60 seconds after trailing-stop
- **Log**: `⛔ BLOCKED: Trailing-stop cooldown active for SPY | 30s remaining`

### Safeguard 9: Duplicate Order Protection (30 seconds)
- **Blocks if**: Same symbol order within 30 seconds
- **Log**: `⛔ BLOCKED: Duplicate order protection | 15s < 30s`

### Safeguard 10: Max Daily Trades (20)
- **Blocks if**: Daily trades ≥ 20
- **Log**: `⛔ BLOCKED: Max daily trades (20) reached`

---

## ✅ What Happens When Trade is ALLOWED

1. **Symbol Selection**: Best symbol chosen (SPY/QQQ/SPX)
2. **Strike Selection**: ATM strike calculated
3. **Position Sizing**: Risk-adjusted size calculated
4. **Order Execution**: Alpaca order submitted
5. **Telegram Alert**: 🟢 Entry alert sent
6. **Position Tracking**: Added to open_positions

**Log example:**
```
✅ TRADE_OPENED | symbol=SPY | option=SPY251210C00680000 | premium=$1.42 | qty=10
✅ NEW ENTRY: 10x SPY251210C00680000 @ $1.42 premium
```

---

## ⛔ What Happens When Trade is BLOCKED

1. **Reason Logged**: Specific safeguard that blocked
2. **Telegram Alert**: ⛔ Block alert sent (if significant)
3. **Agent Continues**: Waits for next iteration
4. **No Position Opened**: Trade skipped

**Log example:**
```
⛔ BLOCKED: SPY (SPY251210C00680000) | Reason: ⛔ BLOCKED: Max concurrent positions (3) reached | Current: 3/3
```

---

## 📋 How to Validate Trade Decisions

### Method 1: Check Logs
```bash
fly logs --app mike-agent-project | grep -E "(BLOCKED|TRADE_OPENED|RL.*action)"
```

### Method 2: Use Validation Script
```bash
python validate_trade_decisions.py
```

### Method 3: Monitor Telegram
- 🟢 Entry alerts = Trades picked
- ⛔ Block alerts = Trades rejected
- 🔴 Exit alerts = Positions closed

---

## 🔍 Common Scenarios

### Scenario 1: Market Closed
- **RL Actions**: None (market closed)
- **Blocked Trades**: None (no signals)
- **Executed Trades**: None (market closed)
- **Status**: ✅ Normal - agent waiting for market open

### Scenario 2: All Trades Blocked
- **RL Actions**: Many BUY signals
- **Blocked Trades**: All blocked (safeguards active)
- **Executed Trades**: None
- **Status**: ⚠️ Check which safeguard is blocking

### Scenario 3: Trades Executing
- **RL Actions**: BUY signals
- **Blocked Trades**: Some blocked (cooldowns, etc.)
- **Executed Trades**: Some executed
- **Status**: ✅ Normal - agent trading within limits

---

## 💡 Key Points

1. **RL Model decides WHAT to trade** (CALL/PUT/HOLD)
2. **Safeguards decide IF trade is allowed** (risk limits)
3. **Both must pass for trade to execute**
4. **Blocked trades are logged with reason**
5. **Telegram alerts show both picks and rejections**

---

## 🎯 What to Monitor

- **During Market Hours**: Look for RL actions and trade executions
- **Blocked Trades**: Check if safeguards are too restrictive
- **Telegram Alerts**: Real-time notifications of all decisions
- **Logs**: Detailed reasoning for each decision

---

## ✅ Validation Checklist

- [ ] RL model is outputting actions (0/1/2)
- [ ] Safeguards are checking trades
- [ ] Blocked trades show clear reasons
- [ ] Executed trades show entry details
- [ ] Telegram alerts are working
- [ ] Agent is running 24/7

---

**Status**: If market is closed, no trades will execute. Check again during market hours (9:30 AM - 4:00 PM ET).





