# 🔍 HOW TO TRACE SPECIFIC TRADES

## Quick Guide to Understanding Trade Decisions

### Method 1: Use the Trace Script

```bash
python3 trace_trade_decisions.py
```

This will:
- Get recent trades from database
- Show complete decision flow for each trade
- Explain data sources and calculations

### Method 2: Check Logs for Specific Trade

```bash
# Get logs and search for specific symbol
fly logs --app mike-agent-project | grep "SPY251212C00682000"

# Or search for RL signals
fly logs --app mike-agent-project | grep "RL Action.*SPY"

# Or search for symbol selection
fly logs --app mike-agent-project | grep "SYMBOL SELECTION.*SPY"
```

### Method 3: Use Monitoring Script

```bash
# Run monitor and watch for specific trades
python3 monitor_agent.py | grep "SPY"
```

---

## What to Look For in Logs

### 1. Data Collection
```
Look for: "get_market_data" or "Insufficient data"
Shows: What data was collected, how many bars
```

### 2. RL Inference
```
Look for: "🔍 SPY RL Action=1, Strength=0.720"
Shows: 
- Action (0-5)
- Confidence level (0.0-1.0)
- Whether it passed threshold (>= 0.65)
```

### 3. Ensemble Signal
```
Look for: "🎯 SPY Ensemble: action=1, confidence=0.780"
Shows:
- Combined signal from 7 agents
- Final confidence
- Individual agent signals
```

### 4. Symbol Selection
```
Look for: "✅ Symbol selected: SPY" or "SYMBOL SELECTION"
Shows:
- Which symbol was chosen
- Why (strength, rotation)
- What other symbols were considered
```

### 5. Safeguard Checks
```
Look for: "⛔ BLOCKED" or "check_safeguards"
Shows:
- Which safeguard failed (if any)
- Why trade was rejected
```

### 6. Position Sizing
```
Look for: "calculate_max_contracts" or "Position size"
Shows:
- How many contracts calculated
- Regime-based sizing
- Greeks limits
```

### 7. Trade Execution
```
Look for: "TRADE_OPENED" or "NEW ENTRY"
Shows:
- Final trade details
- Entry premium
- Quantity
- Strike price
```

---

## Example: Tracing a Real Trade

### Step 1: Find the Trade
```bash
# Get recent trades
python3 analyze_trades_simple.py
```

Output:
```
Symbol: SPY251212C00682000
P&L: $-45.20 (-18.08%)
Entry: $2.50 → Exit: $2.05
Reason: stop_loss
Time: 10:35:22
```

### Step 2: Search Logs for Decision Process
```bash
# Search for RL signal around that time
fly logs --app mike-agent-project | grep -A 5 -B 5 "10:35.*SPY.*RL"
```

Expected output:
```
[10:35:20] 🔍 SPY Observation: shape=(20, 23), min=-1.0, max=1.0
[10:35:20] 🔍 SPY RL Action=1, Strength=0.720 (temperature-calibrated)
[10:35:21] 🎯 SPY Ensemble: action=1, confidence=0.780
[10:35:22] ✅ Symbol selected: SPY (strength=0.756)
[10:35:22] ✅ TRADE_OPENED | symbol=SPY | option=SPY251212C00682000
```

### Step 3: Understand the Decision

**Data Collected:**
- 2 days of 1-minute bars for SPY
- Last 20 bars used for RL inference
- Observation shape: (20, 23)

**RL Inference:**
- Action: 1 (BUY CALL)
- Strength: 0.72 (72% confidence)
- Status: ✅ PASS (0.72 > 0.65 threshold)

**Ensemble:**
- Combined confidence: 0.756 (75.6%)
- Multiple agents confirmed signal

**Selection:**
- SPY chosen (strongest signal)
- No existing positions
- No cooldowns active

**Safeguards:**
- All 13 checks passed
- Within all limits

**Execution:**
- 35 contracts @ $2.50
- Total cost: $8,750
- Trade executed

---

## Understanding Why Trades Lose

### Check Exit Reason
```bash
# Search for exit logs
fly logs --app mike-agent-project | grep "STOP-LOSS\|TAKE-PROFIT\|EXIT"
```

### Analyze Entry vs Exit
1. **Entry Premium:** $2.50
2. **Exit Premium:** $2.05
3. **Loss:** -18% (stop-loss triggered)
4. **Why:** Price moved against position quickly

### Check Market Conditions
```bash
# Search for VIX and regime
fly logs --app mike-agent-project | grep "VIX\|regime"
```

---

## Key Questions to Answer

For each trade, ask:

1. **What data was used?**
   - How many bars?
   - What time period?
   - Data quality?

2. **What was RL confidence?**
   - Above 65% threshold?
   - How strong was signal?

3. **Did ensemble confirm?**
   - Multiple agents agree?
   - Combined confidence?

4. **Why was symbol selected?**
   - Rotation priority?
   - Strongest signal?
   - No conflicts?

5. **Did all safeguards pass?**
   - Which checks were made?
   - Any close calls?

6. **How was size calculated?**
   - Regime-based risk?
   - IV adjustment?
   - Greeks limits?

7. **Why did it exit?**
   - Stop-loss?
   - Take-profit?
   - Time-based?

---

## Tools Available

1. **analyze_trades_simple.py** - Quick trade summary
2. **trace_trade_decisions.py** - Detailed decision flow
3. **monitor_agent.py** - Real-time activity
4. **TRADE_EXAMPLES_WITH_DATA.md** - Complete examples

---

**Use these tools to understand exactly why each trade was picked!**





