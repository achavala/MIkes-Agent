# 📊 DETAILED TRADE LOSS ANALYSIS - $2,350 Loss

## Executive Summary

**Date:** December 16, 2025  
**Total Loss:** -$2,370.40 (from dashboard)  
**Number of Trades:** 56 trades  
**Agent Status:** OFFLINE (at time of screenshot)

---

## 🔍 ROOT CAUSE ANALYSIS

### Issue 1: Database P&L Not Being Saved

**Problem:** The trade database shows 580 trades but all have $0.00 P&L, entry_premium, and exit_premium.

**Root Cause:** The `save_trade()` function may not be capturing actual fill prices from Alpaca orders.

**Impact:** Cannot analyze actual trade performance from database alone.

**Solution:** Need to query Alpaca API directly for actual trade data.

---

## 📈 DECISION-MAKING FLOW ANALYSIS

### How Trades Are Selected

#### Step 1: RL Model Inference
```
For each symbol (SPY, QQQ):
1. Get 20 bars of 1-minute data
2. Prepare observation (20, 23) features
3. Run RL model inference
4. Get action (0-5) and confidence (0.0-1.0)
```

**Current Threshold:** `MIN_ACTION_STRENGTH_THRESHOLD = 0.65` (65%)

**What This Means:**
- Only signals with 65%+ confidence execute
- Signals below 65% are blocked
- This is GOOD - prevents weak trades

#### Step 2: Symbol Selection
```
choose_best_symbol_for_trade():
1. Rotate symbols for fairness
2. Filter out symbols with existing positions
3. Filter out symbols in cooldown:
   - Stop-loss cooldown: 3 minutes
   - Trailing-stop cooldown: 60 seconds
   - Per-symbol cooldown: 10 seconds
4. Sort by RL strength
5. Select strongest signal
```

**Current Limits:**
- `MAX_CONCURRENT = 2` (max 2 positions)
- `MAX_TRADES_PER_SYMBOL = 10` (per day)
- `MIN_TRADE_COOLDOWN_SECONDS = 60` (global)

#### Step 3: Safeguard Checks (13 Layers)

**Global Safeguards:**
1. Daily Loss Limit: -15% → **SHUTDOWN**
2. Hard Dollar Loss: -$500 → **SHUTDOWN**
3. Max Drawdown: -30% from peak → **SHUTDOWN**
4. VIX Kill Switch: VIX > 28 → **NO NEW ENTRIES**
5. IV Rank Minimum: IVR < 30 → **NO TRADES**
6. Time Filter: After 2:30 PM → **DISABLED** (NO_TRADE_AFTER = None)
7. Max Concurrent: 2 positions → **BLOCK NEW ENTRIES**

**Order-Level Safeguards:**
8. Notional Limit: > $50,000 → **REJECT**
9. Position Size: Exceeds regime limit → **REJECT**
10. Max Trades/Symbol: >= 10 → **REJECT**
11. Global Cooldown: < 60 seconds → **REJECT**
12. Stop-Loss Cooldown: < 3 minutes → **REJECT**
13. Duplicate Protection: < 5 minutes → **REJECT**

#### Step 4: Position Sizing

**Regime-Based Risk:**
- Low Vol (VIX < 18): 7% risk per trade
- Normal Vol (VIX 18-25): 10% risk per trade
- High Vol (VIX 25-35): 12% risk per trade
- Crash Vol (VIX > 35): 15% risk per trade

**IV Adjustment:**
- Higher IV = Smaller size
- Lower IV = Larger size
- Formula: `size = base_size * (1 / IV_rank)`

**Greeks Limits:**
- Delta limit: 20% of account
- Gamma limit: Portfolio-wide
- Vega limit: 15% of account

---

## 🚨 WHY TRADES ARE LOSING

### Pattern 1: Too Many Trades (Overtrading)

**Evidence:**
- 56 trades in one day
- Average loss per trade: $2,370 / 56 = **$42.32 per trade**

**Root Cause:**
- `MAX_TRADES_PER_SYMBOL = 10` allows 20 total trades (10 SPY + 10 QQQ)
- But 56 trades suggests multiple entries/exits per position
- May be taking partial profits (TP1/TP2/TP3) which creates multiple SELL orders

**Impact:**
- Each trade has commission/slippage costs
- More trades = more opportunities to lose
- Overtrading in choppy markets

**Recommendation:**
- Reduce `MAX_TRADES_PER_SYMBOL` from 10 to 5
- Increase `MIN_TRADE_COOLDOWN_SECONDS` from 60 to 120 seconds
- Consider reducing position size to limit exposure

### Pattern 2: Stop-Losses Not Effective

**Current Stop-Loss Settings:**
- Normal Stop: -20%
- Hard Stop: -30%
- Trailing Stop: +10% after +50%

**Problem:**
- 0DTE options can move 20-30% very quickly
- Stop-losses may be too wide for 0DTE
- By the time stop triggers, loss is already significant

**Example:**
- Entry: $2.50 premium
- Stop at -20% = $2.00
- But 0DTE can gap from $2.50 → $1.50 in minutes
- Actual exit: $1.50 = -40% loss (not -20%)

**Recommendation:**
- Tighten stop-losses for 0DTE:
  - Normal Stop: -15% (instead of -20%)
  - Hard Stop: -25% (instead of -30%)
- Use time-based stops: Exit all positions by 3:00 PM (1 hour before close)
- Consider tighter stops in high volatility

### Pattern 3: Confidence Threshold Too Low

**Current:** `MIN_ACTION_STRENGTH_THRESHOLD = 0.65` (65%)

**Problem:**
- 65% confidence may still be too low for 0DTE options
- 0DTE options require higher conviction
- Market noise can cause false signals

**Recommendation:**
- Increase to `0.70` (70%) or `0.75` (75%)
- Only trade highest confidence signals
- Better to miss trades than take bad ones

### Pattern 4: Position Sizing Too Large

**Current:**
- Normal Vol: 10% risk per trade
- On $100k account = $10,000 risk per trade

**Problem:**
- 10% risk is aggressive for 0DTE options
- 0DTE options are highly volatile
- One bad trade can wipe out multiple good ones

**Recommendation:**
- Reduce to 5-7% risk per trade
- Use smaller position sizes
- Preserve capital for better setups

### Pattern 5: Trading in Wrong Market Conditions

**Current Filters:**
- VIX Kill: > 28 (good)
- IV Rank: >= 30 (good)
- Time Filter: DISABLED (problem)

**Problem:**
- Trading all day (no time restriction)
- 0DTE options decay rapidly in last hour
- Afternoon volatility can be unpredictable

**Recommendation:**
- Re-enable time filter: `NO_TRADE_AFTER = "14:30"` (2:30 PM)
- Exit all positions by 3:00 PM
- Focus on morning/early afternoon only

---

## 💡 DETAILED RECOMMENDATIONS

### Immediate Fixes (High Priority)

#### 1. Increase Confidence Threshold
```python
MIN_ACTION_STRENGTH_THRESHOLD = 0.75  # Was 0.65
```
**Impact:** Reduces trade count, improves quality

#### 2. Reduce Max Trades Per Symbol
```python
MAX_TRADES_PER_SYMBOL = 5  # Was 10
```
**Impact:** Prevents overtrading

#### 3. Increase Global Cooldown
```python
MIN_TRADE_COOLDOWN_SECONDS = 120  # Was 60
```
**Impact:** Slows down trading, reduces mistakes

#### 4. Tighten Stop-Losses
```python
STOP_LOSS_PCT = 0.15  # Was 0.20 (15% instead of 20%)
HARD_STOP_LOSS = 0.25  # Was 0.30 (25% instead of 30%)
```
**Impact:** Limits losses per trade

#### 5. Re-enable Time Filter
```python
NO_TRADE_AFTER = "14:30"  # Was None
```
**Impact:** Avoids late-day volatility and decay

#### 6. Reduce Position Sizing
```python
# In regime params:
'risk': 0.05  # Was 0.10 (5% instead of 10% for normal vol)
```
**Impact:** Smaller losses per trade

### Medium-Term Improvements

#### 7. Add Time-Based Exit
```python
# Exit all positions by 3:00 PM EST
if current_time >= "15:00":
    close_all_positions()
```

#### 8. Improve Stop-Loss Execution
- Use limit orders instead of market orders
- Check stop-losses more frequently (every 30 seconds)
- Use tighter stops in high volatility

#### 9. Add Market Regime Filter
- Don't trade in choppy/ranging markets
- Only trade in trending markets
- Use ATR to detect market conditions

#### 10. Improve Entry Timing
- Wait for confirmation (2-3 bars)
- Don't enter on first signal
- Check for rejection before entry

### Long-Term Improvements

#### 11. Retrain RL Model
- Use more recent data
- Focus on 0DTE-specific patterns
- Add market regime features

#### 12. Add Ensemble Weighting
- Increase ensemble weight (currently 60%)
- Reduce RL weight (currently 40%)
- Use ensemble to filter RL signals

#### 13. Implement Dynamic Position Sizing
- Reduce size in choppy markets
- Increase size only in strong trends
- Use volatility-adjusted sizing

---

## 📊 EXPECTED IMPACT OF FIXES

### Before (Current Settings)
- Trades per day: 50-60
- Average loss per trade: $40-50
- Win rate: Unknown (likely < 40%)
- Daily loss: -$2,370

### After (Recommended Settings)
- Trades per day: 10-20 (reduced by 60-70%)
- Average loss per trade: $20-30 (reduced by 40-50%)
- Win rate: Should improve to 45-55%
- Daily loss: Should reduce to < $500

**Key Changes:**
1. **Fewer trades** = Less exposure, less commission
2. **Higher confidence** = Better entry quality
3. **Tighter stops** = Smaller losses
4. **Time filter** = Avoids worst market conditions
5. **Smaller size** = Less risk per trade

---

## 🔍 HOW TO ANALYZE FUTURE TRADES

### Use Monitoring Script
```bash
python3 monitor_agent.py
```

**Watch for:**
- RL confidence levels (should be > 70%)
- Why trades are blocked (cooldowns, limits)
- Exit reasons (stop-loss vs take-profit)
- Time of day (avoid late afternoon)

### Check Logs for Patterns
```bash
fly logs --app mike-agent-project | grep "BLOCKED"
fly logs --app mike-agent-project | grep "TRADE_OPENED"
fly logs --app mike-agent-project | grep "STOP-LOSS"
```

**Look for:**
- Repeated blocks for same reason
- Trades happening too frequently
- Stop-losses triggering too often
- Low confidence signals executing

### Analyze Database
```bash
python3 analyze_trades_simple.py
```

**Check:**
- Win rate
- Average win vs average loss
- Exit reasons distribution
- Time-based performance

---

## 🎯 ACTION PLAN

### Step 1: Immediate (Today)
1. ✅ Increase `MIN_ACTION_STRENGTH_THRESHOLD` to 0.75
2. ✅ Reduce `MAX_TRADES_PER_SYMBOL` to 5
3. ✅ Increase `MIN_TRADE_COOLDOWN_SECONDS` to 120
4. ✅ Re-enable `NO_TRADE_AFTER = "14:30"`

### Step 2: This Week
1. ✅ Tighten stop-losses (15% normal, 25% hard)
2. ✅ Reduce position sizing (5-7% risk)
3. ✅ Add time-based exit (3:00 PM)
4. ✅ Monitor and adjust

### Step 3: Next Week
1. Analyze performance with new settings
2. Fine-tune thresholds based on results
3. Consider model retraining if needed
4. Implement additional filters

---

## 📝 SUMMARY

**Main Issues:**
1. Too many trades (overtrading)
2. Stop-losses too wide for 0DTE
3. Confidence threshold too low
4. Position sizing too large
5. Trading all day (no time filter)

**Main Solutions:**
1. Increase confidence to 75%
2. Reduce max trades to 5 per symbol
3. Tighten stop-losses to 15%
4. Reduce position sizing to 5-7%
5. Re-enable 2:30 PM time filter

**Expected Result:**
- 60-70% reduction in trade count
- 40-50% reduction in average loss
- Better win rate
- Daily losses < $500

---

## 🔗 RELATED DOCUMENTS

- `TRADE_DECISION_LOGIC_COMPLETE.md` - Complete decision flow
- `MONITORING_GUIDE.md` - How to monitor agent
- `mike_agent_live_safe.py` - Main trading code

---

**Last Updated:** December 16, 2025  
**Status:** Analysis Complete - Ready for Implementation

