# 🧠 COMPLETE TRADE DECISION LOGIC - DETAILED DOCUMENTATION

## 📋 TABLE OF CONTENTS

1. [System Overview](#system-overview)
2. [Data Flow Architecture](#data-flow-architecture)
3. [Trade Selection Process](#trade-selection-process)
4. [Trade Rejection Criteria](#trade-rejection-criteria)
5. [Safeguard Layers](#safeguard-layers)
6. [Examples](#examples)
7. [Function Call Hierarchy](#function-call-hierarchy)
8. [Complete Flowchart](#complete-flowchart)

---

## 🎯 SYSTEM OVERVIEW

**Mike Agent v3** is a Reinforcement Learning (RL) based trading system that:
- Uses a trained LSTM model (RecurrentPPO) to make trading decisions
- Trades 0DTE options on SPY and QQQ
- Implements 13+ layers of risk safeguards
- Executes trades automatically via Alpaca API
- Never deletes trade history (permanent database)

**Core Philosophy:**
- **Quality over Quantity**: Only trades with 65%+ confidence
- **Risk First**: Every trade must pass 13+ safeguard checks
- **Adaptive**: Position sizing and stops adjust to volatility regime
- **Preserve Capital**: Multiple circuit breakers prevent catastrophic losses

---

## 📊 DATA FLOW ARCHITECTURE

### Step 1: Market Data Collection

**Function:** `get_market_data(symbol, period="2d", interval="1m")`

**Process:**
1. Tries Massive API first (if available)
2. Falls back to yfinance
3. Returns DataFrame with OHLCV data
4. Minimum 20 bars required for RL inference

**Data Sources:**
- **Primary**: Massive API (real-time, paid)
- **Fallback**: yfinance (free, delayed)

**Example:**
```python
# For SPY
sym_hist = get_market_data("SPY", period="2d", interval="1m")
# Returns: DataFrame with columns [Open, High, Low, Close, Volume]
# Shape: (2880, 5) for 2 days of 1-minute bars
```

---

### Step 2: Observation Preparation

**Function:** `prepare_observation(hist_data, risk_mgr, symbol=sym)`

**Process:**
1. Takes last 20 bars (LOOKBACK=20)
2. Normalizes all features to [-1, 1] range
3. Creates observation tensor of shape (20, 23)

**Observation Features (23 total):**

**Price Features (4):**
- Open % change (normalized)
- High % change (normalized)
- Low % change (normalized)
- Close % change (normalized)

**Volume Features (1):**
- Volume (normalized by max volume)

**VIX Features (2):**
- VIX level (normalized: VIX/50)
- VIX delta (normalized)

**Technical Indicators (16):**
- EMA 9/20 difference (normalized)
- VWAP distance (normalized)
- RSI (scaled to [-1, 1])
- MACD histogram (normalized)
- ATR (normalized)
- Momentum features
- Trend strength
- Volatility features

**Output:**
- NumPy array shape: `(20, 23)`
- All values normalized to [-1, 1]
- Ready for RL model inference

**Example:**
```python
obs = prepare_observation(sym_hist, risk_mgr, symbol="SPY")
# Returns: np.array shape (20, 23)
# Values: All between -1.0 and 1.0
```

---

### Step 3: RL Model Inference

**Function:** `model.predict(obs)` or `model.policy.get_distribution(obs_tensor)`

**Process:**
1. **RecurrentPPO Models** (LSTM):
   - Uses `model.predict()` directly
   - Handles LSTM states internally
   - Returns: `(action, lstm_state)`
   - Action strength estimated: 0.65 for BUY, 0.50 for HOLD

2. **Standard PPO Models**:
   - Uses `model.policy.get_distribution()` for temperature calibration
   - Extracts logits from distribution
   - Applies temperature (0.7) for calibration
   - Softmax to get probabilities
   - Action = argmax(probabilities)
   - Action strength = probability of selected action

**Action Mapping:**
- `0` = HOLD (no trade)
- `1` = BUY CALL
- `2` = BUY PUT
- `3` = TRIM 50%
- `4` = TRIM 70%
- `5` = FULL EXIT

**Temperature Calibration:**
- Temperature = 0.7 (sweet spot for live inference)
- Makes model less confident (prevents overconfidence)
- Formula: `probs = softmax(logits / 0.7)`

**Example:**
```python
# RecurrentPPO
action, _ = model.predict(obs, deterministic=False)
# action = 1 (BUY CALL)
# action_strength = 0.65 (estimated)

# Standard PPO with temperature
probs = softmax(logits / 0.7)
# probs = [0.15, 0.70, 0.10, 0.03, 0.01, 0.01]
# action = 1 (argmax)
# action_strength = 0.70 (actual probability)
```

---

### Step 4: Ensemble Signal (Optional)

**Function:** `meta_router.route(data, vix, symbol, current_price, strike)`

**Process:**
1. Combines multiple trading agents:
   - Risk Agent (portfolio risk)
   - Macro Agent (market regime)
   - Volatility Agent (VIX-based)
   - Gamma Agent (options Greeks)
   - Trend Agent (momentum)
   - Reversal Agent (mean reversion)
   - RL Agent (your model)

2. Hierarchical weighting:
   - Risk Agent: 25%
   - Macro Agent: 20%
   - Volatility Agent: 15%
   - Gamma Agent: 15%
   - Trend Agent: 10%
   - Reversal Agent: 5%
   - RL Agent: 10%

3. Final decision:
   - Combines all signals
   - Returns: `(action, confidence, details)`

**Example:**
```python
ensemble_action, ensemble_confidence, details = meta_router.route(
    data=ensemble_data,
    vix=20.5,
    symbol="SPY",
    current_price=677.50,
    strike=678
)
# Returns: (1, 0.75, {'regime': 'normal', 'signals': {...}})
# action = 1 (BUY CALL)
# confidence = 0.75 (75%)
```

---

### Step 5: Signal Combination

**Function:** Combines RL + Ensemble signals

**Process:**
1. **If Ensemble Available:**
   - RL Weight: 40%
   - Ensemble Weight: 60%
   - Combines signals hierarchically
   - Uses higher confidence signal if one is weak

2. **If Ensemble Unavailable:**
   - Uses RL signal only
   - Action strength from RL

3. **Confidence Threshold Check:**
   - **MIN_ACTION_STRENGTH_THRESHOLD = 0.65**
   - If final confidence < 0.65 → **BLOCKED**
   - Only high-confidence signals execute

**Example:**
```python
# RL signal: action=1, strength=0.70
# Ensemble signal: action=1, confidence=0.80
# Combined: action=1, confidence=0.76 (weighted average)
# Result: ✅ PASS (0.76 > 0.65 threshold)

# RL signal: action=1, strength=0.50
# Ensemble signal: action=1, confidence=0.60
# Combined: action=1, confidence=0.56
# Result: ❌ BLOCKED (0.56 < 0.65 threshold)
```

---

### Step 6: Symbol Selection

**Function:** `choose_best_symbol_for_trade(iteration, symbol_actions, target_action, open_positions, risk_mgr)`

**Process:**
1. **Rotation for Fairness:**
   - Rotates priority: [SPY, QQQ] → [QQQ, SPY] → [SPY, QQQ]...
   - Ensures fair distribution across symbols

2. **Filter Candidates:**
   - Must have target action (1=BUY CALL, 2=BUY PUT)
   - Must not have existing position
   - Must not be in stop-loss cooldown (3 min)
   - Must not be in trailing-stop cooldown (60 sec)
   - Must pass portfolio Greek limits (if available)

3. **Select by Strength:**
   - Sorts candidates by action_strength (descending)
   - Picks strongest signal
   - Maintains rotation as tiebreaker

**Example:**
```python
# Available symbols: SPY, QQQ
# SPY: action=1, strength=0.75
# QQQ: action=1, strength=0.68
# Rotation: SPY first this iteration
# Result: ✅ SPY selected (strongest + rotation priority)
```

---

### Step 7: Safeguard Checks (13 Layers)

**Function:** `risk_mgr.check_safeguards(api)` + `risk_mgr.check_order_safety(symbol, qty, premium, api)`

**Safeguard 1: Daily Loss Limit**
- **Check:** `daily_pnl <= -15%`
- **Action:** Close all positions, shutdown
- **Example:** Equity drops from $100k to $85k → **BLOCKED**

**Safeguard 1.5: Hard Dollar Loss Limit**
- **Check:** `daily_pnl_dollar < -$500`
- **Action:** Trading halted for day
- **Example:** Lost $600 today → **BLOCKED**

**Safeguard 2: Max Drawdown**
- **Check:** `equity < 70% of peak`
- **Action:** Full shutdown
- **Example:** Peak was $100k, now $69k → **BLOCKED**

**Safeguard 3: VIX Kill Switch**
- **Check:** `VIX > 28`
- **Action:** No new entries
- **Example:** VIX = 30 (crash) → **BLOCKED**

**Safeguard 4: IV Rank Minimum**
- **Check:** `IV Rank < 30`
- **Action:** No trades
- **Example:** IV Rank = 25 → **BLOCKED**

**Safeguard 5: Time Filter**
- **Check:** `NO_TRADE_AFTER = None` (currently disabled)
- **Action:** No trades after time
- **Example:** Trading after 2:30 PM → **BLOCKED** (if enabled)

**Safeguard 6: Max Concurrent Positions**
- **Check:** `len(open_positions) >= 2`
- **Action:** No new entries
- **Example:** Already have 2 positions → **BLOCKED**

**Safeguard 7: Order Size Sanity**
- **Check:** `notional > $50,000`
- **Action:** Reject order
- **Example:** Trying to buy $60k notional → **BLOCKED**

**Safeguard 8: Max Position Size (Regime-Adjusted)**
- **Check:** `current_exposure + notional > max_notional`
- **Action:** Reject order
- **Regime Limits:**
  - Low Vol: 20% of equity
  - Normal Vol: 25% of equity
  - High Vol: 30% of equity
  - Crash Vol: 35% of equity
- **Example:** Equity=$100k, Normal Vol → Max $25k per position → **BLOCKED** if trying $30k

**Safeguard 8.5: Max Trades Per Symbol**
- **Check:** `symbol_trade_count >= 10`
- **Action:** Reject order
- **Example:** Already traded SPY 10 times today → **BLOCKED**

**Safeguard 8.6: Global Trade Cooldown**
- **Check:** `time_since_last_trade < 60 seconds`
- **Action:** Reject order
- **Example:** Last trade was 30 seconds ago → **BLOCKED**

**Safeguard 8.7: Stop-Loss Cooldown**
- **Check:** `time_since_stop_loss < 3 minutes`
- **Action:** Reject order
- **Example:** SPY hit stop-loss 2 minutes ago → **BLOCKED**

**Safeguard 8.8: Per-Symbol Cooldown**
- **Check:** `time_since_last_symbol_trade < 10 seconds`
- **Action:** Reject order
- **Example:** Last SPY trade was 5 seconds ago → **BLOCKED**

**Safeguard 8.9: Trailing-Stop Cooldown**
- **Check:** `time_since_trailing_stop < 60 seconds`
- **Action:** Reject order
- **Example:** Trailing stop triggered 30 seconds ago → **BLOCKED**

**Safeguard 9: Duplicate Order Protection**
- **Check:** `time_since_last_order < 5 minutes` (same symbol)
- **Action:** Reject order
- **Example:** Same option traded 3 minutes ago → **BLOCKED**

**Safeguard 10: Max Daily Trades**
- **Check:** `daily_trades >= 20`
- **Action:** Reject order
- **Example:** Already 20 trades today → **BLOCKED**

**Safeguard 11: Minimum Confidence Threshold**
- **Check:** `action_strength < 0.65`
- **Action:** Reject order
- **Example:** RL confidence = 0.60 → **BLOCKED**

---

### Step 8: Position Sizing

**Function:** `risk_mgr.calculate_max_contracts(api, strike, regime)`

**Process:**
1. **Regime-Based Risk:**
   - Low Vol: 7% risk per trade
   - Normal Vol: 10% risk per trade
   - High Vol: 12% risk per trade
   - Crash Vol: 15% risk per trade

2. **IV-Adjusted Sizing:**
   - Higher IV = smaller size
   - Lower IV = larger size
   - Formula: `size = base_size * (1 / IV_rank)`

3. **Greeks-Based Adjustment:**
   - Checks portfolio delta limit (20% of account)
   - Checks portfolio gamma limit
   - Checks portfolio vega limit (15% of account)
   - Reduces size if limits would be exceeded

4. **Final Size:**
   - Minimum: 1 contract
   - Maximum: Regime limit
   - Formula: `qty = min(available_capital * risk_pct / premium_cost, max_by_greeks)`

**Example:**
```python
# Equity: $100,000
# Regime: Normal Vol (10% risk)
# Premium: $2.50
# IV Rank: 50 (moderate)
# Available risk: $10,000
# Premium cost per contract: $250
# Base size: $10,000 / $250 = 40 contracts
# IV adjustment: 40 * (1 / 0.5) = 40 (no change)
# Greeks limit: 35 contracts (delta limit)
# Final size: min(40, 35) = 35 contracts
```

---

### Step 9: Order Execution

**Function:** `api.submit_order(symbol, qty, side, type, time_in_force)`

**Process:**
1. **Generate Option Symbol:**
   - Format: `{UNDERLYING}{YYMMDD}{C/P}{STRIKE*1000}`
   - Example: `SPY251216C00678000` (SPY, Dec 16 2025, Call, $678 strike)

2. **Submit Market Order:**
   - Side: 'buy' (entry)
   - Type: 'market'
   - Time in force: 'day'
   - Quantity: Calculated from position sizing

3. **Save to Database:**
   - All trades saved to SQLite database
   - Never deleted (INSERT OR IGNORE)
   - Includes: timestamp, symbol, P&L, regime, VIX

4. **Send Telegram Alert:**
   - Entry alert with details
   - Rate limited (5 min per symbol)

**Example:**
```python
api.submit_order(
    symbol="SPY251216C00678000",
    qty=35,
    side='buy',
    type='market',
    time_in_force='day'
)
# Result: Order executed at market price
# Saved to database
# Telegram alert sent
```

---

## 🚫 TRADE REJECTION CRITERIA

### Rejection Point 1: Market Data

**Rejection:** Insufficient data
- **Check:** `len(sym_hist) < 20`
- **Reason:** "Insufficient data for RL inference"
- **Action:** Skip symbol, try next

**Example:**
```
SPY: 15 bars available (need 20)
→ ⛔ BLOCKED: Insufficient data
→ Try QQQ instead
```

---

### Rejection Point 2: RL Inference

**Rejection:** RL action = HOLD
- **Check:** `rl_action == 0`
- **Reason:** "RL model says HOLD"
- **Action:** No trade, wait for next bar

**Example:**
```
RL Inference: action=0 (HOLD), strength=0.50
→ ⛔ BLOCKED: No trading signal
→ Wait for next minute
```

---

### Rejection Point 3: Confidence Threshold

**Rejection:** Action strength too low
- **Check:** `action_strength < 0.65`
- **Reason:** "Signal confidence too low"
- **Action:** Skip trade, wait for stronger signal

**Example:**
```
RL Inference: action=1 (BUY CALL), strength=0.60
→ ⛔ BLOCKED: Confidence 0.60 < 0.65 threshold
→ Wait for stronger signal
```

---

### Rejection Point 4: Symbol Selection

**Rejection:** No eligible symbols
- **Check:** All symbols filtered out
- **Reasons:**
  - Symbol has existing position
  - Symbol in stop-loss cooldown (3 min)
  - Symbol in trailing-stop cooldown (60 sec)
  - Symbol exceeds portfolio Greek limits

**Example:**
```
SPY: Has existing position → Filtered
QQQ: In stop-loss cooldown (2 min remaining) → Filtered
→ ⛔ BLOCKED: No eligible symbols
→ Wait for cooldown to expire
```

---

### Rejection Point 5: Safeguard Checks

**Rejection:** Any safeguard fails
- **Check:** 13 different safeguard checks
- **Action:** Trade rejected, reason logged

**Example:**
```
Trying to buy SPY call
→ Check 1: Daily loss = -14% ✅ PASS
→ Check 2: VIX = 25 ✅ PASS
→ Check 3: Max concurrent = 1 ✅ PASS
→ Check 4: Max trades per symbol = 8 ✅ PASS
→ Check 5: Global cooldown = 70s ✅ PASS
→ Check 6: Position size = $30k, limit = $25k
→ ⛔ BLOCKED: Position would exceed 25% limit
```

---

### Rejection Point 6: Order Safety

**Rejection:** Order-level checks fail
- **Check:** `check_order_safety()` returns False
- **Reasons:**
  - Notional > $50k
  - Would exceed position size limit
  - Max trades per symbol reached
  - Cooldown active
  - Duplicate order protection

**Example:**
```
Trying to buy 50 contracts @ $3.00 premium
→ Notional: 50 * $3.00 * 100 = $15,000 ✅ PASS
→ Position size check: $15k < $25k limit ✅ PASS
→ Max trades: 8 < 10 ✅ PASS
→ Cooldown: 70s > 60s ✅ PASS
→ Duplicate: Last order was 6 min ago ✅ PASS
→ ✅ ALL CHECKS PASS → Order executed
```

---

## 📊 COMPLETE EXAMPLE: TRADE EXECUTION FLOW

### Scenario: Market Open, SPY Rising

**Time: 9:35 AM EST**

**Step 1: Data Collection**
```
get_market_data("SPY", period="2d", interval="1m")
→ Returns: 2880 bars of OHLCV data
→ ✅ PASS
```

**Step 2: Observation Preparation**
```
prepare_observation(spy_hist, risk_mgr, symbol="SPY")
→ Shape: (20, 23)
→ Features normalized to [-1, 1]
→ ✅ PASS
```

**Step 3: RL Inference**
```
model.predict(obs)
→ Action: 1 (BUY CALL)
→ Strength: 0.72 (72% confidence)
→ ✅ PASS (0.72 > 0.65 threshold)
```

**Step 4: Ensemble Signal (Optional)**
```
meta_router.route(...)
→ Action: 1 (BUY CALL)
→ Confidence: 0.78
→ ✅ PASS
```

**Step 5: Signal Combination**
```
RL: 0.72 * 0.40 = 0.288
Ensemble: 0.78 * 0.60 = 0.468
Combined: 0.756 (75.6%)
→ ✅ PASS (0.756 > 0.65)
```

**Step 6: Symbol Selection**
```
choose_best_symbol_for_trade(...)
→ SPY: action=1, strength=0.756, no position, no cooldown
→ QQQ: action=0 (HOLD)
→ Selected: SPY
→ ✅ PASS
```

**Step 7: Safeguard Checks**
```
check_safeguards(api)
→ Daily loss: -2% ✅ PASS
→ Max drawdown: -5% ✅ PASS
→ VIX: 20.5 ✅ PASS
→ Max concurrent: 0 < 2 ✅ PASS
→ ✅ ALL SAFEGUARDS PASS
```

**Step 8: Position Sizing**
```
calculate_max_contracts(api, strike=678, regime="normal")
→ Equity: $100,000
→ Risk: 10% = $10,000
→ Premium: $2.50
→ Base size: $10,000 / $250 = 40 contracts
→ IV adjustment: 40 * (1/0.5) = 40
→ Greeks limit: 35 contracts
→ Final: min(40, 35) = 35 contracts
→ ✅ PASS
```

**Step 9: Order Safety Check**
```
check_order_safety("SPY251216C00678000", qty=35, premium=2.50, api)
→ Notional: $8,750 < $50k ✅ PASS
→ Position size: $8,750 < $25k ✅ PASS
→ Max trades: 3 < 10 ✅ PASS
→ Cooldown: 70s > 60s ✅ PASS
→ Duplicate: 6 min > 5 min ✅ PASS
→ ✅ ALL CHECKS PASS
```

**Step 10: Order Execution**
```
api.submit_order(
    symbol="SPY251216C00678000",
    qty=35,
    side='buy',
    type='market'
)
→ ✅ ORDER EXECUTED
→ Saved to database
→ Telegram alert sent
```

---

## 🚫 COMPLETE EXAMPLE: TRADE REJECTION FLOW

### Scenario: Low Confidence Signal

**Time: 10:15 AM EST**

**Step 1-3: Data & RL Inference**
```
✅ Data collected
✅ Observation prepared
✅ RL Inference: action=1, strength=0.58
```

**Step 4: Confidence Check**
```
action_strength = 0.58
MIN_ACTION_STRENGTH_THRESHOLD = 0.65
0.58 < 0.65
→ ⛔ BLOCKED: Confidence too low (0.58 < 0.65)
→ Log: "⛔ BLOCKED: Selected symbol SPY confidence too low"
→ Action: Skip trade, wait for next bar
```

---

### Scenario: Symbol in Cooldown

**Time: 10:20 AM EST**

**Step 1-6: All Checks Pass**
```
✅ Data collected
✅ RL Inference: action=1, strength=0.75
✅ Confidence check: 0.75 > 0.65 ✅
✅ Symbol selection: SPY selected
```

**Step 7: Symbol Cooldown Check**
```
SPY in symbol_stop_loss_cooldown
Time since stop-loss: 90 seconds
STOP_LOSS_COOLDOWN_MINUTES = 3 (180 seconds)
90 < 180
→ ⛔ BLOCKED: Stop-loss cooldown active (1.5 min remaining)
→ Log: "⛔ BLOCKED: Selected symbol SPY in stop-loss cooldown"
→ Action: Wait for cooldown to expire
```

---

### Scenario: Position Size Limit

**Time: 10:25 AM EST**

**Step 1-8: All Checks Pass**
```
✅ Data collected
✅ RL Inference: action=1, strength=0.80
✅ Confidence check: PASS
✅ Symbol selection: SPY selected
✅ Safeguards: ALL PASS
✅ Position sizing: 50 contracts calculated
```

**Step 9: Order Safety Check**
```
Current exposure: $20,000
New position: 50 * $3.00 * 100 = $15,000
Total: $35,000
Max notional (Normal Vol): $25,000
$35,000 > $25,000
→ ⛔ BLOCKED: Position would exceed 25% limit
→ Log: "⛔ BLOCKED: Position would exceed 25% limit"
→ Action: Reduce size or skip trade
```

---

## 🔄 FUNCTION CALL HIERARCHY

### Main Loop
```
run_safe_live_trading()
├── init_alpaca()
├── load_rl_model()
├── TradeDatabase() (if available)
└── while True:
    ├── get_market_data() [per symbol]
    ├── prepare_observation() [per symbol]
    ├── model.predict() [per symbol]
    ├── meta_router.route() [per symbol, optional]
    ├── choose_best_symbol_for_trade()
    ├── risk_mgr.check_safeguards()
    ├── risk_mgr.calculate_max_contracts()
    ├── risk_mgr.check_order_safety()
    ├── api.submit_order()
    ├── trade_db.save_trade()
    └── check_stop_losses() [every iteration]
```

### Per-Symbol Processing
```
For each symbol (SPY, QQQ):
├── get_market_data(symbol)
│   ├── massive_client.get_bars() [if available]
│   └── yfinance.Ticker().history() [fallback]
│
├── prepare_observation(hist_data, risk_mgr, symbol)
│   ├── Get last 20 bars
│   ├── Normalize OHLCV
│   ├── Calculate technical indicators
│   ├── Get VIX data
│   └── Return: np.array(20, 23)
│
├── model.predict(obs) [RL Inference]
│   ├── RecurrentPPO: model.predict() directly
│   └── Standard PPO: model.policy.get_distribution()
│       ├── Extract logits
│       ├── Apply temperature (0.7)
│       ├── Softmax
│       └── Argmax for action
│
└── meta_router.route() [Optional Ensemble]
    ├── Risk Agent.analyze()
    ├── Macro Agent.analyze()
    ├── Volatility Agent.analyze()
    ├── Gamma Agent.analyze()
    ├── Trend Agent.analyze()
    ├── Reversal Agent.analyze()
    └── Combine with hierarchical weights
```

### Symbol Selection
```
choose_best_symbol_for_trade()
├── Rotate symbol priority
├── For each symbol:
│   ├── Check if has target action
│   ├── Check if has existing position
│   ├── Check stop-loss cooldown
│   ├── Check trailing-stop cooldown
│   └── Check portfolio Greek limits
├── Filter candidates
└── Sort by strength, return strongest
```

### Safeguard Checks
```
risk_mgr.check_safeguards(api)
├── Get equity
├── Calculate daily P&L
├── Check daily loss limit (-15%)
├── Check hard dollar loss (-$500)
├── Check max drawdown (-30%)
├── Check VIX kill switch (>28)
├── Check IV rank minimum (>=30)
└── Return: (can_trade, reason)

risk_mgr.check_order_safety(symbol, qty, premium, api)
├── Check notional limit ($50k)
├── Check position size limit (regime-based)
├── Check max trades per symbol (10)
├── Check global cooldown (60s)
├── Check stop-loss cooldown (3 min) [if entry]
├── Check per-symbol cooldown (10s) [if entry]
├── Check trailing-stop cooldown (60s) [if entry]
├── Check duplicate order protection (5 min)
└── Check max daily trades (20)
```

### Position Sizing
```
risk_mgr.calculate_max_contracts(api, strike, regime)
├── Get equity
├── Get regime risk % (7-15%)
├── Calculate base size: equity * risk_pct / premium_cost
├── IV adjustment: base_size * (1 / IV_rank)
├── Greeks adjustment: min(size, greeks_limit)
└── Return: (max_contracts, available_notional)
```

---

## 📈 COMPLETE FLOWCHART

```
┌─────────────────────────────────────────────────────────────────┐
│                    START: Main Trading Loop                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ Get Market Data │
                    │ (SPY, QQQ)     │
                    └────────┬───────┘
                             │
                             ▼
              ┌───────────────────────────────┐
              │ For Each Symbol:              │
              │ 1. get_market_data()         │
              │ 2. prepare_observation()      │
              │ 3. model.predict() [RL]       │
              │ 4. meta_router.route() [Ens]  │
              │ 5. Combine signals            │
              └───────────────┬───────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Filter by Action │
                    │ (BUY CALL/PUT)   │
                    └────────┬─────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │ choose_best_symbol_for_trade()│
              │ - Rotation                   │
              │ - Filter positions           │
              │ - Filter cooldowns            │
              │ - Sort by strength           │
              └──────────────┬───────────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
            ┌──────────────┐   ┌──────────────┐
            │ Symbol Found │   │ No Symbol    │
            └──────┬───────┘   └──────┬───────┘
                   │                  │
                   │                  │
                   ▼                  │
        ┌──────────────────────┐     │
        │ Check Confidence      │     │
        │ >= 0.65?             │     │
        └──────┬───────────────┘     │
               │                     │
        ┌──────┴──────┐              │
        │             │              │
        ▼             ▼              │
    ┌──────┐    ┌──────────┐        │
    │ PASS │    │ BLOCKED  │        │
    └───┬──┘    └────┬─────┘        │
        │            │               │
        │            └───────────────┘
        │
        ▼
┌───────────────────────────────┐
│ check_safeguards()            │
│ - Daily loss limit            │
│ - Max drawdown                │
│ - VIX kill switch             │
│ - Max concurrent              │
└───────┬───────────────────────┘
        │
   ┌────┴────┐
   │         │
   ▼         ▼
┌──────┐ ┌──────────┐
│ PASS │ │ BLOCKED  │
└───┬──┘ └────┬─────┘
    │         │
    │         └──────────────┐
    │                        │
    ▼                        │
┌──────────────────────┐    │
│ calculate_max_        │    │
│ contracts()          │    │
│ - Regime risk        │    │
│ - IV adjustment      │    │
│ - Greeks limits      │    │
└───────┬──────────────┘    │
        │                    │
        ▼                    │
┌──────────────────────┐    │
│ check_order_safety()  │    │
│ - Notional limit      │    │
│ - Position size       │    │
│ - Max trades/symbol   │    │
│ - Cooldowns           │    │
│ - Duplicate protect   │    │
└───────┬──────────────┘    │
        │                    │
   ┌────┴────┐               │
   │         │               │
   ▼         ▼               │
┌──────┐ ┌──────────┐       │
│ PASS │ │ BLOCKED  │       │
└───┬──┘ └────┬─────┘       │
    │         │             │
    │         └─────────────┘
    │
    ▼
┌──────────────────────┐
│ api.submit_order()   │
│ - Generate symbol    │
│ - Submit market buy  │
└───────┬──────────────┘
        │
        ▼
┌──────────────────────┐
│ trade_db.save_trade()│
│ - Save to database   │
│ - Never deleted     │
└───────┬──────────────┘
        │
        ▼
┌──────────────────────┐
│ send_entry_alert()   │
│ - Telegram alert     │
│ - Rate limited       │
└───────┬──────────────┘
        │
        ▼
┌──────────────────────┐
│ check_stop_losses()  │
│ - Monitor positions  │
│ - Execute stops/TPs   │
└───────┬──────────────┘
        │
        ▼
┌──────────────────────┐
│ Sleep 10 seconds     │
│ Loop continues...    │
└──────────────────────┘
```

---

## 🎯 KEY DECISION POINTS

### Decision Point 1: RL Action
- **Input:** Market data (20 bars, 23 features)
- **Process:** LSTM model inference
- **Output:** Action (0-5), Strength (0.0-1.0)
- **Threshold:** Strength >= 0.65 to proceed

### Decision Point 2: Symbol Selection
- **Input:** All symbols with BUY signals
- **Process:** Filter by positions, cooldowns, limits
- **Output:** Best symbol or None
- **Criteria:** Strongest signal + rotation + no conflicts

### Decision Point 3: Safeguard Checks
- **Input:** Current account state
- **Process:** 13 safeguard checks
- **Output:** Can trade (True/False), Reason
- **Criteria:** All safeguards must pass

### Decision Point 4: Position Sizing
- **Input:** Equity, regime, premium, IV, Greeks
- **Process:** Calculate max contracts
- **Output:** Quantity (1-max)
- **Criteria:** Within risk limits + Greeks limits

### Decision Point 5: Order Safety
- **Input:** Symbol, quantity, premium
- **Process:** 10 order-level checks
- **Output:** Is safe (True/False), Reason
- **Criteria:** All checks must pass

---

## 📊 SUMMARY STATISTICS

**Total Safeguard Layers:** 13+
- 3 Global safeguards (daily loss, drawdown, VIX)
- 4 Position-level safeguards (size, concurrent, IV rank, time)
- 6 Order-level safeguards (notional, cooldowns, duplicates, limits)

**Confidence Threshold:** 0.65 (65%)
- Only high-confidence signals execute
- Filters out weak signals (0.50-0.64)
- Prevents overtrading

**Trade Limits:**
- Max 10 trades per symbol per day
- Max 20 trades total per day
- Max 2 concurrent positions
- 60 seconds minimum between trades

**Position Sizing:**
- Regime-based risk: 7-15% per trade
- IV-adjusted: Higher IV = smaller size
- Greeks-limited: Delta/Gamma/Vega caps

**Cooldowns:**
- Global: 60 seconds
- Per-symbol: 10 seconds
- Stop-loss: 3 minutes
- Trailing-stop: 60 seconds
- Duplicate: 5 minutes

---

## 🔍 DEBUGGING TIPS

**If trades aren't executing, check logs for:**
1. "⛔ BLOCKED" messages (shows rejection reason)
2. "🔍 RL Action" logs (shows confidence level)
3. "✅ Symbol selected" (confirms symbol selection)
4. "⛔ BLOCKED: Confidence too low" (below 0.65 threshold)

**Common rejection reasons:**
- Confidence < 0.65 (most common)
- Symbol in cooldown
- Max trades per symbol reached
- Position size limit exceeded
- Global cooldown active

---

## ✅ CONCLUSION

The system is designed to be **extremely selective**:
- Only trades high-confidence signals (65%+)
- Multiple layers of protection
- Conservative position sizing
- Extensive cooldown periods
- Complete trade history preservation

**Result:** Fewer trades, higher quality, better risk management.





