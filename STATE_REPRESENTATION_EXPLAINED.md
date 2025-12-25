# 🎯 STATE REPRESENTATION - Complete Visual Explanation

**Date:** December 21, 2025  
**Purpose:** Explain exactly how data flows from market → state → RL model → action

---

## 📊 COMPLETE DATA FLOW

### **Step 1: Fetch Market Data**

**Function:** `get_market_data("SPY", period="2d", interval="1m")`

**What It Returns:**
```python
# DataFrame with OHLCV data
DataFrame:
  Index: [Timestamp, Timestamp, ...]  # 2000+ bars (2 days of 1-minute data)
  Columns:
    - Open:   [677.50, 678.20, 678.80, ...]
    - High:   [678.00, 678.50, 679.20, ...]
    - Low:    [677.20, 678.00, 678.50, ...]
    - Close:  [677.80, 678.30, 678.90, ...]
    - Volume: [1000000, 1200000, 1100000, ...]

Shape: (N, 5) where N = ~2000 bars
```

**This is RAW market data - just prices and volume.**

---

### **Step 2: Prepare Observation (State)**

**Function:** `prepare_observation(hist, risk_mgr, symbol='SPY')`

**Process:**

#### **2.1: Extract Last 20 Bars**
```python
recent = hist.tail(20)  # Last 20 minutes of data
# Shape: (20, 5)
```

#### **2.2: Extract OHLCV Values**
```python
closes = [677.50, 678.20, 678.80, ..., 687.00]  # 20 values
highs  = [678.00, 678.50, 679.20, ..., 688.00]  # 20 values
lows   = [677.20, 678.00, 678.50, ..., 686.50]  # 20 values
opens  = [677.30, 678.10, 678.70, ..., 686.80]  # 20 values
vols   = [1000000, 1200000, 1100000, ..., 1500000]  # 20 values
```

#### **2.3: Normalize OHLCV to Percentage Changes**
```python
base = closes[0]  # First bar = 677.50 (base price)

# Normalize to % change from base
o = (opens - base) / base * 100.0
# Result: [0.00, 0.09, 0.18, ..., 1.37]  # % change from base

h = (highs - base) / base * 100.0
# Result: [0.07, 0.15, 0.25, ..., 1.55]

l = (lows - base) / base * 100.0
# Result: [-0.04, 0.07, 0.15, ..., 1.33]

c = (closes - base) / base * 100.0
# Result: [0.00, 0.10, 0.19, ..., 1.40]  # Strong upward trend!

v = vols / vols.max()  # Normalize volume
# Result: [0.67, 0.80, 0.73, ..., 1.00]
```

#### **2.4: Calculate VIX Features (2 features)**
```python
vix_value = risk_mgr.get_current_vix()  # e.g., 15.8

# Feature 5: VIX normalized
vix_norm = 15.8 / 50.0 = 0.316
# Replicated across all 20 timesteps:
vix_norm = [0.316, 0.316, 0.316, ..., 0.316]  # 20 values

# Feature 6: VIX delta (change from previous)
vix_delta_norm = 0.0  # Live: no history
# Replicated:
vix_delta_norm = [0.0, 0.0, 0.0, ..., 0.0]  # 20 values
```

#### **2.5: Calculate Technical Indicators (11 features)**

**Feature 7: EMA 9/20 Difference**
```python
ema9 = [677.8, 678.1, 678.5, ..., 686.2]   # 9-period EMA
ema20 = [677.5, 677.7, 677.9, ..., 685.8]  # 20-period EMA
ema_diff = tanh(((ema9 - ema20) / base * 100.0) / 0.5)
# Result: [0.12, 0.14, 0.16, ..., 0.18]  # Bullish (EMA9 > EMA20)
```

**Feature 8: VWAP Distance**
```python
vwap = [677.6, 677.9, 678.2, ..., 686.5]   # Volume-weighted average
vwap_dist = tanh(((closes - vwap) / base * 100.0) / 0.5)
# Result: [0.08, 0.10, 0.12, ..., 0.15]  # Price above VWAP (bullish)
```

**Feature 9: RSI Scaled**
```python
rsi = [58, 60, 62, ..., 65]  # RSI values (0-100)
rsi_scaled = (100 - (100 / (1 + rs)) - 50) / 50
# Result: [0.16, 0.20, 0.24, ..., 0.30]  # Moderately bullish
```

**Feature 10: MACD Histogram**
```python
macd = [0.2, 0.3, 0.4, ..., 0.8]  # MACD line
signal = [0.15, 0.20, 0.25, ..., 0.65]  # Signal line
macd_hist = tanh(((macd - signal) / base * 100.0) / 0.3)
# Result: [0.10, 0.12, 0.14, ..., 0.18]  # Positive MACD (bullish)
```

**Feature 11: ATR Scaled**
```python
atr = [0.5, 0.6, 0.7, ..., 0.9]  # Average True Range
atr_scaled = tanh(((atr / base) * 100.0) / 1.0)
# Result: [0.04, 0.05, 0.06, ..., 0.07]  # Low volatility
```

**Feature 12: Candle Body Ratio**
```python
body_ratio = abs(closes - opens) / (highs - lows)
# Result: [0.60, 0.65, 0.70, ..., 0.75]  # Strong candle bodies
```

**Feature 13: Candle Wick Ratio**
```python
wick_ratio = (highs - lows - abs(closes - opens)) / (highs - lows)
# Result: [0.40, 0.35, 0.30, ..., 0.25]  # Small wicks
```

**Feature 14: Pullback**
```python
roll_high = [677.5, 678.2, 678.8, ..., 688.0]  # Rolling high
pullback = tanh(((closes - roll_high) / roll_high * 100.0) / 0.5)
# Result: [-0.02, -0.01, 0.00, ..., 0.01]  # Slight pullback
```

**Feature 15: Breakout**
```python
prior_high = [677.0, 677.5, 678.0, ..., 686.0]  # Prior 10-bar high
breakout = tanh(((closes - prior_high) / atr) / 1.5)
# Result: [0.15, 0.18, 0.20, ..., 0.25]  # Breaking above prior high
```

**Feature 16: Trend Slope**
```python
slope = polyfit(range(20), closes, 1)[0]  # Linear trend slope
trend_slope = tanh(((slope / base) * 100.0) / 0.05)
# Result: [0.20, 0.22, 0.24, ..., 0.28]  # Strong upward trend (replicated)
```

**Feature 17: Momentum Burst**
```python
vol_z = (volume - volume.mean()) / volume.std()  # Volume z-score
impulse = abs(price_change) / base * 100.0
burst = tanh((vol_z * impulse) / 2.0)
# Result: [0.12, 0.14, 0.16, ..., 0.20]  # Moderate momentum
```

**Feature 18: Trend Strength**
```python
trend_strength = tanh((abs(ema_diff) + abs(macd_hist) + abs(vwap_dist)) / 1.5)
# Result: [0.40, 0.42, 0.44, ..., 0.48]  # Strong trend
```

#### **2.6: Calculate Greeks (4 features)**

**If no position:**
```python
delta = [0.0, 0.0, 0.0, ..., 0.0]  # 20 values
gamma = [0.0, 0.0, 0.0, ..., 0.0]  # 20 values
theta = [0.0, 0.0, 0.0, ..., 0.0]  # 20 values
vega  = [0.0, 0.0, 0.0, ..., 0.0]  # 20 values
```

**If position exists:**
```python
# Calculate Greeks for current position
greeks = calculate_greeks(S=closes[-1], K=strike, T=time_to_exp, ...)
delta = [0.5, 0.5, 0.5, ..., 0.5]  # Replicated across 20 timesteps
gamma = [0.02, 0.02, 0.02, ..., 0.02]
theta = [-0.1, -0.1, -0.1, ..., -0.1]
vega  = [0.05, 0.05, 0.05, ..., 0.05]
```

---

### **Step 3: Combine into State Matrix**

**Code (Line 3033-3054):**
```python
obs = np.column_stack([
    o, h, l, c, v,                    # 5 features: OHLCV
    vix_norm,                         # 1 feature: VIX
    vix_delta_norm,                   # 1 feature: VIX delta
    ema_diff,                         # 1 feature: EMA 9/20 diff
    vwap_dist,                        # 1 feature: VWAP distance
    rsi_scaled,                       # 1 feature: RSI
    macd_hist,                        # 1 feature: MACD histogram
    atr_scaled,                       # 1 feature: ATR
    body_ratio,                       # 1 feature: Candle body ratio
    wick_ratio,                       # 1 feature: Candle wick ratio
    pullback,                         # 1 feature: Pullback
    breakout,                         # 1 feature: Breakout
    trend_slope,                      # 1 feature: Trend slope
    burst,                            # 1 feature: Momentum burst
    trend_strength,                   # 1 feature: Trend strength
    greeks[:,0],                      # 1 feature: Delta
    greeks[:,1],                      # 1 feature: Gamma
    greeks[:,2],                      # 1 feature: Theta
    greeks[:,3],                      # 1 feature: Vega
]).astype(np.float32)
```

**Result:**
```python
# Shape: (20, 23)
# - 20 timesteps (last 20 bars)
# - 23 features per timestep

obs = np.array([
    # Timestep 0 (oldest bar)
    [0.00, 0.07, -0.04, 0.00, 0.67,  # OHLCV (5)
     0.316, 0.0,                      # VIX (2)
     0.12, 0.08, 0.16, 0.10, 0.04,   # Technical (5)
     0.60, 0.40, -0.02, 0.15,        # Technical (4)
     0.20, 0.12, 0.40,               # Technical (3)
     0.0, 0.0, 0.0, 0.0],            # Greeks (4)
    
    # Timestep 1
    [0.09, 0.15, 0.07, 0.10, 0.80,    # OHLCV
     0.316, 0.0,                      # VIX
     0.14, 0.10, 0.20, 0.12, 0.05,   # Technical
     0.65, 0.35, -0.01, 0.18,         # Technical
     0.22, 0.14, 0.42,                # Technical
     0.0, 0.0, 0.0, 0.0],             # Greeks
    
    # ... (18 more timesteps)
    
    # Timestep 19 (newest bar)
    [1.37, 1.55, 1.33, 1.40, 1.00,    # OHLCV
     0.316, 0.0,                      # VIX
     0.18, 0.15, 0.30, 0.18, 0.07,    # Technical
     0.75, 0.25, 0.01, 0.25,          # Technical
     0.28, 0.20, 0.48,                 # Technical
     0.0, 0.0, 0.0, 0.0]              # Greeks
], dtype=np.float32)
```

**This (20, 23) matrix is the STATE that the RL model sees!**

---

## 🎯 HOW RL MODEL USES STATE

### **Step 4: RL Model Inference**

**Input:**
```python
obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
# Shape: (1, 20, 23)
# - Batch size: 1
# - Sequence length: 20
# - Features: 23
```

**Process:**
```python
# Forward pass through policy network
action_dist = model.policy.get_distribution(obs_tensor)

# Get raw logits (before softmax)
logits = action_dist.distribution.logits
# Shape: (1, 6)
# Values: [0.2, 1.5, -0.3, -0.5, -0.8, -1.0]
#         HOLD, CALL, PUT, TRIM50, TRIM70, EXIT

# Apply temperature softmax
temperature = 0.7
probs = torch.softmax(logits / temperature, dim=-1)
# Shape: (1, 6)
# Values: [0.15, 0.55, 0.10, 0.08, 0.07, 0.05]

# Select action
rl_action = int(np.argmax(probs))  # Action 1 (BUY CALL)
action_strength = float(probs[0][rl_action])  # 0.55 (55% confidence)
```

**Output:**
- **Action:** 1 (BUY CALL)
- **Confidence:** 0.55 (55%)

---

## ⚠️ IMPORTANT CLARIFICATION

### **The Model Does NOT Predict Price**

**Common Misconception:**
- ❌ "Model predicts price"
- ❌ "Model gives price to reward model"

**Reality:**
- ✅ **Model predicts ACTIONS** (HOLD, BUY CALL, BUY PUT, etc.)
- ✅ **State (observation) is given to RL model** to make action decision
- ✅ **Reward is calculated during TRAINING only** (not live trading)

### **During Training:**
```
1. Fetch market data
2. Prepare state (20, 23)
3. RL model predicts action
4. Environment executes action
5. Calculate reward (based on PnL, Sharpe, win rate, drawdown)
6. Update model weights using reward
```

### **During Live Trading:**
```
1. Fetch market data
2. Prepare state (20, 23)
3. RL model predicts action
4. Execute action in real market
5. NO reward calculation (model already trained)
```

---

## 📊 VISUAL STATE REPRESENTATION

### **Complete State Matrix (20 × 23)**

```
Timestep | Feature 0-4 (OHLCV)        | Feature 5-6 (VIX) | Feature 7-17 (Technical)              | Feature 18-21 (Greeks)
---------|----------------------------|------------------|-------------------------------------|------------------------
    0    | [0.00, 0.07, -0.04, 0.00, 0.67] | [0.316, 0.0] | [0.12, 0.08, 0.16, 0.10, 0.04, ...] | [0.0, 0.0, 0.0, 0.0]
    1    | [0.09, 0.15, 0.07, 0.10, 0.80] | [0.316, 0.0] | [0.14, 0.10, 0.20, 0.12, 0.05, ...] | [0.0, 0.0, 0.0, 0.0]
    2    | [0.18, 0.25, 0.15, 0.19, 0.73] | [0.316, 0.0] | [0.16, 0.12, 0.24, 0.14, 0.06, ...] | [0.0, 0.0, 0.0, 0.0]
   ...   | ...                        | ...              | ...                                 | ...
   19    | [1.37, 1.55, 1.33, 1.40, 1.00] | [0.316, 0.0] | [0.18, 0.15, 0.30, 0.18, 0.07, ...] | [0.0, 0.0, 0.0, 0.0]
```

**Each row = 1 timestep (1 minute of market data)**  
**Each column = 1 feature (price, indicator, etc.)**

---

## 🔍 WHAT THE MODEL LEARNS

**The model learns patterns in the state matrix:**

1. **Price Patterns:**
   - Rising prices (c[0] = 0.00 → c[19] = 1.40) → BUY CALL
   - Falling prices → BUY PUT
   - Sideways → HOLD

2. **Technical Patterns:**
   - EMA9 > EMA20 (ema_diff > 0) → Bullish
   - Price > VWAP (vwap_dist > 0) → Bullish
   - RSI > 50 (rsi_scaled > 0) → Bullish
   - Positive MACD (macd_hist > 0) → Bullish

3. **Combined Signals:**
   - Multiple bullish indicators → High confidence BUY CALL
   - Mixed signals → Lower confidence
   - Bearish signals → BUY PUT or HOLD

---

## 📝 SUMMARY

**Data Flow:**
1. **Fetch:** `get_market_data()` → DataFrame (N, 5) with OHLCV
2. **Transform:** `prepare_observation()` → State matrix (20, 23)
3. **Predict:** RL model → Action (0-5) + Confidence (0.0-1.0)
4. **Execute:** Action → Trade in real market

**State Representation:**
- **Shape:** (20, 23)
- **20 timesteps:** Last 20 minutes of market data
- **23 features:** OHLCV (5) + VIX (2) + Technical (11) + Greeks (4)

**Key Points:**
- ✅ Model predicts **ACTIONS**, not prices
- ✅ State is **given to RL model** to make decisions
- ✅ Reward is calculated **during training only**
- ✅ Live trading uses **learned policy** (no reward calculation)

---

**This is the complete state representation!**


