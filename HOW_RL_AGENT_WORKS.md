# 🧠 How the RL Agent Works - Complete Explanation

## Overview

Your trading agent uses **PPO (Proximal Policy Optimization)**, a reinforcement learning algorithm that learns to trade 0DTE options by maximizing rewards (profit) while managing risk.

---

## 🔄 Complete Flow: From Market Data to Trading Action

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. DATA COLLECTION                                                │
│     • Fetch SPY/QQQ/SPX 1-minute bars                              │
│     • Last 20 bars (LOOKBACK window)                               │
│     • Features: Open, High, Low, Close, Volume                     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│  2. STATE PREPARATION (Observation Space)                          │
│                                                                     │
│     Input: Raw market data (20 bars × 5 features)                  │
│     Output: Observation tensor (1, 20, 5)                          │
│                                                                     │
│     Example:                                                       │
│       [Bar 1:  O=450.0, H=450.5, L=449.8, C=450.2, V=1000000]     │
│       [Bar 2:  O=450.2, H=450.8, L=450.1, C=450.6, V=1200000]     │
│       ...                                                          │
│       [Bar 20: O=452.0, H=452.5, L=451.8, C=452.3, V=1300000]     │
│                                                                     │
│     Location: `mike_agent_live_safe.py::prepare_observation()`     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│  3. PPO MODEL PROCESSING                                           │
│                                                                     │
│     ┌──────────────────────────────────────────────┐               │
│     │    SHARED BACKBONE (Feature Extraction)      │               │
│     │                                               │               │
│     │    Input:  (1, 20, 5) = 100 features         │               │
│     │      ↓                                         │               │
│     │    Flatten: (1, 100)                          │               │
│     │      ↓                                         │               │
│     │    Dense Layer 1: 64 neurons → ReLU          │               │
│     │      ↓                                         │               │
│     │    Dense Layer 2: 64 neurons → ReLU          │               │
│     │      ↓                                         │               │
│     │    Feature Vector: (1, 64)                    │               │
│     └──────────────────┬────────────────────────────┘               │
│                        │                                             │
│          ┌─────────────┴─────────────┐                              │
│          ↓                           ↓                              │
│     ┌──────────────┐         ┌──────────────┐                      │
│     │ ACTOR        │         │ CRITIC       │                      │
│     │ (Policy)     │         │ (Value)      │                      │
│     │              │         │              │                      │
│     │ Output:      │         │ Output:      │                      │
│     │ • Mean (μ)   │         │ • Value V(s) │                      │
│     │ • Std (σ)    │         │   (Expected  │                      │
│     │              │         │    return)   │                      │
│     └──────────────┘         └──────────────┘                      │
│                                                                     │
│     Location: `stable_baselines3.PPO`                              │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│  4. ACTION SAMPLING                                                │
│                                                                     │
│     Actor Output:                                                  │
│       • Mean (μ) = -0.3  ← Slight bearish bias                    │
│       • Std (σ) = 0.5   ← Exploration/randomness                  │
│                                                                     │
│     Sampling (Live Trading - Deterministic):                       │
│       action_raw = μ = -0.3  ← No randomness, use mean            │
│                                                                     │
│     Sampling (Training - Stochastic):                              │
│       action_raw ~ N(μ=-0.3, σ=0.5)  ← Sample from distribution   │
│       Example: action_raw = -0.45                                  │
│                                                                     │
│     Clip to [-1.0, 1.0]:                                          │
│       action_raw = -0.45  ← Final continuous value                │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│  5. ACTION MAPPING (Continuous → Discrete)                         │
│                                                                     │
│     Input: action_raw = -0.45 (continuous, -1.0 to +1.0)          │
│                                                                     │
│     Mapping Logic:                                                 │
│       if abs(action_raw) < 0.35:                                   │
│           action = 0  # HOLD (no conviction)                       │
│       elif action_raw > 0:                                         │
│           action = 1  # BUY CALL (bullish)                         │
│       else:                                                        │
│           action = 2  # BUY PUT (bearish)                          │
│                                                                     │
│     If position exists AND action_raw >= 0.5:                      │
│       if action_raw < 0.75:                                        │
│           action = 3  # TRIM 50%                                   │
│       elif action_raw < 0.9:                                       │
│           action = 4  # TRIM 70%                                   │
│       else:                                                        │
│           action = 5  # FULL EXIT                                  │
│                                                                     │
│     Example: action_raw = -0.45                                    │
│       → abs(-0.45) = 0.45 > 0.35  ✓ Not HOLD                      │
│       → -0.45 < 0  ✓ Negative                                     │
│       → action = 2  # BUY PUT                                      │
│                                                                     │
│     Location: `mike_agent_live_safe.py` lines 1337-1352           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│  6. RISK CHECKS & CONSTRAINTS                                      │
│                                                                     │
│     Before executing action:                                       │
│       ✓ Check daily loss limit (-15%)                              │
│       ✓ Check position size limits                                 │
│       ✓ Check VIX kill switch (>28)                                │
│       ✓ Check max concurrent positions                             │
│       ✓ Check gap-based override (first 60 min)                    │
│                                                                     │
│     Location: `mike_agent_live_safe.py::run_safe_live_trading()`   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│  7. TRADE EXECUTION                                                │
│                                                                     │
│     Example: action = 2 (BUY PUT)                                  │
│                                                                     │
│     Steps:                                                          │
│       1. Get current SPY price: $452.30                            │
│       2. Find ATM strike: $452 (round down)                        │
│       3. Calculate position size:                                  │
│          • Risk: 10% of equity = $10,000                           │
│          • Premium: $0.45                                          │
│          • Contracts: $10,000 / ($0.45 × 100) = 22 contracts      │
│       4. Get option symbol: SPY251205P00452000                      │
│       5. Submit order to Alpaca:                                   │
│          • Symbol: SPY251205P00452000                              │
│          • Quantity: 22                                            │
│          • Side: BUY                                               │
│          • Type: MARKET                                            │
│       6. Track in risk_mgr.open_positions                          │
│                                                                     │
│     Location: `mike_agent_live_safe.py` lines 1421-1530           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│  8. POSITION MONITORING (Every 30-60 seconds)                      │
│                                                                     │
│     For each open position:                                        │
│                                                                     │
│     A. Stop Losses (Highest Priority):                             │
│        • Absolute -15% stop → FORCED FULL EXIT                     │
│        • Hard stop -35% → Exit remaining                            │
│        • Normal stop -20% → Damage control (sell 50%)              │
│                                                                     │
│     B. Take Profits (Sequential, One Per Tick):                    │
│        • TP1: +40% → Sell 50%                                      │
│        • TP2: +80% → Sell 60% of remaining                         │
│        • TP3: +150% → Full exit                                    │
│                                                                     │
│     C. Trailing Stops (After TP1/TP2):                             │
│        • After TP1: Trail at TP1 - 20% = +20%                      │
│          → Sell 80% of remaining, keep 20% as runner              │
│        • Runner exits at EOD or -15% stop                          │
│                                                                     │
│     Location: `mike_agent_live_safe.py::check_stop_losses()`      │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│  9. NEXT ITERATION (Loop Continues)                                │
│                                                                     │
│     Wait 30-60 seconds → Fetch new market data → Repeat            │
└─────────────────────────────────────────────────────────────────────┘

```

---

## 📊 Key Components

### 1. **Observation Space** (What the model sees)

**Shape:** `(1, 20, 5)`
- **20 bars**: Last 20 minutes of market data
- **5 features**: Open, High, Low, Close, Volume

**Location:** `mike_agent_live_safe.py::prepare_observation()` (line 1020)

**Code:**
```python
def prepare_observation(data: pd.DataFrame, risk_mgr: RiskManager, symbol: str = 'SPY') -> np.ndarray:
    # Get last 20 bars
    recent = data.tail(LOOKBACK)  # LOOKBACK = 20
    
    # Extract OHLCV features
    obs_data = recent[['open', 'high', 'low', 'close', 'volume']]
    
    # Normalize volume
    if obs_data['volume'].max() > 0:
        obs_data['volume'] = obs_data['volume'] / obs_data['volume'].max()
    
    # Convert to numpy array
    state = obs_data.values.astype(np.float32)  # Shape: (20, 5)
    state = state.reshape(1, 20, 5)  # Add batch dimension
    
    return state  # Final shape: (1, 20, 5)
```

---

### 2. **PPO Model Architecture**

**Algorithm:** Proximal Policy Optimization (PPO)

**Network Structure:**
```
Input: (1, 20, 5) → Flatten to (1, 100)
    ↓
Dense Layer 1: 64 neurons → ReLU
    ↓
Dense Layer 2: 64 neurons → ReLU
    ↓
    ├─→ ACTOR (Policy Network)
    │     Output: Mean (μ), Std (σ)
    │
    └─→ CRITIC (Value Network)
          Output: Value V(s)
```

**Actor (Policy):**
- Outputs action distribution: `μ` (mean) and `σ` (standard deviation)
- Used to sample actions
- Higher `σ` = more exploration/randomness

**Critic (Value):**
- Estimates state value `V(s)` = expected future return
- Used during training to reduce variance

**Location:** `mike_agent_live_safe.py::load_rl_model()` (line 459)

**Code:**
```python
def load_rl_model():
    model = PPO.load(MODEL_PATH)
    return model
```

---

### 3. **Action Space**

**Raw Output:** Continuous value from `-1.0` to `+1.0`

**Mapped Actions:**
- `0`: HOLD
- `1`: BUY CALL
- `2`: BUY PUT
- `3`: TRIM 50%
- `4`: TRIM 70%
- `5`: FULL EXIT

**Mapping Logic:**
```python
# Extract raw action value
action_value = float(action_raw[0])  # e.g., -0.45

# Map to discrete actions
if abs(action_value) < 0.35:
    action = 0  # HOLD (near zero = no conviction)
elif action_value > 0:
    action = 1  # Positive → BUY CALL
else:
    action = 2  # Negative → BUY PUT

# If position exists and action is strong (>= 0.5):
if action_value >= 0.5 and risk_mgr.open_positions:
    if action_value < 0.75:
        action = 3  # TRIM 50%
    elif action_value < 0.9:
        action = 4  # TRIM 70%
    else:
        action = 5  # FULL EXIT
```

**Location:** `mike_agent_live_safe.py` lines 1337-1352

---

### 4. **Reward Function** (Training Only)

**During training**, the model learns by maximizing rewards:

```python
# Simplified reward (in mike_rl_agent.py)
reward = action[0] * 0.001  # Very basic

# Enhanced reward (in historical_training_system.py)
reward = (
    realized_pnl / capital * 10 +      # Profit reward
    sharpe_ratio * 0.1 +                # Risk-adjusted return
    win_rate * 0.2 +                    # Win rate bonus
    -drawdown * 0.5                     # Drawdown penalty
)
```

**Reward Components:**
- **Realized P&L**: Profit from closed trades
- **Sharpe Ratio**: Risk-adjusted return
- **Win Rate**: Percentage of winning trades
- **Drawdown**: Maximum loss from peak

**Location:** 
- Simple: `mike_rl_agent.py::MikeTradingEnv.step()` (line 36)
- Enhanced: `historical_training_system.py::HistoricalTradingEnv._calculate_reward()` (line 751)

---

### 5. **Training Process**

**Training File:** `train_historical_model.py`

**Steps:**
1. **Load historical data** (SPY, QQQ, SPX from 2002-present)
2. **Create environment** (`HistoricalTradingEnv`)
3. **Train PPO model** for 5,000,000 timesteps
4. **Save model** to `models/mike_rl_model.zip`

**Code:**
```python
# Create environment
env = HistoricalTradingEnv(
    data=historical_data,
    vix_data=vix_data,
    symbol='SPY',
    window_size=20,
    use_greeks=True
)

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64
)

# Train
model.learn(total_timesteps=5000000)

# Save
model.save("models/mike_rl_model.zip")
```

**Location:** `train_historical_model.py`

---

### 6. **Live Trading Loop**

**Main Loop:** `mike_agent_live_safe.py::run_safe_live_trading()`

**Steps (Every 30-60 seconds):**
1. Fetch latest market data (SPY, QQQ, SPX)
2. Prepare observation (last 20 bars)
3. Get RL action: `action, _ = model.predict(obs, deterministic=True)`
4. Map continuous action to discrete action
5. Apply risk checks and constraints
6. Execute trade (if action != 0 and checks pass)
7. Monitor positions for stop-loss/take-profit
8. Repeat

**Code:**
```python
while True:
    # 1. Fetch market data
    hist = fetch_market_data(symbol='SPY', interval='1m')
    
    # 2. Prepare observation
    obs = prepare_observation(hist, risk_mgr)
    
    # 3. Get RL action
    action_raw, _ = model.predict(obs, deterministic=True)
    
    # 4. Map to discrete action
    action = map_action(action_raw)
    
    # 5. Execute if valid
    if action == 1:  # BUY CALL
        execute_buy_call(...)
    elif action == 2:  # BUY PUT
        execute_buy_put(...)
    
    # 6. Check stop-losses/take-profits
    check_stop_losses(api, risk_mgr, current_price)
    
    # 7. Wait before next iteration
    time.sleep(30)
```

**Location:** `mike_agent_live_safe.py::run_safe_live_trading()` (line 1131)

---

## 🎯 Key Concepts

### **Entropy & Randomness**

**Entropy** controls exploration vs. exploitation:
- **High entropy** (large `σ`): More random actions → explores more
- **Low entropy** (small `σ`): More deterministic → exploits learned patterns

**In Live Trading:**
- `deterministic=True` → Use mean (`μ`) only, no randomness
- Ensures consistent, reproducible actions

**In Training:**
- `deterministic=False` → Sample from distribution `N(μ, σ)`
- Allows exploration of new strategies

---

### **Actor-Critic Architecture**

**Actor (Policy Network):**
- Decides **what action** to take
- Outputs probability distribution over actions

**Critic (Value Network):**
- Estimates **how good** the current state is
- Used during training to reduce variance

**Why Both?**
- Actor learns to take good actions
- Critic helps actor learn faster by providing baseline

---

### **Observation vs. Action**

**Observation (Input):**
- **What the model sees**: Market data (OHLCV)
- Shape: `(1, 20, 5)` = 20 bars × 5 features
- Updated every 30-60 seconds

**Action (Output):**
- **What the model decides**: Trading action (HOLD, BUY CALL, BUY PUT, etc.)
- Continuous: `-1.0` to `+1.0` (raw)
- Discrete: `0-5` (mapped)

---

## 📁 Key Files

1. **`mike_agent_live_safe.py`** - Main live trading agent
   - Observation preparation (line 1020)
   - Action mapping (line 1337)
   - Trade execution (line 1421)
   - Stop-loss/take-profit (line 497)

2. **`mike_rl_agent.py`** - Simple RL environment (legacy)
   - Basic training environment
   - Simple reward function

3. **`historical_training_system.py`** - Advanced training environment
   - Historical data simulation
   - 0DTE options pricing
   - Enhanced reward function

4. **`train_historical_model.py`** - Training script
   - Orchestrates training
   - Handles multiple symbols
   - Regime-aware balancing

5. **`RL_SYSTEM_END_TO_END_GUIDE.md`** - Detailed documentation
   - Complete system flow
   - PPO architecture details
   - Control methods (RLHF, constraints)

---

## 🔄 Summary

**In Simple Terms:**

1. **The model looks at** the last 20 minutes of market data (OHLCV)
2. **It processes** this data through a neural network (PPO)
3. **It outputs** a continuous value (-1.0 to +1.0)
4. **This value is mapped** to a trading action (HOLD, BUY CALL, BUY PUT, etc.)
5. **The action is executed** if risk checks pass
6. **Positions are monitored** for stop-losses and take-profits
7. **The process repeats** every 30-60 seconds

**The model learned** from historical data (2002-present) to maximize rewards (profit) while managing risk.

---

## 📚 Further Reading

- `RL_SYSTEM_END_TO_END_GUIDE.md` - Complete technical deep-dive
- `RL_SYSTEM_FLOW_DIAGRAM.md` - Visual flow diagrams
- `README_RL_GUIDES.md` - Quick index

---

**Questions?** The RL agent uses PPO to learn optimal trading strategies from historical data and applies them to live trading with comprehensive risk management.

