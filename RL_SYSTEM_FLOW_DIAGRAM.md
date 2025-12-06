# 🎯 RL SYSTEM FLOW - Visual Diagram

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        STEP 1: DATA COLLECTION                          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ↓
        ┌────────────────────────────────────────┐
        │  Market Data (SPY)                     │
        │  • Open, High, Low, Close, Volume      │
        │  • Last 20 bars (1-minute intervals)   │
        └─────────────────┬──────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    STEP 2: STATE PREPARATION                            │
│                                                                         │
│  Input: Raw market data                                                 │
│  Output: Observation tensor (1, 20, 5)                                  │
│                                                                         │
│  [Bar 1: O=450.0, H=450.5, L=449.8, C=450.2, V=1000000]               │
│  [Bar 2: O=450.2, H=450.8, L=450.1, C=450.6, V=1200000]               │
│  [Bar 3: O=450.6, H=451.0, L=450.4, C=450.9, V=1100000]               │
│  ...                                                                    │
│  [Bar 20: O=452.0, H=452.5, L=451.8, C=452.3, V=1300000]              │
│                                                                         │
│  Shape: (1, 20, 5)  ← Batch dimension for VecEnv                       │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                   STEP 3: PPO MODEL PROCESSING                          │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │          SHARED BACKBONE NETWORK                             │     │
│  │                                                               │     │
│  │  Input: (1, 20, 5)                                           │     │
│  │    ↓                                                          │     │
│  │  Flatten: (1, 100)  [20 bars × 5 features]                  │     │
│  │    ↓                                                          │     │
│  │  Dense Layer 1: 64 neurons → ReLU activation                │     │
│  │    ↓                                                          │     │
│  │  Dense Layer 2: 64 neurons → ReLU activation                │     │
│  │    ↓                                                          │     │
│  │  Feature Vector: (1, 64)  ← Learned representations         │     │
│  └────────────────────────┬─────────────────────────────────────┘     │
│                           │                                             │
│              ┌────────────┴────────────┐                                │
│              ↓                         ↓                                │
│  ┌─────────────────────┐  ┌─────────────────────┐                     │
│  │  ACTOR NETWORK      │  │  CRITIC NETWORK     │                     │
│  │  (Policy)           │  │  (Value Function)   │                     │
│  │                     │  │                     │                     │
│  │  Feature (64)       │  │  Feature (64)       │                     │
│  │    ↓                │  │    ↓                │                     │
│  │  Dense Layer        │  │  Dense Layer        │                     │
│  │    ↓                │  │    ↓                │                     │
│  │  OUTPUT:            │  │  OUTPUT:            │                     │
│  │  • Mean (μ): -0.3   │  │  • Value: 0.12     │                     │
│  │  • Std (σ): 0.5     │  │    (Expected +12%  │                     │
│  │                     │  │     return)         │                     │
│  └──────────┬──────────┘  └─────────────────────┘                     │
│             │                                                          │
└─────────────┼──────────────────────────────────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│              STEP 4: ACTION SAMPLING (with Entropy)                     │
│                                                                         │
│  Actor Output:                                                          │
│    Mean (μ) = -0.3  ← Slight bearish bias                              │
│    Std (σ) = 0.5   ← Randomness/exploration level                      │
│                                                                         │
│  ┌──────────────────────────────────────────┐                         │
│  │  Sampling Methods:                       │                         │
│  │                                          │                         │
│  │  Deterministic=True (Live Trading):      │                         │
│  │    action_raw = μ = -0.3  ← No randomness│                         │
│  │                                          │                         │
│  │  Deterministic=False (Training):         │                         │
│  │    action_raw ~ N(μ=-0.3, σ=0.5)        │                         │
│  │    = -0.45  (example sample)             │                         │
│  │                                          │                         │
│  │  Clip to [-1.0, 1.0]:                   │                         │
│  │    action_raw = -0.45  ← Final value    │                         │
│  └──────────────────────────────────────────┘                         │
│                                                                         │
│  Entropy Controls Randomness:                                          │
│    • High entropy (σ large) → More exploration                         │
│    • Low entropy (σ small) → More exploitation                         │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    STEP 5: ACTION MAPPING                               │
│                                                                         │
│  Input: action_raw = -0.45 (continuous)                                 │
│                                                                         │
│  Mapping Logic:                                                         │
│    if abs(action_raw) < 0.35:                                           │
│        action = 0  # HOLD                                               │
│    elif action_raw > 0:                                                 │
│        action = 1  # BUY CALL                                           │
│    else:                                                                │
│        action = 2  # BUY PUT                                            │
│                                                                         │
│  For action_raw = -0.45:                                                │
│    abs(-0.45) = 0.45 > 0.35  ✓ Not HOLD                                │
│    -0.45 < 0  ✓ Negative                                               │
│    → action = 2  # BUY PUT                                              │
│                                                                         │
│  If position exists and action_raw >= 0.5:                              │
│    if action_raw < 0.75:                                                │
│        action = 3  # TRIM 50%                                           │
│    elif action_raw < 0.9:                                               │
│        action = 4  # TRIM 70%                                           │
│    else:                                                                │
│        action = 5  # FULL EXIT                                          │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                STEP 6: CONSTRAINT LAYER (Human Control)                 │
│                                                                         │
│  Input: action = 2 (BUY PUT)                                            │
│                                                                         │
│  Apply Constraints:                                                     │
│    ✓ Check if action violates human rules                               │
│    ✓ Block early exits before TP1                                       │
│    ✓ Enforce stop losses                                                │
│    ✓ Respect position limits                                            │
│                                                                         │
│  Example Constraint:                                                    │
│    if action in [3, 4, 5] and pnl < 0.40:                               │
│        action = 0  # Force HOLD (too early to exit)                     │
│                                                                         │
│  Output: action = 2  (No constraint violation)                          │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    STEP 7: TRADE EXECUTION                              │
│                                                                         │
│  Action: 2 (BUY PUT)                                                    │
│                                                                         │
│  Execution Steps:                                                       │
│    1. Find ATM strike: $452 (current price)                            │
│    2. Calculate position size: 5 contracts                             │
│    3. Get option symbol: SPY251205P00452000                            │
│    4. Submit order to Alpaca:                                          │
│       • Symbol: SPY251205P00452000                                     │
│       • Quantity: 5                                                    │
│       • Side: BUY                                                      │
│       • Type: MARKET                                                   │
│    5. Track position in risk_mgr.open_positions                        │
│                                                                         │
│  Order Status: ✓ FILLED                                                 │
│  Entry Premium: $0.45                                                   │
│  Entry Price: $452.30                                                   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│              STEP 8: POSITION MONITORING & EXITS                        │
│                                                                         │
│  Every 30-60 seconds:                                                   │
│    1. Check Stop Losses:                                                │
│       • Absolute -15% stop (highest priority)                          │
│       • Hard stop -35%                                                  │
│       • Normal stop -20%                                                │
│                                                                         │
│    2. Check Take Profits:                                               │
│       • TP1: +40% → Sell 50%                                           │
│       • TP2: +80% → Sell 60% of remaining                              │
│       • TP3: +150% → Full exit                                         │
│                                                                         │
│    3. Check Trailing Stops:                                             │
│       • After TP1/TP2: Trail at TP - 20%                               │
│                                                                         │
│  Example:                                                               │
│    Current Premium: $0.68 (+51% profit)                                │
│    TP1 Triggered: Sell 50% (2.5 contracts)                             │
│    Remaining: 2.5 contracts                                            │
│    Trail Stop Activated: Exit 80% at TP1 - 20% = +20%                  │
│    Runner: 0.5 contracts until EOD or -15% stop                        │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    STEP 9: REWARD CALCULATION                           │
│                    (Only During Training)                               │
│                                                                         │
│  Reward Components:                                                     │
│    • Realized P&L: +$125 (2.5 contracts × $0.23 profit)                │
│    • Sharpe Ratio: 2.5                                                  │
│    • Win Rate: 75%                                                      │
│    • Drawdown: -5%                                                      │
│                                                                         │
│  Reward Formula:                                                        │
│    reward = (pnl/capital × 10) + (sharpe × 0.1) +                      │
│             (win_rate × 0.2) - (drawdown × 0.5)                        │
│                                                                         │
│    reward = (125/100000 × 10) + (2.5 × 0.1) +                          │
│             (0.75 × 0.2) - (0.05 × 0.5)                                │
│    reward = 0.0125 + 0.25 + 0.15 - 0.025 = 0.3875                      │
│                                                                         │
│  This reward is used to update the PPO model weights                   │
│  (Only happens during training, not live trading)                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Components Explained

### 1. Observation (State)
- **Shape:** (1, 20, 5) = Batch × Bars × Features
- **Content:** Last 20 bars of OHLCV data
- **Purpose:** Represents current market condition

### 2. PPO Model
- **Actor:** Outputs action distribution (mean + std)
- **Critic:** Estimates expected return from state
- **Shared Backbone:** Learns feature representations

### 3. Action Sampling
- **Deterministic=True:** Uses mean only (no randomness)
- **Deterministic=False:** Samples from distribution (has randomness)
- **Entropy:** Controls exploration vs exploitation

### 4. Action Mapping
- **Continuous → Discrete:** -1.0 to +1.0 → 0-5
- **Interpretation:** HOLD, BUY CALL, BUY PUT, TRIM, EXIT

### 5. Constraints
- **Human Rules:** Block unwanted behaviors
- **Priority:** Stop losses > TP system > RL actions
- **Preserves Randomness:** Constrains final action, not raw output

### 6. Execution
- **Real Trading:** Via Alpaca API
- **Position Tracking:** In `risk_mgr.open_positions`
- **Risk Management:** Position sizing, limits, safeguards

### 7. Rewards (Training Only)
- **Components:** P&L, Sharpe, win rate, drawdown
- **Purpose:** Update model weights to improve performance
- **Not Used Live:** Live trading uses pre-trained weights

---

## Entropy & Randomness Flow

```
┌─────────────────────────────────────────────────────────┐
│           TRAINING (Exploration Mode)                   │
└─────────────────────────────────────────────────────────┘
              │
              ↓
    Actor Output: μ=-0.3, σ=0.5
              │
              ↓
    Sample: action_raw ~ N(-0.3, 0.5)
              │
              ↓
    Result: action_raw = -0.45 (random)
              │
              ↓
    Map: action = 2 (BUY PUT)
              │
              ↓
    Execute: Buy 5 PUT contracts
              │
              ↓
    Calculate Reward: Based on outcome
              │
              ↓
    Update Model: Learn from experience


┌─────────────────────────────────────────────────────────┐
│          LIVE TRADING (Exploitation Mode)               │
└─────────────────────────────────────────────────────────┘
              │
              ↓
    Actor Output: μ=-0.3, σ=0.5
              │
              ↓
    Deterministic: action_raw = μ = -0.3 (no randomness)
              │
              ↓
    Map: action = 2 (BUY PUT)
              │
              ↓
    Constraints: Check human rules
              │
              ↓
    Execute: Buy 5 PUT contracts (if allowed)
```

---

## Controlling Without Losing Randomness

### Method 1: Constraint Layer (Recommended)

```
Raw Action (with randomness) → Constraint Check → Final Action
     ↓                              ↓
  action = 5                   if pnl < 0.40:
  (FULL EXIT)                     action = 0 (HOLD)
```

**Benefits:**
- ✅ Keeps randomness in raw output
- ✅ Applies human rules after sampling
- ✅ No retraining needed

### Method 2: Reward Shaping

```
Bad Action → Negative Reward → Model Learns to Avoid
```

**Benefits:**
- ✅ Model learns preferred behavior
- ✅ Requires retraining
- ✅ More permanent change

### Method 3: RLHF (Full Control)

```
Human Preferences → Reward Model → Shaped Rewards → Better Policy
```

**Benefits:**
- ✅ Complete control over behavior
- ✅ Complex to implement
- ✅ Requires preference data

---

**This diagram shows the complete flow from market data to trade execution!**

