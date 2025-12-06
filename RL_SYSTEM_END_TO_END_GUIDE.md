# 🧠 RL SYSTEM END-TO-END GUIDE - Step by Step

## Overview

This document explains your complete Reinforcement Learning system from input (market data) to output (trading actions), including PPO architecture, critic/value functions, action selection with entropy, and how to control behavior without losing randomness.

---

## PART 1: WHAT YOUR MODEL RECEIVES (Environment & States)

### 1.1 Environment Setup

**File:** `mike_rl_agent.py`

```python
class MikeTradingEnv(gym.Env):
    def __init__(self, data, window_size=20):
        self.data = data  # Historical OHLCV data
        self.window_size = 20  # Lookback period
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size, 5),  # 20 bars × 5 features
            dtype=np.float32
        )
```

### 1.2 Observation Space (State)

**What the model sees:**
- **Shape:** `(20, 5)` = 20 bars of historical data × 5 features
- **Features:**
  1. `open` - Opening price
  2. `high` - High price
  3. `low` - Low price
  4. `close` - Closing price
  5. `volume` - Trading volume (normalized)

**Current Implementation:**
```python
def _get_obs(self):
    # Get last 20 bars of data
    window = self.data.iloc[start:end]  # 20 rows
    # Extract OHLCV columns
    return window[['open','high','low','close','volume']].values  # Shape: (20, 5)
```

**In Live Trading:**
```python
# mike_agent_live_safe.py, line 1010-1053
def prepare_observation(data: pd.DataFrame, risk_mgr: RiskManager) -> np.ndarray:
    recent = data.tail(LOOKBACK)  # Last 20 bars
    obs_data = recent[['open', 'high', 'low', 'close', 'volume']]
    state = obs_data.values.astype(np.float32)  # Shape: (20, 5)
    state = state.reshape(1, 20, 5)  # Add batch dimension for VecEnv
    return state  # Final shape: (1, 20, 5)
```

### 1.3 Action Space

**What the model can output:**
- **Type:** Continuous `Box` space
- **Range:** `-1.0` to `+1.0` (single float value)
- **Interpretation:** Raw action signal (later mapped to discrete actions)

**Current Mapping:**
```python
# Raw output: -1.0 to +1.0 (continuous)
if abs(action_value) < 0.35:
    action = 0  # HOLD
elif action_value > 0:
    action = 1  # BUY CALL
else:
    action = 2  # BUY PUT

# If position exists and action_value >= 0.5:
if action_value >= 0.5:
    if action_value < 0.75:
        action = 3  # TRIM 50%
    elif action_value < 0.9:
        action = 4  # TRIM 70%
    else:
        action = 5  # FULL EXIT
```

---

## PART 2: PPO ARCHITECTURE - How It Works

### 2.1 PPO Overview

**PPO (Proximal Policy Optimization)** has **two neural networks**:

1. **Actor (Policy Network)** - Decides actions
2. **Critic (Value Network)** - Estimates state value

### 2.2 Network Architecture

**Your Current Setup:**
```python
model = PPO("MlpPolicy", env, verbose=1, device="cpu")
# "MlpPolicy" = Multi-Layer Perceptron (feed-forward neural network)
```

**Hidden Structure (Stable-Baselines3 Default):**
```
INPUT: State (20, 5) → Flatten to (100,)
    ↓
DENSE LAYER 1: 64 neurons → ReLU activation
    ↓
DENSE LAYER 2: 64 neurons → ReLU activation
    ↓
SPLIT INTO TWO BRANCHES:
    ↓                    ↓
ACTOR HEAD         CRITIC HEAD
(Output: action)   (Output: value)
```

### 2.3 Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. STATE PREPARATION                                        │
│    Market Data (SPY OHLCV) → Last 20 bars                  │
│    Shape: (1, 20, 5)                                        │
└───────────────────────────────┬─────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. FEATURE EXTRACTION (Shared Backbone)                     │
│    Input: (1, 20, 5)                                        │
│    ↓                                                         │
│    Flatten: (1, 100)  [20×5 = 100 features]                │
│    ↓                                                         │
│    Dense Layer 1: 64 neurons → ReLU                         │
│    ↓                                                         │
│    Dense Layer 2: 64 neurons → ReLU                         │
│    ↓                                                         │
│    Feature Vector: (1, 64)                                  │
└───────────────────────────────┬─────────────────────────────┘
                                ↓
                ┌───────────────┴───────────────┐
                ↓                               ↓
┌───────────────────────────┐  ┌───────────────────────────┐
│ 3. ACTOR NETWORK          │  │ 4. CRITIC NETWORK         │
│    (Policy)               │  │    (Value Estimator)      │
│                           │  │                           │
│    Feature Vector (64)    │  │    Feature Vector (64)    │
│    ↓                      │  │    ↓                      │
│    Dense Layer            │  │    Dense Layer            │
│    ↓                      │  │    ↓                      │
│    OUTPUT:                │  │    OUTPUT:                │
│    Action Distribution    │  │    State Value (scalar)   │
│    (Mean + Std)           │  │    V(s) = expected return │
└───────────┬───────────────┘  └───────────┬───────────────┘
            ↓                               ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. ACTION SAMPLING (with Entropy/Randomness)                │
│                                                             │
│    Actor Output:                                            │
│      - Mean (μ): -0.3 (slight bearish bias)                │
│      - Std (σ): 0.5 (randomness/exploration)               │
│                                                             │
│    Sample from Normal Distribution:                         │
│      action_raw ~ N(μ=-0.3, σ=0.5)                         │
│      → Example: action_raw = -0.45                          │
│                                                             │
│    Clip to [-1, 1]:                                         │
│      action_raw = -0.45 (within bounds)                     │
└───────────────────────────────┬─────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. ACTION MAPPING                                           │
│    action_raw = -0.45                                       │
│    ↓                                                         │
│    abs(-0.45) = 0.45 > 0.35 → Not HOLD                     │
│    -0.45 < 0 → BUY PUT (Action 2)                          │
└───────────────────────────────┬─────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. EXECUTION LOGIC                                          │
│    Execute: BUY PUT                                         │
│    ↓                                                         │
│    Calculate Position Size                                  │
│    Submit Order to Alpaca                                   │
│    ↓                                                         │
│    Wait for next bar (30-60 seconds)                        │
│    ↓                                                         │
│    Calculate Reward (P&L change)                            │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Actor Network (Policy)

**Purpose:** Outputs action probability distribution

**Output Format:**
- **Mean (μ):** Center of action distribution (e.g., -0.3 = slight bearish)
- **Std (σ):** Spread/randomness (higher = more exploration)

**Sampling:**
```python
# During training (exploration):
action_raw = np.random.normal(mean=μ, std=σ)  # Sample from distribution
action_raw = np.clip(action_raw, -1.0, 1.0)    # Clip to valid range

# During inference (exploitation):
action_raw = μ  # Use mean directly (deterministic=True)
```

**Current Code:**
```python
# Line 1242: mike_agent_live_safe.py
action_raw, _ = model.predict(obs, deterministic=True)
# deterministic=True → Uses mean only (no sampling)
# deterministic=False → Samples from distribution (more random)
```

### 2.5 Critic Network (Value Function)

**Purpose:** Estimates expected future return from current state

**What It Predicts:**
```
V(s) = Expected Total Reward from state s
     = E[R_t + γR_{t+1} + γ²R_{t+2} + ...]
```

**How It's Used:**
- **Advantage Calculation:** `Advantage = Actual Reward - V(s)`
  - Positive advantage → Action was better than expected
  - Negative advantage → Action was worse than expected
- **Policy Update:** Adjust actor to favor actions with positive advantage

**Example:**
```python
State: Market looks bullish (SPY up 0.5%)
Critic Output: V(s) = 0.12  # Expects +12% return from this state

You take action: BUY CALL
Actual reward: +0.25  # +25% profit
Advantage: 0.25 - 0.12 = +0.13  # Better than expected! ✅

Result: Actor learns to favor BUY CALL in similar states
```

### 2.6 Entropy and Randomness

**What is Entropy?**
- Measures **randomness/uncertainty** in action selection
- High entropy = More exploration (try random actions)
- Low entropy = More exploitation (use best known actions)

**How PPO Controls It:**
```python
# PPO Objective includes entropy bonus:
loss = policy_loss - entropy_coefficient * entropy

# Higher entropy_coefficient → More exploration
# Lower entropy_coefficient → More exploitation
```

**Your Current Setting:**
```python
# Default PPO entropy_coefficient = 0.01
# This balances exploration vs exploitation
```

**During Training:**
- **Early:** High entropy (explore many actions)
- **Later:** Low entropy (exploit best actions)

**During Live Trading:**
```python
# Line 1242: deterministic=True
action_raw, _ = model.predict(obs, deterministic=True)
# This uses mean only (zero randomness)
# For exploration: set deterministic=False
```

---

## PART 3: REWARD CALCULATION & EXECUTION LOGIC

### 3.1 Current Reward Function (Simplified)

**Training Environment (`mike_rl_agent.py`):**
```python
def step(self, action):
    reward = action[0] * 0.001  # Very simple reward
    return self._get_obs(), reward, done, False, {}
```

**Problem:** This is too simple! Real rewards should be based on:
- Realized P&L
- Risk-adjusted returns
- Win rate
- Drawdown

### 3.2 Ideal Reward Function (From README)

```python
reward = (
    realized_pnl / capital * 10 +        # Profit reward
    sharpe_ratio * 0.1 +                 # Risk-adjusted return
    win_rate * 0.2 +                     # Win rate bonus
    -drawdown * 0.5                      # Drawdown penalty
)
```

**How It Would Work:**
```python
def calculate_reward(self, action, pnl, capital, sharpe, win_rate, drawdown):
    # Component 1: Direct profit
    profit_component = (pnl / capital) * 10
    
    # Component 2: Risk-adjusted (better Sharpe = better reward)
    sharpe_component = sharpe * 0.1
    
    # Component 3: Consistency (higher win rate = better)
    winrate_component = win_rate * 0.2
    
    # Component 4: Penalty for large drawdowns
    drawdown_penalty = -drawdown * 0.5
    
    total_reward = profit_component + sharpe_component + winrate_component + drawdown_penalty
    return total_reward
```

### 3.3 Execution Logic Flow

**Current Implementation (`mike_agent_live_safe.py`):**

```python
# 1. Get observation (market state)
obs = prepare_observation(hist, risk_mgr)  # Shape: (1, 20, 5)

# 2. Get action from PPO model
action_raw, _ = model.predict(obs, deterministic=True)
# action_raw: Continuous value [-1, 1]

# 3. Map to discrete action
if abs(action_value) < 0.35:
    action = 0  # HOLD
elif action_value > 0:
    action = 1  # BUY CALL
else:
    action = 2  # BUY PUT

# 4. Execute action
if action == 1:
    # Buy call option
    strike = find_atm_strike(current_price)
    symbol = get_option_symbol('SPY', strike, 'call')
    qty = calculate_position_size(...)
    api.submit_order(symbol=symbol, qty=qty, side='buy', ...)

# 5. Check stop losses / take profits
check_stop_losses(api, risk_mgr, current_price, trade_db)

# 6. Wait for next iteration (30-60 seconds)
time.sleep(30)
```

**Note:** Currently, rewards are NOT calculated in live trading. The model uses pre-trained weights. To improve the model, you'd need to:
1. Record states, actions, and outcomes
2. Calculate rewards offline
3. Retrain the model with new data

---

## PART 4: CONTROLLING PPO WITHOUT LOSING RANDOMNESS (RLHF)

### 4.1 The Problem

**Current Situation:**
- PPO outputs actions based on historical training
- Sometimes outputs actions you don't want (e.g., early exits)
- But you still want some randomness for exploration

**Challenge:** How to guide the model while preserving exploration?

### 4.2 Solution: Reinforcement Learning from Human Feedback (RLHF)

**Concept:** Use human preferences to shape reward function

**Three Approaches:**

#### Approach 1: Reward Shaping (Simplest)

**Idea:** Modify reward function to penalize unwanted behaviors

**Example:**
```python
def calculate_reward_with_shaping(action, pnl, ...):
    base_reward = calculate_reward(action, pnl, ...)
    
    # Penalty for early exits (before TP1)
    if action in [3, 4, 5] and pnl < 0.40:  # Exiting before +40%
        penalty = -0.5  # Strong penalty
        base_reward += penalty
    
    # Penalty for not using stop loss
    if pnl < -0.15:  # Should have stopped at -15%
        penalty = -1.0  # Very strong penalty
        base_reward += penalty
    
    return base_reward
```

**Implementation:**
1. Modify `MikeTradingEnv.step()` to use new reward function
2. Retrain model with shaped rewards
3. Model learns to avoid penalized behaviors

#### Approach 2: Preference-Based Learning (Full RLHF)

**Idea:** Train a reward model from human preferences

**Steps:**

**Step 1: Collect Preference Data**
```python
# Present human with two scenarios:
Scenario A: Exit at +10% profit
Scenario B: Hold until TP1 at +40%

Human chooses: Scenario B is better
# Record preference: (state, action_A, action_B, preference=B)
```

**Step 2: Train Reward Model**
```python
# Train a neural network to predict human preferences
class PreferenceRewardModel:
    def __init__(self):
        self.model = MLP(input_dim=state_dim, output_dim=1)
    
    def predict_reward(self, state, action):
        # Returns reward score
        return self.model([state, action])
    
    def train(self, preference_data):
        # Learn from human preferences
        for state, action_A, action_B, preference in preference_data:
            if preference == 'A':
                # Action A is better
                reward_A = self.predict_reward(state, action_A)
                reward_B = self.predict_reward(state, action_B)
                loss = max(0, 1 - (reward_A - reward_B))  # Reward A should be higher
            # ... train model
```

**Step 3: Use Reward Model in PPO**
```python
class MikeTradingEnv(gym.Env):
    def __init__(self, data, reward_model):
        self.reward_model = reward_model  # Human preference model
    
    def step(self, action):
        # Get base reward (P&L based)
        base_reward = calculate_base_reward(...)
        
        # Get preference-based reward
        preference_reward = self.reward_model.predict_reward(self.state, action)
        
        # Combine (weighted average)
        total_reward = 0.7 * base_reward + 0.3 * preference_reward
        
        return next_state, total_reward, done, info
```

#### Approach 3: Constraint-Based Control (Safest)

**Idea:** Add constraints without changing reward function

**Implementation:**
```python
def get_action_with_constraints(obs, model, risk_mgr):
    # Get action from PPO
    action_raw, _ = model.predict(obs, deterministic=False)  # Keep randomness
    
    # Apply constraints
    action = map_to_discrete(action_raw)
    
    # Constraint 1: Block early exits
    if action in [3, 4, 5]:  # TRIM/EXIT
        # Check if position is at TP1 or higher
        for symbol, pos_data in risk_mgr.open_positions.items():
            pnl_pct = calculate_pnl(symbol, pos_data)
            if pnl_pct < 0.40:  # Not at TP1 yet
                action = 0  # Force HOLD (but keep randomness in next step)
    
    # Constraint 2: Enforce stop loss (hard constraint)
    # This already happens in check_stop_losses()
    
    return action
```

**Benefits:**
- ✅ Keeps model randomness (exploration)
- ✅ Adds human control (constraints)
- ✅ No retraining needed
- ✅ Easy to adjust

### 4.3 Recommended Approach for Your System

**Use Constraint-Based Control + Reward Shaping:**

```python
# 1. Add constraints in live trading (immediate fix)
def get_safe_action(obs, model, risk_mgr, deterministic=False):
    # Get action with randomness
    action_raw, _ = model.predict(obs, deterministic=deterministic)
    action = map_to_discrete(action_raw)
    
    # Apply constraints
    action = apply_trading_constraints(action, risk_mgr)
    
    return action

# 2. Retrain with reward shaping (long-term improvement)
def train_with_shaped_rewards():
    env = MikeTradingEnv(data, reward_shaping=True)
    model = PPO("MlpPolicy", env)
    model.learn(total_timesteps=100000)
    model.save("mike_rl_model_v2.zip")
```

---

## PART 5: PRACTICAL IMPLEMENTATION

### 5.1 Current System Summary

**What Works:**
- ✅ State preparation (OHLCV from last 20 bars)
- ✅ PPO model loading and inference
- ✅ Action mapping (continuous → discrete)
- ✅ Trade execution via Alpaca

**What Needs Improvement:**
- ⚠️ Reward function is too simple (line 36 in `mike_rl_agent.py`)
- ⚠️ No reward calculation in live trading
- ⚠️ No human feedback mechanism
- ⚠️ Constraints applied AFTER action (should be integrated)

### 5.2 Recommended Next Steps

**Step 1: Enhance Reward Function (Training)**
```python
# Modify mike_rl_agent.py
class MikeTradingEnv(gym.Env):
    def step(self, action):
        # Calculate real reward based on P&L
        reward = self._calculate_real_reward(action)
        return self._get_obs(), reward, done, False, {}
    
    def _calculate_real_reward(self, action):
        # Get current position P&L
        pnl_pct = self._get_position_pnl()
        
        # Reward components
        profit_reward = pnl_pct * 10  # Scale profit
        risk_penalty = -abs(pnl_pct) * 0.5 if pnl_pct < 0 else 0
        
        return profit_reward + risk_penalty
```

**Step 2: Add Constraint Layer (Live Trading)**
```python
# Modify mike_agent_live_safe.py
def get_constrained_action(obs, model, risk_mgr, deterministic=False):
    # Get action with optional randomness
    action_raw, _ = model.predict(obs, deterministic=deterministic)
    action_value = float(action_raw[0])
    
    # Map to discrete action
    action = map_continuous_to_discrete(action_value, risk_mgr)
    
    # Apply constraints BEFORE execution
    action = apply_human_constraints(action, action_value, risk_mgr)
    
    return action, action_value

def apply_human_constraints(action, action_value, risk_mgr):
    # Constraint 1: Block early exits before TP1
    if action in [3, 4, 5]:  # TRIM/EXIT
        for symbol, pos_data in risk_mgr.open_positions.items():
            pnl = calculate_pnl(symbol, pos_data)
            if pnl < 0.40:  # Not at TP1
                return 0  # Force HOLD
    
    # Constraint 2: Preserve randomness for exploration
    # If deterministic=False, action_value already has randomness
    # We just constrain the final action, not the raw value
    
    return action
```

**Step 3: Add Exploration Mode (Optional)**
```python
# Allow model to explore occasionally
EXPLORATION_PROB = 0.1  # 10% of time, use random action

def get_action_with_exploration(obs, model, risk_mgr):
    if random.random() < EXPLORATION_PROB:
        # Random exploration
        action_value = random.uniform(-1, 1)
        action = map_continuous_to_discrete(action_value, risk_mgr)
    else:
        # Use model (with constraints)
        action, action_value = get_constrained_action(
            obs, model, risk_mgr, 
            deterministic=False  # Keep some randomness
        )
    
    return action, action_value
```

---

## SUMMARY

### Your RL Pipeline:

```
1. MARKET DATA → Observation (20 bars × 5 features)
2. OBSERVATION → PPO Model (Actor + Critic)
3. ACTOR → Action Distribution (Mean + Std)
4. SAMPLING → Action Raw (-1 to +1) [with entropy/randomness]
5. MAPPING → Discrete Action (0-5)
6. CONSTRAINTS → Human Rules Applied
7. EXECUTION → Trade via Alpaca
8. REWARD → P&L (only in training, not live)
```

### Key Points:

1. **States:** Last 20 bars of OHLCV data (shape: 20×5)
2. **Actions:** Continuous value [-1, 1] mapped to discrete [0-5]
3. **PPO:** Two networks (Actor = actions, Critic = value estimation)
4. **Entropy:** Controls randomness (higher = more exploration)
5. **RLHF:** Use constraints + reward shaping to guide without losing randomness

### To Control Without Losing Randomness:

- ✅ **Constraints:** Block unwanted actions, but keep raw randomness
- ✅ **Reward Shaping:** Penalize bad behaviors during training
- ✅ **Exploration Mode:** Allow occasional random actions
- ✅ **Deterministic Flag:** Control randomness level (False = more random)

---

**Ready to implement any of these? Let me know which approach you want to start with!**

