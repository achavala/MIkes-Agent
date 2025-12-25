# 🏗️ RL MODEL ARCHITECTURE - Complete Explanation

**Date:** December 21, 2025  
**Purpose:** Explain the exact architecture and data flow of the RL model

---

## 📊 ACTUAL MODEL ARCHITECTURE

### **PPO (Proximal Policy Optimization) Model Structure**

The model uses **stable-baselines3 PPO**, which has this architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT: State (20, 23)                 │
│              (Observation matrix from market data)       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              FEATURE EXTRACTOR (MLP/LSTM)                │
│  - Processes raw observation (20, 23)                    │
│  - Extracts meaningful features                          │
│  - Outputs: Feature vector (e.g., 64 dimensions)        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ├──────────────────────┐
                       │                      │
                       ▼                      ▼
            ┌──────────────────┐    ┌──────────────────┐
            │  ACTION NETWORK   │    │  VALUE NETWORK    │
            │     (Actor)       │    │     (Critic)     │
            │                   │    │                  │
            │ Outputs:          │    │ Outputs:         │
            │ - Action logits   │    │ - State value    │
            │   (6 actions)     │    │   (scalar)       │
            └──────────────────┘    └──────────────────┘
                       │
                       ▼
            ┌──────────────────┐
            │   ACTION OUTPUT   │
            │  (0-5) + Confidence│
            └──────────────────┘
```

---

## 🔄 COMPLETE DATA FLOW

### **Step 1: Fetch Market Data**

```python
# Function: get_market_data("SPY", period="2d", interval="1m")
# Returns: DataFrame (N, 5) with OHLCV
hist = get_market_data("SPY")
# Shape: (2000+, 5)
```

### **Step 2: Prepare Observation (State)**

```python
# Function: prepare_observation(hist, risk_mgr, symbol='SPY')
# Returns: numpy array (20, 23)
obs = prepare_observation(hist, risk_mgr, symbol='SPY')
# Shape: (20, 23)
# - 20 timesteps (last 20 bars)
# - 23 features per timestep
```

**This is the STATE that goes into the model.**

---

### **Step 3: Model Forward Pass (Feature Extraction + Action Prediction)**

**Code (Line 3634-3653):**
```python
# Convert observation to tensor
obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
# Shape: (1, 20, 23) - batch size 1, sequence 20, features 23

# Get action distribution from policy
action_dist = model.policy.get_distribution(obs_tensor)
```

**What Happens Inside `model.policy.get_distribution()`:**

#### **3.1: Feature Extractor (Hidden Layer)**
```python
# The policy network has a feature extractor (MLP or LSTM)
# This processes the raw observation (20, 23)

# For MLP (standard PPO):
features = mlp_extractor(obs_tensor)
# Process: (1, 20, 23) → Flatten → MLP layers → (1, 64)
# Output: Feature vector (e.g., 64 dimensions)

# For LSTM (RecurrentPPO):
features, lstm_state = lstm_extractor(obs_tensor)
# Process: (1, 20, 23) → LSTM layers → (1, 64)
# Output: Feature vector + LSTM hidden state
```

**This is the ENCODED STATE (feature vector).**

#### **3.2: Action Network (Actor Head)**
```python
# The action network takes the feature vector and outputs action logits
action_logits = action_net(features)
# Input: (1, 64) - feature vector
# Output: (1, 6) - logits for 6 actions
# Values: [0.2, 1.5, -0.3, -0.5, -0.8, -1.0]
#         HOLD, CALL, PUT, TRIM50, TRIM70, EXIT
```

**This is the ACTION OUTPUT (logits).**

#### **3.3: Apply Temperature Softmax**
```python
# Convert logits to probabilities
temperature = 0.7
probs = torch.softmax(action_logits / temperature, dim=-1)
# Output: (1, 6)
# Values: [0.15, 0.55, 0.10, 0.08, 0.07, 0.05]

# Select action
rl_action = int(np.argmax(probs))  # Action 1 (BUY CALL)
action_strength = float(probs[0][rl_action])  # 0.55 (55% confidence)
```

**This is the FINAL ACTION (0-5) + CONFIDENCE (0.0-1.0).**

---

### **Step 4: Reward Calculation (TRAINING ONLY)**

**During Training:**
```python
# 1. Agent takes action
action = model.predict(obs)  # e.g., action = 1 (BUY CALL)

# 2. Environment executes action
env.step(action)
# - Executes trade in simulated environment
# - Updates position, PnL, etc.

# 3. Calculate reward
reward = calculate_reward(
    realized_pnl=350,
    capital=10000,
    sharpe_ratio=0.5,
    win_rate=0.65,
    drawdown=0.1
)
# reward = (350/10000)*10 + 0.5*0.1 + 0.65*0.2 - 0.1*0.5
# reward = 0.35 + 0.05 + 0.13 - 0.05 = 0.48

# 4. Update model weights
model.learn(total_timesteps=1, reset_num_timesteps=False)
# - Uses reward to update policy network weights
# - Improves action selection for future states
```

**During Live Trading:**
```python
# 1. Agent takes action
action = model.predict(obs)  # e.g., action = 1 (BUY CALL)

# 2. Execute action in real market
execute_trade(action)  # Place order via Alpaca API

# 3. NO reward calculation (model already trained)
# 4. NO weight updates (model is frozen)
```

---

## 🎯 CLARIFICATION: Your Understanding vs. Reality

### **Your Understanding:**
```
1. Get data → Pass to trained model → Outputs action and state
2. Pass state to RL head → Gives action
3. Calculate reward
```

### **Actual Architecture:**
```
1. Get data → Prepare observation (state) (20, 23)
2. Pass state to model.policy:
   a. Feature Extractor (MLP/LSTM) → Encoded state (feature vector)
   b. Action Network (Actor) → Action logits → Action probabilities → Action (0-5)
3. During training only: Calculate reward → Update model weights
```

---

## 📊 DETAILED ARCHITECTURE BREAKDOWN

### **Model Components:**

#### **1. Feature Extractor (MLP or LSTM)**
```python
# Standard PPO (MLP):
mlp_extractor = nn.Sequential(
    nn.Linear(20*23, 256),  # Flatten (20,23) → 460 → 256
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64)      # Output: 64-dim feature vector
)

# RecurrentPPO (LSTM):
lstm_extractor = nn.LSTM(
    input_size=23,          # Features per timestep
    hidden_size=64,         # LSTM hidden size
    num_layers=2,
    batch_first=True
)
# Output: (1, 20, 64) → (1, 64) after processing sequence
```

**Purpose:** Extract meaningful patterns from raw observation

#### **2. Action Network (Actor)**
```python
action_net = nn.Sequential(
    nn.Linear(64, 32),      # Feature vector → 32
    nn.ReLU(),
    nn.Linear(32, 6)        # Output: 6 action logits
)
# Output: (1, 6) - logits for 6 actions
```

**Purpose:** Predict which action to take

#### **3. Value Network (Critic) - Training Only**
```python
value_net = nn.Sequential(
    nn.Linear(64, 32),      # Feature vector → 32
    nn.ReLU(),
    nn.Linear(32, 1)        # Output: State value (scalar)
)
# Output: (1, 1) - estimated value of current state
```

**Purpose:** Estimate state value (used for training, not inference)

---

## 🔍 COMPLETE FLOW DIAGRAM

### **During Inference (Live Trading):**

```
┌─────────────────────────────────────────────────────────┐
│ STEP 1: Fetch Market Data                                │
│ get_market_data("SPY") → DataFrame (N, 5)                │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 2: Prepare Observation (State)                      │
│ prepare_observation(hist) → numpy array (20, 23)        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Model Forward Pass                               │
│                                                          │
│  obs (20, 23)                                           │
│     │                                                    │
│     ▼                                                    │
│  Feature Extractor (MLP/LSTM)                           │
│     │                                                    │
│     ▼                                                    │
│  Feature Vector (64 dim)                                │
│     │                                                    │
│     ▼                                                    │
│  Action Network (Actor)                                  │
│     │                                                    │
│     ▼                                                    │
│  Action Logits (6 dim)                                   │
│     │                                                    │
│     ▼                                                    │
│  Temperature Softmax                                    │
│     │                                                    │
│     ▼                                                    │
│  Action Probabilities (6 dim)                           │
│     │                                                    │
│     ▼                                                    │
│  Action (0-5) + Confidence (0.0-1.0)                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 4: Execute Action                                   │
│ Execute trade in real market (Alpaca API)                │
│ NO reward calculation (model already trained)            │
└─────────────────────────────────────────────────────────┘
```

### **During Training:**

```
┌─────────────────────────────────────────────────────────┐
│ STEP 1-3: Same as inference                             │
│ (Fetch data → Prepare state → Model forward pass)       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 4: Execute Action in Environment                    │
│ env.step(action) → Simulated trade execution            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 5: Calculate Reward                                 │
│ reward = f(realized_pnl, sharpe, win_rate, drawdown)   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 6: Update Model Weights                            │
│ model.learn() → Backpropagation → Update policy         │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 KEY POINTS

### **1. State is NOT Output by Model**
- ❌ **Wrong:** "Model outputs action and state"
- ✅ **Correct:** State is INPUT to model, action is OUTPUT

### **2. Feature Extractor is Part of Policy**
- The feature extractor (MLP/LSTM) is INSIDE the policy network
- It's not a separate model - it's part of the PPO architecture
- It processes observation → feature vector → action

### **3. Reward is Training-Only**
- ✅ **During training:** Reward calculated → Model weights updated
- ❌ **During live trading:** No reward calculation, model is frozen

### **4. Architecture is End-to-End**
```
Observation (20, 23)
    ↓
Feature Extractor (MLP/LSTM)
    ↓
Feature Vector (64 dim)
    ↓
Action Network (Actor)
    ↓
Action (0-5) + Confidence
```

**All in one model - no separate "RL head" that takes state as input.**

---

## 🎯 SUMMARY

**Your Understanding:**
- Get data → Model → Action + State → RL Head → Action → Reward

**Actual Architecture:**
- Get data → Prepare state (20, 23) → Model (Feature Extractor + Action Network) → Action → (Training: Reward → Update weights)

**Key Differences:**
1. State is INPUT, not OUTPUT
2. Feature extractor is INSIDE the model, not separate
3. Reward is TRAINING-ONLY, not calculated during live trading
4. It's an end-to-end model, not a two-stage system

---

**The model is a single PPO network that takes state as input and outputs actions directly!**


