# 🔥 **LSTM MODEL RETRAINING GUIDE**

**Date**: 2025-12-12  
**Goal**: Retrain RL model with LSTM backbone for temporal intelligence

---

## ✅ **PREREQUISITES**

### **1. Verify RecurrentPPO is Available**

```bash
python3 -c "from sb3_contrib import RecurrentPPO; print('✅ RecurrentPPO available')"
```

If not available:
```bash
pip install sb3-contrib
```

### **2. Verify MASSIVE_API_KEY is Set**

```bash
# Check environment
echo $MASSIVE_API_KEY

# Or check .env file
cat .env | grep MASSIVE_API_KEY
```

If not set, add to `.env`:
```bash
MASSIVE_API_KEY=your_polygon_api_key_here
```

---

## 🚀 **QUICK START (Automated)**

### **Option 1: Use the Script**

```bash
./retrain_lstm_model.sh
```

This script:
- ✅ Checks prerequisites
- ✅ Loads API key from .env
- ✅ Starts training with correct parameters
- ✅ Logs progress to file

---

## 🚀 **MANUAL START (Step-by-Step)**

### **Step 1: Verify Training Script Will Use LSTM**

The training script will automatically detect and use RecurrentPPO if available. It will print:
```
✅ RecurrentPPO available - Using LSTM Policy with action masking (temporal intelligence enabled)
```

### **Step 2: Start Training**

```bash
python3 -u train_historical_model.py \
  --symbols SPY,QQQ,SPX \
  --timesteps 500000 \
  --model-name mike_momentum_model_v3_lstm \
  --human-momentum \
  --data-source massive \
  --intraday-days 60 \
  --learning-rate 3e-5 \
  --ent-coef 0.08 \
  --gamma 0.92 \
  --n-steps 512
```

### **Step 3: Monitor Progress**

Training logs will show:
- LSTM configuration (hidden size, layers)
- Training progress (steps, rewards, diagnostics)
- Checkpoint saves (every 50k steps)

Watch for:
```
✅ RecurrentPPO Configuration:
   - Hidden Size: 256
   - N Steps: 512 (optimized for LSTM)
   - Batch Size: 128 (optimized for LSTM)
```

---

## 📊 **TRAINING PARAMETERS EXPLAINED**

| Parameter | Value | Why |
|-----------|-------|-----|
| `--timesteps` | 500000 | Standard for intraday scalping (was validated in tune2) |
| `--human-momentum` | Enabled | Uses 20×23 observation space with momentum features |
| `--data-source massive` | Polygon | Real 1-minute intraday bars (not daily) |
| `--intraday-days 60` | 60 days | ~2 months of 1-minute data for training |
| `--learning-rate` | 3e-5 | Lower for LSTM stability (was 3e-4 for MLP) |
| `--ent-coef` | 0.08 | High entropy to prevent HOLD collapse (tuned in tune2) |
| `--gamma` | 0.92 | Lower discount for short-horizon scalping |
| `--n-steps` | 512 | Optimized for LSTM (was 2048 for MLP) |

---

## ⏱️ **EXPECTED TRAINING TIME**

- **500k timesteps**: ~4-8 hours (depending on CPU/GPU)
- **Checkpoints**: Saved every 50k steps
- **Final model**: `models/mike_momentum_model_v3_lstm_500000.zip`

---

## 🔍 **HOW TO VERIFY LSTM IS ACTIVE**

### **During Training**:

Look for these log messages:
```
✅ RecurrentPPO available - Using LSTM Policy with action masking
✅ RecurrentPPO Configuration:
   - Hidden Size: 256
   - N Steps: 512
   - Batch Size: 128
```

### **After Training**:

```bash
python3 -c "
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import os

model_path = 'models/mike_momentum_model_v3_lstm_500000.zip'
if os.path.exists(model_path):
    try:
        model = RecurrentPPO.load(model_path)
        print('✅ Model loaded: RecurrentPPO (LSTM)')
        print(f'Policy: {type(model.policy).__name__}')
        print(f'Features: {type(model.policy.features_extractor).__name__}')
        if hasattr(model.policy.features_extractor, 'lstm'):
            print('✅ LSTM detected in features extractor')
        else:
            print('⚠️  No LSTM attribute found (may be in policy network)')
    except Exception as e:
        print(f'❌ Error loading model: {e}')
else:
    print(f'❌ Model file not found: {model_path}')
"
```

---

## 📋 **TRAINING MONITORING**

### **Watch Training Logs**:

```bash
# Real-time monitoring
tail -f logs/training/mike_momentum_model_v3_lstm_500000.log

# Or check diagnostics
grep "MomentumDiagnostics" logs/training/mike_momentum_model_v3_lstm_500000.log | tail -20
```

### **Key Metrics to Watch**:

1. **HOLD %** - Should decrease over time (target: 30-40% by 100k steps)
2. **BUY %** - Should increase (target: 60-70% by 100k steps)
3. **Strong-Setup BUY Rate** - Should rise (target: 75-85% by 100k steps)
4. **Reward** - Should stabilize (not collapse to negative)

---

## 🎯 **AFTER TRAINING COMPLETE**

### **Step 1: Validate Model**

```bash
python3 validate_model.py \
  --model models/mike_momentum_model_v3_lstm_500000.zip \
  --offline-eval \
  --intraday \
  --symbols SPY,QQQ,SPX \
  --intraday-days 10 \
  --stochastic
```

### **Step 2: Update Live Agent**

Edit `mike_agent_live_safe.py`:
```python
MODEL_PATH = "models/mike_momentum_model_v3_lstm_500000.zip"
```

### **Step 3: Test Live Agent**

```bash
python3 mike_agent_live_safe.py
```

The agent should now:
- ✅ Load LSTM model successfully
- ✅ Use temporal intelligence for decisions
- ✅ Show improved pattern recognition

---

## 🚨 **TROUBLESHOOTING**

### **Issue: "RecurrentPPO not available"**

**Solution**:
```bash
pip install sb3-contrib
```

### **Issue: "MASSIVE_API_KEY missing"**

**Solution**:
```bash
# Add to .env file
echo "MASSIVE_API_KEY=your_key_here" >> .env
```

### **Issue: Training is slow**

**Solution**:
- This is normal for LSTM (more computation)
- Can reduce `--timesteps` to 100000 for faster initial test
- Can run in background: `nohup ./retrain_lstm_model.sh &`

### **Issue: Model still uses MLP after training**

**Check**:
- Did you see "RecurrentPPO available" message during training?
- Check model file: `python3 -c "from sb3_contrib import RecurrentPPO; m = RecurrentPPO.load('models/...'); print(type(m))"`

---

## 📊 **EXPECTED IMPROVEMENTS WITH LSTM**

### **Temporal Intelligence**:
- ✅ Pattern recognition across time
- ✅ Regime transition detection
- ✅ Trend continuation/break detection
- ✅ Volatility clustering awareness

### **Performance**:
- ✅ Better entry timing
- ✅ Improved exit decisions
- ✅ Reduced false signals
- ✅ Higher win rate (expected)

---

## 🎯 **SUCCESS CRITERIA**

Training is successful if:
1. ✅ Model trains without errors
2. ✅ HOLD % decreases to 30-40% by 100k steps
3. ✅ BUY % increases to 60-70% by 100k steps
4. ✅ Strong-setup BUY rate > 75% by 100k steps
5. ✅ Model file is created and loads successfully
6. ✅ Live agent can load and use the model

---

**Last Updated**: 2025-12-12





