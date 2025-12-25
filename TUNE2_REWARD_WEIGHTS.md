# 🎯 **TUNE2 REWARD WEIGHTS & HYPERPARAMETERS**

**Date**: 2025-12-12  
**Model**: `mike_momentum_model_v2_intraday_tune2`  
**Purpose**: Prevent slow recollapse observed in tune1 (HOLD 55% → 61% at 10k)

---

## 📊 **UPDATED REWARD WEIGHTS**

| Reward Component | Tune1 Value | Tune2 Value | Change | Reason |
|------------------|-------------|------------|--------|--------|
| **good_buy_bonus (CALL)** | +0.10 | **+0.12** | +0.02 | Strengthen BUY bias on strong setups |
| **good_buy_bonus (PUT)** | +0.08 | **+0.10** | +0.02 | Match calls, maintain strong BUY bias |
| **missed_opportunity_penalty** | -0.05 | **-0.06** | -0.01 | Sweet spot for scalper-style agent |
| **bad_chase_penalty** | -0.03 | **-0.03** | No change | Already optimal (avoid over-punishment) |
| **hold_tax (per step)** | -0.001 | **-0.0005** | +0.0005 | Softer tax to avoid overwhelming PPO early |

---

## ⚙️ **UPDATED HYPERPARAMETERS**

| Parameter | Tune1 Value | Tune2 Value | Change | Reason |
|-----------|-------------|-------------|--------|--------|
| **ent_coef** | 0.06 | **0.08** | +0.02 | PPO with action masking + 23-feature obs needs 0.08-0.12 to stay exploratory |

---

## 📈 **TUNE1 RESULTS (For Reference)**

### **Step 5,000** ✅
- HOLD: 54.9% (good improvement from 87%)
- BUY_CALL: 24.1%
- BUY_PUT: 21.0%
- Strong-setup BUY rate: 45.9% (very good)

### **Step 10,000** ⚠️
- HOLD: 61.4% (slow recollapse starting)
- BUY_CALL: 22.1%
- BUY_PUT: 16.4%
- Strong-setup BUY rate: 39.2% (weakening)

**Diagnosis**: Reward balance still too shallow, entropy pressure still too weak.

---

## 🎯 **TUNE2 SUCCESS CRITERIA**

### **By Step 10,000**
- ✅ HOLD ≤ **55%** (must not rise above tune1's 5k level)
- ✅ BUY_CALL + BUY_PUT ≥ **45%**
- ✅ Strong-setup BUY rate ≥ **50-60%**

### **By Step 25,000**
- ✅ HOLD ≤ **40%** (must continue trending down)
- ✅ BUY_CALL + BUY_PUT ≥ **60%**
- ✅ Strong-setup BUY rate ≥ **70%**

### **If HOLD Still Rises Past 60% at 10k**
- Increase `ent_coef` to **0.10** for tune3
- Consider increasing `good_buy_bonus` to **+0.15** for calls

---

## 🚀 **TUNE2 TRAINING COMMAND**

```bash
python3 -u train_historical_model.py \
  --symbols SPY,QQQ,SPX \
  --timesteps 100000 \
  --model-name mike_momentum_model_v2_intraday_tune2 \
  --human-momentum \
  --data-source massive \
  --intraday-days 60 \
  --learning-rate 3e-5 \
  --ent-coef 0.08 \
  --gamma 0.92 \
  --n-steps 512
```

---

## 📝 **CHANGES IMPLEMENTED**

### **Files Modified**:
1. `historical_training_system.py`:
   - Line 868: `missed_opportunity_penalty`: -0.05 → -0.06
   - Line 872: `hold_tax`: -0.001 → -0.0005
   - Line 878: `good_buy_bonus (CALL)`: +0.10 → +0.12
   - Line 895: `good_buy_bonus (PUT)`: +0.08 → +0.10

2. `train_historical_model.py`:
   - Line 565: Default `ent_coef` for human_momentum: 0.06 → 0.08

---

**END OF TUNE2 CONFIGURATION**





