# 🚨 **ANTI-COLLAPSE FIXES APPLIED**

**Date**: 2025-12-12  
**Issue**: Model collapsing into HOLD mode (87.2% at 85k steps)  
**Status**: ✅ **FIXES IMPLEMENTED**

---

## 📋 **CHANGES MADE**

### **1. Increased Exploration (Entropy Coefficient)**

**File**: `train_historical_model.py` (line 565)

- **Before**: `ent_coef = 0.02` (too low, causing collapse)
- **After**: `ent_coef = 0.06` (default for human_momentum mode)
- **Impact**: Model will explore BUY_CALL/BUY_PUT more aggressively instead of defaulting to HOLD

---

### **2. Strengthened Good-Buy Bonus**

**File**: `historical_training_system.py` (lines 877-878, 894-895)

- **Calls**: `+0.05` → `+0.10` (doubled)
- **Puts**: `+0.02` → `+0.08` (quadrupled)
- **Impact**: Strong setups now provide much better reward signal for taking entries

---

### **3. Strengthened Missed-Opportunity Penalty**

**File**: `historical_training_system.py` (line 868)

- **Before**: `-0.02` (too weak)
- **After**: `-0.05` (2.5x stronger)
- **Impact**: Model will be penalized more for HOLDing during strong setups

---

### **4. Added Per-Step HOLD Tax**

**File**: `historical_training_system.py` (line 871)

- **New**: `-0.001` per step when HOLDing while flat (not in strong setup)
- **Impact**: Makes "doing nothing all day" slightly worse than taking some risk
- **Note**: Over a full trading day (390 minutes), this accumulates to ~`-0.39`, which is still small compared to scalp rewards (+0.3 to +2.0)

---

### **5. Reduced Bad-Chase Penalty**

**File**: `historical_training_system.py` (line 845)

- **Before**: `-0.07` (too harsh, could drive model back to HOLD)
- **After**: `-0.03` (reduced to avoid over-punishment)
- **Impact**: Model won't be terrified of buying anything, preventing retreat to HOLD

---

## 🎯 **EXPECTED BEHAVIOR AFTER FIXES**

### **Target Metrics (at 25k-50k steps)**

- **FLAT action distribution**:
  - HOLD: **30-40%** (down from 87%)
  - BUY_CALL + BUY_PUT: **60-70%** (up from 13%)

- **Strong-setup behavior**:
  - BUY rate on strong setups: **70-85%** (up from 14.5%)
  - HOLD rate on strong setups: **< 20-25%** (down from 85.5%)

- **Trigger counts**:
  - `good_buy_bonus` should dominate `missed_opportunity`
  - `bad_chase_penalty` should remain small

---

## 🚀 **NEXT STEPS**

### **Step 1: Run 100k Tuning Run (Recommended First)**

```bash
python3 -u train_historical_model.py \
  --symbols SPY,QQQ,SPX \
  --timesteps 100000 \
  --model-name mike_momentum_model_v2_intraday_tune1 \
  --human-momentum \
  --data-source massive \
  --intraday-days 60 \
  --learning-rate 3e-5 \
  --ent-coef 0.06 \
  --gamma 0.92 \
  --n-steps 512
```

**Watch diagnostics at**: 5k, 10k, 25k, 50k steps

**Success criteria**:
- HOLD trending **down** toward 30-40%
- BUY_CALL + BUY_PUT trending **up** to 60-70%
- Strong-setup BUY rate **> 70%**

---

### **Step 2: If Tuning Run Shows Healthy Behavior → Full 500k Run**

```bash
python3 -u train_historical_model.py \
  --symbols SPY,QQQ,SPX \
  --timesteps 500000 \
  --model-name mike_momentum_model_v2_intraday_full \
  --human-momentum \
  --data-source massive \
  --intraday-days 60 \
  --learning-rate 3e-5 \
  --ent-coef 0.06 \
  --gamma 0.92 \
  --n-steps 512
```

---

### **Step 3: If Still Collapsing (HOLD > 70% at 25k steps)**

**Additional tuning options**:

1. **Increase entropy further**: `--ent-coef 0.08` or `0.10`
2. **Increase good-buy bonus**: Change `+0.10` → `+0.12` or `+0.15` in code
3. **Increase missed-op penalty**: Change `-0.05` → `-0.06` or `-0.08` in code
4. **Increase per-step HOLD tax**: Change `-0.001` → `-0.002` in code

---

## 📊 **MONITORING COMMANDS**

### **Check Training Progress**

```bash
tail -f logs/training/mike_momentum_model_v2_intraday_tune1_100k.log | grep -A 10 "MomentumDiagnostics"
```

### **Check Process Status**

```bash
ps aux | grep train_historical_model | grep -v grep
```

### **Check Latest Diagnostics**

```bash
grep "MomentumDiagnostics" logs/training/mike_momentum_model_v2_intraday_tune1_100k.log | tail -5
```

---

## ✅ **VALIDATION CHECKLIST**

After 25k-50k steps, verify:

- [ ] HOLD % < 50%
- [ ] BUY_CALL + BUY_PUT % > 50%
- [ ] Strong-setup BUY rate > 60%
- [ ] `good_buy_bonus` count > `missed_opportunity` count
- [ ] No reward explosion (NaN/inf)
- [ ] Loss curves stable (not diverging)

---

**END OF FIXES DOCUMENT**





