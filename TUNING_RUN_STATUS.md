# 🎯 **100K TUNING RUN STATUS**

**Model**: `mike_momentum_model_v2_intraday_tune1`  
**Started**: 2025-12-12  
**Purpose**: Validate anti-collapse fixes before full 500k run

---

## 📊 **SUCCESS CRITERIA**

### **Target Metrics (at 25k-50k steps)**

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **HOLD %** | < 50% | TBD | ⏳ Monitoring |
| **BUY_CALL + BUY_PUT %** | > 50% | TBD | ⏳ Monitoring |
| **Strong-setup BUY rate** | > 70% | TBD | ⏳ Monitoring |
| **Illegal actions (TRIM/EXIT while flat)** | 0% | TBD | ⏳ Monitoring |
| **Value loss** | Stable (no NaN/0) | TBD | ⏳ Monitoring |
| **Reward magnitude** | Stable (no explosion) | TBD | ⏳ Monitoring |

---

## 🔍 **MONITORING CHECKPOINTS**

### **Step 5,000** ⏳
- Expected: HOLD should be < 70% (down from 87%)
- Expected: BUY actions should be > 30%

### **Step 10,000** ⏳
- Expected: HOLD trending down toward 50-60%
- Expected: BUY actions trending up toward 40-50%

### **Step 25,000** ⏳
- Expected: HOLD < 50%
- Expected: BUY_CALL + BUY_PUT > 50%
- Expected: Strong-setup BUY rate > 60%

### **Step 50,000** ⏳
- Expected: HOLD < 40%
- Expected: BUY_CALL + BUY_PUT > 60%
- Expected: Strong-setup BUY rate > 70%

---

## 🚀 **COMMANDS**

### **Check Training Status**
```bash
./monitor_tuning_run.sh
```

### **Watch Live Logs**
```bash
tail -f logs/training/mike_momentum_model_v2_intraday_tune1_100k.log | grep -E "(MomentumDiagnostics|time/|train/)"
```

### **Extract Latest Diagnostics**
```bash
grep "MomentumDiagnostics" logs/training/mike_momentum_model_v2_intraday_tune1_100k.log | tail -1
```

### **Check Process**
```bash
ps aux | grep train_historical_model | grep tune1 | grep -v grep
```

---

## 📝 **DIAGNOSTICS LOG**

### **Step 5,000**
```
[Will be populated when checkpoint reached]
```

### **Step 10,000**
```
[Will be populated when checkpoint reached]
```

### **Step 25,000**
```
[Will be populated when checkpoint reached]
```

### **Step 50,000**
```
[Will be populated when checkpoint reached]
```

---

## ⚠️ **IF STILL COLLAPSING (HOLD > 70% at 25k)**

**Additional tuning options**:

1. **Increase entropy**: `--ent-coef 0.08` or `0.10`
2. **Increase good-buy bonus**: Edit `historical_training_system.py` line 878: `+0.10` → `+0.12` or `+0.15`
3. **Increase missed-op penalty**: Edit `historical_training_system.py` line 868: `-0.05` → `-0.06` or `-0.08`
4. **Increase per-step HOLD tax**: Edit `historical_training_system.py` line 872: `-0.001` → `-0.002`

---

## ✅ **NEXT STEPS AFTER TUNING RUN**

1. **If metrics look healthy** → Start full 500k run with same hyperparameters
2. **If still collapsing** → Apply additional tuning (see above)
3. **If metrics are good but need refinement** → Fine-tune reward weights

---

**Last Updated**: [Auto-updated when diagnostics are checked]





