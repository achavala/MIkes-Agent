# 🚀 **500K FULL INTRADAY TRAINING STATUS**

**Model**: `mike_momentum_model_v2_intraday_full`  
**Started**: 2025-12-12  
**Purpose**: Full-scale training with validated Tune2 parameters

---

## 📊 **TRAINING CONFIGURATION**

| Parameter | Value | Source |
|-----------|-------|--------|
| **Timesteps** | 500,000 | Full training |
| **Model Name** | `mike_momentum_model_v2_intraday_full` | Production model |
| **Symbols** | SPY, QQQ, SPX | Multi-symbol |
| **Data Source** | Massive/Polygon (1m bars) | Intraday |
| **Data Window** | Last 60 days | Recent data |
| **Human Momentum Mode** | ✅ Enabled | MikeInvesting-style |
| **Learning Rate** | 3e-5 | Optimized |
| **Entropy Coefficient** | 0.08 | Validated from Tune2 |
| **Gamma** | 0.92 | Scalping discount |
| **N-Steps** | 512 | PPO batch size |

---

## 📈 **EXPECTED METRICS (Based on Tune2 Pattern)**

### **At 25,000 Steps**
- HOLD: ~45-55%
- Strong-setup BUY rate: 55-65%

### **At 50,000 Steps**
- HOLD: ~40-50%
- Strong-setup BUY rate: 60-70%

### **At 100,000 Steps**
- HOLD: ~35-45%
- Strong-setup BUY rate: 70-75%

### **At 250,000 Steps**
- HOLD: ~35-40%
- Strong-setup BUY rate: 75-80%

### **At 500,000 Steps** (Final)
- HOLD: **30-40%** ✅
- BUY_CALL + BUY_PUT: **60-70%+** ✅
- Strong-setup BUY rate: **75-85%** ✅

---

## 🎯 **SUCCESS CRITERIA**

### **Training Health Checks**
- ✅ HOLD trending down (not rising)
- ✅ Strong-setup BUY rate trending up
- ✅ No value loss collapse (NaN/0)
- ✅ Reward magnitude stable (no explosion)
- ✅ Illegal actions (TRIM/EXIT while flat) = 0%

### **Final Model Quality (At 500k)**
- ✅ HOLD ≤ 40%
- ✅ Combined BUY actions ≥ 60%
- ✅ Strong-setup BUY rate ≥ 75%
- ✅ Good trigger balance (good_buy >> missed_opportunity)

---

## 🔍 **MONITORING COMMANDS**

### **Quick Status Check**
```bash
./monitor_tuning_run.sh
```

### **Watch Live Diagnostics**
```bash
tail -f logs/training/mike_momentum_model_v2_intraday_full_500k.log | grep -E "(MomentumDiagnostics|time/|train/)"
```

### **Extract Latest Diagnostics**
```bash
grep "MomentumDiagnostics" logs/training/mike_momentum_model_v2_intraday_full_500k.log | tail -1
```

### **Check Specific Checkpoint**
```bash
grep -A 6 "MomentumDiagnostics @ step=25,000" logs/training/mike_momentum_model_v2_intraday_full_500k.log
```

### **Check Process Status**
```bash
ps aux | grep train_historical_model | grep full | grep -v grep
```

---

## 📝 **KEY CHECKPOINTS TO MONITOR**

| Checkpoint | Expected HOLD % | Expected Strong-Setup BUY % | Action |
|------------|-----------------|----------------------------|--------|
| **5k** | ~50-55% | ~45-50% | Baseline |
| **10k** | ~50-60% | ~45-55% | May see early exploration dip |
| **25k** | ~45-55% | ~55-65% | Recovery phase |
| **50k** | ~40-50% | ~60-70% | Should match Tune2 pattern |
| **100k** | ~35-45% | ~70-75% | Approaching targets |
| **250k** | ~35-40% | ~75-80% | Stabilization |
| **500k** | **30-40%** | **75-85%** | Final model |

---

## 🧪 **POST-TRAINING: OFFLINE EVALUATION**

After 500k training completes, run:

```bash
python3 validate_model.py \
  --model models/mike_momentum_model_v2_intraday_full.zip \
  --offline-eval \
  --intraday \
  --symbols SPY,QQQ,SPX \
  --intraday-days 10 \
  --stochastic
```

### **Evaluation Criteria**
- ✅ Trades/day ≥ 10-30
- ✅ No trade exceeds -15% (seatbelt working)
- ✅ Good BUY patterns on momentum
- ✅ TP1/TP2/TP3 hit rates
- ✅ Model doesn't overtrade chop
- ✅ Good symbol balancing across SPY/QQQ/SPX

---

## 📊 **TUNE2 VALIDATION (Reference)**

Tune2 achieved at 100k:
- HOLD: **38.2%** ✅
- Combined BUY: **61.8%** ✅
- Strong-setup BUY: **68.3%** ✅

**Pattern**: Early dip at 10k (63% HOLD) → Recovery by 25k → Continued improvement → Targets met at 100k

**Conclusion**: Tune2 is healthy and ready for full-scale training.

---

## ⚠️ **IF METRICS DEVIATE FROM EXPECTATIONS**

### **If HOLD > 60% at 25k**
- Check if it's a temporary exploration dip (like Tune2 at 10k)
- If it persists past 50k, consider increasing `ent_coef` to 0.10

### **If Strong-Setup BUY Rate < 50% at 50k**
- May need to increase `good_buy_bonus` further
- Or increase `missed_opportunity_penalty`

### **If Value Loss Collapses (NaN/0)**
- Stop training immediately
- Check for reward explosion or observation space issues

---

## 📅 **ESTIMATED COMPLETION TIME**

Based on Tune2 performance (~217 fps):
- **100k steps**: ~8-10 minutes
- **500k steps**: ~40-50 minutes

**Note**: Actual time may vary based on system load and data loading.

---

## ✅ **NEXT STEPS AFTER TRAINING**

1. **Validate training metrics** (check 500k diagnostics)
2. **Run offline evaluation** (see command above)
3. **If eval passes** → Move to paper mode
4. **If paper mode successful** → Deploy to live trading

---

**Last Updated**: Training started 2025-12-12





