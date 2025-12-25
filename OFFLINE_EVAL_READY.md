# 🧪 **OFFLINE EVALUATION - READY CHECKLIST**

**Date**: 2025-12-12  
**Model**: `mike_momentum_model_v2_intraday_full`  
**Status**: ⏳ **Waiting for 500k training completion**

---

## ✅ **PRE-EVALUATION CHECKLIST**

### **Training Status**
- [x] Training started successfully
- [x] Early diagnostics validated (5k, 10k, 25k, 50k)
- [x] Training pattern confirmed healthy
- [ ] Training completed to 500k steps
- [ ] Final diagnostics extracted (100k, 250k, 500k)
- [ ] Model file saved: `models/mike_momentum_model_v2_intraday_full.zip`

### **Model Validation**
- [ ] Model file exists and is loadable
- [ ] Model has correct observation space (20×23)
- [ ] Model has correct action space (6 discrete actions)
- [ ] Action masking is properly configured

---

## 🚀 **OFFLINE EVALUATION COMMAND**

Once training completes, run:

```bash
python3 validate_model.py \
  --model models/mike_momentum_model_v2_intraday_full.zip \
  --offline-eval \
  --intraday \
  --symbols SPY,QQQ,SPX \
  --intraday-days 10 \
  --stochastic
```

---

## 📊 **EXPECTED RESULTS**

### **Trade Frequency**
- ✅ **10-30 trades/day** (scalper-like frequency)
- ✅ **Not overtrading** (< 50 trades/day)
- ✅ **Not undertrading** (> 5 trades/day)

### **Risk Management**
- ✅ **No SL breaches > -15%** (seatbelt working)
- ✅ **Average loss per losing trade < -10%**
- ✅ **Max drawdown per trade < -15%**

### **Entry Quality**
- ✅ **Momentum-based entries** (not random)
- ✅ **Strong-setup entries** (high setup_score)
- ✅ **Good entry timing** (not chasing)

### **Exit Quality**
- ✅ **TP1/TP2/TP3 hit rates** (tiered exits working)
- ✅ **Proper trim behavior** (not holding too long)
- ✅ **Stop-loss enforcement** (exits on invalidation)

### **Symbol Rotation**
- ✅ **Good symbol balancing** (SPY/QQQ/SPX all traded)
- ✅ **No single-symbol bias**
- ✅ **Cooldown respect** (not over-trading one symbol)

### **Profit Distribution**
- ✅ **Profit clustering on strong setups**
- ✅ **Wins > Losses** (or at least balanced)
- ✅ **Average win > Average loss**

---

## 🔍 **WHAT TO LOOK FOR IN EVAL OUTPUT**

### **Good Signs** ✅
- High trade frequency (10-30/day)
- Good win rate (> 50%)
- Strong-setup trades more profitable
- No catastrophic losses
- Symbol diversity

### **Warning Signs** ⚠️
- Too few trades (< 5/day) → Model too conservative
- Too many trades (> 50/day) → Model overtrading
- Many losses > -15% → Stop-loss not working
- Single symbol dominance → Rotation broken
- All trades losing → Reward shaping issue

---

## 📝 **POST-EVALUATION STEPS**

### **If Evaluation Passes** ✅
1. **Extract action_probs distribution** for confidence calibration
2. **Integrate into live paper-mode agent**
3. **Monitor for 1-2 sessions**
4. **Compare to human scalper behavior**
5. **Fine-tune if needed**

### **If Evaluation Fails** ⚠️
1. **Identify specific failure mode**:
   - Too few trades → Increase entropy or good-buy bonus
   - Too many trades → Increase missed-op penalty or hold tax
   - Losses > -15% → Check stop-loss enforcement
   - Poor entry timing → Check setup_score calculation
2. **Adjust reward weights** if needed
3. **Re-run short tuning run** (50k-100k steps)
4. **Re-evaluate**

---

## 🔧 **CONFIDENCE CALIBRATION (Post-Eval)**

After offline eval, we'll:

1. **Extract mean action_probs distribution**
2. **Analyze confidence levels**:
   - Strong setups → Should have high BUY probability
   - Weak setups → Should have high HOLD probability
   - Chop → Should have balanced probabilities
3. **Adjust softmax temperature** in live inference:
   - Boost strong signals (higher confidence on strong setups)
   - Flatten weak ones (lower confidence during chop)
4. **Re-test in paper mode**

**Benefits**:
- ✅ Fewer marginal BUYs during chop
- ✅ More decisive BUYs during strong momentum
- ✅ Cleaner separation between HOLD and BUY

---

## 📋 **EVALUATION OUTPUT TEMPLATE**

Save evaluation results to:
```
evaluation_results/mike_momentum_model_v2_intraday_full_eval_YYYYMMDD_HHMMSS.txt
```

Include:
- Total trades
- Trades per day
- Win rate
- Average win/loss
- Max loss
- TP1/TP2/TP3 hit rates
- Symbol distribution
- Strong-setup vs weak-setup performance

---

**Last Updated**: 2025-12-12





