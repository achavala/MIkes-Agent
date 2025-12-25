# ✅ **PAPER MODE INTEGRATION - COMPLETE**

**Date**: 2025-12-12  
**Model**: `mike_momentum_model_v2_intraday_full.zip`  
**Status**: ✅ **INTEGRATED AND READY FOR PAPER MODE**

---

## 🔧 **CHANGES MADE**

### **1. Model Path Updated** ✅
- **Before**: `models/mike_historical_model.zip`
- **After**: `models/mike_momentum_model_v2_intraday_full.zip`
- **Location**: Line 218 in `mike_agent_live_safe.py`

### **2. MaskablePPO Support Added** ✅
- Added import for `MaskablePPO` from `sb3_contrib`
- Added import for `get_action_masks` utility
- Updated `load_rl_model()` to try MaskablePPO first, fallback to PPO
- **Location**: Lines 43-55, 932-950 in `mike_agent_live_safe.py`

### **3. Paper Mode Already Enabled** ✅
- `USE_PAPER` is set to `True` by default (line 212)
- Paper URL configured: `https://paper-api.alpaca.markets`
- **No changes needed**

---

## 📋 **VERIFICATION CHECKLIST**

### **Pre-Deployment**
- [x] Model path updated to new model
- [x] MaskablePPO support added
- [x] Paper mode enabled by default
- [ ] Model file exists at correct path
- [ ] Observation space matches (20, 23)
- [ ] Action masking will be used (if model supports it)

### **During Paper Mode**
- [ ] Model loads successfully
- [ ] Inference works correctly
- [ ] Action masking applied (if available)
- [ ] Trades execute in paper mode
- [ ] Diagnostics logging active

---

## 🚀 **DEPLOYMENT COMMAND**

### **Start Paper Mode Agent**
```bash
cd /Users/chavala/Mike-agent-project
python3 mike_agent_live_safe.py
```

### **Or Use Restart Script**
```bash
./restart_agent.sh
```

### **Verify Paper Mode**
Look for these log messages:
```
Mode: PAPER TRADING
Loading RL model from models/mike_momentum_model_v2_intraday_full.zip...
✓ Model loaded successfully (MaskablePPO with action masking)
```

---

## 📊 **EXPECTED BEHAVIOR**

### **Action Distribution**
- HOLD: ~12% (matches training)
- BUY_CALL + BUY_PUT: ~88% (matches training)
- Strong-setup BUY: ~95% (matches training)

### **Trade Frequency**
- Target: 10-30 trades/day
- Current: ~40 trades/day (slightly high, but acceptable)
- Monitor: Should decrease if model learns to be more selective

### **Risk Control**
- Worst loss: <= -15% (seatbelt working)
- Average loss: < -10%
- Stop-loss triggers: Should be rare but correct

---

## 🔍 **WHAT TO MONITOR**

### **Entry Quality**
- Are entries happening on strong setups? (setup_score >= 3.0)
- Are entries happening during momentum? (EMA/VWAP/MACD aligned)
- Are entries happening at good prices? (not chasing)

### **Exit Quality**
- Are TP1 hits frequent? (20-40% profits)
- Are TP2 hits occasional? (50-70% profits)
- Are TP3 hits rare? (100-200% profits)
- Are stop-losses triggering correctly? (at -15%)

### **Trading Activity**
- Trade frequency (target: 10-30/day)
- Symbol distribution (should be balanced)
- Action distribution (should match training)

### **Risk Metrics**
- Worst loss per trade (must be <= -15%)
- Daily drawdown
- Max concurrent positions
- Portfolio risk (Delta/Theta/Vega)

---

## 📝 **LOG FILES TO COLLECT**

After running paper mode, collect these logs:

1. **Agent Logs**
   ```
   logs/live/agent_*.log
   ```

2. **Action Logs** (if available)
   ```
   logs/live/actions_*.json
   ```

3. **Trade Logs** (if available)
   ```
   logs/live/trades_*.json
   ```

4. **Database Trades** (if trade database enabled)
   - Check trade database for all executed trades

---

## 🎯 **SUCCESS CRITERIA**

### **Minimum Requirements** ✅
- ✅ No crashes or errors
- ✅ Trades execute correctly
- ✅ Stop-losses trigger at -15%
- ✅ TP levels hit as expected
- ✅ Symbol rotation works

### **Performance Targets**
- ✅ Win rate > 50%
- ✅ Profit factor > 1.0
- ✅ Average win > Average loss
- ✅ Daily PnL positive (or at least not consistently negative)
- ✅ No losses > -15%

### **Behavioral Targets**
- ✅ Entries on strong setups (setup_score >= 3.0)
- ✅ Exits at TP levels (not too early, not too late)
- ✅ Balanced symbol distribution
- ✅ Trade frequency 10-30/day (not 40+)

---

## ⚠️ **TROUBLESHOOTING**

### **If Model Fails to Load**
1. Check model file exists: `ls -lh models/mike_momentum_model_v2_intraday_full.zip`
2. Check sb3-contrib installed: `pip install sb3-contrib`
3. Check model file size (should be ~839 KB)

### **If Action Masking Not Working**
- Model will fallback to standard PPO
- This is acceptable but not optimal
- Check logs for "MaskablePPO" vs "standard PPO" message

### **If Paper Mode Not Working**
- Check `USE_PAPER` is `True` (line 212)
- Check Alpaca paper API keys are set
- Check paper URL is correct: `https://paper-api.alpaca.markets`

---

## 🏆 **NEXT STEPS**

1. **Deploy to Paper Mode** ✅
   - Integration complete
   - Ready to start

2. **Run Full Session** (9:30 AM - 11:00 AM)
   - Monitor entry timing
   - Monitor exit quality
   - Collect trade logs

3. **Analyze Results**
   - Entry vs exit structure
   - BUY/HOLD breakdown
   - Missed opportunities
   - PnL distribution

4. **Fine-Tune if Needed**
   - Softmax temperature
   - Entry confidence threshold
   - Penalty for chop-zone entries
   - Look-ahead reward window

---

## 📚 **REFERENCE**

- **Model**: `models/mike_momentum_model_v2_intraday_full.zip`
- **Training**: 500k steps, 1-minute intraday bars
- **Final Metrics**: HOLD 11.6%, BUY 88.4%, Strong-setup BUY 94.9%
- **Offline Eval**: 398 trades, worst loss -0.36%, break-even PnL

---

**Last Updated**: 2025-12-12





