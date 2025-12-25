# 🚀 **PAPER MODE DEPLOYMENT GUIDE**

**Date**: 2025-12-12  
**Model**: `mike_momentum_model_v2_intraday_full`  
**Status**: ✅ **Ready for Paper Mode Deployment**

---

## ✅ **PRE-DEPLOYMENT VALIDATION**

### **Training** ✅
- ✅ 500k training completed
- ✅ Final metrics: HOLD 11.6%, BUY 88.4%, Strong-setup BUY 94.9%
- ✅ Model converged to scalper behavior

### **Offline Evaluation** ✅
- ✅ Model loads correctly
- ✅ Risk control excellent (worst loss -0.36%)
- ✅ Trading activity good (398 trades, all symbols)
- ⚠️ Break-even performance (needs monitoring)

### **Infrastructure** ✅
- ✅ sb3-contrib installed
- ✅ Model file validated
- ✅ Observation/action spaces correct

---

## 🔧 **DEPLOYMENT STEPS**

### **Step 1: Verify Model Integration**

Check that `mike_agent_live_safe.py` (or your live agent) can:
- ✅ Load the model: `models/mike_momentum_model_v2_intraday_full.zip`
- ✅ Use MaskablePPO for inference
- ✅ Apply action masking correctly
- ✅ Handle observation space (20, 23)

### **Step 2: Enable Paper Trading Mode**

Ensure your live agent has:
- ✅ Paper trading mode enabled (Alpaca paper account)
- ✅ Real-time data feed (Polygon/Massive)
- ✅ Full diagnostics logging
- ✅ Trade execution logging

### **Step 3: Configure Monitoring**

Set up monitoring for:
- ✅ Entry timing (action, confidence, setup_score)
- ✅ Exit timing (TP/SL hits, trim behavior)
- ✅ Symbol selection (SPY/QQQ/SPX distribution)
- ✅ PnL tracking (per trade, daily)
- ✅ Risk metrics (max loss, drawdown)

### **Step 4: Start Paper Trading**

Run your live agent in paper mode:
```bash
# Example (adjust to your actual command)
python3 mike_agent_live_safe.py --paper-mode --model models/mike_momentum_model_v2_intraday_full.zip
```

---

## 📊 **WHAT TO MONITOR DURING PAPER MODE**

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
- Trade frequency (target: 10-30/day, current: ~40/day)
- Symbol distribution (should be balanced)
- Action distribution (should match training: ~88% BUY, ~12% HOLD)

### **Risk Metrics**
- Worst loss per trade (must be <= -15%)
- Daily drawdown
- Max concurrent positions
- Portfolio risk (Delta/Theta/Vega)

### **Performance Metrics**
- Win rate
- Average win vs average loss
- Profit factor
- Daily PnL
- Cumulative PnL

---

## 🎯 **SUCCESS CRITERIA FOR PAPER MODE**

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

## ⚠️ **WARNING SIGNS TO WATCH FOR**

### **If You See These, Stop and Analyze**

1. **Many losses > -10%**
   - Stop-loss may not be working
   - Check hard -15% enforcement

2. **All trades losing**
   - Reward shaping may be wrong
   - Check entry/exit logic

3. **Single symbol dominance**
   - Symbol rotation broken
   - Check cooldown logic

4. **Overtrading (> 50 trades/day)**
   - Model too aggressive
   - May need to increase missed-op penalty or hold tax

5. **No trades (< 5 trades/day)**
   - Model too conservative
   - May need to increase good-buy bonus or decrease entropy

---

## 📝 **PAPER MODE CHECKLIST**

### **Pre-Deployment**
- [x] Model validated and loads correctly
- [x] Offline evaluation passed
- [ ] Model integrated into live agent
- [ ] Paper trading mode enabled
- [ ] Diagnostics logging active
- [ ] Real-time monitoring setup

### **During Paper Mode**
- [ ] Monitor entry quality
- [ ] Monitor exit quality
- [ ] Monitor trading activity
- [ ] Monitor risk metrics
- [ ] Monitor performance metrics
- [ ] Collect detailed trade data

### **Post-Paper Mode**
- [ ] Analyze entry/exit timing
- [ ] Analyze PnL distribution
- [ ] Analyze win rate
- [ ] Analyze TP/SL hit rates
- [ ] Compare to offline eval
- [ ] Decide: proceed to live or fine-tune

---

## 🔧 **INTEGRATION CODE SNIPPETS**

### **Load Model in Live Agent**
```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

# Load model
model = MaskablePPO.load("models/mike_momentum_model_v2_intraday_full.zip")

# Get action masks (when flat: only HOLD/BUY allowed)
action_masks = env.action_masks()

# Predict with masking
action, _ = model.predict(obs, action_masks=action_masks, deterministic=False)
```

### **Enable Paper Trading (Alpaca)**
```python
# Use paper API key
api_key = os.getenv("ALPACA_PAPER_API_KEY")
api_secret = os.getenv("ALPACA_PAPER_SECRET_KEY")
base_url = "https://paper-api.alpaca.markets"  # Paper endpoint
```

---

## 📊 **EXPECTED BEHAVIOR**

Based on training and offline eval:

### **Action Distribution**
- HOLD: ~12% (matches training)
- BUY_CALL + BUY_PUT: ~88% (matches training)
- Strong-setup BUY: ~95% (matches training)

### **Trade Frequency**
- Target: 10-30 trades/day
- Current: ~40 trades/day (slightly high)
- Monitor: Should decrease if model learns to be more selective

### **Risk Control**
- Worst loss: <= -15% (seatbelt working)
- Average loss: < -10%
- Stop-loss triggers: Should be rare but correct

### **Profitability**
- Target: Positive daily PnL
- Current: Break-even (needs monitoring)
- Monitor: Entry/exit quality

---

## 🎯 **NEXT STEPS AFTER PAPER MODE**

### **If Paper Mode Successful** ✅
1. Analyze detailed trade data
2. Fine-tune if needed (entry/exit timing)
3. Scale to small live trading (1 contract)
4. Gradually increase position size

### **If Paper Mode Shows Issues** ⚠️
1. Identify specific problems:
   - Entry quality? → Check setup_score calculation
   - Exit quality? → Check TP/SL logic
   - Overtrading? → Adjust reward weights
   - Undertrading? → Adjust reward weights
2. Fine-tune reward shaping if needed
3. Re-run short training (50k-100k steps)
4. Re-evaluate

---

## 🏆 **CONCLUSION**

**Model is ready for paper mode deployment** with:
- ✅ Excellent risk control
- ✅ Good trading activity
- ✅ Model stability
- ⚠️ Break-even performance (needs monitoring)

**Deploy to paper mode and monitor closely for 1-2 sessions to collect detailed trade data.**

---

**Last Updated**: 2025-12-12





