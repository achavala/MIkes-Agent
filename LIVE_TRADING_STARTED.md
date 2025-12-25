# 🚀 **LIVE TRADING - STARTED**

**Date**: 2025-12-12  
**Time**: Started  
**Mode**: **PAPER TRADING**  
**Model**: `mike_momentum_model_v2_intraday_full.zip`

---

## ✅ **AGENT STATUS**

### **Process**
- ✅ **Agent Started**: Running in background
- ✅ **Mode**: Paper Trading (safe for testing)
- ✅ **Model**: `mike_momentum_model_v2_intraday_full.zip` (840 KB)

### **Configuration**
- ✅ **Symbols**: SPY, QQQ, SPX
- ✅ **Position Sizing**: Regime-adjusted (no 10-contract cap)
- ✅ **Stop-Loss**: Hard -15% seatbelt
- ✅ **Take-Profit**: TP1/TP2/TP3 dynamic system
- ✅ **Risk Limits**: All active

---

## 📊 **EXPECTED BEHAVIOR**

### **Action Distribution**
- HOLD: ~12% (matches training: 11.6%)
- BUY_CALL: ~70% (matches training: 69.8%)
- BUY_PUT: ~19% (matches training: 18.6%)
- Strong-setup BUY: ~95% (matches training: 94.9%)

### **Trade Frequency**
- Expected: **35-50 trades** in full trading day
- This is aggressive scalper behavior
- Will tune after first session if needed

### **Risk Control**
- Hard SL: **-15%** (always executes)
- Expected worst: **< -1%** (based on offline eval: -0.36%)

---

## 🔍 **MONITORING COMMANDS**

### **Watch Live Logs**
```bash
tail -f logs/live/agent_*.log | grep -E "(TRADE|INFO|WARNING|ERROR)"
```

### **Check Agent Status**
```bash
ps aux | grep mike_agent_live_safe | grep -v grep
```

### **View Recent Trades**
```bash
grep "TRADE\|EXECUTED\|EXIT" logs/live/agent_*.log | tail -20
```

### **Check Current Positions**
```bash
grep "Open Positions\|Position:" logs/live/agent_*.log | tail -10
```

### **Monitor RL Actions**
```bash
grep "RL Inference\|Action=" logs/live/agent_*.log | tail -20
```

---

## 📋 **WHAT TO WATCH**

### **Entry Quality** ✅
- Entries on EMA/VWAP reclaim?
- Entries on retests?
- Entries on momentum bursts?
- Entries on consolidation → breakout?

### **Exit Quality** ✅
- TP1 hits (20-40% profits)?
- TP2 hits (50-70% profits)?
- TP3 hits (100-200% profits)?
- Stop-losses trigger at -15%?

### **Trading Activity** ✅
- Trade frequency (target: 10-30/day, current: ~40/day expected)
- Symbol distribution (SPY/QQQ/SPX balanced?)
- Action distribution (matches training?)

### **Risk Metrics** ✅
- Worst loss per trade (must be <= -15%)
- Daily drawdown
- Max concurrent positions
- Portfolio risk

---

## 📝 **LOG FILES**

### **Agent Logs**
```
logs/live/agent_YYYYMMDD_HHMMSS.log
```

### **After Session - Collect Logs**
```bash
./collect_paper_mode_logs.sh
```

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

---

## ⚠️ **IF ISSUES OCCUR**

### **If Agent Crashes**
1. Check logs for error messages
2. Verify API keys are set
3. Verify model file exists
4. Restart agent

### **If No Trades**
1. Check if market is open
2. Check if risk limits are blocking
3. Check RL action outputs
4. Verify data feed is working

### **If Too Many Trades**
1. This is expected (35-50/day)
2. Will tune after first session
3. Monitor for overtrading in chop

---

## 🏆 **READY FOR LIVE TRADING**

**Agent is now running in paper mode!**

**Monitor the logs and watch for:**
- Entry timing
- Exit quality
- Trade frequency
- Risk control
- Symbol rotation

**After session, collect logs for analysis and optimization.**

---

**Last Updated**: 2025-12-12 (Agent started)





