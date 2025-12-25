# 🚀 **LIVE AGENT - RUNNING**

**Date**: 2025-12-12  
**Time**: Started ~10:01 AM  
**Status**: ✅ **RUNNING IN PAPER MODE**

---

## ✅ **AGENT STATUS**

### **Process**
- ✅ **PID**: 44607
- ✅ **Status**: Running (56.7% CPU - actively working)
- ✅ **Mode**: Paper Trading
- ✅ **Model**: `mike_momentum_model_v2_intraday_full.zip`

### **Fixes Applied**
- ✅ **Observation shape**: Fixed to (20, 23) - matches training
- ✅ **10-contract limit**: Removed - system decides
- ✅ **MaskablePPO**: Enabled with action masking

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

## 🔍 **MONITORING**

### **Check Agent Status**
```bash
ps aux | grep mike_agent_live_safe | grep -v grep
```

### **View Logs**
```bash
tail -f logs/live/agent_*.log | grep -E "(TRADE|INFO|WARNING|ERROR)"
```

### **Check Recent Trades**
```bash
grep "TRADE\|EXECUTED\|EXIT" logs/live/agent_*.log | tail -20
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

## 📝 **AFTER SESSION**

### **Collect Logs**
```bash
./collect_paper_mode_logs.sh
```

### **Send for Analysis**
- Top 10 profitable trades
- Top 10 losing trades
- Any -15% stops triggered
- Action probability snapshots
- BUY/HOLD patterns

---

## 🏆 **AGENT IS LIVE AND TRADING**

**The agent is now running in paper mode and ready to trade!**

**Monitor the logs and watch for trading activity throughout the day.**

---

**Last Updated**: 2025-12-12 (Agent started)





