# 🎯 **TODAY'S ACTION PLAN** (Citadel-Grade Roadmap)

**Date**: 2025-12-12  
**Priority**: Paper Mode Testing FIRST (before activating execution/greeks)

---

## ✅ **PHASE 0: COMPLETE** ✅

- [x] LSTM training complete
- [x] Model validated
- [x] Live agent updated
- [x] Integration tested

---

## 🚀 **PHASE 1: PAPER MODE TESTING** (DO THIS NOW)

### **Why First**:
> "Do NOT change 3 major variables at once" - Standard systematic trading discipline

We have:
- ✅ New LSTM model (just trained)
- ⏳ ExecutionModel (not yet activated)
- ⏳ PortfolioGreeksManager (not yet activated)

**Correct Order**: Test LSTM FIRST, then activate execution/greeks

---

## 📋 **STEP-BY-STEP: START PAPER MODE**

### **Step 1: Start Agent** (2 minutes)

```bash
python3 mike_agent_live_safe.py
```

**What to Watch**:
- ✅ Model loads: Should see "RecurrentPPO with LSTM temporal intelligence"
- ✅ No errors on startup
- ✅ Agent starts monitoring market

---

### **Step 2: Monitor Health Metrics** (Continuous)

**Option A: Use Monitoring Script** (Recommended):
```bash
# Run every 10-15 minutes
./monitor_paper_mode.sh
```

**Option B: Manual Monitoring**:
```bash
# Watch live logs
tail -f logs/live/agent_*.log | grep "RL Inference"

# Check action distribution
grep "RL Inference" logs/live/agent_*.log | grep -oP "action=\K[0-5]" | sort | uniq -c
```

---

### **Step 3: Validate 6 Health Metrics**

| # | Metric | Target | How to Check |
|---|--------|--------|--------------|
| 1 | **HOLD Rate** | < 40% | `./monitor_paper_mode.sh` or count HOLD actions |
| 2 | **BUY Rate** | 40-60% | Count BUY_CALL + BUY_PUT |
| 3 | **EXIT Rate** | > 30% | Count FULL EXIT actions |
| 4 | **Confidence Scores** | 0.55-0.85 | Check "Strength:" values in logs |
| 5 | **Signal Timing** | 1-3 bars | Manual: Check if actions change after trend shifts |
| 6 | **Action Distribution** | Diverse | Should see multiple action types |

---

### **Step 4: Red Flags to Watch**

**Stop and investigate if you see**:

- ❌ **HOLD > 60%** = Model too conservative or collapsed
- ❌ **BUY < 20%** = Underfitting (model not learning)
- ❌ **EXIT > 60%** = Fear behavior (exiting too aggressively)
- ❌ **BUY > 70%** = Reckless bias (entering too aggressively)
- ❌ **All actions same strength** = Model collapsed
- ❌ **No actions for >30 minutes** = Model stuck

---

### **Step 5: Validation Criteria** (After 1-2 Sessions)

**Pass if**:
- ✅ HOLD Rate < 40%
- ✅ BUY Rate 20-70% (healthy range)
- ✅ EXIT Rate 30-60% (temporal intelligence working)
- ✅ Confidence scores 0.55-0.85
- ✅ Actions change within 1-3 bars of trend shifts
- ✅ Diverse action distribution

**Fail if**:
- ❌ Model is too conservative (HOLD > 60%)
- ❌ Model is too aggressive (BUY > 70%)
- ❌ Model collapsed (all same action)
- ❌ No temporal behavior (doesn't react to trends)

---

## ⏳ **AFTER PAPER MODE VALIDATION** (This Week)

### **If Paper Mode PASSES**:

1. **Activate ExecutionModel** (2-3 hours)
   - Integrate into backtester
   - Add to RL environment
   - Add to live trading

2. **Activate PortfolioGreeksManager** (1-2 hours)
   - Import into live agent
   - Add portfolio checks
   - Update on trades

### **If Paper Mode FAILS**:

1. **Analyze failure reason**:
   - Too conservative? → Retrain with higher entropy
   - Too aggressive? → Retrain with lower entropy
   - Collapsed? → Check training logs

2. **Retrain if needed**:
   - Adjust reward weights
   - Adjust entropy coefficient
   - Retrain for 100k-200k steps

---

## 📊 **MONITORING TOOLS**

### **Quick Status Check**:
```bash
./monitor_paper_mode.sh
```

### **Watch Live**:
```bash
tail -f logs/live/agent_*.log | grep -E "RL Inference|SIGNAL|EXECUTED"
```

### **Check Action Distribution**:
```bash
grep "RL Inference" logs/live/agent_*.log | \
  grep -oP "action=\K[0-5]" | \
  sort | uniq -c | \
  awk '{print "Action "$2": "$1" times"}'
```

### **Check Confidence Scores**:
```bash
grep "Strength:" logs/live/agent_*.log | \
  grep -oP "Strength: \K[0-9.]+" | \
  awk '{sum+=$1; count++; if($1<min||min==0) min=$1; if($1>max) max=$1} \
       END {print "Avg: "sum/count", Min: "min", Max: "max}'
```

---

## 🎯 **SUCCESS CRITERIA FOR TODAY**

### **Minimum**:
- [ ] Agent started in paper mode
- [ ] Model loads successfully
- [ ] RL inference working (actions logged)
- [ ] No critical errors

### **Ideal**:
- [ ] 1 full market session monitored
- [ ] All 6 health metrics checked
- [ ] Action distribution validated
- [ ] Temporal behavior confirmed

---

## 🚀 **READY TO START**

**Command to start**:
```bash
python3 mike_agent_live_safe.py
```

**Command to monitor** (in another terminal):
```bash
./monitor_paper_mode.sh
```

**Command to watch live** (in another terminal):
```bash
tail -f logs/live/agent_*.log | grep "RL Inference"
```

---

**Status**: ✅ **READY TO START PAPER MODE TESTING**

**Next**: After 1-2 sessions, proceed to Phase 2 (Execution & Portfolio Greeks)

---

**Last Updated**: 2025-12-12





