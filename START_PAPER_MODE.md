# 🚀 **START PAPER MODE TESTING**

**Quick Start Guide**

---

## ✅ **READY TO START**

Your LSTM model is:
- ✅ Trained (500k steps)
- ✅ Validated (loads successfully)
- ✅ Integrated (MODEL_PATH updated)
- ✅ Ready for testing

---

## 🚀 **START COMMAND**

```bash
python3 mike_agent_live_safe.py
```

---

## 📊 **MONITORING** (Run in Separate Terminal)

### **Option 1: Health Monitor** (Recommended)
```bash
# Run every 10-15 minutes
./monitor_paper_mode.sh
```

### **Option 2: Watch Live Logs**
```bash
tail -f logs/live/agent_*.log | grep "RL Inference"
```

---

## ✅ **WHAT TO LOOK FOR**

### **Good Signs** ✅:
- Model loads: "RecurrentPPO with LSTM temporal intelligence"
- Actions logged: Multiple action types (not all HOLD)
- Confidence scores: 0.55-0.85 range
- Actions change: Responds to market conditions

### **Red Flags** ⚠️:
- HOLD > 60% = Too conservative
- BUY < 20% = Underfitting
- EXIT > 60% = Fear behavior
- All same action = Collapsed

---

## 📋 **VALIDATION CHECKLIST**

After 1-2 market sessions, check:

- [ ] HOLD Rate < 40%
- [ ] BUY Rate 20-70%
- [ ] EXIT Rate 30-60%
- [ ] Confidence scores 0.55-0.85
- [ ] Temporal behavior (reacts to trends)
- [ ] Diverse action distribution

---

## ⏭️ **AFTER VALIDATION**

**If PASSES** → Proceed to Phase 2 (Execution & Portfolio Greeks)  
**If FAILS** → Analyze and retrain if needed

---

**Ready? Start now**: `python3 mike_agent_live_safe.py`





