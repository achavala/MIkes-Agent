# 🔍 AGENT LOGS DIAGNOSIS

## Problem: No Agent Logs Visible

**Issue:** When running `fly logs`, you only see Streamlit warnings, but no agent logs.

---

## 🔍 Root Cause Analysis

### **Why No Output from grep Commands:**

1. **Agent runs on different machine:**
   - Streamlit dashboard: Machine `28630ddce66198`
   - Trading agent: Machine `48ed77ece94d18`
   - Your grep was looking for agent logs, but they're on a different machine

2. **Agent might be crashing silently:**
   - Agent runs in background (`python mike_agent_live_safe.py &`)
   - If it crashes, errors might not be captured
   - `start_cloud.sh` continues even if agent fails

3. **Logs might have rotated:**
   - Fly.io logs rotate after a certain time
   - Older logs might not be visible

---

## ✅ Fixes Applied

### **1. Improved Error Logging in start_cloud.sh**
- Agent output now redirected to `/tmp/agent.log`
- Startup output is displayed in logs
- Errors are captured and shown

### **2. Better Error Handling**
- Increased wait time from 2s to 5s
- Shows agent startup output in logs
- Displays errors if agent fails

---

## 🔍 How to Check Agent Status

### **Method 1: Check All Logs (Recommended)**
```bash
fly logs --app mike-agent-project --no-tail | grep -v "use_container_width\|Please replace" | tail -100
```

### **Method 2: Check Agent Machine Specifically**
```bash
fly logs --app mike-agent-project --no-tail | grep "app\[48ed77ece94d18\]" | tail -50
```

### **Method 3: Check for Agent Activity**
```bash
fly logs --app mike-agent-project --no-tail | grep -iE "agent|model|trading|alpaca|starting" | tail -30
```

### **Method 4: Check Recent Logs Only**
```bash
fly logs --app mike-agent-project --no-tail | tail -50
```

---

## 📋 What to Look For

### **✅ Good Signs:**
```
🤖 Starting trading agent...
✅ Agent started (PID: XXXX)
📋 Agent startup output:
Loading RL model from models/mike_historical_model.zip...
✓ Model loaded successfully (standard PPO, no action masking)
🧪 Starting Agent in PAPER mode...
✓ Connected to Alpaca (PAPER)
🤖 Trading agent running
```

### **❌ Bad Signs:**
```
⚠️ Agent may have failed to start, checking logs...
Error: ...
Exception: ...
Traceback: ...
```

---

## 🎯 Next Steps

1. **Wait 30 seconds** after deployment
2. **Check logs** using Method 1 above
3. **Look for agent startup output** in the logs
4. **If agent failed**, check the error message in logs

---

## ⚠️ Common Issues

### **Issue 1: Model Loading Error**
**Symptom:** "Model not found" or "Failed to load model"  
**Fix:** Model should be in Docker image (we fixed this)

### **Issue 2: Import Error**
**Symptom:** "No module named X"  
**Fix:** Check requirements.txt includes all dependencies

### **Issue 3: Alpaca Connection Error**
**Symptom:** "Failed to connect to Alpaca"  
**Fix:** Check ALPACA_KEY and ALPACA_SECRET secrets are set

### **Issue 4: Silent Crash**
**Symptom:** Agent starts but immediately dies  
**Fix:** Check `/tmp/agent.log` in the container (now captured in logs)

---

## 📝 Summary

**Status:** ✅ **Error logging improved**

**What changed:**
- Agent output now captured in logs
- Startup output displayed
- Errors are visible

**Next:** Check logs after deployment to see agent startup output.





