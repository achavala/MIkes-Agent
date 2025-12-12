# 🔄 How to Restart the Agent

---

## 🚀 **Quick Restart (Recommended)**

### **Simply run:**
```bash
./restart_agent.sh
```

This script will:
- ✅ Stop any existing agent processes
- ✅ Check dependencies
- ✅ Validate code compiles
- ✅ Start the agent in the background
- ✅ Show you the PID and log file

---

## 📋 **Step-by-Step Manual Restart**

### **Step 1: Stop the existing agent**
```bash
# Stop gracefully
pkill -f mike_agent_live_safe.py

# Wait a moment
sleep 2

# Verify it's stopped
ps aux | grep mike_agent_live_safe | grep -v grep
# (Should show nothing if stopped)
```

### **Step 2: Install dependencies (if needed)**
```bash
pip3 install stable-baselines3 alpaca-trade-api yfinance
```

### **Step 3: Start the agent**
```bash
# Create logs directory
mkdir -p logs

# Start in background
nohup python3 mike_agent_live_safe.py > logs/agent_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Note the PID (shown after &)
echo "Agent started with PID: $!"
```

---

## 📊 **Monitor the Agent**

### **View live logs:**
```bash
# Follow the latest log
tail -f $(ls -t logs/agent_*.log | head -1)

# Or follow all agent logs
tail -f logs/agent_*.log
```

### **Filter for important messages:**
```bash
# Stop-loss triggers
tail -f logs/agent_*.log | grep -i "STOP-LOSS"

# Trades
tail -f logs/agent_*.log | grep -i "TRADE\|BUY\|SELL"

# RL decisions
tail -f logs/agent_*.log | grep -i "RL Inference"

# Errors
tail -f logs/agent_*.log | grep -i "ERROR\|CRITICAL"
```

---

## ✅ **Verify Agent is Running**

```bash
# Check process
ps aux | grep mike_agent_live_safe | grep -v grep

# Check recent logs
tail -20 logs/agent_*.log
```

---

## 🛑 **Stop the Agent**

```bash
# Graceful stop
pkill -f mike_agent_live_safe.py

# Force stop (if needed)
pkill -9 -f mike_agent_live_safe.py
```

---

## ⚠️ **Common Issues**

### **Issue: "stable-baselines3 not installed"**
**Solution:**
```bash
pip3 install stable-baselines3 alpaca-trade-api yfinance
```

### **Issue: "ModuleNotFoundError"**
**Solution:** Install missing module
```bash
pip3 install <module-name>
```

### **Issue: Agent starts but stops immediately**
**Check logs:**
```bash
tail -50 logs/agent_*.log
```

**Common causes:**
- Missing API keys in `config.py`
- Alpaca API connection issue
- Import errors

---

## 🎯 **Expected Behavior After Restart**

### **On startup (first 10 seconds):**
```
✅ Loading RL model from models/mike_ppo_model_historical.zip
✅ Model loaded successfully
✅ CURRENT REGIME: CALM (VIX: 15.2)
✅ 13/13 SAFEGUARDS: ACTIVE
✅ Agent initialized and ready
```

### **Within 1 minute (if -88.79% position exists):**
```
🚨 STEP 1 STOP-LOSS (ALPACA PnL): SPY251210C00688000 @ -88.79% → FORCING IMMEDIATE CLOSE
✓ Position closed: SPY251210C00688000
```

### **Multi-symbol RL logs:**
```
🧠 SPY RL Inference: action=1 (BUY CALL) | Source: RL
🧠 QQQ RL Inference: action=0 (HOLD) | Source: RL
🧠 SPX RL Inference: action=1 (BUY CALL) | Source: RL
```

---

## 📝 **Quick Commands Reference**

| Task | Command |
|------|---------|
| **Restart** | `./restart_agent.sh` |
| **Stop** | `pkill -f mike_agent_live_safe.py` |
| **Monitor** | `tail -f logs/agent_*.log` |
| **Check Status** | `ps aux \| grep mike_agent_live_safe` |
| **View Errors** | `tail -f logs/agent_*.log \| grep ERROR` |
| **Install Deps** | `pip3 install stable-baselines3 alpaca-trade-api yfinance` |

---

**Ready?** Run: `./restart_agent.sh`

