# 🔄 How to Restart the Agent

**Quick Guide**: Restart the Mike Agent trading system

---

## 🚀 **Method 1: Using the Restart Script (Recommended)**

### **Step 1: Run the restart script**
```bash
./restart_agent.sh
```

This script will:
- ✅ Stop any existing agent processes
- ✅ Validate the code compiles
- ✅ Start the agent in the background
- ✅ Show you the PID and log file location

---

## 🛠️ **Method 2: Manual Restart**

### **Step 1: Stop the existing agent**
```bash
# Find running processes
ps aux | grep mike_agent_live_safe

# Stop gracefully
pkill -f mike_agent_live_safe.py

# If needed, force kill
pkill -9 -f mike_agent_live_safe.py
```

### **Step 2: Verify code compiles**
```bash
python3 -m py_compile mike_agent_live_safe.py
```

### **Step 3: Start the agent**
```bash
# Create logs directory
mkdir -p logs

# Start in background with logging
nohup python3 mike_agent_live_safe.py > logs/agent_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## 📊 **Monitor the Agent**

### **View real-time logs:**
```bash
# Find the latest log file
ls -t logs/agent_*.log | head -1

# Follow the log
tail -f logs/agent_*.log

# Or follow the most recent
tail -f $(ls -t logs/agent_*.log | head -1)
```

### **Watch for specific messages:**
```bash
# Watch for stop-loss triggers
tail -f logs/agent_*.log | grep -i "STOP-LOSS"

# Watch for trades
tail -f logs/agent_*.log | grep -i "TRADE"

# Watch for RL decisions
tail -f logs/agent_*.log | grep -i "RL Inference"

# Watch for errors
tail -f logs/agent_*.log | grep -i "ERROR\|CRITICAL"
```

---

## ✅ **Verify Agent is Running**

### **Check if agent process is running:**
```bash
ps aux | grep mike_agent_live_safe | grep -v grep
```

You should see:
```
chavala  12345  ...  python3 mike_agent_live_safe.py
```

### **Check recent log output:**
```bash
tail -20 logs/agent_*.log
```

You should see recent log entries with timestamps.

---

## 🛑 **Stop the Agent**

### **Graceful stop:**
```bash
pkill -f mike_agent_live_safe.py
```

### **Force stop (if needed):**
```bash
pkill -9 -f mike_agent_live_safe.py
```

---

## 🔍 **Expected Log Messages After Restart**

### **On startup, you should see:**
```
✅ Loading RL model from models/mike_ppo_model_historical.zip
✅ Model loaded successfully
✅ CURRENT REGIME: CALM (VIX: 15.2)
✅ 13/13 SAFEGUARDS: ACTIVE
✅ Agent initialized and ready
```

### **Within 1 minute, if -88.79% position exists:**
```
🚨 STEP 1 STOP-LOSS (ALPACA PnL): SPY251210C00688000 @ -88.79% → FORCING IMMEDIATE CLOSE
✓ Position closed: SPY251210C00688000
```

### **For multi-symbol RL:**
```
🧠 SPY RL Inference: action=1 (BUY CALL) | Source: RL
🧠 QQQ RL Inference: action=0 (HOLD) | Source: RL
🧠 SPX RL Inference: action=1 (BUY CALL) | Source: RL
```

---

## ⚠️ **Troubleshooting**

### **If agent doesn't start:**
1. Check for syntax errors:
   ```bash
   python3 -m py_compile mike_agent_live_safe.py
   ```

2. Check if port is in use:
   ```bash
   lsof -i :8501  # For GUI
   ```

3. Check logs for errors:
   ```bash
   tail -50 logs/agent_*.log
   ```

### **If agent starts but stops immediately:**
- Check config.py for API keys
- Verify Alpaca API connection
- Check for import errors

### **If stop-loss doesn't trigger:**
- Check log for "STEP 1/2/3/4 STOP-LOSS" messages
- Verify position exists in Alpaca
- Check premium data availability

---

## 📝 **Quick Reference**

| Action | Command |
|--------|---------|
| **Restart** | `./restart_agent.sh` |
| **Stop** | `pkill -f mike_agent_live_safe.py` |
| **Monitor** | `tail -f logs/agent_*.log` |
| **Check Status** | `ps aux \| grep mike_agent_live_safe` |
| **View Errors** | `tail -f logs/agent_*.log \| grep ERROR` |

---

**Ready to restart?** Run: `./restart_agent.sh`

