# 🛡️ LIVE AGENT PRIORITY SYSTEM

**Date:** December 23, 2025  
**Purpose:** Ensure live trading agent always runs during market hours, preventing backtest interference

---

## 🎯 PROBLEM SOLVED

**Issue:** Backtest scripts could interfere with live trading agent during market hours, causing:
- Live agent not running when market opens
- Backtest consuming resources needed for live trading
- No trades executed during market hours

**Solution:** Implemented a priority system that:
- ✅ Ensures live agent always runs during market hours
- ✅ Prevents backtest from running during market hours
- ✅ Automatically starts live agent if not running
- ✅ Kills backtest processes during market hours

---

## 🏗️ SYSTEM ARCHITECTURE

### **1. Live Agent Lock File**
- **Location:** `/tmp/mike_agent_live.lock`
- **Created by:** Live agent on startup
- **Removed by:** Live agent on shutdown
- **Purpose:** Indicates live agent is running

### **2. Watchdog Process**
- **Script:** `ensure_live_agent_running.py`
- **Purpose:** Monitors and ensures live agent runs during market hours
- **Actions:**
  - Checks if market is open (9:30 AM - 4:00 PM EST, Mon-Fri)
  - Checks if live agent is running
  - Kills any backtest processes during market hours
  - Starts live agent if not running during market hours

### **3. Backtest Protection**
- **Script:** `backtest_lock.py` (can be imported by backtest scripts)
- **Phase 0 Backtest:** Already updated with protection
- **Purpose:** Prevents backtest from running during market hours

---

## 🚀 USAGE

### **Start Live Agent with Watchdog (Recommended)**

```bash
./start_live_agent_with_watchdog.sh
```

This will:
1. Check if watchdog is already running
2. Check if live agent is already running
3. Start watchdog in background
4. Watchdog will automatically start live agent when market opens

### **Start Watchdog Manually**

```bash
python3 ensure_live_agent_running.py
```

### **Start Live Agent Manually (Without Watchdog)**

```bash
python3 mike_agent_live_safe.py
```

The live agent will:
- Create lock file on startup
- Remove lock file on shutdown
- Log to `logs/live_agent_YYYYMMDD.log`

---

## 🔒 BACKTEST PROTECTION

### **Phase 0 Backtest**

The Phase 0 backtest (`phase0_backtest/run_phase0.py`) now includes protection:

- **Checks market hours** before running
- **Checks if live agent is running**
- **Blocks backtest** if:
  - Market is open AND live agent is running
- **Warns user** if:
  - Market is open BUT live agent is not running

### **Other Backtest Scripts**

To add protection to other backtest scripts, import and call:

```python
from backtest_lock import prevent_backtest_during_market_hours

# At the start of your backtest script
prevent_backtest_during_market_hours()
```

---

## 📊 HOW IT WORKS

### **Market Hours Detection**

```python
def is_market_open():
    # Check if weekday (Monday=0, Friday=4)
    if now_est.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check market hours (9:30 AM - 4:00 PM EST)
    market_open = now_est.replace(hour=9, minute=30, second=0)
    market_close = now_est.replace(hour=16, minute=0, second=0)
    
    return market_open <= now_est < market_close
```

### **Watchdog Loop**

1. **Check if market is open**
2. **If market is open:**
   - Check if backtest is running → Kill it
   - Check if live agent is running → Start if not
3. **If market is closed:**
   - Log status but don't interfere

### **Live Agent Lock**

Live agent creates lock file on startup:
```python
LIVE_AGENT_LOCK_FILE = "/tmp/mike_agent_live.lock"
with open(LIVE_AGENT_LOCK_FILE, 'w') as f:
    f.write(f"Live agent lock - {datetime.now().isoformat()}\n")
    f.write(f"PID: {os.getpid()}\n")
```

Lock file is removed on shutdown (via `atexit` and signal handlers).

---

## 🔍 MONITORING

### **Check if Watchdog is Running**

```bash
ps aux | grep ensure_live_agent_running
```

### **Check if Live Agent is Running**

```bash
ps aux | grep mike_agent_live_safe
```

### **Check Lock File**

```bash
cat /tmp/mike_agent_live.lock
cat /tmp/mike_agent_live.pid
```

### **View Watchdog Logs**

```bash
tail -f logs/watchdog.log
```

### **View Live Agent Logs**

```bash
tail -f logs/live_agent_$(date +%Y%m%d).log
```

---

## 🛑 STOPPING

### **Stop Watchdog**

```bash
pkill -f ensure_live_agent_running.py
```

### **Stop Live Agent**

```bash
pkill -f mike_agent_live_safe.py
```

Or use the PID file:
```bash
kill $(cat /tmp/mike_agent_live.pid)
```

### **Stop Both**

```bash
pkill -f ensure_live_agent_running.py
pkill -f mike_agent_live_safe.py
```

---

## ⚙️ CONFIGURATION

### **Watchdog Settings**

Edit `ensure_live_agent_running.py`:

```python
CHECK_INTERVAL = 60  # Check every 60 seconds
LIVE_AGENT_SCRIPT = "mike_agent_live_safe.py"
BACKTEST_SCRIPTS = ["run_phase0.py", "phase0_backtest/run_phase0.py"]
```

### **Lock File Locations**

```python
LIVE_AGENT_LOCK_FILE = "/tmp/mike_agent_live.lock"
LIVE_AGENT_PID_FILE = "/tmp/mike_agent_live.pid"
```

---

## ✅ VERIFICATION

### **Test 1: Watchdog Starts Live Agent**

1. Stop live agent: `pkill -f mike_agent_live_safe.py`
2. Start watchdog: `python3 ensure_live_agent_running.py`
3. Wait for market open (or manually set time)
4. Verify live agent starts automatically

### **Test 2: Backtest Blocked During Market Hours**

1. Ensure live agent is running
2. Try to run backtest: `python3 phase0_backtest/run_phase0.py`
3. Verify backtest is blocked with error message

### **Test 3: Watchdog Kills Backtest**

1. Start backtest during market hours
2. Start watchdog
3. Verify watchdog kills backtest and starts live agent

---

## 📝 FILES CREATED

1. **`ensure_live_agent_running.py`** - Watchdog script
2. **`backtest_lock.py`** - Backtest protection module
3. **`start_live_agent_with_watchdog.sh`** - Startup script
4. **`LIVE_AGENT_PRIORITY_SYSTEM.md`** - This documentation

---

## 🔄 UPDATED FILES

1. **`mike_agent_live_safe.py`** - Added lock file creation/removal
2. **`phase0_backtest/run_phase0.py`** - Added market hours protection

---

## 🎯 SUMMARY

**The system ensures:**
- ✅ Live agent always runs during market hours
- ✅ Backtest cannot interfere with live trading
- ✅ Automatic recovery if live agent stops
- ✅ Clear separation between backtest and live trading

**To use:**
```bash
./start_live_agent_with_watchdog.sh
```

**That's it!** The watchdog will handle everything automatically.

---

**Last Updated:** December 23, 2025


