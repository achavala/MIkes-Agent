# 🛡️ BULLETPROOF TRAINING GUIDE

**Purpose:** Ensure training continues even with:
- ❌ No internet connection
- ❌ Lid closed
- ❌ System trying to sleep
- ❌ Terminal closure
- ❌ Process interruption

---

## ✅ BULLETPROOF FEATURES

### 1. **Offline Operation** ✅
- Training uses **only local data files** (no internet required)
- All data pre-collected: `data/historical/enriched/*.pkl`
- Model checkpoints saved locally
- **Works completely offline**

### 2. **Lid-Closed Operation** ✅
- Uses `caffeinate -u` flag (prevents sleep when lid closed)
- Requires MacBook to be **plugged into power**
- Training continues even with lid closed

### 3. **Sleep Prevention** ✅
- Multiple layers of sleep prevention:
  - `caffeinate -d -i -m -s -u` (all flags)
  - System-level: `pmset -a sleep 0`
  - System-level: `pmset -a disablesleep 1`
- **Maximum protection against sleep**

### 4. **Process Resilience** ✅
- Uses `nohup` (survives terminal closure)
- Background process (doesn't require terminal)
- Auto-restart on failure (up to 10 times)
- Monitoring loop checks every minute

### 5. **Checkpoint Recovery** ✅
- Saves checkpoint every 100,000 steps
- Can resume from latest checkpoint
- Automatic recovery on restart

---

## 🚀 QUICK START

### **One Command to Start:**

```bash
cd /Users/chavala/Mike-agent-project
./bulletproof_training.sh
```

**That's it!** The script handles everything:
- ✅ Sleep prevention
- ✅ Process monitoring
- ✅ Auto-restart
- ✅ Offline operation

---

## 📋 DETAILED STEPS

### **Step 1: Pre-Flight Checks**

```bash
# 1. Ensure MacBook is PLUGGED INTO POWER (CRITICAL!)
# 2. Check data files exist (local - no internet needed)
ls -lh data/historical/enriched/*.pkl

# Expected:
# SPY_enriched_2002-01-01_latest.pkl  (~3.5 MB)
# QQQ_enriched_2002-01-01_latest.pkl  (~3.5 MB)
# SPX_enriched_2002-01-01_latest.pkl  (~3.5 MB)
```

### **Step 2: Start Training**

```bash
cd /Users/chavala/Mike-agent-project
./bulletproof_training.sh
```

**What happens:**
1. Checks power status
2. Prevents sleep (multiple methods)
3. Verifies local data files
4. Starts training in background
5. Starts monitoring loop
6. Auto-restarts if training stops

### **Step 3: Verify It's Running**

```bash
# Wait 2-3 minutes, then check:
./check_training_status.sh
```

**Expected output:**
```
✅ Training: RUNNING (PID: xxxxx)
✅ Sleep Prevention: ACTIVE
✅ Power: PLUGGED IN
📁 Latest Checkpoint: mike_rl_model_100000.zip
```

### **Step 4: Close Lid / Disconnect Internet**

**After confirming training is running:**
- ✅ **Close laptop lid** - Training continues!
- ✅ **Disconnect internet** - Training continues! (uses local data)
- ✅ **Close terminal** - Training continues! (uses nohup)

---

## 🔍 MONITORING

### **Quick Status Check:**

```bash
./check_training_status.sh
```

**Shows:**
- Training process status
- Sleep prevention status
- Power status
- Latest checkpoint
- Log file info

### **Watch Log in Real-Time:**

```bash
tail -f training_*.log
```

### **Check Process:**

```bash
ps aux | grep train_historical_model
ps aux | grep caffeinate
```

### **Check Checkpoints:**

```bash
ls -lth models/*.zip | head -5
```

---

## 🛡️ BULLETPROOF GUARANTEES

### **✅ Works Offline**
- Training uses **only local files**
- No internet connection required
- All data pre-collected

### **✅ Works with Lid Closed**
- `caffeinate -u` prevents sleep when lid closed
- Requires power adapter (MacBook must be plugged in)
- System stays awake even with lid closed

### **✅ Survives Sleep Attempts**
- Multiple sleep prevention methods:
  - `caffeinate` with all flags
  - System-level `pmset` settings
  - Process-level protection
- **Maximum protection**

### **✅ Survives Terminal Closure**
- Uses `nohup` (no hang up)
- Background process
- Continues even if terminal closes

### **✅ Auto-Recovery**
- Monitoring loop checks every minute
- Auto-restarts if training stops
- Up to 10 restart attempts
- Resumes from latest checkpoint

---

## 🔧 TROUBLESHOOTING

### **Problem: Training Stops When Lid Closes**

**Check:**
```bash
# 1. Is MacBook plugged in?
pmset -g batt | grep "AC Power"

# 2. Is caffeinate running?
./check_training_status.sh

# 3. Check caffeinate flags
ps aux | grep caffeinate
# Should see: caffeinate -d -i -m -s -u
```

**Solution:**
- Ensure MacBook is **plugged into power**
- Restart sleep prevention: `./prevent_sleep.sh start`
- Or restart training: `./bulletproof_training.sh`

### **Problem: Training Stops When Internet Disconnects**

**Check:**
```bash
# Check if training is using internet
lsof -p $(cat .training.pid) | grep -i tcp
```

**Solution:**
- Training should **NOT** use internet (uses local files)
- If it does, check `train_historical_model.py` for any network calls
- Training should work completely offline

### **Problem: Training Process Dies**

**Check:**
```bash
# Check log for errors
tail -100 training_*.log | grep -i error

# Check if out of memory
tail -100 training_*.log | grep -i memory

# Check restart count
cat .restart_count
```

**Solution:**
- Training will auto-restart (up to 10 times)
- Check log for specific error
- May need to free up memory/disk space

### **Problem: System Goes to Sleep**

**Check:**
```bash
# Check caffeinate
ps aux | grep caffeinate

# Check system settings
pmset -g
```

**Solution:**
- Restart sleep prevention: `./prevent_sleep.sh start`
- Or use bulletproof script: `./bulletproof_training.sh`
- May need sudo for system-level settings

---

## 📊 EXPECTED BEHAVIOR

### **Normal Operation:**
- ✅ Training runs continuously
- ✅ Checkpoints saved every 100K steps
- ✅ Log file grows over time
- ✅ CPU usage ~100% (normal)
- ✅ Works offline
- ✅ Works with lid closed
- ✅ Survives terminal closure

### **If Training Stops:**
- ✅ Auto-restarts (up to 10 times)
- ✅ Resumes from latest checkpoint
- ✅ Monitoring loop continues
- ✅ Logs restart attempts

---

## 🎯 SUCCESS CRITERIA

Training is successful when:
- ✅ Final checkpoint exists: `models/mike_rl_model_5000000.zip`
- ✅ Log shows "Training completed successfully!"
- ✅ Model can be loaded without errors
- ✅ No manual intervention needed

---

## 📝 FILES CREATED

### **During Training:**
- `training_YYYYMMDD_HHMMSS.log` - Training log
- `models/mike_rl_model_*.zip` - Checkpoints (every 100K steps)
- `.training.pid` - Training process ID
- `.caffeinate.pid` - Caffeinate process ID
- `.training.lock` - Lock file
- `.restart_count` - Restart attempt counter
- `.monitor.pid` - Monitor process ID

### **Scripts:**
- `bulletproof_training.sh` - Main launcher (bulletproof)
- `check_training_status.sh` - Status checker
- `prevent_sleep.sh` - Sleep prevention (updated)

---

## 🚀 QUICK REFERENCE

### **Start Training:**
```bash
./bulletproof_training.sh
```

### **Check Status:**
```bash
./check_training_status.sh
```

### **Watch Log:**
```bash
tail -f training_*.log
```

### **Stop Training:**
```bash
kill $(cat .training.pid)
./prevent_sleep.sh stop
```

---

## ✅ FINAL CHECKLIST

Before closing lid / disconnecting internet:

- [ ] MacBook is **plugged into power**
- [ ] Training started successfully
- [ ] Status check shows "RUNNING"
- [ ] Log file is growing
- [ ] Checkpoint files are being created
- [ ] Sleep prevention is active

**Then:**
- ✅ Close lid - Training continues!
- ✅ Disconnect internet - Training continues!
- ✅ Close terminal - Training continues!

---

**🎉 Your training is now bulletproof!**

**Run:**
```bash
./bulletproof_training.sh
```

**Then close your lid and let it run! 🚀**

