# 🚀 COMPLETE TRAINING PLAN - LID CLOSED OPERATION

**Purpose:** Start historical model training and ensure it completes even when laptop lid is closed  
**Estimated Time:** 7 days (CPU) or 1.75 days (GPU)  
**Status:** Ready to execute

---

## ✅ PRE-FLIGHT CHECKLIST

### 1. **System Requirements**
- [ ] MacBook is **plugged into power adapter** (CRITICAL - won't work on battery)
- [ ] At least **10 GB free disk space** for checkpoints and logs
- [ ] Virtual environment activated and dependencies installed
- [ ] Historical data files exist (SPY, QQQ, SPX enriched data)

### 2. **Verify Data Files**
```bash
cd /Users/chavala/Mike-agent-project
ls -lh data/historical/enriched/*.pkl
```

**Expected output:**
```
SPY_enriched_2002-01-01_latest.pkl  (~3.5 MB)
QQQ_enriched_2002-01-01_latest.pkl  (~3.5 MB)
SPX_enriched_2002-01-01_latest.pkl  (~3.5 MB)
```

### 3. **Check Disk Space**
```bash
df -h .
```

**Need:** At least 10 GB free

### 4. **System Preferences (Optional but Recommended)**
1. **System Settings → Battery → Options:**
   - Set "Turn display off after" to "Never"
   - Uncheck "Prevent automatic sleeping on power adapter when display is off"
   - Uncheck "Enable Power Nap" (if available)

2. **System Settings → Battery:**
   - Ensure "Low Power Mode" is OFF

---

## 🎯 STEP-BY-STEP EXECUTION PLAN

### **STEP 1: Prepare Environment**

```bash
# Navigate to project directory
cd /Users/chavala/Mike-agent-project

# Activate virtual environment
source venv/bin/activate

# Verify Python and dependencies
python --version  # Should be Python 3.x
pip list | grep stable-baselines3  # Should show installed
```

### **STEP 2: Prevent Sleep (Lid-Closed Compatible)**

```bash
# Start caffeinate with lid-closed support
./prevent_sleep.sh start

# Verify it's running
./prevent_sleep.sh status
```

**Expected output:**
```
✅ caffeinate is RUNNING (PID: xxxxx)
   Your Mac will not sleep.
```

**What this does:**
- `-d`: Prevents display from sleeping
- `-i`: Prevents system from idle sleeping
- `-m`: Prevents disk from idle sleeping
- `-s`: Prevents system from sleeping (even on AC power)
- `-u`: **Prevents system from sleeping when lid is closed** (CRITICAL)

### **STEP 3: Start Training**

```bash
# Start training (automatically handles sleep prevention)
./start_training.sh
```

**What this does:**
1. Checks if training is already running
2. Activates virtual environment
3. Verifies data files exist
4. Starts `caffeinate` (if not already running)
5. Launches training in background with `nohup`
6. Saves PID to `.training.pid`
7. Redirects output to `training_YYYYMMDD_HHMMSS.log`

**Expected output:**
```
🚀 STARTING TRAINING
═══════════════════════════════════════════════════════════

📦 Activating virtual environment...
📊 Checking data files...
☕ Preventing Mac from sleeping...
💾 Checking disk space...
🎯 Starting training...
   Symbols: SPY,QQQ,SPX
   Timesteps: 5000000
   Save frequency: Every 100000 steps
   Log file: training_20251207_143022.log

✅ Training started!

Process ID: 12345
Log file: training_20251207_143022.log
PID file: .training.pid

📊 Monitor training:
   tail -f training_20251207_143022.log

📁 Check checkpoints:
   ls -lth models/*.zip | head -5

🛑 Stop training:
   kill 12345
   ./prevent_sleep.sh stop
```

### **STEP 4: Close Laptop Lid**

**After training starts:**
1. ✅ **Ensure MacBook is plugged into power** (CRITICAL)
2. ✅ **Wait 2-3 minutes** to ensure training started successfully
3. ✅ **Check training log** to confirm it's running:
   ```bash
   tail -20 training_*.log
   ```
4. ✅ **Close laptop lid** - Training will continue!

---

## 📊 MONITORING TRAINING (While Lid Closed)

### **Method 1: SSH from Another Device**

If you have another computer/phone:

```bash
# From another device, SSH into your Mac
ssh username@your-mac-ip-address

# Then check training status
cd /Users/chavala/Mike-agent-project
tail -f training_*.log
```

### **Method 2: Open Lid Periodically**

1. Open laptop lid
2. Check training status:
   ```bash
   cd /Users/chavala/Mike-agent-project
   tail -50 training_*.log
   ```
3. Check if process is running:
   ```bash
   ps aux | grep train_historical_model
   ```
4. Close lid again

### **Method 3: Check Training Progress**

```bash
# Check latest checkpoint
ls -lth models/*.zip | head -1

# Check log file size (growing = training active)
ls -lh training_*.log

# Check process status
cat .training.pid
ps -p $(cat .training.pid)
```

---

## 🔍 VERIFY TRAINING IS RUNNING

### **Quick Health Check**

```bash
cd /Users/chavala/Mike-agent-project

# 1. Check if caffeinate is running
./prevent_sleep.sh status

# 2. Check if training process is running
if [ -f .training.pid ]; then
    TRAINING_PID=$(cat .training.pid)
    if ps -p "$TRAINING_PID" > /dev/null 2>&1; then
        echo "✅ Training is RUNNING (PID: $TRAINING_PID)"
    else
        echo "❌ Training process NOT running"
    fi
else
    echo "❌ No training PID file found"
fi

# 3. Check latest log entries
echo "Latest log entries:"
tail -10 training_*.log | tail -5

# 4. Check checkpoint files
echo "Latest checkpoint:"
ls -lth models/*.zip 2>/dev/null | head -1 || echo "No checkpoints yet"
```

---

## ⏱️ ESTIMATED TIMELINE

### **Training Progress Milestones**

| Timesteps | Estimated Time | Checkpoint Saved |
|-----------|----------------|------------------|
| 100,000   | ~3.3 hours     | ✅ `models/mike_rl_model_100000.zip` |
| 500,000   | ~16.7 hours    | ✅ `models/mike_rl_model_500000.zip` |
| 1,000,000 | ~33.3 hours    | ✅ `models/mike_rl_model_1000000.zip` |
| 2,000,000 | ~66.7 hours    | ✅ `models/mike_rl_model_2000000.zip` |
| 5,000,000 | ~167 hours (7 days) | ✅ `models/mike_rl_model_5000000.zip` |

**Note:** Times are for CPU-only. GPU would be ~4x faster.

---

## 🛑 STOPPING TRAINING

### **If You Need to Stop:**

```bash
cd /Users/chavala/Mike-agent-project

# 1. Stop training process
if [ -f .training.pid ]; then
    TRAINING_PID=$(cat .training.pid)
    kill "$TRAINING_PID"
    rm .training.pid
    echo "✅ Training stopped"
fi

# 2. Stop caffeinate
./prevent_sleep.sh stop

# 3. Verify processes are stopped
ps aux | grep train_historical_model
ps aux | grep caffeinate
```

---

## 📁 FILES CREATED DURING TRAINING

### **Log Files**
- `training_YYYYMMDD_HHMMSS.log` - Main training log
- Contains: Progress, loss, rewards, checkpoints

### **Checkpoint Files**
- `models/mike_rl_model_100000.zip` - Every 100K steps
- `models/mike_rl_model_200000.zip`
- `models/mike_rl_model_500000.zip`
- ... (continues every 100K steps)
- `models/mike_rl_model_5000000.zip` - Final model

### **PID Files**
- `.training.pid` - Training process ID
- `.caffeinate.pid` - Caffeinate process ID

---

## 🔧 TROUBLESHOOTING

### **Problem: Training Stops When Lid Closes**

**Solution:**
1. Ensure MacBook is **plugged into power**
2. Verify `caffeinate -u` flag is used:
   ```bash
   ./prevent_sleep.sh status
   ps aux | grep caffeinate
   # Should see: caffeinate -d -i -m -s -u
   ```
3. Check system settings:
   ```bash
   pmset -g
   # Look for: sleep = 0 or sleep prevented
   ```

### **Problem: Training Process Dies**

**Check:**
```bash
# Check log for errors
tail -100 training_*.log | grep -i error

# Check if out of memory
tail -100 training_*.log | grep -i memory

# Restart training
./start_training.sh
```

### **Problem: Disk Space Full**

**Check:**
```bash
df -h .
```

**Solution:**
- Delete old checkpoints (keep latest 3-5)
- Delete old log files
- Free up space

### **Problem: Can't Connect After Closing Lid**

**Solution:**
- Open lid to check status
- Use SSH from another device
- Check if MacBook is still on (power light)

---

## ✅ COMPLETION CHECKLIST

### **When Training Completes:**

1. **Verify Final Model:**
   ```bash
   ls -lh models/mike_rl_model_5000000.zip
   ```

2. **Check Training Log:**
   ```bash
   tail -50 training_*.log
   # Should see: "Training completed successfully!"
   ```

3. **Stop Caffeinate:**
   ```bash
   ./prevent_sleep.sh stop
   ```

4. **Restore System Settings:**
   - System Settings → Battery → Restore normal sleep settings

5. **Test Model:**
   ```bash
   python -c "from stable_baselines3 import PPO; model = PPO.load('models/mike_rl_model_5000000.zip'); print('✅ Model loaded successfully!')"
   ```

---

## 🎯 QUICK START COMMANDS

### **Complete Setup (Copy-Paste Ready):**

```bash
# 1. Navigate to project
cd /Users/chavala/Mike-agent-project

# 2. Activate environment
source venv/bin/activate

# 3. Verify data exists
ls -lh data/historical/enriched/*.pkl

# 4. Start training (handles everything)
./start_training.sh

# 5. Wait 2-3 minutes, then check log
tail -20 training_*.log

# 6. If everything looks good, close lid!
# (Make sure MacBook is plugged into power)
```

### **Quick Status Check:**

```bash
cd /Users/chavala/Mike-agent-project
./prevent_sleep.sh status
ps -p $(cat .training.pid 2>/dev/null) && echo "✅ Training running" || echo "❌ Training stopped"
tail -5 training_*.log
```

---

## 📝 NOTES

1. **Power Adapter Required:** MacBook must be plugged in for lid-closed operation
2. **Checkpoints Every 100K Steps:** Model is saved every 100,000 steps
3. **Training Can Resume:** If interrupted, can resume from latest checkpoint
4. **Log File Grows:** Training log will grow to several MB over 7 days
5. **CPU Usage:** Training will use 100% CPU (this is normal)

---

## 🎉 SUCCESS CRITERIA

Training is successful when:
- ✅ Final checkpoint exists: `models/mike_rl_model_5000000.zip`
- ✅ Log shows "Training completed successfully!"
- ✅ Model can be loaded without errors
- ✅ Model file size is reasonable (~10-50 MB)

---

**Ready to start? Run:**
```bash
cd /Users/chavala/Mike-agent-project
./start_training.sh
```

**Then close your lid and let it run! 🚀**

