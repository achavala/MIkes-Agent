# 📊 **TRAINING STATUS MONITORING GUIDE**

**Quick reference for checking LSTM training progress**

---

## 🚀 **QUICK STATUS CHECK**

### **Option 1: Use the Status Script** (Easiest)

```bash
./check_training_status.sh
```

This shows:
- ✅ Process status (running/stopped)
- 📊 Latest log output
- 🔍 Key indicators (LSTM active, diagnostics, checkpoints)
- ⚠️ Errors if any

### **Option 2: Manual Commands**

#### **Check if Process is Running**:
```bash
ps aux | grep train_historical_model | grep -v grep
```

If running, you'll see the process with PID, CPU%, and Memory%.

#### **View Latest Logs**:
```bash
tail -50 training_output.log
```

#### **Watch Live Progress**:
```bash
tail -f training_output.log
```
(Press `Ctrl+C` to stop watching)

---

## 📊 **KEY INDICATORS TO LOOK FOR**

### **✅ Good Signs**:

1. **LSTM is Active**:
   ```
   ✅ RecurrentPPO available - Using LSTM Policy with action masking
   ```

2. **Training Started**:
   ```
   🏋️ Training for 500,000 timesteps...
   ```

3. **Diagnostics Appearing** (every 5k steps):
   ```
   📊 MomentumDiagnostics @ step=5,000
   FLAT action distribution (count / %):
     action 0: XXXX (YY.Y%)
     action 1: XXXX (YY.Y%)
   ```

4. **Checkpoints Saving** (every 50k steps):
   ```
   Saving model checkpoint to models/checkpoints/mike_momentum_model_v3_lstm_50000_steps.zip
   ```

5. **Progress Updates**:
   ```
   | 50000/500000 | ...
   ```

### **⚠️ Warning Signs**:

1. **Process Not Running**:
   ```bash
   ps aux | grep train_historical_model | grep -v grep
   # Returns nothing = process stopped
   ```

2. **Errors in Logs**:
   ```bash
   grep -i error training_output.log | tail -10
   ```

3. **Training Stuck** (no new log entries for >10 minutes):
   - Check CPU usage
   - Check if data is downloading

---

## 🔍 **DETAILED MONITORING**

### **1. Check Process Details**:

```bash
# Find the process
ps aux | grep train_historical_model | grep -v grep

# Get detailed info
top -p $(pgrep -f train_historical_model)
```

### **2. Monitor Training Progress**:

```bash
# Watch last 20 lines continuously
watch -n 5 'tail -20 training_output.log'

# Or use tail with follow
tail -f training_output.log
```

### **3. Check Diagnostics** (Training Health):

```bash
# Show all diagnostics
grep "MomentumDiagnostics" training_output.log

# Show latest 5 diagnostics
grep "MomentumDiagnostics" training_output.log | tail -5

# Show with context (6 lines after each)
grep -A 6 "MomentumDiagnostics" training_output.log | tail -30
```

### **4. Check Training Log File**:

```bash
# Find the training log
ls -t logs/training/mike_momentum_model_v3_lstm_*.log | head -1

# View it
tail -f logs/training/mike_momentum_model_v3_lstm_500000.log
```

### **5. Check for Errors**:

```bash
# All errors
grep -i error training_output.log

# Recent errors
grep -i error training_output.log | tail -20

# Exceptions/Tracebacks
grep -A 10 "Exception\|Traceback" training_output.log | tail -50
```

### **6. Check Checkpoint Progress**:

```bash
# List saved checkpoints
ls -lh models/checkpoints/mike_momentum_model_v3_lstm_*.zip

# Check latest checkpoint
ls -t models/checkpoints/mike_momentum_model_v3_lstm_*.zip | head -1
```

---

## 📈 **EXPECTED PROGRESS TIMELINE**

| Time | Expected Progress |
|------|-------------------|
| **0-5 min** | Data loading, environment setup |
| **5-10 min** | Training starts, first diagnostics at 5k steps |
| **30-60 min** | First checkpoint at 50k steps |
| **2-4 hours** | 100k-200k steps (check diagnostics) |
| **4-8 hours** | 500k steps complete, final model saved |

---

## 🎯 **WHAT TO MONITOR**

### **During First Hour**:

1. ✅ Process is running
2. ✅ LSTM detected in logs
3. ✅ Training started message
4. ✅ First diagnostics at 5k steps
5. ✅ No errors

### **Every Hour After**:

1. ✅ New diagnostics appearing
2. ✅ Checkpoints being saved
3. ✅ HOLD % decreasing (should be <50% by 25k steps)
4. ✅ BUY % increasing (should be >50% by 25k steps)
5. ✅ No errors

### **Key Metrics to Watch**:

From diagnostics (every 5k steps):
- **HOLD %**: Should decrease (target: 30-40% by 100k)
- **BUY %**: Should increase (target: 60-70% by 100k)
- **Strong-Setup BUY Rate**: Should rise (target: 75%+ by 100k)

---

## 🛑 **IF TRAINING STOPS**

### **Check Why**:

```bash
# Check end of log for errors
tail -100 training_output.log

# Check if process crashed
dmesg | tail -20

# Check system resources
df -h  # Disk space
free -h  # Memory
```

### **Common Issues**:

1. **Out of Memory**:
   - Reduce `--intraday-days` (e.g., 30 instead of 60)
   - Reduce `--timesteps` for test run

2. **API Key Issues**:
   - Check `.env` file has `MASSIVE_API_KEY`
   - Verify key is valid

3. **Data Download Failed**:
   - Check internet connection
   - Check API quota

### **Restart Training**:

```bash
# If training stopped, restart with same command
nohup ./retrain_lstm_model.sh > training_output.log 2>&1 &
```

---

## ✅ **WHEN TRAINING COMPLETES**

### **Check Completion**:

```bash
# Look for completion message
grep -i "complete\|finished\|saved" training_output.log | tail -5

# Check if final model exists
ls -lh models/mike_momentum_model_v3_lstm_500000.zip
```

### **Verify Model**:

```bash
python3 -c "
from sb3_contrib import RecurrentPPO
model = RecurrentPPO.load('models/mike_momentum_model_v3_lstm_500000.zip')
print('✅ LSTM model loaded successfully')
print(f'Policy: {type(model.policy).__name__}')
"
```

---

## 📋 **QUICK REFERENCE COMMANDS**

```bash
# Status check
./check_training_status.sh

# Watch live
tail -f training_output.log

# Check diagnostics
grep "MomentumDiagnostics" training_output.log | tail -5

# Check errors
grep -i error training_output.log | tail -10

# Check process
ps aux | grep train_historical_model | grep -v grep

# Check checkpoints
ls -lh models/checkpoints/mike_momentum_model_v3_lstm_*.zip

# Stop training (if needed)
pkill -f train_historical_model
```

---

**Last Updated**: 2025-12-12





