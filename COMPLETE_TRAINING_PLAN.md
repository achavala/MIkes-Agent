# 🚀 COMPLETE TRAINING PLAN

**Purpose:** Comprehensive guide for training the 0DTE RL model  
**Dataset:** 23.9 years (6,022 days × 3 symbols)  
**Estimated Time:** See calculations below

---

## ⏱️ TRAINING TIME ESTIMATES

### Dataset Size
- **Symbols:** SPY, QQQ, SPX
- **Rows per symbol:** 6,022
- **Columns per symbol:** 77
- **Total data points:** ~1.4M
- **Default timesteps:** 5,000,000

### Time Estimates

| Hardware | Speed | Total Time | Notes |
|----------|-------|------------|-------|
| **CPU-Only (Mac)** | ~120s per 1K steps | **~167 hours (7 days)** | Conservative estimate |
| **GPU-Accelerated** | ~30s per 1K steps | **~42 hours (1.75 days)** | If GPU available |
| **Realistic (Mixed)** | ~60s per 1K steps | **~83 hours (3.5 days)** | Average scenario |

### Recommended Approach

**Start Small, Scale Up:**
1. **Test Run:** 500K steps (~17 hours CPU) - Validate learning
2. **Medium Run:** 2M steps (~67 hours CPU / ~2.8 days) - Good baseline
3. **Full Run:** 5M steps (~167 hours CPU / ~7 days) - Production model

---

## 💻 PREVENTING MAC FROM SLEEPING

### Method 1: Using `caffeinate` (Recommended) ✅

I've created a script for you: `prevent_sleep.sh`

**Before Training:**
```bash
./prevent_sleep.sh start
```

**After Training:**
```bash
./prevent_sleep.sh stop
```

**Check Status:**
```bash
./prevent_sleep.sh status
```

**What it does:**
- Prevents display from sleeping
- Prevents system from idle sleeping
- Prevents disk from idle sleeping
- Prevents system from sleeping (even on AC power)

### Method 2: Manual `caffeinate` Command

**Start:**
```bash
caffeinate -d -i -m -s &
```

**Stop:**
```bash
pkill caffeinate
```

### Method 3: System Preferences

**Energy Saver Settings:**
1. System Preferences → Battery
2. Set "Turn display off after" to "Never"
3. Uncheck "Prevent automatic sleeping when display is off"

**Or use command line:**
```bash
# Prevent sleep for 100 hours (adjust as needed)
pmset noidle
# Or set sleep time to never (until reboot)
sudo pmset -a sleep 0
```

---

## 📋 COMPLETE TRAINING CHECKLIST

### Pre-Training Setup

- [ ] **Activate virtual environment**
  ```bash
  source venv/bin/activate
  ```

- [ ] **Verify data files exist**
  ```bash
  ls -lh data/historical/enriched/*.pkl
  ```
  Should see: SPY, QQQ, SPX enriched files (~3.5 MB each)

- [ ] **Check disk space**
  ```bash
  df -h
  ```
  Need at least **10 GB free** for checkpoints and logs

- [ ] **Prevent Mac from sleeping**
  ```bash
  ./prevent_sleep.sh start
  ```

- [ ] **Close unnecessary applications**
  - Free up RAM
  - Close browsers, IDEs, etc.

- [ ] **Connect to power adapter**
  - Don't train on battery
  - Ensure stable power

- [ ] **Check system resources**
  ```bash
  # Check available RAM
  vm_stat | grep "Pages free"
  
  # Check CPU cores
  sysctl -n hw.ncpu
  ```

### Training Execution

- [ ] **Start training with checkpoints**
  ```bash
  python train_historical_model.py \
      --symbols SPY,QQQ,SPX \
      --start-date 2002-01-01 \
      --timesteps 5000000 \
      --use-greeks \
      --regime-balanced \
      --save-freq 100000
  ```

- [ ] **Monitor training progress**
  - Check logs every few hours
  - Monitor disk space
  - Watch for errors

- [ ] **Use `nohup` for background execution**
  ```bash
  nohup python train_historical_model.py \
      --symbols SPY,QQQ,SPX \
      --start-date 2002-01-01 \
      --timesteps 5000000 \
      --use-greeks \
      --regime-balanced \
      --save-freq 100000 \
      > training.log 2>&1 &
  ```

### Post-Training

- [ ] **Stop caffeinate**
  ```bash
  ./prevent_sleep.sh stop
  ```

- [ ] **Verify model saved**
  ```bash
  ls -lh models/
  ```

- [ ] **Check training logs**
  ```bash
  tail -100 training.log
  ```

---

## 🎯 RECOMMENDED TRAINING STRATEGY

### Phase 1: Validation Run (Day 1)

**Purpose:** Verify everything works, model is learning

```bash
python train_historical_model.py \
    --symbols SPY \
    --start-date 2020-01-01 \
    --timesteps 500000 \
    --use-greeks \
    --save-freq 50000
```

**Expected Time:** ~17 hours (CPU)  
**What to check:**
- ✅ Training starts without errors
- ✅ Model saves checkpoints
- ✅ Loss decreases over time
- ✅ No memory leaks

### Phase 2: Medium Run (Days 2-5)

**Purpose:** Train on full dataset with moderate timesteps

```bash
python train_historical_model.py \
    --symbols SPY,QQQ,SPX \
    --start-date 2002-01-01 \
    --timesteps 2000000 \
    --use-greeks \
    --regime-balanced \
    --save-freq 100000
```

**Expected Time:** ~67 hours (~2.8 days CPU)  
**What to expect:**
- ✅ Model learns patterns
- ✅ Performance improves
- ✅ Checkpoints every 100K steps

### Phase 3: Full Production Run (Days 6-13)

**Purpose:** Maximum training for best model

```bash
python train_historical_model.py \
    --symbols SPY,QQQ,SPX \
    --start-date 2002-01-01 \
    --timesteps 5000000 \
    --use-greeks \
    --regime-balanced \
    --save-freq 100000
```

**Expected Time:** ~167 hours (~7 days CPU)  
**What to expect:**
- ✅ Fully trained model
- ✅ Best performance
- ✅ Production-ready

---

## 📊 MONITORING DURING TRAINING

### Check Training Progress

```bash
# Watch log file in real-time
tail -f training.log

# Check checkpoint files
ls -lth models/ | head -10

# Monitor system resources
top -l 1 | grep -E "CPU|Mem"

# Check disk space
df -h .
```

### Create Monitoring Script

Save as `monitor_training.sh`:

```bash
#!/bin/bash
while true; do
    clear
    echo "=== Training Monitor ==="
    echo ""
    echo "Latest log entries:"
    tail -5 training.log
    echo ""
    echo "Latest checkpoints:"
    ls -lth models/*.zip 2>/dev/null | head -3
    echo ""
    echo "Disk space:"
    df -h . | tail -1
    echo ""
    echo "Press Ctrl+C to exit"
    sleep 60
done
```

Run: `chmod +x monitor_training.sh && ./monitor_training.sh`

---

## 🔄 RESUME TRAINING FROM CHECKPOINT

If training is interrupted, resume from last checkpoint:

```bash
python train_historical_model.py \
    --symbols SPY,QQQ,SPX \
    --start-date 2002-01-01 \
    --timesteps 5000000 \
    --use-greeks \
    --regime-balanced \
    --load-model models/mike_rl_model_step_XXXXXX.zip \
    --save-freq 100000
```

---

## 🌐 ALTERNATIVE: CLOUD TRAINING

If local training is too slow, consider cloud options:

### Google Colab (Free GPU)
- Free GPU access
- ~10x faster than CPU
- 5M steps in ~4-8 hours
- **Guide:** See `CLOUD_TRAINING_GUIDE.md`

### AWS/GCP (Paid)
- Professional GPU instances
- ~20-50x faster than CPU
- 5M steps in ~2-4 hours
- Cost: ~$10-50 for full training

---

## 📝 TRAINING SCRIPT TEMPLATE

Save as `start_training.sh`:

```bash
#!/bin/bash

# Training Configuration
SYMBOLS="SPY,QQQ,SPX"
START_DATE="2002-01-01"
TIMESTEPS=5000000
SAVE_FREQ=100000
LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"

echo "🚀 Starting Training..."
echo "Symbols: $SYMBOLS"
echo "Timesteps: $TIMESTEPS"
echo "Log file: $LOG_FILE"
echo ""

# Activate venv
source venv/bin/activate

# Prevent sleep
./prevent_sleep.sh start

# Start training with nohup
nohup python train_historical_model.py \
    --symbols "$SYMBOLS" \
    --start-date "$START_DATE" \
    --timesteps "$TIMESTEPS" \
    --use-greeks \
    --regime-balanced \
    --save-freq "$SAVE_FREQ" \
    > "$LOG_FILE" 2>&1 &

TRAINING_PID=$!
echo "Training started (PID: $TRAINING_PID)"
echo "Monitor with: tail -f $LOG_FILE"
echo ""
echo "To stop training:"
echo "  kill $TRAINING_PID"
echo "  ./prevent_sleep.sh stop"
```

Make executable: `chmod +x start_training.sh`

---

## ✅ QUICK START GUIDE

### 1. Prevent Sleep
```bash
./prevent_sleep.sh start
```

### 2. Start Training (Background)
```bash
nohup python train_historical_model.py \
    --symbols SPY,QQQ,SPX \
    --start-date 2002-01-01 \
    --timesteps 5000000 \
    --use-greeks \
    --regime-balanced \
    --save-freq 100000 \
    > training.log 2>&1 &
```

### 3. Monitor Progress
```bash
tail -f training.log
```

### 4. Check Status
```bash
# Training process
ps aux | grep train_historical_model

# Sleep prevention
./prevent_sleep.sh status

# Latest checkpoint
ls -lth models/*.zip | head -1
```

### 5. Stop Training (when complete)
```bash
# Stop training
pkill -f train_historical_model

# Restore normal sleep
./prevent_sleep.sh stop
```

---

## 🎯 EXPECTED TIMELINE

### Full 5M Steps Training (CPU)

| Day | Hours | Progress | Checkpoint |
|-----|-------|----------|------------|
| 1 | 0-24 | Steps 0-714K | Every 100K |
| 2 | 24-48 | Steps 714K-1.43M | Every 100K |
| 3 | 48-72 | Steps 1.43M-2.14M | Every 100K |
| 4 | 72-96 | Steps 2.14M-2.86M | Every 100K |
| 5 | 96-120 | Steps 2.86M-3.57M | Every 100K |
| 6 | 120-144 | Steps 3.57M-4.29M | Every 100K |
| 7 | 144-167 | Steps 4.29M-5.0M | Final model |

**Total:** ~7 days continuous training

---

## 🛡️ SAFETY MEASURES

1. **Checkpoint Frequency:** Save every 100K steps (can resume if interrupted)
2. **Log Rotation:** Monitor log file size, rotate if needed
3. **Disk Space:** Check weekly, clean old checkpoints if needed
4. **Backup:** Copy final model to external drive after training
5. **Validation:** Test model after training before live use

---

## 📞 TROUBLESHOOTING

### Mac Still Sleeping?

```bash
# Check caffeinate status
./prevent_sleep.sh status

# Check system sleep settings
pmset -g

# Force prevent sleep
sudo pmset -a sleep 0
sudo pmset -a disablesleep 1
```

### Training Stopped?

```bash
# Check log for errors
tail -100 training.log

# Check if process is still running
ps aux | grep train_historical_model

# Resume from last checkpoint
python train_historical_model.py --load-model models/mike_rl_model_step_XXXXXX.zip ...
```

### Out of Memory?

```bash
# Reduce batch size in training script
# Or reduce number of parallel environments
# Or train on single symbol first
```

---

## 🎉 SUCCESS CRITERIA

Training is successful when:

- ✅ Model saves final checkpoint
- ✅ Training loss decreases over time
- ✅ No errors in log file
- ✅ Model file exists in `models/` directory
- ✅ Model size is reasonable (~10-100 MB)

---

**Good luck with training! 🚀**

**Estimated Start Time:** Now  
**Estimated Completion:** 7 days from now (CPU)  
**Recommended:** Start with 500K step test run first

