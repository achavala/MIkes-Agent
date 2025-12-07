# 🖥️ PARALLEL TRAINING SETUP GUIDE

**Strategy:** Run training on desktop Mac for 7 days, continue daily live testing on M2 Mac laptop

---

## ✅ YES, THIS IS POSSIBLE AND RECOMMENDED!

### Benefits:
- ✅ **Desktop Mac:** Dedicated 7-day training (no interruptions)
- ✅ **M2 Mac Laptop:** Continue daily live trading/testing
- ✅ **No conflicts:** Training and live trading are separate
- ✅ **Faster:** Can use both machines simultaneously

---

## 📋 SETUP OVERVIEW

### Desktop Mac (Training Machine)
- **Purpose:** Run 7-day training uninterrupted
- **What it needs:** Data files, training script, dependencies
- **Output:** Trained model files

### M2 Mac Laptop (Daily Work Machine)
- **Purpose:** Continue daily live trading/testing
- **What it needs:** Current code, live trading scripts
- **Output:** Daily trading logs, results

---

## 🔧 SETUP INSTRUCTIONS

### Step 1: Prepare Files for Desktop Mac

On your **M2 Mac laptop** (current machine):

```bash
cd /Users/chavala/Mike-agent-project

# Create a package for desktop Mac
tar -czf training_package.tar.gz \
    data/historical/enriched/*.pkl \
    historical_training_system.py \
    train_historical_model.py \
    quant_features_collector.py \
    institutional_features.py \
    greeks_calculator.py \
    prevent_sleep.sh \
    start_training.sh \
    requirements.txt \
    venv/ \
    --exclude="venv/__pycache__" \
    --exclude="*.pyc"
```

**OR create a simpler sync script:**

```bash
# Create sync package
./prepare_for_desktop.sh
```

---

### Step 2: Transfer to Desktop Mac

**Option A: Using USB Drive**
```bash
# Copy to USB drive
cp training_package.tar.gz /Volumes/USB_DRIVE/

# On desktop Mac:
# 1. Insert USB drive
# 2. Copy training_package.tar.gz to Desktop
# 3. Extract: tar -xzf training_package.tar.gz -C ~/Desktop/mike-agent-training/
```

**Option B: Using Network Sharing (Same WiFi)**
```bash
# On M2 Mac laptop - enable file sharing
# System Preferences → Sharing → File Sharing → ON

# On desktop Mac - connect to M2 Mac:
# Finder → Go → Connect to Server
# Enter: smb://M2_MAC_IP_ADDRESS
# Or: smb://M2_MAC_NAME.local

# Copy files over network
```

**Option C: Using Cloud Storage (iCloud/Dropbox/Google Drive)**
```bash
# Upload to cloud from M2 Mac
cp training_package.tar.gz ~/iCloud\ Drive/

# Download on desktop Mac from cloud
```

---

### Step 3: Setup Desktop Mac for Training

On your **Desktop Mac**, extract and setup:

```bash
# Create training directory
mkdir -p ~/mike-agent-training
cd ~/mike-agent-training

# Extract package (if using tar)
tar -xzf training_package.tar.gz

# OR if using git/clone, clone the repo:
# git clone YOUR_REPO_URL ~/mike-agent-training
```

**Setup Virtual Environment:**

```bash
cd ~/mike-agent-training

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install RL dependencies specifically
pip install "stable-baselines3[extra]" gymnasium

# Verify data files exist
ls -lh data/historical/enriched/*.pkl
# Should see: SPY, QQQ, SPX enriched files (~3.5 MB each)
```

**Prevent Sleep:**

```bash
# Make script executable
chmod +x prevent_sleep.sh

# Start preventing sleep
./prevent_sleep.sh start
```

---

### Step 4: Start Training on Desktop Mac

```bash
cd ~/mike-agent-training
source venv/bin/activate

# Start training (background)
./start_training.sh

# OR manually:
nohup python train_historical_model.py \
    --symbols SPY,QQQ,SPX \
    --start-date 2002-01-01 \
    --timesteps 5000000 \
    --use-greeks \
    --regime-balanced \
    --save-freq 100000 \
    > training.log 2>&1 &

# Monitor
tail -f training.log
```

---

### Step 5: Continue Daily Work on M2 Mac

Your **M2 Mac laptop** continues working normally:

**Daily Live Trading (as usual):**
```bash
cd /Users/chavala/Mike-agent-project
source venv/bin/activate

# Run live trading agent
python mike_agent_live_safe.py

# Or dashboard
streamlit run app.py
```

**No conflicts because:**
- ✅ Training runs on desktop (separate machine)
- ✅ Live trading runs on laptop (separate machine)
- ✅ Different directories/processes
- ✅ No shared resources

---

## 📊 MONITORING BOTH MACHINES

### Desktop Mac (Training) - Check Progress

**Via SSH (if enabled):**
```bash
# From M2 Mac laptop
ssh user@desktop-mac-ip

# Then check training
tail -f ~/mike-agent-training/training.log
```

**Via Screen Sharing (if enabled):**
- System Preferences → Sharing → Screen Sharing
- Connect from M2 Mac to view desktop Mac screen

**Check Remotely:**
```bash
# On desktop Mac - create status file
cat > ~/mike-agent-training/check_status.sh << 'EOF'
#!/bin/bash
echo "=== Training Status ==="
echo "Time: $(date)"
echo ""
echo "Latest log entries:"
tail -5 training.log
echo ""
echo "Latest checkpoint:"
ls -lth models/*.zip 2>/dev/null | head -1
echo ""
echo "Process running:"
ps aux | grep train_historical_model | grep -v grep
EOF

chmod +x ~/mike-agent-training/check_status.sh
```

---

## 🔄 SYNC TRAINED MODEL BACK TO M2 MAC

After training completes (7 days later):

### Option 1: USB Drive
```bash
# On desktop Mac
cp models/mike_rl_model_final.zip /Volumes/USB_DRIVE/

# On M2 Mac laptop
cp /Volumes/USB_DRIVE/mike_rl_model_final.zip \
   /Users/chavala/Mike-agent-project/models/
```

### Option 2: Network/Cloud
```bash
# Same method as initial transfer
# Copy models/*.zip from desktop to laptop
```

### Option 3: Git (if using version control)
```bash
# On desktop Mac
git add models/mike_rl_model_final.zip
git commit -m "Trained model after 7 days"
git push

# On M2 Mac laptop
git pull
```

---

## 📁 FILE STRUCTURE

### Desktop Mac (Training)
```
~/mike-agent-training/
├── data/
│   └── historical/
│       └── enriched/
│           ├── SPY_enriched_2002-01-01_latest.pkl
│           ├── QQQ_enriched_2002-01-01_latest.pkl
│           └── SPX_enriched_2002-01-01_latest.pkl
├── historical_training_system.py
├── train_historical_model.py
├── quant_features_collector.py
├── prevent_sleep.sh
├── start_training.sh
├── requirements.txt
├── venv/
├── models/                    # ← Trained models saved here
│   ├── mike_rl_model_step_100000.zip
│   ├── mike_rl_model_step_200000.zip
│   └── ...
└── training.log
```

### M2 Mac Laptop (Daily Work)
```
/Users/chavala/Mike-agent-project/
├── mike_agent_live_safe.py    # ← Daily live trading
├── app.py                      # ← Dashboard
├── models/                     # ← Will receive trained model later
│   └── (current model)
├── logs/                       # ← Daily trading logs
└── ... (all your current files)
```

---

## ✅ CHECKLIST

### Desktop Mac Setup
- [ ] Create training directory
- [ ] Transfer data files (enriched .pkl files)
- [ ] Transfer training scripts
- [ ] Setup virtual environment
- [ ] Install dependencies
- [ ] Verify data files exist
- [ ] Prevent sleep: `./prevent_sleep.sh start`
- [ ] Start training: `./start_training.sh`
- [ ] Verify training is running: `ps aux | grep train`

### M2 Mac Laptop (Continue Normal Use)
- [ ] Continue daily live trading (no changes needed)
- [ ] Run dashboard as usual
- [ ] Monitor daily P&L
- [ ] No interference with training

### After 7 Days
- [ ] Check training completed on desktop Mac
- [ ] Copy trained model to M2 Mac laptop
- [ ] Test new model
- [ ] Update live trading to use new model
- [ ] Stop sleep prevention on desktop Mac

---

## 🚨 IMPORTANT NOTES

### 1. Keep Desktop Mac Plugged In
- Desktop Mac should stay plugged in for 7 days
- Don't let battery drain

### 2. Keep Desktop Mac Awake
- Use `prevent_sleep.sh` script
- Close lid OK if plugged in (but keep awake)

### 3. Network Sharing (Optional)
- Enable Screen Sharing or SSH for remote monitoring
- Not required, but helpful

### 4. Both Machines Independent
- Training on desktop doesn't affect laptop
- Live trading on laptop doesn't affect training
- Can use both simultaneously

### 5. Model File Size
- Final model will be ~10-50 MB
- Easy to transfer via USB/network/cloud

---

## 🎯 QUICK SETUP SCRIPT

I'll create a script to package everything for easy transfer:

```bash
# prepare_for_desktop.sh
# Packages all training files for desktop Mac
```

---

## 📞 REMOTE MONITORING

### Option 1: SSH (Best)
```bash
# Enable SSH on desktop Mac:
# System Preferences → Sharing → Remote Login → ON

# From M2 Mac laptop:
ssh user@desktop-mac-ip
cd ~/mike-agent-training
tail -f training.log
```

### Option 2: Screen Sharing
```bash
# Enable on desktop Mac:
# System Preferences → Sharing → Screen Sharing → ON

# Connect from M2 Mac:
# Finder → Go → Connect to Server
# Enter: vnc://desktop-mac-ip
```

### Option 3: Simple Status File
```bash
# On desktop Mac - create web-accessible status
# (if you have a simple web server)
```

---

## ✅ SUMMARY

**This setup is PERFECT for your needs:**

1. ✅ **Desktop Mac:** Runs 7-day training uninterrupted
2. ✅ **M2 Mac Laptop:** Continue daily live trading/testing
3. ✅ **No conflicts:** Completely separate
4. ✅ **Easy transfer:** Simple file copy after training
5. ✅ **Remote monitoring:** Optional but helpful

**Next Steps:**
1. Package files for desktop Mac
2. Transfer to desktop Mac
3. Setup on desktop Mac
4. Start training on desktop Mac
5. Continue daily work on M2 Mac laptop
6. After 7 days: Copy trained model back to laptop

---

**Ready to setup? Let me create the packaging script!**

