# ⚡ QUICK PARALLEL SETUP GUIDE

**Goal:** Run 7-day training on desktop Mac, continue daily live trading on M2 Mac laptop

---

## ✅ YES, THIS WORKS PERFECTLY!

### Your Setup:
- **Desktop Mac:** 7-day training (uninterrupted)
- **M2 Mac Laptop:** Daily live trading/testing (normal use)
- **No conflicts:** Completely separate machines

---

## 🚀 3-STEP SETUP

### Step 1: Package Files (On M2 Mac Laptop - NOW)

```bash
cd /Users/chavala/Mike-agent-project
./prepare_for_desktop.sh
```

This creates: `mike-agent-training-package_YYYYMMDD_HHMMSS.tar.gz`

**Size:** ~15-20 MB (includes all data files)

---

### Step 2: Transfer to Desktop Mac

**Option A: USB Drive**
```bash
# Copy to USB
cp mike-agent-training-package_*.tar.gz /Volumes/USB_DRIVE/
```

**Option B: Cloud (iCloud/Dropbox/Google Drive)**
```bash
# Upload to cloud
cp mike-agent-training-package_*.tar.gz ~/iCloud\ Drive/
```

**Option C: Network Sharing**
```bash
# Enable file sharing on M2 Mac, then copy over network
```

---

### Step 3: Setup Desktop Mac (One Time)

```bash
# Extract package
tar -xzf mike-agent-training-package_*.tar.gz
cd mike-agent-training-package_*/

# Run automated setup
./setup_desktop_mac.sh

# Start training
./prevent_sleep.sh start
./start_training.sh

# Monitor
tail -f training.log
```

**That's it!** Training runs for 7 days.

---

## 📱 CONTINUE DAILY WORK ON M2 MAC

**No changes needed!** Continue as usual:

```bash
cd /Users/chavala/Mike-agent-project
source venv/bin/activate

# Daily live trading
python mike_agent_live_safe.py

# Or dashboard
streamlit run app.py
```

**No conflicts because:**
- ✅ Different machines
- ✅ Different directories
- ✅ Completely separate

---

## 📊 AFTER 7 DAYS

### Step 1: Get Trained Model from Desktop Mac

```bash
# On desktop Mac
cp models/mike_rl_model_step_5000000.zip /Volumes/USB_DRIVE/
# OR upload to cloud
```

### Step 2: Copy to M2 Mac Laptop

```bash
# On M2 Mac laptop
cp /Volumes/USB_DRIVE/mike_rl_model_step_5000000.zip \
   /Users/chavala/Mike-agent-project/models/
```

### Step 3: Update Live Trading to Use New Model

```bash
# Update mike_agent_live_safe.py to use new model path
# Or just rename:
mv models/mike_rl_model_step_5000000.zip models/mike_rl_model.zip
```

---

## ⏱️ TIMELINE

| Day | Desktop Mac | M2 Mac Laptop |
|-----|-------------|---------------|
| **Day 0** | Setup & start training | Package files, transfer |
| **Day 1-7** | Training runs 24/7 | Daily live trading/testing |
| **Day 7** | Training completes | Continue daily work |
| **Day 8** | Copy model to laptop | Receive & test new model |

---

## ✅ CHECKLIST

### On M2 Mac Laptop (Now):
- [ ] Run `./prepare_for_desktop.sh`
- [ ] Transfer package to desktop Mac
- [ ] Continue daily work (no changes needed)

### On Desktop Mac (One Time):
- [ ] Extract package
- [ ] Run `./setup_desktop_mac.sh`
- [ ] Start training: `./start_training.sh`
- [ ] Leave running for 7 days

### After 7 Days:
- [ ] Copy trained model from desktop to laptop
- [ ] Test new model
- [ ] Update live trading

---

## 💡 TIPS

1. **Keep Desktop Mac Plugged In:** Don't let battery drain
2. **Use prevent_sleep.sh:** Prevents Mac from sleeping
3. **Monitor Remotely (Optional):** Enable SSH/Screen Sharing
4. **Check Progress Daily:** `tail -f training.log` on desktop Mac

---

## 🎯 SUMMARY

**This setup is perfect for you:**

1. ✅ Desktop Mac runs training for 7 days uninterrupted
2. ✅ M2 Mac laptop continues daily live trading/testing
3. ✅ No interference between the two
4. ✅ Simple file transfer after training

**Ready to start? Run:**

```bash
./prepare_for_desktop.sh
```

Then transfer the package to your desktop Mac!

---

**Full details:** See `PARALLEL_TRAINING_SETUP.md`

