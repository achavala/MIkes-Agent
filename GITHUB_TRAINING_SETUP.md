# 🔄 GITHUB-BASED TRAINING SETUP

**Strategy:** Use GitHub to sync between M2 Mac laptop and Desktop Mac for parallel training

---

## ✅ YES, THIS IS THE BEST APPROACH!

### Benefits:
- ✅ **GitHub sync:** Easy push/pull between machines
- ✅ **Version control:** Track all changes
- ✅ **Cursor on both:** Edit and validate from either machine
- ✅ **Automatic sync:** Changes sync automatically
- ✅ **Backup:** Everything stored in GitHub
- ✅ **No manual file transfer:** Much cleaner workflow

---

## 📋 SETUP OVERVIEW

### M2 Mac Laptop (Development Machine)
- Push code to GitHub
- Continue daily live trading/testing
- Monitor training progress via GitHub

### Desktop Mac (Training Machine)
- Clone from GitHub
- Install Cursor
- Run 7-day training
- Commit training progress/logs to GitHub

---

## 🚀 STEP-BY-STEP SETUP

### Step 1: Prepare GitHub Repository (On M2 Mac)

**1.1 Check current Git status:**

```bash
cd /Users/chavala/Mike-agent-project
git status
```

**1.2 Update .gitignore (if needed):**

Make sure training outputs are tracked but large data files aren't:

```bash
# Check .gitignore
cat .gitignore
```

**Good .gitignore should include:**
```
# Training outputs (track these for progress)
!models/*.zip
!training*.log

# Large data files (don't track - too big)
data/historical/enriched/*.pkl
*.pkl
```

**1.3 Add all files to Git:**

```bash
# Add all training files
git add historical_training_system.py
git add train_historical_model.py
git add quant_features_collector.py
git add institutional_features.py
git add greeks_calculator.py
git add prevent_sleep.sh
git add start_training.sh
git add COMPLETE_TRAINING_PLAN.md
git add PARALLEL_TRAINING_SETUP.md
git add GITHUB_TRAINING_SETUP.md

# Add setup scripts
git add prepare_for_desktop.sh
git add comprehensive_pre_training_validation.py
git add TRAINING_TIME_ESTIMATE.py

# Commit
git commit -m "Add complete training system and parallel setup"

# Push to GitHub
git push origin main
# OR if branch is master:
git push origin master
```

**1.4 Verify on GitHub:**

Go to your GitHub repository and verify all files are there.

---

### Step 2: Setup Desktop Mac

**2.1 Install Cursor (if not already):**

- Download from: https://cursor.sh
- Install normally

**2.2 Clone Repository:**

```bash
# Navigate to desired location
cd ~/Desktop  # or wherever you want

# Clone repository
git clone YOUR_GITHUB_REPO_URL mike-agent-training
# Example:
# git clone https://github.com/yourusername/mike-agent-project.git mike-agent-training

cd mike-agent-training
```

**2.3 Setup Virtual Environment:**

```bash
# Create venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install RL dependencies
pip install "stable-baselines3[extra]" gymnasium
```

**2.4 Download Data Files (Only Once):**

Since `.pkl` files are too large for GitHub, you need to download them:

**Option A: From M2 Mac via Network/Cloud**
```bash
# On M2 Mac, share data folder
# Then on Desktop Mac, copy:
# cp -r /path/to/shared/data/historical/enriched data/historical/
```

**Option B: Use Data Collection Script**
```bash
# On Desktop Mac, re-run data collection (takes 10-20 minutes)
source venv/bin/activate
python collect_quant_features.py --symbols SPY,QQQ,SPX --start-date 2002-01-01 --interval 1d
```

**Option C: Use Git LFS (Large File Storage)**
```bash
# On M2 Mac
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git add data/historical/enriched/*.pkl
git commit -m "Add data files via LFS"
git push

# On Desktop Mac (after cloning)
git lfs install
git lfs pull
```

---

### Step 3: Open in Cursor (Desktop Mac)

**3.1 Open Repository in Cursor:**

```bash
# From terminal
cd ~/Desktop/mike-agent-training
cursor .
```

**OR:**
- Open Cursor
- File → Open Folder
- Select `~/Desktop/mike-agent-training`

**3.2 Verify Setup:**

In Cursor terminal:
```bash
source venv/bin/activate
python --version
python -c "import stable_baselines3; print('✅ RL libraries installed')"
ls -lh data/historical/enriched/*.pkl
```

---

### Step 4: Start Training (Desktop Mac)

**4.1 Prevent Sleep:**

```bash
./prevent_sleep.sh start
```

**4.2 Start Training:**

```bash
# In Cursor terminal or regular terminal
source venv/bin/activate
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
```

**4.3 Monitor in Cursor:**

You can watch logs directly in Cursor:
```bash
tail -f training.log
```

---

### Step 5: Sync Training Progress (Both Machines)

**5.1 Commit Training Progress (Desktop Mac):**

Create a script to auto-commit training progress:

```bash
# Create auto-commit script
cat > commit_training_progress.sh << 'EOF'
#!/bin/bash
cd ~/Desktop/mike-agent-training

# Add training logs and checkpoints
git add training*.log
git add models/*.zip 2>/dev/null || true

# Commit
git commit -m "Training progress: $(date +%Y-%m-%d\ %H:%M:%S)" || echo "No changes"

# Push
git push origin main 2>/dev/null || git push origin master 2>/dev/null || echo "Push failed"
EOF

chmod +x commit_training_progress.sh
```

**5.2 Schedule Auto-Commits (Desktop Mac):**

```bash
# Add to crontab for hourly commits
(crontab -l 2>/dev/null; echo "0 * * * * cd ~/Desktop/mike-agent-training && ./commit_training_progress.sh >> /tmp/git_commit.log 2>&1") | crontab -
```

**5.3 Pull Updates (M2 Mac):**

From your M2 Mac, you can pull training progress:

```bash
cd /Users/chavala/Mike-agent-project
git pull

# View training progress
tail -50 training*.log
ls -lth models/*.zip | head -5
```

---

## 🔄 WORKFLOW SUMMARY

### Daily Workflow:

**On Desktop Mac (Training):**
```bash
# Training runs automatically
# Progress auto-commits to GitHub hourly
# You can monitor in Cursor anytime
```

**On M2 Mac (Daily Work):**
```bash
# Continue live trading as usual
cd /Users/chavala/Mike-agent-project
git pull  # Get latest training progress
tail -50 training*.log  # View progress
```

**Both Machines:**
- Open Cursor
- Edit/view code
- Commit changes
- Push/pull via Git

---

## 📊 MONITORING TRAINING FROM BOTH MACHINES

### Option 1: View Logs in GitHub

Training logs are committed to GitHub, view them there:
```
training.log → View in GitHub web interface
```

### Option 2: Pull and View Locally

**On M2 Mac:**
```bash
cd /Users/chavala/Mike-agent-project
git pull
tail -f training*.log  # Watch latest logs
```

### Option 3: Use Cursor on Both Machines

- **Desktop Mac:** Open training repo in Cursor, view logs locally
- **M2 Mac:** Open main repo in Cursor, pull and view training logs

---

## 🎯 GITHUB REPOSITORY STRUCTURE

```
your-github-repo/
├── historical_training_system.py
├── train_historical_model.py
├── quant_features_collector.py
├── institutional_features.py
├── greeks_calculator.py
├── prevent_sleep.sh
├── start_training.sh
├── requirements.txt
├── .gitignore
├── README.md
├── models/              # ← Trained models (tracked)
│   └── *.zip
├── training*.log        # ← Training logs (tracked)
├── data/                # ← Not tracked (too large)
│   └── historical/
│       └── enriched/
│           └── *.pkl   # ← Downloaded separately
└── ...
```

---

## ✅ CHECKLIST

### Initial Setup (M2 Mac):
- [ ] Verify Git repository is set up
- [ ] Update .gitignore (exclude large .pkl files)
- [ ] Commit all training scripts
- [ ] Push to GitHub
- [ ] Verify files on GitHub web

### Desktop Mac Setup:
- [ ] Install Cursor
- [ ] Clone repository from GitHub
- [ ] Setup virtual environment
- [ ] Install dependencies
- [ ] Download data files (.pkl) - via network/cloud/LFS
- [ ] Verify setup
- [ ] Open in Cursor

### Start Training (Desktop Mac):
- [ ] Prevent sleep: `./prevent_sleep.sh start`
- [ ] Start training: `./start_training.sh`
- [ ] Verify training is running
- [ ] Setup auto-commit script (optional)

### Monitor (Both Machines):
- [ ] Pull updates on M2 Mac: `git pull`
- [ ] View logs in Cursor
- [ ] Check GitHub for progress

---

## 🔧 ADVANCED: GIT LFS FOR DATA FILES

If you want to store data files in GitHub:

**1. Install Git LFS:**
```bash
# On both machines
brew install git-lfs  # macOS
# OR download from: https://git-lfs.github.com
```

**2. Setup Git LFS (On M2 Mac):**
```bash
cd /Users/chavala/Mike-agent-project
git lfs install
git lfs track "data/historical/enriched/*.pkl"
git add .gitattributes
git add data/historical/enriched/*.pkl
git commit -m "Add data files via LFS"
git push
```

**3. Pull LFS Files (On Desktop Mac):**
```bash
git lfs install
git clone YOUR_REPO_URL
cd mike-agent-training
git lfs pull  # Downloads large files
```

**Note:** Git LFS may require GitHub paid plan for large files.

---

## 💡 BEST PRACTICES

1. **Regular Commits:** Commit training progress regularly
2. **Pull Before Work:** Always `git pull` before starting work
3. **Separate Branches:** Consider using branches for training vs. development
4. **Monitor GitHub:** Check GitHub web for training logs
5. **Backup Models:** Keep model files in GitHub (they're tracked)

---

## 🚀 QUICK START COMMANDS

### On M2 Mac (Initial Setup):
```bash
cd /Users/chavala/Mike-agent-project
git add .
git commit -m "Add training system"
git push
```

### On Desktop Mac (Clone & Setup):
```bash
git clone YOUR_REPO_URL ~/Desktop/mike-agent-training
cd ~/Desktop/mike-agent-training
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install "stable-baselines3[extra]" gymnasium
cursor .
```

### After Training Starts:
```bash
# Desktop Mac - Auto-commit progress
./commit_training_progress.sh

# M2 Mac - Check progress
git pull
tail -50 training*.log
```

---

## ✅ SUMMARY

**This GitHub-based approach is PERFECT:**

1. ✅ **One command sync:** `git pull` / `git push`
2. ✅ **Cursor on both:** Edit from either machine
3. ✅ **Version control:** Track all changes
4. ✅ **Easy monitoring:** View progress from anywhere
5. ✅ **No manual transfer:** Everything via Git
6. ✅ **Automatic backup:** GitHub stores everything

**Ready to setup? Follow the steps above!**

---

## 📞 TROUBLESHOOTING

### "Repository not found"
- Check GitHub URL is correct
- Verify you have access
- Use HTTPS or SSH URL

### "Large files rejected"
- Use Git LFS for .pkl files
- OR exclude them and download separately

### "Training logs not syncing"
- Check auto-commit script is running
- Manually commit: `git add training*.log && git commit -m "Progress" && git push`

### "Data files missing on Desktop Mac"
- Download from M2 Mac via network/cloud
- OR re-run data collection script
- OR use Git LFS

---

**Full guide created! Follow step-by-step for GitHub-based training setup!**

