# 🚀 Mike Agent v3 - Deployment Guide

**Complete guide to deploy Mike Agent to GitHub & Railway with mobile access**

---

## 📋 Prerequisites

- GitHub account: https://github.com/achavala
- Railway account: https://railway.app (free tier works)
- Alpaca API keys (paper or live)
- Git installed (`brew install git` on macOS)

---

## Step 1: Prepare Project for GitHub

### 1.1 Verify `.gitignore` is correct

The `.gitignore` file should exclude:
- `config.py` (contains API keys)
- `*.zip` (model files)
- `logs/`, `data/`, `*.csv`, `*.log`
- `venv/`, `__pycache__/`

**Already done!** ✓

### 1.2 Create GitHub Repository

1. Go to https://github.com/achavala
2. Click **"+"** → **"New repository"**
3. Name: `mike-agent` (or `mike-agent-project`)
4. Description: "RL Trading Bot with 12 Safeguards – December 2025"
5. Choose **Public** or **Private**
6. **Don't** initialize with README (you already have one)
7. Click **"Create repository"**

### 1.3 Upload to GitHub

Run these commands in your terminal:

```bash
cd /Users/chavala/Mike-agent-project

# Initialize git (if not already done)
git init

# Add all files (respects .gitignore)
git add .

# Commit
git commit -m "Initial commit: Mike Agent v3 Final Form - Production Ready"

# Set main branch
git branch -M main

# Add remote (replace YOUR_USERNAME with achavala)
git remote add origin https://github.com/achavala/mike-agent.git

# Push to GitHub
git push -u origin main
```

**If you get "remote already exists" error:**
```bash
git remote remove origin
git remote add origin https://github.com/achavala/mike-agent.git
git push -u origin main
```

**Result:** Your code is now at https://github.com/achavala/mike-agent 🎉

---

## Step 2: Deploy to Railway

### 2.1 Sign Up for Railway

1. Go to https://railway.app/dashboard
2. Click **"Sign Up"** → **"Login with GitHub"**
3. Authorize Railway to access your GitHub repos

### 2.2 Create New Project

1. In Railway dashboard, click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Authorize Railway (if prompted)
4. Select repository: `achavala/mike-agent`
5. Railway auto-detects Python/Streamlit → starts building

### 2.3 Configure Environment Variables

**Critical:** Add your Alpaca API keys here (not in code!)

1. In Railway dashboard → Your project → **"Variables"** tab
2. Click **"New Variable"** and add:

```
ALPACA_KEY = PKXX2KTB6QGJ7EW4CG7YFX4XUF
ALPACA_SECRET = 5U2MjLpCRLKfBDhrz5X93ZMtuxJJ2k9Y4H5FXgHqoNKo
ALPACA_BASE_URL = https://paper-api.alpaca.markets
MODE = paper
```

**For live trading later:**
```
ALPACA_BASE_URL = https://api.alpaca.markets
MODE = live
```

3. Click **"Save"** → Railway redeploys automatically

### 2.4 Verify Deployment

1. Go to **"Deployments"** tab → Watch build logs
2. Wait for **"Build successful"** message
3. Click **"View Logs"** → Should see Streamlit starting
4. Your app URL appears at top: `https://mike-agent-XXXX.up.railway.app`

**If build fails:**
- Check logs for missing dependencies
- Update `requirements_railway.txt` if needed
- Push to GitHub → Railway auto-redeploys

---

## Step 3: Access from Mobile Phone

### 3.1 Get Your Railway URL

1. In Railway dashboard → Your project → **"Settings"** → **"Networking"**
2. Copy the generated URL: `https://mike-agent-XXXX.up.railway.app`

### 3.2 Open on Mobile

**iOS (iPhone/iPad):**
1. Open **Safari** browser
2. Paste Railway URL
3. Tap **"Share"** → **"Add to Home Screen"**
4. Name it "Mike Agent"
5. Now you have an app icon on your home screen!

**Android:**
1. Open **Chrome** browser
2. Paste Railway URL
3. Tap **Menu** (3 dots) → **"Add to Home Screen"**
4. Name it "Mike Agent"
5. App icon appears on home screen

### 3.3 Test Mobile Access

- Open the app icon → Streamlit dashboard loads
- Tap **"Start Trading"** → Monitor live trades
- Charts, logs, PnL all work on mobile
- Refresh to see latest data

**Mobile-optimized:** Streamlit dashboards are responsive and work great on phones!

---

## Step 4: Update Code (Auto-Deploy)

### 4.1 Make Changes Locally

```bash
cd /Users/chavala/Mike-agent-project

# Make your changes to files
# ...

# Commit and push
git add .
git commit -m "Update: Added new feature"
git push origin main
```

### 4.2 Railway Auto-Redeploys

- Railway watches your GitHub repo
- On every push → Auto-builds and redeploys
- Your mobile app updates automatically (refresh browser)

**No manual deployment needed!** 🚀

---

## Troubleshooting

### Build Fails

**Error:** `ModuleNotFoundError: No module named 'X'`

**Fix:**
1. Add missing package to `requirements_railway.txt`
2. Push to GitHub
3. Railway redeploys automatically

### API Errors

**Error:** `401 Unauthorized` or `Invalid API key`

**Fix:**
1. Check Railway **Variables** tab
2. Verify `ALPACA_KEY` and `ALPACA_SECRET` are correct
3. Ensure no extra spaces/quotes
4. Redeploy

### Port Binding Error

**Error:** `Port already in use`

**Fix:**
- Ensure `Procfile` has `--server.port=$PORT`
- Railway sets `$PORT` automatically
- Already configured! ✓

### Model Not Found

**Error:** `FileNotFoundError: mike_rl_model.zip`

**Fix:**
1. Upload model file to GitHub (if < 100MB)
2. Or fetch dynamically from S3/cloud storage
3. Or train model on Railway (add training script)

### Mobile App Slow

**Fix:**
- Railway free tier has 512MB RAM (fine for testing)
- Upgrade to $5/month plan for more resources
- Or optimize Streamlit dashboard (reduce chart updates)

---

## Security Checklist

- ✅ `config.py` is in `.gitignore` (never committed)
- ✅ API keys stored in Railway environment variables
- ✅ Repository can be private (if you want)
- ✅ No hardcoded secrets in code
- ✅ `config.py.example` shows structure (safe to commit)

---

## Cost Estimate

| Service | Free Tier | Paid Tier |
|---------|-----------|-----------|
| **GitHub** | Unlimited repos | Free |
| **Railway** | 500 hours/month, 512MB RAM | $5/month (1GB RAM) |

**For testing:** Free tier is perfect  
**For production:** $5/month gives you 24/7 uptime

---

## Next Steps

1. ✅ Push to GitHub
2. ✅ Deploy to Railway
3. ✅ Test on mobile
4. ✅ Go live: Switch Railway vars to live keys
5. ✅ Watch it compound!

---

## Support

- **Railway Docs:** https://docs.railway.app
- **Streamlit Docs:** https://docs.streamlit.io
- **GitHub Issues:** Create issue in your repo

---

**Your Mike Agent is now cloud-native, mobile-accessible, and eternal.** 🚀

**The empire expands.**

