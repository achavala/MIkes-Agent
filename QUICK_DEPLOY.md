# ⚡ Quick Deploy Guide - 5 Minutes

## Step 1: Push to GitHub (2 min)

```bash
cd /Users/chavala/Mike-agent-project
chmod +x deploy_to_github.sh
./deploy_to_github.sh
```

**Or manually:**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/achavala/mike-agent.git
git push -u origin main
```

## Step 2: Deploy to Railway (2 min)

1. Go to https://railway.app/dashboard
2. **"New Project"** → **"Deploy from GitHub repo"**
3. Select `achavala/mike-agent`
4. Wait for build (auto-detects Streamlit)

## Step 3: Add Environment Variables (1 min)

In Railway dashboard → **"Variables"** tab → Add:

```
ALPACA_KEY = YOUR_PAPER_KEY
ALPACA_SECRET = YOUR_PAPER_SECRET
ALPACA_BASE_URL = https://paper-api.alpaca.markets
MODE = paper
```

**Save** → Auto-redeploys

## Step 4: Get Mobile URL

Railway dashboard → **"Settings"** → **"Networking"** → Copy URL

**Open on phone:**
- iOS: Safari → Paste URL → Share → Add to Home Screen
- Android: Chrome → Paste URL → Menu → Add to Home Screen

**Done!** 🎉

---

**Full guide:** See `DEPLOYMENT_GUIDE.md`

