# 🚀 Complete Guide: Deploy to Railway from GitHub

**Step-by-step guide to connect GitHub repository to Railway for automatic deployments**

---

## 📋 Overview

Railway automatically pulls code from your GitHub repository and deploys it. Here's how it works:

```
Your Computer → GitHub → Railway → Live Website
   (push code)  (stores)  (deploys)  (accessible)
```

---

## Step 1: Push Your Code to GitHub

### 1.1 Check Current Git Status

```bash
cd /Users/chavala/Mike-agent-project
git status
```

### 1.2 Initialize Git (if needed)

If you see "not a git repository":

```bash
git init
git branch -M main
```

### 1.3 Check Remote Repository

See if GitHub remote is already set up:

```bash
git remote -v
```

**If you see your GitHub URL**, skip to Step 1.5.

**If you see "no remotes" or nothing**, continue to Step 1.4.

### 1.4 Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `mike-agent` (or your preferred name)
3. Description: "RL Trading Bot with 12 Safeguards"
4. Choose **Public** or **Private**
5. **DO NOT** check "Initialize with README"
6. Click **"Create repository"**

### 1.5 Connect Local Code to GitHub

**If remote doesn't exist:**

```bash
git remote add origin https://github.com/YOUR_USERNAME/mike-agent.git
```

**If remote exists but wrong URL:**

```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/mike-agent.git
```

Replace `YOUR_USERNAME` with your actual GitHub username (e.g., `achavala`).

### 1.6 Push Code to GitHub

```bash
# Stage all changes
git add .

# Commit changes
git commit -m "Initial commit: Mike Agent v3 - Production Ready"

# Push to GitHub
git push -u origin main
```

**If you get authentication errors:**
- Use GitHub Personal Access Token instead of password
- Or use SSH: `git remote set-url origin git@github.com:YOUR_USERNAME/mike-agent.git`

**Success:** Your code is now on GitHub! ✅

---

## Step 2: Connect GitHub to Railway

### 2.1 Sign Up for Railway

1. Go to https://railway.app
2. Click **"Start a New Project"** or **"Login"**
3. Click **"Login with GitHub"**
4. Authorize Railway to access your GitHub account

### 2.2 Create New Project in Railway

1. In Railway dashboard, click **"+ New Project"**
2. Select **"Deploy from GitHub repo"**
3. Authorize Railway (if prompted)
4. You'll see a list of your GitHub repositories
5. Find and select **`mike-agent`** (or your repo name)
6. Click **"Deploy Now"**

**Railway automatically:**
- Detects it's a Python/Streamlit project
- Reads `Procfile` and `requirements_railway.txt`
- Starts building your application

---

## Step 3: Configure Railway Deployment

### 3.1 Wait for Initial Build

Railway will show:
- Building...
- Installing dependencies...
- Starting application...

**First build takes 2-5 minutes.**

### 3.2 Check Build Status

Click on your project → **"Deployments"** tab

**Success indicators:**
- ✅ Green checkmark
- "Build successful" message
- Application URL appears

**If build fails:**
- Check logs for errors
- Usually missing dependencies or configuration issues
- See Troubleshooting section below

### 3.3 Get Your Application URL

1. Click on your project
2. Go to **"Settings"** tab
3. Scroll to **"Networking"** section
4. Your URL: `https://mike-agent-XXXX.up.railway.app`

**Copy this URL** - you'll need it!

---

## Step 4: Add Environment Variables

**Critical:** Add your Alpaca API keys here (not in code!)

### 4.1 Open Variables Tab

1. In Railway dashboard → Your project
2. Click **"Variables"** tab
3. Click **"+ New Variable"**

### 4.2 Add Required Variables

Add each variable one by one:

**Variable 1:**
- Key: `ALPACA_KEY`
- Value: Your Alpaca API key (starts with `PK...`)
- Click **"Add"**

**Variable 2:**
- Key: `ALPACA_SECRET`
- Value: Your Alpaca secret key
- Click **"Add"**

**Variable 3:**
- Key: `ALPACA_BASE_URL`
- Value: `https://paper-api.alpaca.markets`
- Click **"Add"**

**Variable 4:**
- Key: `MODE`
- Value: `paper`
- Click **"Add"**

### 4.3 Save and Redeploy

After adding all variables:
- Railway automatically redeploys
- Wait 1-2 minutes for redeploy
- Your app now has API access!

---

## Step 5: Verify Deployment

### 5.1 Open Your Application

1. Go to your Railway URL: `https://mike-agent-XXXX.up.railway.app`
2. Streamlit dashboard should load
3. You should see your trading dashboard!

### 5.2 Test Functionality

- ✅ Dashboard loads
- ✅ Can see portfolio (if connected)
- ✅ Can see positions (if any)
- ✅ Logs are accessible

**If something doesn't work:**
- Check Railway logs: **"Deployments"** → Click latest deployment → **"View Logs"**
- Check environment variables are set correctly
- Verify API keys are valid

---

## Step 6: Automatic Deployments (The Magic!)

### How Auto-Deploy Works

Once connected, Railway watches your GitHub repo:

```
You push code to GitHub
       ↓
Railway detects change
       ↓
Railway builds new version
       ↓
Railway deploys automatically
       ↓
Your app updates!
```

### Making Updates

**To update your app:**

1. Make changes locally
2. Commit and push:

```bash
cd /Users/chavala/Mike-agent-project

# Make your changes...
# Edit files...

# Stage changes
git add .

# Commit
git commit -m "Update: Added new feature"

# Push to GitHub
git push origin main
```

3. Railway automatically:
   - Detects the push
   - Starts building
   - Deploys new version
   - Your app updates! (takes 2-3 minutes)

**No manual deployment needed!** 🎉

---

## Step 7: Access from Mobile

### 7.1 Get Railway URL

1. Railway dashboard → Your project → **"Settings"**
2. Copy URL: `https://mike-agent-XXXX.up.railway.app`

### 7.2 Add to Mobile Home Screen

**iPhone/iPad:**
1. Open Safari
2. Paste Railway URL
3. Tap **Share** button (square with arrow)
4. Tap **"Add to Home Screen"**
5. Name it "Mike Agent"
6. Tap **"Add"**

**Android:**
1. Open Chrome
2. Paste Railway URL
3. Tap menu (3 dots)
4. Tap **"Add to Home Screen"**
5. Name it "Mike Agent"
6. Tap **"Add"**

**Now you have a mobile app!** 📱

---

## 🔧 Troubleshooting

### Build Fails

**Error:** `ModuleNotFoundError: No module named 'X'`

**Fix:**
1. Add missing package to `requirements_railway.txt`
2. Commit and push:

```bash
git add requirements_railway.txt
git commit -m "Add missing dependency"
git push origin main
```

3. Railway auto-redeploys

### API Errors

**Error:** `401 Unauthorized` or `Invalid API key`

**Fix:**
1. Railway dashboard → **"Variables"** tab
2. Check `ALPACA_KEY` and `ALPACA_SECRET` are correct
3. Make sure no extra spaces/quotes
4. Redeploy: **"Deployments"** → **"Redeploy"**

### Port Binding Error

**Error:** `Port already in use`

**Fix:**
- Check `Procfile` has `--server.port=$PORT`
- Already configured! ✅
- If still fails, check Railway logs

### App Not Loading

**Check:**
1. Railway dashboard → **"Deployments"** → Check status
2. Click deployment → **"View Logs"** → Look for errors
3. Check if build succeeded (green checkmark)
4. Verify URL is correct

### Connection Issues

**If Railway can't connect to GitHub:**
1. Railway dashboard → **"Settings"** → **"Connected Accounts"**
2. Re-authorize GitHub connection
3. Try deploying again

---

## 📁 Required Files for Railway

Railway needs these files to work:

### ✅ `Procfile`
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
```
**Purpose:** Tells Railway how to run your app

### ✅ `requirements_railway.txt`
```
streamlit==1.51.0
alpaca-trade-api==0.48
...
```
**Purpose:** Lists all Python dependencies

### ✅ `app.py`
**Purpose:** Your Streamlit dashboard (main file)

**All of these are already in your project!** ✅

---

## 🔒 Security Checklist

- ✅ `config.py` is in `.gitignore` (never committed)
- ✅ API keys stored in Railway environment variables
- ✅ Repository can be private
- ✅ No hardcoded secrets in code

---

## 💰 Cost Estimate

| Service | Free Tier | Paid Tier |
|---------|-----------|-----------|
| **GitHub** | Unlimited repos | Free |
| **Railway** | 500 hours/month, 512MB RAM | $5/month (1GB RAM) |

**For testing:** Free tier is perfect!  
**For production:** $5/month for 24/7 uptime

---

## 📝 Quick Reference

### GitHub Commands

```bash
# Check status
git status

# Stage changes
git add .

# Commit
git commit -m "Your message"

# Push to GitHub
git push origin main

# Check remote
git remote -v
```

### Railway Dashboard

- **Projects:** List of all deployments
- **Deployments:** Build history and logs
- **Variables:** Environment variables (API keys)
- **Settings:** Configuration and URL
- **Metrics:** Resource usage

---

## ✅ Deployment Checklist

Before deploying, ensure:

- [ ] Code is pushed to GitHub
- [ ] Railway project is created
- [ ] GitHub repo is connected
- [ ] Environment variables are set
- [ ] Build is successful
- [ ] App URL is accessible
- [ ] Mobile app is added to home screen

---

## 🎯 Next Steps

1. **Test locally first** - Make sure everything works
2. **Push to GitHub** - Get code online
3. **Deploy to Railway** - Go live
4. **Set environment variables** - Add API keys
5. **Test on mobile** - Access from phone
6. **Monitor** - Watch for any issues

---

## 🆘 Need Help?

**Railway Support:**
- Docs: https://docs.railway.app
- Discord: https://discord.gg/railway

**Common Issues:**
- Check deployment logs first
- Verify environment variables
- Check GitHub repository access
- Review `Procfile` and `requirements_railway.txt`

---

**You're all set! Your code will automatically deploy to Railway whenever you push to GitHub.** 🚀

