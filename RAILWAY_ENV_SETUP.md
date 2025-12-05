# Railway Environment Variables Setup

## ⚠️ IMPORTANT: Set These in Railway

The mobile app error was caused by missing `config.py` (which is correctly excluded from git for security).

**Solution:** Use Railway environment variables instead.

## 📋 Required Environment Variables

Go to Railway Dashboard → Your Project → **Variables** tab and add:

### 1. ALPACA_KEY
```
ALPACA_KEY=PKXX2KTB6QGJ7EW4CG7YFX4XUF
```
(Replace with your actual paper trading key)

### 2. ALPACA_SECRET
```
ALPACA_SECRET=5U2MjLpCRLKfBDhrz5X93ZMtuxJJ2k9Y4H5FXgHqoNKo
```
(Replace with your actual paper trading secret)

### 3. ALPACA_BASE_URL
```
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```
(For paper trading - change to `https://api.alpaca.markets` for live)

### 4. (Optional) FORCE_CAPITAL
```
FORCE_CAPITAL=10000
```
(If you want to force a specific capital amount)

## 🔧 How to Add Variables in Railway

1. Go to Railway dashboard: https://railway.app
2. Click on your project: **fabulous-magic**
3. Click **Settings** tab
4. Click **Variables** section
5. Click **"New Variable"** for each one:
   - Name: `ALPACA_KEY`
   - Value: `your_paper_key_here`
   - Click **"Add"**
6. Repeat for `ALPACA_SECRET` and `ALPACA_BASE_URL`
7. Railway will automatically redeploy

## ✅ After Adding Variables

1. Railway will detect the new variables
2. Automatically redeploy your app
3. Wait 2-3 minutes for deployment
4. Test on mobile - should work now!

## 🔍 Verify Variables Are Set

In Railway dashboard:
- Go to **Variables** tab
- You should see:
  - ✅ ALPACA_KEY
  - ✅ ALPACA_SECRET
  - ✅ ALPACA_BASE_URL

## 🐛 If Still Not Working

1. **Check Railway logs:**
   - Go to **Deployments** tab
   - Click on latest deployment
   - Check **Deploy Logs** for errors

2. **Verify variable names:**
   - Must be exactly: `ALPACA_KEY` (not `ALPACA_KEY_` or `ALPACA-KEY`)
   - No spaces before/after the `=`

3. **Redeploy manually:**
   - In Railway, click **"Redeploy"** button
   - Or push a new commit to trigger redeploy

## 📱 Test on Mobile

After Railway redeploys:
1. Open Safari on iPhone
2. Go to: `https://web-production-6d4fd.up.railway.app`
3. Should load without errors now!
4. Add to home screen if you want

---

**The fix is pushed to GitHub. Just add the environment variables in Railway!**

