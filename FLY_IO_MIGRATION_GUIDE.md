# 🚀 Fly.io Migration Guide

**Target Tag:** `Freeze-for-Paper-Trade-Deployment-Ready`  
**Platform:** Fly.io  
**Mode:** Paper Trading  
**Budget:** <$15/month

---

## ✅ Prerequisites

1. **Verify you're on the frozen tag:**
   ```bash
   git checkout Freeze-for-Paper-Trade-Deployment-Ready
   git describe --tags --exact-match
   ```
   Should output: `Freeze-for-Paper-Trade-Deployment-Ready`

2. **Required files (now created):**
   - ✅ `Dockerfile` - Docker image definition
   - ✅ `start_cloud.sh` - Startup script for agent + dashboard
   - ✅ `requirements.txt` - Python dependencies

---

## 📋 Step-by-Step Migration

### STEP 1: Install Fly.io CLI

```bash
curl -L https://fly.io/install.sh | sh
fly auth login
fly version
```

### STEP 2: Create Fly.io App

From repo root (on the frozen tag):

```bash
fly launch
```

**Answer prompts:**
- **App name:** `mikes-agent-paper`
- **Region:** `ord` (Chicago) or your preferred region
- **Dockerfile detected?** Yes
- **Postgres?** No
- **Redis?** No
- **Deploy now?** **No** (we'll set secrets first)

This creates `fly.toml` configuration file.

### STEP 3: Set Environment Variables

```bash
fly secrets set \
  ALPACA_KEY=YOUR_PAPER_KEY \
  ALPACA_SECRET=YOUR_PAPER_SECRET \
  MODE=paper \
  TZ=America/New_York
```

**Verify:**
```bash
fly secrets list
```

### STEP 4: Deploy

Since you're on the frozen tag, this deploys exactly that code:

```bash
fly deploy
```

This will:
- Build Docker image from `Dockerfile`
- Start a Linux VM
- Run `start_cloud.sh` (starts both agent and dashboard)
- Keep it alive 24/7

### STEP 5: Verify Deployment

**Check status:**
```bash
fly status
```

Should show: `1 machine running`

**Check logs:**
```bash
fly logs
```

Look for:
- ✅ "Starting Cloud Deployment..."
- ✅ "Starting Agent in PAPER mode..."
- ✅ "Starting Streamlit dashboard..."
- ✅ Market status messages

**Get dashboard URL:**
```bash
fly info
```

Open the URL in your browser (works on mobile too).

### STEP 6: Optimize Costs (Recommended)

```bash
fly scale vm shared-cpu-1x --memory 1024
```

**Expected costs:**
- **$5–$10/month** (idle)
- **$10–$15/month** (active trading)

---

## 🛡️ What You Get

✅ **24/7 Execution** - Runs even if laptop is off  
✅ **No Sleeping** - Continuous operation  
✅ **Immutable Deployment** - Exact tag, no branch confusion  
✅ **Full Linux VM** - Complete environment  
✅ **Deterministic Behavior** - Same code every time  

---

## 🚦 Market Open Behavior

- **Pre-market:** Logs show "Market closed — waiting for open..."
- **9:30 AM ET:** Trading starts automatically
- **Market hours:** Continuous trading loop
- **After hours:** Waits for next market open

**No manual intervention needed.**

---

## 🔒 Important Rules

1. **Never deploy from `main` branch**
   - Always checkout the tag first:
     ```bash
     git checkout Freeze-for-Paper-Trade-Deployment-Ready
     fly deploy
     ```

2. **Never reuse tags**
   - New deployment → new tag

3. **Always verify tag before deploying:**
   ```bash
   git describe --tags --exact-match
   ```

---

## 🐛 Troubleshooting

### Agent not starting?
```bash
fly logs | grep -i "error\|exception\|failed"
```

### Dashboard not accessible?
```bash
fly status
fly logs | grep -i "streamlit\|dashboard"
```

### Check if both processes are running:
```bash
fly ssh console
ps aux | grep -E "python|streamlit"
```

### Restart the app:
```bash
fly apps restart mikes-agent-paper
```

### View real-time logs:
```bash
fly logs --app mikes-agent-paper
```

---

## 📊 Monitoring

**Dashboard:** Access via `fly info` URL  
**Logs:** `fly logs`  
**Status:** `fly status`  
**SSH Access:** `fly ssh console` (for debugging)

---

## 🏁 Migration Complete

Once `fly deploy` finishes successfully:

✅ Railway is no longer needed  
✅ Paper trading is cloud-native  
✅ Production-ready deployment  
✅ 24/7 autonomous operation  

---

## 📝 Notes

- The `start_cloud.sh` script runs both:
  - Trading agent (`mike_agent_live_safe.py`)
  - Streamlit dashboard (`dashboard_app.py`)

- Both processes run in the background
- If either crashes, Fly.io will restart the container
- Logs are aggregated and viewable via `fly logs`

---

## 🔄 Updating Deployment

To deploy a new version:

1. Create a new tag:
   ```bash
   git tag -a new-tag-name -m "Description"
   git push origin new-tag-name
   ```

2. Checkout the new tag:
   ```bash
   git checkout new-tag-name
   ```

3. Deploy:
   ```bash
   fly deploy
   ```

**Never modify code directly on Fly.io - always use tags.**

