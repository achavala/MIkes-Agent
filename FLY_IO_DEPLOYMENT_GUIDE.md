# 🚀 FLY.IO DEPLOYMENT GUIDE - AUTO-START ON MARKET OPEN

**Date:** December 22, 2025  
**Status:** ✅ **READY** - Agent automatically starts when market opens

---

## ✅ DEPLOYMENT STATUS

### **Automatic Market Open Detection:**
- ✅ Agent checks Alpaca clock every iteration
- ✅ Automatically waits when market is closed
- ✅ Automatically resumes when market opens (9:30 AM EST)
- ✅ No manual intervention needed

### **Fly.io Configuration:**
- ✅ Machine stays running 24/7 (`auto_stop_machines = false`)
- ✅ Auto-starts if machine stops (`auto_start_machines = true`)
- ✅ Minimum 1 machine always running
- ✅ Agent starts automatically on container startup

---

## 🚀 DEPLOYMENT STEPS

### **1. Set Fly.io Secrets**

**Option A: Use Setup Script (Recommended)**
```bash
./setup_fly_secrets.sh
```

This script:
- Reads secrets from `.env` file
- Sets them in Fly.io automatically
- Verifies all required secrets are set

**Option B: Manual Setup**
```bash
# Set Alpaca credentials (REQUIRED)
fly secrets set ALPACA_KEY=your_alpaca_key ALPACA_SECRET=your_alpaca_secret

# Set Massive API key (OPTIONAL but recommended)
fly secrets set MASSIVE_API_KEY=your_massive_api_key

# Or use POLYGON_API_KEY
fly secrets set POLYGON_API_KEY=your_polygon_key
```

### **2. Verify Secrets**
```bash
fly secrets list
```

You should see:
- `ALPACA_KEY` ✅
- `ALPACA_SECRET` ✅
- `MASSIVE_API_KEY` ✅ (optional)

### **3. Deploy to Fly.io**
```bash
fly deploy
```

### **4. Verify Deployment**
```bash
# Check app status
fly status

# View logs
fly logs

# SSH into container (optional)
fly ssh console
```

---

## ⏰ HOW MARKET OPEN DETECTION WORKS

### **Automatic Behavior:**

1. **Container Starts:**
   - Agent starts immediately when container starts
   - Dashboard starts on port 8080

2. **Market Closed:**
   - Agent checks Alpaca clock: `clock.is_open`
   - If `clock.is_open = False`:
     - Logs: "⏸️ Market is CLOSED"
     - Shows: "Next open: [time]"
     - Sleeps for 60 seconds
     - Repeats check

3. **Market Opens:**
   - Agent checks Alpaca clock: `clock.is_open`
   - If `clock.is_open = True`:
     - Logs: "✅ Market is OPEN"
     - Immediately starts trading loop
     - Begins processing trades

4. **Market Closes:**
   - Agent detects market close
   - Logs: "⏸️ Market is CLOSED"
   - Sleeps until next market open

### **No Manual Intervention Needed:**
- ✅ Agent automatically detects market status
- ✅ No cron jobs needed
- ✅ No scheduled tasks needed
- ✅ Works 24/7, automatically

---

## 📊 MONITORING

### **Check Agent Status:**
```bash
# View real-time logs
fly logs

# Check if agent is running
fly ssh console -C "ps aux | grep mike_agent_live_safe"
```

### **Expected Log Messages:**

**When Market is Closed:**
```
⏸️  Market is CLOSED (Alpaca clock) | Next open: 2025-12-23 14:30:00+00:00 | Next close: 2025-12-23 21:00:00+00:00
```

**When Market Opens:**
```
⏰ ALPACA_CLOCK_EST = 2025-12-23 09:30:00 EST | Market Open: True | Today: 2025-12-23
✅ Market is OPEN - Starting trading loop
```

---

## 🔧 TROUBLESHOOTING

### **Agent Not Starting:**
1. Check secrets are set:
   ```bash
   fly secrets list
   ```

2. Check logs:
   ```bash
   fly logs
   ```

3. Verify Alpaca credentials:
   ```bash
   fly ssh console -C "python3 -c 'import os; print(\"ALPACA_KEY:\", bool(os.getenv(\"ALPACA_KEY\")))'"
   ```

### **Market Not Detecting:**
1. Check Alpaca clock:
   - Agent uses Alpaca clock as authoritative source
   - If Alpaca API is down, agent will log error

2. Check timezone:
   - Container timezone is set to `America/New_York` (EST/EDT)
   - All timestamps use EST

### **Machine Stopped:**
1. Check machine status:
   ```bash
   fly status
   ```

2. Restart machine:
   ```bash
   fly apps restart mike-agent-project
   ```

---

## 📋 REQUIRED SECRETS

### **Required:**
- `ALPACA_KEY` - Alpaca API key
- `ALPACA_SECRET` - Alpaca API secret

### **Optional (Recommended):**
- `MASSIVE_API_KEY` - Massive/Polygon API key (for better data)
- `POLYGON_API_KEY` - Alternative name for MASSIVE_API_KEY

---

## ✅ VERIFICATION CHECKLIST

Before deployment, verify:

- [ ] `.env` file has all required keys
- [ ] `setup_fly_secrets.sh` script is executable
- [ ] Secrets are set in Fly.io (`fly secrets list`)
- [ ] `fly.toml` has `auto_start_machines = true`
- [ ] `fly.toml` has `auto_stop_machines = false`
- [ ] `Dockerfile` includes all necessary files
- [ ] `start_cloud.sh` is executable

---

## 🎯 SUMMARY

**The agent is fully automated:**

1. ✅ **Starts automatically** when container starts
2. ✅ **Detects market status** using Alpaca clock
3. ✅ **Waits for market open** automatically
4. ✅ **Resumes trading** when market opens
5. ✅ **Sleeps when market closes** automatically
6. ✅ **No manual intervention** needed

**Just deploy and it works!**

---

**Deploy with:**
```bash
./setup_fly_secrets.sh  # Set secrets
fly deploy              # Deploy to Fly.io
fly logs                # Monitor logs
```

**The agent will automatically start trading when the market opens!**


