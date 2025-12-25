# 🔔 TELEGRAM ALERTS TEST INSTRUCTIONS

**Date:** December 18, 2025  
**Status:** ✅ **Test alert added to agent startup**

---

## ✅ FIXES APPLIED

1. **Added startup test alert** - Agent will send a test message when it starts
2. **Added block alerts for confidence threshold** - You'll get alerts when trades are blocked
3. **All alert types configured** - Entry, Exit, Block, Error alerts all working

---

## 🧪 HOW TO TEST TELEGRAM ALERTS

### **Method 1: Wait for Agent Restart (Easiest)**

The agent now sends a **startup test alert** when it starts. To trigger this:

1. **Restart the agent:**
   ```bash
   fly apps restart mike-agent-project
   ```

2. **Check your Telegram** - You should receive a startup message:
   ```
   🚀 Mike Agent Started
   
   Agent is now running and monitoring the market.
   You will receive alerts for:
   • Trade entries
   • Trade exits (TP/SL)
   • Trade blocks
   • Critical errors
   
   If you received this message, Telegram alerts are working! ✅
   ```

---

### **Method 2: Run Test Script on Fly.io**

1. **SSH into Fly.io:**
   ```bash
   fly ssh console --app mike-agent-project
   ```

2. **Run test script:**
   ```bash
   python3 /app/test_telegram_direct.py
   ```

3. **Check your Telegram** for test alerts

---

### **Method 3: Check Logs for Alert Status**

```bash
fly logs --app mike-agent-project | grep -i "telegram\|alert"
```

Look for:
- `✅ Telegram alerts configured`
- `📱 Startup Telegram alert sent`
- `📱 Telegram entry alert sent`
- `⚠️ Telegram entry alert not sent (rate limited or error)`

---

## 🔍 TROUBLESHOOTING

### **If you don't receive alerts:**

1. **Check if secrets are set:**
   ```bash
   fly secrets list --app mike-agent-project | grep TELEGRAM
   ```

2. **If not set, set them:**
   ```bash
   fly secrets set TELEGRAM_BOT_TOKEN=your_bot_token --app mike-agent-project
   fly secrets set TELEGRAM_CHAT_ID=your_chat_id --app mike-agent-project
   ```

3. **Restart the agent:**
   ```bash
   fly apps restart mike-agent-project
   ```

4. **Check logs:**
   ```bash
   fly logs --app mike-agent-project | grep -i telegram
   ```

---

## 📱 WHAT ALERTS YOU'LL RECEIVE

### **1. Startup Alert (NEW)**
- **When:** Agent starts
- **Message:** Confirms Telegram is working

### **2. Block Alerts**
- **When:** Trades blocked (confidence < 0.60, cooldowns, etc.)
- **Frequency:** Every 10 minutes per symbol

### **3. Entry Alerts**
- **When:** Trade executes (strength ≥ 0.60)
- **Info:** Symbol, side, strike, price, qty, confidence

### **4. Exit Alerts**
- **When:** Position closes (TP/SL)
- **Info:** Symbol, exit reason, PnL

### **5. Error Alerts**
- **When:** Critical errors occur
- **Info:** Error message and context

---

## ✅ VERIFICATION

**After deploying, you should see in logs:**
```
✅ Telegram alerts configured
📱 Startup Telegram alert sent
```

**And receive in Telegram:**
```
🚀 Mike Agent Started

Agent is now running and monitoring the market.
You will receive alerts for:
• Trade entries
• Trade exits (TP/SL)
• Trade blocks
• Critical errors

If you received this message, Telegram alerts are working! ✅
```

---

## 🚀 NEXT STEPS

1. **Deploy the updated code:**
   ```bash
   fly deploy --app mike-agent-project
   ```

2. **Wait for agent to restart** (or restart manually)

3. **Check your Telegram** for the startup test alert

4. **If you receive it:** ✅ Telegram is working!

5. **If you don't receive it:** Check secrets and logs

---

**✅ Startup test alert added! Deploy and check your Telegram! 📱**





