# 🔔 Telegram Trade Alerts — Setup Guide

## ✅ Implementation Complete

Telegram alerts have been fully integrated into the Mike Agent trading system. You'll receive real-time notifications for:

- ✅ **Entry alerts** — When trades are opened (CALL/PUT)
- ✅ **Exit alerts** — When positions are closed (TP/SL)
- ✅ **Block alerts** — When safeguards prevent trades
- ✅ **Error alerts** — Critical errors in the trading loop

---

## 🚀 Quick Setup (5 minutes)

### Step 1: Create Telegram Bot

1. Open Telegram
2. Search for **@BotFather**
3. Send: `/start`
4. Send: `/newbot`
5. Follow prompts:
   - Bot name: `MikeAgent Alerts`
   - Username: `mike_agent_alerts_bot` (or your choice)
6. **Save the BOT TOKEN** (looks like: `123456789:AAHxxxxxx`)

### Step 2: Get Your Chat ID

**Option A (Easiest):**
1. Search Telegram for **@userinfobot**
2. Start the bot
3. It will reply with your Chat ID (e.g., `987654321`)

**Option B (Group Chat):**
1. Add your bot to a group
2. Send a message in the group
3. Visit: `https://api.telegram.org/bot<BOT_TOKEN>/getUpdates`
4. Find `"chat":{"id":...}` in the response

### Step 3: Set Fly.io Secrets

```bash
fly secrets set \
  TELEGRAM_BOT_TOKEN=123456789:AAHxxxxxx \
  TELEGRAM_CHAT_ID=987654321
```

### Step 4: Redeploy

```bash
fly deploy
```

That's it! Alerts will start working automatically.

---

## 📱 What You'll Receive

### Entry Alert Example
```
🟢 MIKE AGENT ALERT

ENTERED SPY251210C00680000
Type: CALL
Strike: $680.00
Expiry: 0DTE
Price: $1.42
Size: 10 contracts
Confidence: 81.00%
Source: RL (high confidence)

⏰ 2025-12-16 15:32:10 UTC
```

### Exit Alert Example
```
🔴 MIKE AGENT ALERT

EXITED SPY251210C00680000
Reason: Take Profit 1 (+40%)
Entry: $1.42
Exit: $1.99
PnL: +40.14%
Size: 10 contracts

⏰ 2025-12-16 15:45:22 UTC
```

### Stop Loss Alert Example
```
🔴 MIKE AGENT ALERT

EXITED SPY251210C00680000
Reason: Stop Loss (-15.0%)
Entry: $1.42
Exit: $1.21
PnL: -14.79%
Size: 10 contracts

⏰ 2025-12-16 15:38:15 UTC
```

### Block Alert Example
```
⛔ MIKE AGENT ALERT

TRADE BLOCKED
Symbol: SPY
Reason: ⛔ BLOCKED: Max concurrent positions (3) reached | Current: 3/3

⏰ 2025-12-16 15:30:00 UTC
```

### Error Alert Example
```
🚨 MIKE AGENT ALERT

CRITICAL ERROR
Connection timeout: Alpaca API
Context: Main trading loop

⏰ 2025-12-16 15:50:00 UTC
```

---

## 🔒 Safety Features

- ✅ **Never blocks trading** — All alerts are fire-and-forget
- ✅ **Network failures ignored** — Trading continues even if Telegram is down
- ✅ **No secrets in code** — Uses environment variables only
- ✅ **Graceful degradation** — Works without Telegram configured
- ✅ **Rate limiting** — Prevents alert spam:
  - Entry alerts: 5 min per symbol
  - Exit alerts: 1 min per symbol
  - Block alerts: 10 min per symbol
  - Error alerts: 5 min per context
  - Daily summary: 1 hour

---

## 🧪 Testing

### Quick Test (Recommended)

After setting secrets, test immediately:

**Option 1: Local Test Script**
```bash
python test_telegram.py
```

**Option 2: Test in Fly.io**
```bash
fly ssh console
python test_telegram.py
```

**Option 3: Python One-Liner**
```python
from utils.telegram_alerts import test_telegram_alert
test_telegram_alert()
```

Expected result:
- ✅ You receive a Telegram message within 1-3 seconds
- ✅ Message says "Telegram alert test successful 🚀"

If you **don't** receive the message:
1. Double-check `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
2. Verify secrets are set: `fly secrets list`
3. Make sure bot is started (send `/start` to your bot)

### Production Testing

After test passes, let the agent run for 1 full market session. You should receive:
- 🟢 Entry alerts when trades open
- 🔴 Exit alerts when positions close
- ⛔ Block alerts if safeguards prevent trades

---

## 📊 Alert Types

| Type | Emoji | When Sent |
|------|-------|-----------|
| Entry | 🟢 | Trade opened (CALL/PUT) |
| Exit | 🔴 | Position closed (TP/SL) |
| Block | ⛔ | Safeguard prevented trade |
| Error | 🚨 | Critical error occurred |
| Info | ℹ️ | General information |
| PnL | 📊 | Daily summary (optional) |

---

## 🎯 Alert Details Included

### Entry Alerts:
- Symbol (option contract)
- Type (CALL/PUT)
- Strike price
- Expiry (0DTE)
- Fill price
- Contract size
- Confidence level
- Action source (RL/Ensemble)

### Exit Alerts:
- Symbol
- Exit reason (TP1/TP2/TP3/SL)
- Entry price
- Exit price
- PnL percentage
- PnL dollar amount
- Contract size

### Block Alerts:
- Symbol
- Block reason (safeguard details)

### Error Alerts:
- Error message
- Context (where it occurred)

---

## 🔧 Troubleshooting

### Alerts not working?

1. **Check secrets are set:**
   ```bash
   fly secrets list
   ```
   Should show `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`

2. **Check logs:**
   ```bash
   fly logs | grep -i telegram
   ```

3. **Verify bot token:**
   - Test with: `curl https://api.telegram.org/bot<TOKEN>/getMe`
   - Should return bot info

4. **Verify chat ID:**
   - Send a test message to your bot
   - Check: `https://api.telegram.org/bot<TOKEN>/getUpdates`

### Common Issues:

- **"Telegram alerts available but not configured"**
  → Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` secrets

- **"No alerts received"**
  → Check bot token and chat ID are correct
  → Verify bot is started (send `/start` to bot)

- **"Alerts delayed"**
  → Normal - alerts are fire-and-forget (non-blocking)
  → Network latency may cause 1-2 second delay

---

## 📝 Code Location

- **Alert utility:** `utils/telegram_alerts.py`
- **Integration:** `mike_agent_live_safe.py`
  - Entry alerts: Lines ~3142, ~3450
  - Exit alerts: Lines ~1410, ~1562, ~1626, ~1640, ~2015
  - Block alerts: Lines ~3155, ~3401
  - Error alerts: Line ~3567

---

## ✅ Status

- ✅ Entry alerts (CALL) — Working
- ✅ Entry alerts (PUT) — Working
- ✅ Exit alerts (TP1/TP2/TP3) — Working
- ✅ Exit alerts (Stop Loss) — Working
- ✅ Block alerts — Working
- ✅ Error alerts — Working
- ✅ Daily summary — Available (not auto-enabled)

---

## 🎉 You're All Set!

Once configured, you'll receive real-time alerts for all trading activity. No manual monitoring needed — just check your Telegram when you want updates!

**Next:** Deploy and start receiving alerts automatically at market open.

