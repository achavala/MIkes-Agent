#!/usr/bin/env python3
"""
Direct Telegram Test - Run this on Fly.io to test alerts
Usage: fly ssh console --app mike-agent-project
Then: python3 /app/test_telegram_direct.py
"""
import os
import sys

# Add app directory to path
sys.path.insert(0, '/app')

print("=" * 80)
print("🔔 TELEGRAM ALERTS TEST")
print("=" * 80)
print()

# Check environment variables
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")

print("📋 Configuration Check:")
print(f"  TELEGRAM_BOT_TOKEN: {'✅ Set' if bot_token else '❌ Not set'}")
if bot_token:
    print(f"    Token: {bot_token[:10]}...{bot_token[-5:] if len(bot_token) > 15 else '***'}")
print(f"  TELEGRAM_CHAT_ID: {'✅ Set' if chat_id else '❌ Not set'}")
if chat_id:
    print(f"    Chat ID: {chat_id}")
print()

if not bot_token or not chat_id:
    print("❌ Telegram secrets not configured!")
    print()
    print("To set secrets on Fly.io:")
    print("  fly secrets set TELEGRAM_BOT_TOKEN=your_token --app mike-agent-project")
    print("  fly secrets set TELEGRAM_CHAT_ID=your_chat_id --app mike-agent-project")
    sys.exit(1)

# Import Telegram module
try:
    from utils.telegram_alerts import (
        test_telegram_alert,
        send_entry_alert,
        send_exit_alert,
        send_block_alert,
        send_info
    )
    print("✅ Telegram module imported")
except ImportError as e:
    print(f"❌ Failed to import Telegram module: {e}")
    sys.exit(1)

print()
print("-" * 80)
print("🧪 TEST 1: Basic Test Alert")
print("-" * 80)
result1 = test_telegram_alert()
if result1:
    print("✅ Test alert sent! Check your Telegram.")
else:
    print("❌ Test alert failed")

print()
print("-" * 80)
print("🧪 TEST 2: Entry Alert")
print("-" * 80)
result2 = send_entry_alert(
    symbol="SPY241202C00450000",
    side="CALL",
    strike=450.00,
    expiry="0DTE",
    fill_price=0.45,
    qty=5,
    confidence=0.60,
    action_source="RL+Ensemble"
)
if result2:
    print("✅ Entry alert sent!")
else:
    print("⚠️ Entry alert not sent (rate limited or error)")

print()
print("-" * 80)
print("🧪 TEST 3: Exit Alert")
print("-" * 80)
result3 = send_exit_alert(
    symbol="SPY241202C00450000",
    exit_reason="Take Profit 1",
    entry_price=0.45,
    exit_price=0.58,
    pnl_pct=28.89,
    qty=5,
    pnl_dollar=65.00
)
if result3:
    print("✅ Exit alert sent!")
else:
    print("⚠️ Exit alert not sent (rate limited or error)")

print()
print("-" * 80)
print("🧪 TEST 4: Block Alert")
print("-" * 80)
result4 = send_block_alert(
    symbol="SPY",
    block_reason="Test: Confidence too low (strength=0.521 < 0.600)"
)
if result4:
    print("✅ Block alert sent!")
else:
    print("⚠️ Block alert not sent (rate limited or error)")

print()
print("-" * 80)
print("🧪 TEST 5: Info Alert")
print("-" * 80)
result5 = send_info("Test info alert from Fly.io - Telegram alerts are working! ✅")
if result5:
    print("✅ Info alert sent!")
else:
    print("⚠️ Info alert not sent (rate limited or error)")

print()
print("=" * 80)
if result1:
    print("✅ TELEGRAM IS WORKING!")
    print("Check your Telegram - you should have received test alerts!")
else:
    print("❌ TELEGRAM TEST FAILED")
    print("Check the error messages above for details")
print("=" * 80)
