#!/usr/bin/env python3
"""
Quick test script to verify Telegram alerts are working.
Run this after setting TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.

Usage:
    python test_telegram.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.telegram_alerts import test_telegram_alert, is_configured

if __name__ == "__main__":
    print("=" * 60)
    print("🔔 TELEGRAM ALERTS TEST")
    print("=" * 60)
    print()
    
    if not is_configured():
        print("❌ Telegram not configured!")
        print()
        print("Please set environment variables:")
        print("  export TELEGRAM_BOT_TOKEN=your_bot_token")
        print("  export TELEGRAM_CHAT_ID=your_chat_id")
        print()
        print("Or for Fly.io:")
        print("  fly secrets set TELEGRAM_BOT_TOKEN=xxx TELEGRAM_CHAT_ID=xxx")
        sys.exit(1)
    
    print("✅ Telegram configured")
    print("📤 Sending test alert...")
    print()
    
    success = test_telegram_alert()
    
    print()
    if success:
        print("=" * 60)
        print("✅ TEST SUCCESSFUL")
        print("=" * 60)
        print()
        print("Check your Telegram - you should have received a test message!")
        print("Your alerts are ready to use. 🚀")
        sys.exit(0)
    else:
        print("=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print()
        print("Please check:")
        print("  1. BOT_TOKEN is correct")
        print("  2. CHAT_ID is correct")
        print("  3. Bot is started (send /start to your bot)")
        print("  4. Network connectivity")
        sys.exit(1)





