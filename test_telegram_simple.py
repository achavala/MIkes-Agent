#!/usr/bin/env python3
"""Simple Telegram test - can be run directly"""
import os
import sys
sys.path.insert(0, '/app')

try:
    from utils.telegram_alerts import test_telegram_alert, is_configured
    
    print("=" * 60)
    print("TELEGRAM TEST")
    print("=" * 60)
    
    if not is_configured():
        print("❌ NOT CONFIGURED")
        print(f"BOT_TOKEN: {'Set' if os.getenv('TELEGRAM_BOT_TOKEN') else 'NOT SET'}")
        print(f"CHAT_ID: {'Set' if os.getenv('TELEGRAM_CHAT_ID') else 'NOT SET'}")
        sys.exit(1)
    
    print("✅ Configured")
    result = test_telegram_alert()
    print(f"Result: {'✅ SUCCESS' if result else '❌ FAILED'}")
    sys.exit(0 if result else 1)
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)





