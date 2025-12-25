#!/usr/bin/env python3
"""
Test Telegram Alerts on Fly.io
This script tests all Telegram alert types to verify they're working
"""

import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_telegram_alerts():
    """Test all Telegram alert types"""
    print("=" * 80)
    print("🔔 TELEGRAM ALERTS TEST")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print()
    
    # Check if Telegram module is available
    try:
        from utils.telegram_alerts import (
            send_entry_alert,
            send_exit_alert,
            send_block_alert,
            send_error_alert,
            send_info,
            test_telegram_alert,
            is_configured
        )
        print("✅ Telegram alerts module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Telegram alerts module: {e}")
        return False
    
    # Check configuration
    print("\n" + "-" * 80)
    print("📋 CONFIGURATION CHECK")
    print("-" * 80)
    
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    print(f"TELEGRAM_BOT_TOKEN: {'✅ Set' if bot_token else '❌ Not set'}")
    if bot_token:
        print(f"   Token: {bot_token[:10]}...{bot_token[-5:] if len(bot_token) > 15 else '***'}")
    print(f"TELEGRAM_CHAT_ID: {'✅ Set' if chat_id else '❌ Not set'}")
    if chat_id:
        print(f"   Chat ID: {chat_id}")
    
    if not is_configured():
        print("\n❌ Telegram is NOT configured!")
        print("   Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables")
        print("   On Fly.io: fly secrets set TELEGRAM_BOT_TOKEN=... --app mike-agent-project")
        print("   On Fly.io: fly secrets set TELEGRAM_CHAT_ID=... --app mike-agent-project")
        return False
    
    print("\n✅ Telegram is configured")
    
    # Test basic alert
    print("\n" + "-" * 80)
    print("🧪 TEST 1: Basic Test Alert")
    print("-" * 80)
    test_result = test_telegram_alert()
    if test_result:
        print("✅ Test alert sent successfully!")
    else:
        print("❌ Test alert failed")
        return False
    
    # Test entry alert
    print("\n" + "-" * 80)
    print("🧪 TEST 2: Entry Alert")
    print("-" * 80)
    try:
        entry_result = send_entry_alert(
            symbol="SPY241202C00450000",
            side="CALL",
            strike=450.00,
            expiry="0DTE",
            fill_price=0.45,
            qty=5,
            confidence=0.60,
            action_source="RL+Ensemble"
        )
        if entry_result:
            print("✅ Entry alert sent successfully!")
        else:
            print("⚠️ Entry alert not sent (rate limited or error)")
    except Exception as e:
        print(f"❌ Entry alert error: {e}")
    
    # Test exit alert
    print("\n" + "-" * 80)
    print("🧪 TEST 3: Exit Alert")
    print("-" * 80)
    try:
        exit_result = send_exit_alert(
            symbol="SPY241202C00450000",
            exit_reason="Take Profit 1",
            entry_price=0.45,
            exit_price=0.58,
            pnl_pct=28.89,
            qty=5,
            pnl_dollar=65.00
        )
        if exit_result:
            print("✅ Exit alert sent successfully!")
        else:
            print("⚠️ Exit alert not sent (rate limited or error)")
    except Exception as e:
        print(f"❌ Exit alert error: {e}")
    
    # Test block alert
    print("\n" + "-" * 80)
    print("🧪 TEST 4: Block Alert")
    print("-" * 80)
    try:
        block_result = send_block_alert(
            symbol="SPY",
            block_reason="Confidence too low (strength=0.521 < 0.600)"
        )
        if block_result:
            print("✅ Block alert sent successfully!")
        else:
            print("⚠️ Block alert not sent (rate limited or error)")
    except Exception as e:
        print(f"❌ Block alert error: {e}")
    
    # Test info alert
    print("\n" + "-" * 80)
    print("🧪 TEST 5: Info Alert")
    print("-" * 80)
    try:
        info_result = send_info("This is a test info alert from Fly.io deployment")
        if info_result:
            print("✅ Info alert sent successfully!")
        else:
            print("⚠️ Info alert not sent (rate limited or error)")
    except Exception as e:
        print(f"❌ Info alert error: {e}")
    
    print("\n" + "=" * 80)
    print("✅ TELEGRAM ALERTS TEST COMPLETE")
    print("=" * 80)
    print("\nCheck your Telegram to see if you received the test alerts!")
    print("If you received them, Telegram alerts are working correctly! 🎉")
    
    return True

if __name__ == "__main__":
    success = test_telegram_alerts()
    sys.exit(0 if success else 1)





