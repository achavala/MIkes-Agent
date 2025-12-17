#!/usr/bin/env python3
"""
Direct Telegram API test - shows detailed error messages
"""
import os
import requests

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

print("=" * 60)
print("🔍 TELEGRAM API DIRECT TEST")
print("=" * 60)
print()

if not BOT_TOKEN:
    print("❌ TELEGRAM_BOT_TOKEN not set")
    exit(1)

if not CHAT_ID:
    print("❌ TELEGRAM_CHAT_ID not set")
    exit(1)

print(f"✅ Bot Token: {BOT_TOKEN[:10]}...{BOT_TOKEN[-5:]}")
print(f"✅ Chat ID: {CHAT_ID}")
print()

# Test 1: Verify bot token
print("Test 1: Verifying bot token...")
try:
    response = requests.get(
        f"https://api.telegram.org/bot{BOT_TOKEN}/getMe",
        timeout=10
    )
    if response.status_code == 200:
        bot_info = response.json()
        if bot_info.get('ok'):
            print(f"✅ Bot verified: @{bot_info['result']['username']}")
        else:
            print(f"❌ Bot API error: {bot_info}")
            exit(1)
    else:
        print(f"❌ HTTP {response.status_code}: {response.text}")
        exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

print()

# Test 2: Send message
print("Test 2: Sending test message...")
try:
    response = requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json={
            "chat_id": CHAT_ID,
            "text": "🧪 Direct API test - if you see this, it works!",
        },
        timeout=10
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        if result.get('ok'):
            print("✅ Message sent successfully!")
            print(f"   Message ID: {result['result']['message_id']}")
            print()
            print("Check your Telegram - you should have received the message!")
        else:
            print(f"❌ API returned error: {result}")
            exit(1)
    else:
        print(f"❌ HTTP {response.status_code}")
        print(f"Response: {response.text}")
        exit(1)
        
except requests.exceptions.HTTPError as e:
    print(f"❌ HTTP Error: {e}")
    if hasattr(e, 'response') and e.response:
        print(f"   Status: {e.response.status_code}")
        print(f"   Response: {e.response.text}")
    exit(1)
except requests.exceptions.RequestException as e:
    print(f"❌ Network Error: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()
print("=" * 60)
print("✅ ALL TESTS PASSED")
print("=" * 60)

