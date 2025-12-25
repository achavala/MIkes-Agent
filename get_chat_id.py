#!/usr/bin/env python3
"""
Get your Telegram Chat ID by sending a message to your bot first.

Steps:
1. Send ANY message to your bot in Telegram (e.g., "/start" or "hello")
2. Run this script
3. It will show your Chat ID
"""
import os
import requests
import sys

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not BOT_TOKEN:
    print("❌ TELEGRAM_BOT_TOKEN not set")
    print("   Set it with: export TELEGRAM_BOT_TOKEN=your_token")
    sys.exit(1)

print("=" * 60)
print("🔍 GET TELEGRAM CHAT ID")
print("=" * 60)
print()
print("📋 INSTRUCTIONS:")
print("1. Open Telegram and send ANY message to your bot")
print("   (e.g., '/start' or 'hello')")
print("2. Then run this script")
print()
print("Waiting for you to send a message...")
print("(Press Enter after you've sent a message to the bot)")
input()

print()
print("Fetching updates...")

try:
    response = requests.get(
        f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
        timeout=10
    )
    
    if response.status_code == 200:
        data = response.json()
        
        if data.get('ok') and data.get('result'):
            updates = data['result']
            
            if not updates:
                print("❌ No messages found!")
                print()
                print("Make sure you:")
                print("  1. Sent a message to your bot")
                print("  2. Used the correct bot token")
                sys.exit(1)
            
            # Get the most recent message
            latest = updates[-1]
            chat = latest['message']['chat']
            
            chat_id = chat['id']
            chat_type = chat['type']
            chat_title = chat.get('title', 'N/A')
            chat_username = chat.get('username', 'N/A')
            chat_first_name = chat.get('first_name', 'N/A')
            
            print()
            print("=" * 60)
            print("✅ CHAT ID FOUND")
            print("=" * 60)
            print()
            print(f"Chat ID: {chat_id}")
            print(f"Type: {chat_type}")
            
            if chat_type == 'private':
                print(f"User: {chat_first_name} (@{chat_username})" if chat_username != 'N/A' else f"User: {chat_first_name}")
            elif chat_type in ['group', 'supergroup']:
                print(f"Group: {chat_title}")
            
            print()
            print("=" * 60)
            print("📋 SET THIS AS YOUR CHAT ID:")
            print("=" * 60)
            print()
            print(f"fly secrets set TELEGRAM_CHAT_ID={chat_id}")
            print()
            print("Or for local testing:")
            print(f"export TELEGRAM_CHAT_ID={chat_id}")
            print()
            
        else:
            print(f"❌ API Error: {data}")
            sys.exit(1)
    else:
        print(f"❌ HTTP {response.status_code}: {response.text}")
        sys.exit(1)
        
except requests.exceptions.RequestException as e:
    print(f"❌ Network Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)





