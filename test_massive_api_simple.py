#!/usr/bin/env python3
"""
Simple Massive API Test - Direct Endpoint Testing
Tests with the known endpoint format from Massive docs
"""

import os
import requests
from datetime import datetime

MASSIVE_API_KEY = os.getenv('MASSIVE_API_KEY', '')

if not MASSIVE_API_KEY:
    print("❌ MASSIVE_API_KEY not set")
    print("   Run: export MASSIVE_API_KEY='your_key'")
    exit(1)

print("=" * 80)
print("🧪 SIMPLE MASSIVE API TEST")
print("=" * 80)
print()
print(f"API Key: {MASSIVE_API_KEY[:10]}...{MASSIVE_API_KEY[-5:]}")
print()

# Test 1: Known working endpoint from docs
print("TEST 1: Dividends endpoint (known working from docs)")
print("-" * 80)
url = "https://api.massive.com/v1/dividends"
params = {"apiKey": MASSIVE_API_KEY}
try:
    response = requests.get(url, params=params, timeout=10)
    print(f"Status: {response.status_code}")
    print(f"URL: {response.url}")
    if response.status_code == 200:
        print("✅ SUCCESS! API connection works")
        data = response.json()
        print(f"Response type: {type(data)}")
        if isinstance(data, (list, dict)) and len(str(data)) < 500:
            print(f"Sample data: {data}")
    elif response.status_code == 401:
        print("❌ Authentication failed - check API key")
    elif response.status_code == 403:
        print("⚠️  Forbidden - subscription may not include dividends")
    else:
        print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"❌ Error: {e}")

print()
print()

# Test 2: Try stocks endpoint
print("TEST 2: Stocks bars endpoint")
print("-" * 80)
url = "https://api.massive.com/v1/bars/SPY"
params = {
    "apiKey": MASSIVE_API_KEY,
    "start": "2025-12-01",
    "end": "2025-12-08",
    "timespan": "minute",
    "multiplier": 1
}
try:
    response = requests.get(url, params=params, timeout=10)
    print(f"Status: {response.status_code}")
    print(f"URL: {response.url}")
    if response.status_code == 200:
        print("✅ SUCCESS! Stocks data works")
        data = response.json()
        print(f"Response type: {type(data)}")
        if isinstance(data, (list, dict)):
            if isinstance(data, list) and len(data) > 0:
                print(f"Sample record: {data[0]}")
            elif isinstance(data, dict):
                print(f"Keys: {list(data.keys())[:10]}")
    elif response.status_code == 401:
        print("❌ Authentication failed")
    elif response.status_code == 403:
        print("⚠️  Forbidden - check 'Stocks Developer' subscription")
    elif response.status_code == 404:
        print("⚠️  Endpoint not found - may need different path")
    else:
        print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"❌ Error: {e}")

print()
print()

# Test 3: Try options endpoint
print("TEST 3: Options chain endpoint")
print("-" * 80)
today = datetime.now().strftime("%Y-%m-%d")
url = "https://api.massive.com/v1/options/chain"
params = {
    "apiKey": MASSIVE_API_KEY,
    "symbol": "SPY",
    "expiry": today
}
try:
    response = requests.get(url, params=params, timeout=10)
    print(f"Status: {response.status_code}")
    print(f"URL: {response.url}")
    if response.status_code == 200:
        print("✅ SUCCESS! Options chain works")
        data = response.json()
        print(f"Response type: {type(data)}")
        if isinstance(data, (list, dict)):
            if isinstance(data, list) and len(data) > 0:
                print(f"Sample option: {data[0]}")
                print(f"Total options: {len(data)}")
            elif isinstance(data, dict):
                print(f"Keys: {list(data.keys())[:10]}")
    elif response.status_code == 401:
        print("❌ Authentication failed")
    elif response.status_code == 403:
        print("⚠️  Forbidden - check 'Options Starter' subscription")
    elif response.status_code == 404:
        print("⚠️  Endpoint not found - may need different path")
    else:
        print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"❌ Error: {e}")

print()
print("=" * 80)
print("📝 NEXT STEPS")
print("=" * 80)
print()
print("1. Check Massive API documentation:")
print("   https://massive.com/docs")
print()
print("2. Check your dashboard for:")
print("   - API base URL")
print("   - Available endpoints")
print("   - Subscription details")
print()
print("3. If endpoints work, we can integrate!")

