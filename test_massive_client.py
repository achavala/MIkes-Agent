#!/usr/bin/env python3
"""
Test Massive API Client
"""

import os
from massive_api_client import MassiveAPIClient
from datetime import datetime, timedelta

API_KEY = os.getenv('MASSIVE_API_KEY', '')
if not API_KEY:
    print("❌ MASSIVE_API_KEY not set")
    print("   Set it (recommended): export MASSIVE_API_KEY='your_polygon_key'")
    print("   Or create a local .env file with: MASSIVE_API_KEY=your_polygon_key")
    raise SystemExit(1)

print("=" * 80)
print("🧪 TESTING MASSIVE API CLIENT")
print("=" * 80)
print()

client = MassiveAPIClient(API_KEY)

# Test 1: Connection
print("TEST 1: Connection Test")
print("-" * 80)
if client.test_connection():
    print("✅ Connection successful!")
else:
    print("❌ Connection failed")
print()

# Test 2: Historical Data
print("TEST 2: Historical Data")
print("-" * 80)
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

try:
    df = client.get_historical_data('SPY', start_date, end_date, interval='1d')
    print(f"✅ Retrieved {len(df)} bars")
    if len(df) > 0:
        print(f"   Columns: {list(df.columns)}")
        print(f"   Latest close: ${df['close'].iloc[-1]:.2f}")
except Exception as e:
    print(f"❌ Error: {e}")
print()

# Test 3: Real-time Price
print("TEST 3: Real-time Price")
print("-" * 80)
try:
    price = client.get_real_time_price('SPY')
    if price:
        print(f"✅ Current SPY price: ${price:.2f}")
    else:
        print("⚠️  Could not get price")
except Exception as e:
    print(f"❌ Error: {e}")
print()

# Test 4: Options Contracts
print("TEST 4: Options Contracts")
print("-" * 80)
try:
    # Get contracts for tomorrow (next trading day)
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    contracts = client.get_options_contracts('SPY', expiration_date=tomorrow, limit=10)
    print(f"✅ Retrieved {len(contracts)} contracts")
    if len(contracts) > 0:
        sample = contracts[0]
        print(f"   Sample contract keys: {list(sample.keys())[:10]}")
except Exception as e:
    print(f"❌ Error: {e}")
print()

print("=" * 80)
print("✅ TESTING COMPLETE")
print("=" * 80)

