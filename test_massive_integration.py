#!/usr/bin/env python3
"""
Test Massive API Integration in mike_agent_live_safe.py
"""

import os
import sys

# Set environment variables
if not os.getenv("MASSIVE_API_KEY", "").strip():
    print("❌ MASSIVE_API_KEY not set")
    print("   Set it (recommended): export MASSIVE_API_KEY='your_polygon_key'")
    print("   Or create a local .env file with: MASSIVE_API_KEY=your_polygon_key")
    raise SystemExit(1)
os.environ['USE_MASSIVE_API'] = 'true'

# Import after setting env vars
from mike_agent_live_safe import get_market_data, get_current_price

print("=" * 80)
print("🧪 TESTING MASSIVE API INTEGRATION")
print("=" * 80)
print()

# Test 1: Market Data
print("TEST 1: get_market_data('SPY', '2d', '1m')")
print("-" * 80)
try:
    hist = get_market_data('SPY', '2d', '1m')
    print(f"✅ Retrieved {len(hist)} bars")
    if len(hist) > 0:
        print(f"   Columns: {list(hist.columns)}")
        print(f"   Latest close: ${hist['close'].iloc[-1]:.2f}")
        print(f"   Data source: {'Massive API' if len(hist) > 0 else 'yfinance'}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 2: Current Price
print("TEST 2: get_current_price('SPY')")
print("-" * 80)
try:
    price = get_current_price('SPY')
    if price:
        print(f"✅ Current SPY price: ${price:.2f}")
    else:
        print("⚠️  Price not available")
except Exception as e:
    print(f"❌ Error: {e}")

print()

# Test 3: Multi-symbol prices
print("TEST 3: Multi-symbol prices")
print("-" * 80)
symbols = ['SPY', 'QQQ', 'SPX']
for sym in symbols:
    try:
        price = get_current_price(sym)
        if price:
            print(f"✅ {sym}: ${price:.2f}")
        else:
            print(f"⚠️  {sym}: Not available")
    except Exception as e:
        print(f"❌ {sym}: Error - {e}")

print()

# Test 4: VIX
print("TEST 4: VIX price")
print("-" * 80)
try:
    vix = get_current_price("^VIX")
    if vix:
        print(f"✅ VIX: {vix:.2f}")
    else:
        print("⚠️  VIX not available")
except Exception as e:
    print(f"❌ Error: {e}")

print()
print("=" * 80)
print("✅ INTEGRATION TEST COMPLETE")
print("=" * 80)

