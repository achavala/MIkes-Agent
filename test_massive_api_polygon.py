#!/usr/bin/env python3
"""
Test Massive API (Polygon.io) Access
Tests with correct Polygon.io endpoints
"""

import os
import requests
from datetime import datetime, timedelta
from datetime import timezone
from zoneinfo import ZoneInfo

MASSIVE_API_KEY = os.getenv('MASSIVE_API_KEY', '')
if not MASSIVE_API_KEY:
    # Fall back to the same key-resolution logic used by the main client
    try:
        from massive_api_client import MassiveAPIClient
        MASSIVE_API_KEY = MassiveAPIClient().api_key
    except Exception:
        print("❌ MASSIVE_API_KEY not set (and could not auto-resolve)")
        print("   Set it: export MASSIVE_API_KEY='your_key'")
        exit(1)

print("=" * 80)
print("🧪 MASSIVE API TEST (Polygon.io Endpoints)")
print("=" * 80)
print()
print("Note: Massive = Polygon.io (rebranded)")
print()

# Test 0: Indices Advanced (SPX 1-minute aggregates)
print("=" * 80)
print("TEST 0: Indices Advanced - SPX 1-minute Aggregates (I:SPX)")
print("=" * 80)
print()

ny = ZoneInfo("America/New_York")
now_ny = datetime.now(tz=ny)
end_date = now_ny.strftime("%Y-%m-%d")
start_date = (now_ny - timedelta(days=2)).strftime("%Y-%m-%d")

url = "https://api.polygon.io/v2/aggs/ticker/I:SPX/range/1/minute/{}/{}".format(start_date, end_date)
params = {"apiKey": MASSIVE_API_KEY, "limit": 50000}

try:
    response = requests.get(url, params=params, timeout=20)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        api_status = data.get("status", "unknown")
        results = data.get("results") or []
        print(f"✅ Response status field: {api_status}")
        print(f"   resultsCount: {data.get('resultsCount', None)} | len(results): {len(results)}")
        if results:
            last = results[-1]
            t_ms = last.get("t")
            last_utc = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)
            last_ny = last_utc.astimezone(ny)
            age_sec = (now_ny.timestamp() * 1000 - t_ms) / 1000.0
            print(f"   Last bar (NY): {last_ny.isoformat()} | close={last.get('c')} | age_sec={age_sec:.1f}")
            if now_ny.hour >= 9 and (now_ny.hour < 16 or (now_ny.hour == 16 and now_ny.minute == 0)):
                if age_sec < 90:
                    print("   ✅ GO: REAL-TIME CHECK PASSED (< 90s)")
                elif age_sec < 900:
                    print("   ⚠️  NO-GO (for realtime): age 90s–15m. Re-run after 2–3 minutes to confirm.")
                else:
                    print("   ❌ NO-GO: Likely delayed (age >= 15m) or market closed")
        indices_works = True
    elif response.status_code == 403:
        print("❌ Forbidden (403) - index entitlement not active for this key")
        indices_works = False
    else:
        print(f"⚠️  Unexpected status: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
        indices_works = False
except Exception as e:
    print(f"❌ Error: {e}")
    indices_works = False

print()

# Test 1: Stock Aggregates (Stocks Developer subscription)
print("=" * 80)
print("TEST 1: Stock Aggregates (Stocks Developer)")
print("=" * 80)
print()

# Get last 7 days
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

url = "https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{}/{}".format(start_date, end_date)
params = {"apiKey": MASSIVE_API_KEY}

try:
    response = requests.get(url, params=params, timeout=10)
    print(f"URL: {response.url[:100]}...")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        results_count = data.get('resultsCount', 0)
        status = data.get('status', 'unknown')
        
        print(f"✅ SUCCESS!")
        print(f"   Status: {status}")
        print(f"   Results: {results_count} bars")
        
        if results_count > 0 and 'results' in data:
            sample = data['results'][0]
            print(f"   Sample data keys: {list(sample.keys())}")
            print(f"   Latest close: ${sample.get('c', 0):.2f}")
        
        print()
        print("   ✅ Stocks Developer subscription works!")
        stocks_works = True
    elif response.status_code == 401:
        print("❌ Authentication failed - check API key")
        stocks_works = False
    elif response.status_code == 403:
        print("⚠️  Forbidden - Stocks Developer subscription may not be active")
        stocks_works = False
    else:
        print(f"⚠️  Unexpected status: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
        stocks_works = False
except Exception as e:
    print(f"❌ Error: {e}")
    stocks_works = False

print()

# Test 1B: SPY vs SPX last-bar timestamp alignment (market-hours validation)
print("=" * 80)
print("TEST 1B: SPY vs I:SPX last-bar timestamp alignment (minute)")
print("=" * 80)
print()

def _last_bar_ms(ticker: str) -> int | None:
    url2 = "https://api.polygon.io/v2/aggs/ticker/{}/range/1/minute/{}/{}".format(ticker, start_date, end_date)
    r = requests.get(url2, params={"apiKey": MASSIVE_API_KEY, "limit": 50000}, timeout=20)
    if r.status_code != 200:
        return None
    d = r.json()
    res = d.get("results") or []
    if not res:
        return None
    return res[-1].get("t")

spy_t = _last_bar_ms("SPY")
spx_t = _last_bar_ms("I:SPX")
now_ny = datetime.now(tz=ZoneInfo("America/New_York"))

if spy_t and spx_t:
    spy_age = (now_ny.timestamp() * 1000 - spy_t) / 1000.0
    spx_age = (now_ny.timestamp() * 1000 - spx_t) / 1000.0
    skew = abs(spy_t - spx_t) / 1000.0
    print(f"SPY age_sec: {spy_age:.1f}")
    print(f"SPX age_sec: {spx_age:.1f}")
    print(f"Timestamp skew_sec: {skew:.1f}")
    if now_ny.hour >= 9 and (now_ny.hour < 16 or (now_ny.hour == 16 and now_ny.minute == 0)):
        if spy_age < 90 and spx_age < 90 and skew < 90:
            print("✅ GO: SPY and SPX are time-aligned (<90s old, <90s skew)")
        else:
            print("❌ NO-GO (for realtime): timestamps not aligned / too old (check delay vs market hours)")
else:
    print("⚠️ Could not compute alignment (missing bars or non-200 response)")

print()

# Test 2: Options Snapshot (Options Starter subscription)
print("=" * 80)
print("TEST 2: Options Snapshot (Options Starter)")
print("=" * 80)
print()

url = "https://api.polygon.io/v2/snapshot/options/SPY"
params = {"apiKey": MASSIVE_API_KEY}

try:
    response = requests.get(url, params=params, timeout=10)
    print(f"URL: {response.url[:100]}...")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        status = data.get('status', 'unknown')
        
        print(f"✅ SUCCESS!")
        print(f"   Status: {status}")
        print(f"   Response keys: {list(data.keys())}")
        
        if 'results' in data:
            results = data['results']
            print(f"   Options contracts: {len(results) if isinstance(results, list) else 'dict'}")
            
            if isinstance(results, list) and len(results) > 0:
                sample = results[0]
                print(f"   Sample option keys: {list(sample.keys())}")
                
                # Check for critical fields
                has_price = any(k in sample for k in ['bid', 'ask', 'last_quote', 'last_trade'])
                has_greeks = any(k in sample for k in ['greeks', 'delta', 'gamma', 'theta', 'vega'])
                has_details = any(k in sample for k in ['details', 'contract_type', 'strike_price'])
                
                print()
                print("   📊 Data Quality Check:")
                print(f"      ✅ Price data: {has_price}")
                print(f"      ✅ Greeks: {has_greeks}")
                print(f"      ✅ Contract details: {has_details}")
                
                if has_price and has_greeks:
                    print()
                    print("   ✅ EXCELLENT: Full options data available!")
                    print("   ✅ INTEGRATION RECOMMENDED")
                elif has_price:
                    print()
                    print("   ✅ GOOD: Options prices available")
                    print("   ✅ INTEGRATION RECOMMENDED")
        elif isinstance(data, dict):
            print(f"   Response structure: {type(data)}")
        
        print()
        print("   ✅ Options Starter subscription works!")
        options_works = True
    elif response.status_code == 401:
        print("❌ Authentication failed - check API key")
        options_works = False
    elif response.status_code == 403:
        print("⚠️  Forbidden - Options Starter subscription may not be active")
        print("   Check your dashboard: https://massive.com/dashboard")
        options_works = False
    elif response.status_code == 404:
        print("⚠️  Endpoint not found - may need different endpoint")
        options_works = False
    else:
        print(f"⚠️  Unexpected status: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
        options_works = False
except Exception as e:
    print(f"❌ Error: {e}")
    options_works = False

print()

# Test 3: Options Chain (alternative endpoint)
print("=" * 80)
print("TEST 3: Options Chain (Alternative)")
print("=" * 80)
print()

today = datetime.now().strftime("%Y-%m-%d")
url = "https://api.polygon.io/v3/reference/options/contracts"
params = {
    "apiKey": MASSIVE_API_KEY,
    "underlying_ticker": "SPY",
    "expiration_date": today,
    "limit": 10
}

try:
    response = requests.get(url, params=params, timeout=10)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Options contracts endpoint works!")
        print(f"   Results: {data.get('count', 0)}")
        options_contracts_works = True
    else:
        print(f"⚠️  Status: {response.status_code}")
        options_contracts_works = False
except Exception as e:
    print(f"⚠️  Error: {e}")
    options_contracts_works = False

print()
print("=" * 80)
print("📊 SUMMARY")
print("=" * 80)
print()

results = {
    "Stocks Developer": stocks_works,
    "Options Starter (Snapshot)": options_works,
    "Options Contracts": options_contracts_works if 'options_contracts_works' in locals() else False
}

for service, works in results.items():
    status = "✅ WORKS" if works else "❌ FAILED"
    print(f"   {status}: {service}")

print()

if stocks_works or options_works:
    print("✅ SUCCESS! At least one subscription is working")
    print()
    print("Next steps:")
    print("   1. Install: pip install polygon-api-client")
    print("   2. I'll create massive_api_client.py wrapper")
    print("   3. Integrate into trading system")
else:
    print("⚠️  Could not verify subscriptions")
    print()
    print("Please check:")
    print("   1. API key is correct")
    print("   2. Subscriptions are active")
    print("   3. Dashboard: https://massive.com/dashboard")

print()
print("=" * 80)

