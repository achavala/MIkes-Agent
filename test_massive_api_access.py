#!/usr/bin/env python3
"""
Test Massive API Access
Verifies what your subscription includes before integration
"""

import os
import sys
import requests
from typing import Dict, Optional
from datetime import datetime

# You'll need to set this in your environment or config
MASSIVE_API_KEY = os.getenv('MASSIVE_API_KEY', '')
# Try common Massive API base URLs
MASSIVE_BASE_URLS = [
    "https://api.massive.com",
    "https://api.massive.com/v1",
    "https://massive.com/api",
    "https://massive.com/api/v1",
    "https://api.massive.com/api",
    "https://api.massive.com/api/v1",
]

def find_correct_base_url():
    """Try to find the correct base URL by testing common endpoints"""
    if not MASSIVE_API_KEY:
        return None, None
    
    # Massive API uses apiKey as query parameter, not Bearer token
    # Based on: curl "https://api.massive.com/v1/dividends?apiKey=YOUR_API_KEY"
    
    # Common test endpoints (try with apiKey param)
    test_endpoints = [
        "/health",
        "/v1/health",
        "/v1/status",
        "/v1/account",
        "/v1/dividends",  # Known working endpoint from docs
        "/v1/stocks/SPY/bars",  # Test stocks endpoint
    ]
    
    print("   Trying to find correct base URL...")
    print("   Using apiKey query parameter authentication...")
    print()
    
    for base_url in MASSIVE_BASE_URLS:
        for endpoint in test_endpoints:
            try:
                url = f"{base_url}{endpoint}"
                # Try with apiKey as query parameter
                params = {"apiKey": MASSIVE_API_KEY}
                
                # Also try common date params for endpoints that need them
                if "bars" in endpoint:
                    params["start"] = "2025-12-01"
                    params["end"] = "2025-12-08"
                
                response = requests.get(
                    url,
                    params=params,
                    timeout=5
                )
                
                # 200 = success, 401 = auth issue, 403 = forbidden but auth works
                if response.status_code == 200:
                    print(f"   ✅ Found working base URL: {base_url}")
                    print(f"   ✅ Endpoint responded: {endpoint} (200 OK)")
                    return base_url, endpoint
                elif response.status_code in [401, 403]:
                    # Auth recognized but may not have access
                    print(f"   ✅ Base URL works: {base_url} (auth recognized: {response.status_code})")
                    print(f"   ✅ Endpoint: {endpoint}")
                    return base_url, endpoint
                elif response.status_code == 404:
                    continue  # Try next
                else:
                    # Got a response
                    print(f"   ⚠️  {base_url}{endpoint}: {response.status_code}")
                    
            except requests.exceptions.RequestException:
                continue  # Try next
    
    return None, None

def test_api_connection():
    """Test basic API connection"""
    print("=" * 80)
    print("🔍 TEST 1: API Connection")
    print("=" * 80)
    print()
    
    if not MASSIVE_API_KEY:
        print("❌ MASSIVE_API_KEY not set")
        print("   Set it in your environment: export MASSIVE_API_KEY='your_key'")
        print("   Or add to config.py: MASSIVE_API_KEY = 'your_key'")
        return False, None
    
    # Try to find correct base URL
    base_url, working_endpoint = find_correct_base_url()
    
    if base_url:
        print()
        print(f"✅ Found working configuration:")
        print(f"   Base URL: {base_url}")
        print(f"   Test endpoint: {working_endpoint}")
        return True, base_url
    else:
        print()
        print("❌ Could not find working base URL")
        print()
        print("   Please check:")
        print("   1. Massive API documentation")
        print("   2. Your API key is correct")
        print("   3. Base URL format")
        print()
        print("   Common formats:")
        print("   - https://api.massive.com/v1/...")
        print("   - https://massive.com/api/v1/...")
        print("   - Check your Massive dashboard for API docs")
        return False, None

def test_stocks_data(base_url):
    """Test Stocks Developer subscription"""
    print("=" * 80)
    print("🔍 TEST 2: Stocks Developer Access")
    print("=" * 80)
    print()
    
    if not MASSIVE_API_KEY or not base_url:
        return False
    
    headers = {
        "Authorization": f"Bearer {MASSIVE_API_KEY}",
        "X-API-Key": MASSIVE_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Test historical data endpoint (adjust based on Massive API docs)
    test_symbol = "SPY"
    
    # Try common endpoint patterns
    endpoints_to_try = [
        f"{base_url}/stocks/{test_symbol}/bars",
        f"{base_url}/v1/stocks/{test_symbol}/bars",
        f"{base_url}/bars/{test_symbol}",
        f"{base_url}/v1/bars/{test_symbol}",
        f"{base_url}/market/stocks/{test_symbol}/bars",
        f"{base_url}/v1/market/stocks/{test_symbol}/bars",
    ]
    
    for endpoint in endpoints_to_try:
        try:
            response = requests.get(
                endpoint,
                params={
                    "start": "2025-12-01",
                    "end": "2025-12-08",
                    "interval": "1m"
                },
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Stocks data access: WORKING")
                print(f"   Endpoint: {endpoint}")
                print(f"   Sample data: {str(data)[:200]}...")
                return True
            elif response.status_code == 403:
                print(f"✅ Stocks endpoint found but FORBIDDEN: {endpoint}")
                print("   Your 'Stocks Developer' subscription may not include this")
                continue
            elif response.status_code == 401:
                print(f"⚠️  Auth issue: {endpoint}")
                continue
            elif response.status_code == 404:
                continue  # Try next endpoint
            else:
                print(f"⚠️  {endpoint}: {response.status_code}")
                continue
                
        except requests.exceptions.RequestException as e:
            continue  # Try next endpoint
    
    print("❌ Could not find working stocks endpoint")
    print("   Check Massive API documentation")
    return False

def test_options_chain(base_url):
    """Test Options Starter subscription - Options Chain"""
    print("=" * 80)
    print("🔍 TEST 3: Options Chain Access (CRITICAL)")
    print("=" * 80)
    print()
    
    if not MASSIVE_API_KEY or not base_url:
        return False
    
    headers = {
        "Authorization": f"Bearer {MASSIVE_API_KEY}",
        "X-API-Key": MASSIVE_API_KEY,
        "Content-Type": "application/json"
    }
    
    test_symbol = "SPY"
    # Today's date for 0DTE
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Try common endpoint patterns
    endpoints_to_try = [
        (f"{base_url}/options/chain", {"symbol": test_symbol, "expiry": today}),
        (f"{base_url}/v1/options/chain", {"symbol": test_symbol, "expiry": today}),
        (f"{base_url}/options/{test_symbol}/chain", {"expiry": today}),
        (f"{base_url}/v1/options/{test_symbol}/chain", {"expiry": today}),
        (f"{base_url}/market/options/{test_symbol}/chain", {"expiry": today}),
        (f"{base_url}/v1/market/options/{test_symbol}/chain", {"expiry": today}),
        (f"{base_url}/options/chain/{test_symbol}", {"expiry": today}),
    ]
    
    for endpoint, params in endpoints_to_try:
        try:
            response = requests.get(
                endpoint,
                params=params,
                headers=headers,
                timeout=10
            )
        
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Options chain access: WORKING")
                print(f"   Endpoint: {endpoint}")
                
                # Check if we have individual option data
                if isinstance(data, list) and len(data) > 0:
                    sample = data[0]
                    print(f"   Sample option: {sample}")
                    
                    # Check what fields are available
                    keys = sample.keys() if isinstance(sample, dict) else []
                    print(f"   Available fields: {list(keys)}")
                    
                    # Check for critical fields
                    has_price = 'bid' in keys or 'ask' in keys or 'price' in keys
                    has_greeks = any(k in keys for k in ['delta', 'gamma', 'theta', 'vega'])
                    has_volume = 'volume' in keys or 'open_interest' in keys
                    
                    print()
                    print("   📊 Data Quality Check:")
                    print(f"      ✅ Individual option prices: {has_price}")
                    print(f"      ✅ Greeks available: {has_greeks}")
                    print(f"      ✅ Volume/OI: {has_volume}")
                    
                    if has_price and has_greeks:
                        print()
                        print("   ✅ EXCELLENT: Full options chain data available!")
                        print("   ✅ INTEGRATION RECOMMENDED")
                        return True
                    elif has_price:
                        print()
                        print("   ✅ GOOD: Options prices available")
                        print("   ✅ INTEGRATION RECOMMENDED (can calculate Greeks)")
                        return True
                    else:
                        print()
                        print("   ⚠️  LIMITED: Only aggregated data")
                        print("   ⚠️  May not be worth $29/month")
                        return False
                else:
                    print(f"   ⚠️  Unexpected data format: {type(data)}")
                    return False
                    
            elif response.status_code == 403:
                print(f"✅ Options endpoint found but FORBIDDEN: {endpoint}")
                print("   Your 'Options Starter' subscription may not include this")
                continue
            elif response.status_code == 401:
                print(f"⚠️  Auth issue: {endpoint}")
                continue
            elif response.status_code == 404:
                continue  # Try next endpoint
            else:
                print(f"⚠️  {endpoint}: {response.status_code}")
                continue
                
        except requests.exceptions.RequestException as e:
            continue  # Try next endpoint
    
    print("❌ Could not find working options endpoint")
    print("   Check Massive API documentation")
    return False

def test_options_greeks(base_url):
    """Test Options Starter subscription - Greeks"""
    print("=" * 80)
    print("🔍 TEST 4: Options Greeks Access")
    print("=" * 80)
    print()
    
    if not MASSIVE_API_KEY or not base_url:
        return None
    
    # Massive API uses apiKey as query parameter
    base_params = {
        "apiKey": MASSIVE_API_KEY,
        "symbol": "SPY",
        "strike": 682,
        "expiry": datetime.now().strftime("%Y-%m-%d"),
        "type": "call"
    }
    
    # Test for specific option Greeks (usually included in chain, but try separate endpoint)
    endpoints_to_try = [
        f"{base_url}/v1/options/greeks",
        f"{base_url}/options/greeks",
    ]
    
    for endpoint in endpoints_to_try:
        try:
            response = requests.get(
                endpoint,
                params=base_params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Options Greeks access: WORKING")
                print(f"   Endpoint: {endpoint}")
                print(f"   Sample: {data}")
                return True
            elif response.status_code == 404:
                continue  # Try next
            else:
                print(f"⚠️  Greeks endpoint: {endpoint} - {response.status_code}")
                continue
                
        except requests.exceptions.RequestException:
            continue
    
    print("⚠️  Greeks endpoint not found (may be included in chain data)")
    return None

def main():
    print()
    print("=" * 80)
    print("🧪 MASSIVE API ACCESS VERIFICATION")
    print("=" * 80)
    print()
    print("This script verifies what your Massive subscription includes")
    print("before we integrate it into the trading system.")
    print()
    
    if not MASSIVE_API_KEY:
        print("⚠️  MASSIVE_API_KEY not set")
        print()
        print("To test, set your API key:")
        print("  export MASSIVE_API_KEY='your_key_here'")
        print("  python3 test_massive_api_access.py")
        print()
        print("Or add to config.py:")
        print("  MASSIVE_API_KEY = 'your_key_here'")
        print()
        return
    
    results = {}
    base_url = None
    
    # Run tests
    connection_result = test_api_connection()
    if isinstance(connection_result, tuple):
        results['connection'], base_url = connection_result
    else:
        results['connection'] = connection_result
    
    print()
    
    if results['connection'] and base_url:
        results['stocks'] = test_stocks_data(base_url)
        print()
        
        results['options_chain'] = test_options_chain(base_url)
        print()
        
        results['options_greeks'] = test_options_greeks(base_url)
        print()
    elif results['connection']:
        print("⚠️  Connection test passed but base URL not found")
        print("   You may need to manually specify the base URL")
        print("   Check Massive API documentation")
    
    # Summary
    print("=" * 80)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 80)
    print()
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  N/A"
        print(f"   {status}: {test_name}")
    
    print()
    
    # Recommendation
    if results.get('options_chain'):
        print("✅ RECOMMENDATION: INTEGRATE MASSIVE API")
        print()
        print("Your subscription includes:")
        print("   ✅ Options chain data")
        print("   ✅ Individual option prices")
        if results.get('options_greeks'):
            print("   ✅ Real-time Greeks")
        print()
        print("This will significantly improve:")
        print("   • Option selection accuracy")
        print("   • Position sizing precision")
        print("   • Risk management")
        print("   • Execution quality")
    elif results.get('connection'):
        print("⚠️  RECOMMENDATION: VERIFY SUBSCRIPTION")
        print()
        print("Could not verify options chain access.")
        print("Please check:")
        print("   1. Massive API documentation")
        print("   2. Your subscription details")
        print("   3. What 'Options Starter' actually includes")
    else:
        print("❌ RECOMMENDATION: FIX API ACCESS FIRST")
        print()
        print("Could not connect to Massive API.")
        print("Check your API key and endpoint URL.")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()

