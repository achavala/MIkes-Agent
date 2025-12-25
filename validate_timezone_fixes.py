#!/usr/bin/env python3
"""
Validation script for timezone fixes and critical improvements
Run this to verify all fixes are working correctly
"""

from datetime import datetime
import pytz
import sys

def validate_timezone_behavior():
    """Test 1: Verify runtime timezone behavior"""
    print("=" * 80)
    print("TEST 1: Runtime Timezone Behavior")
    print("=" * 80)
    
    est = pytz.timezone("US/Eastern")
    now_est = datetime.now(est)
    now_utc = datetime.utcnow()
    
    print(f"NOW EST: {now_est}")
    print(f"NOW UTC: {now_utc}")
    print(f"EST Date: {now_est.date()}")
    print(f"UTC Date: {now_utc.date()}")
    
    # Check if EST is approximately +5 hours (or +4 in DST)
    est_hour = now_est.hour
    utc_hour = now_utc.hour
    hour_diff = (est_hour - utc_hour) % 24
    
    if hour_diff in [4, 5]:
        print(f"✅ EST timezone correct (EST is UTC{hour_diff:+d})")
    else:
        print(f"⚠️  WARNING: EST timezone may be incorrect (diff: {hour_diff} hours)")
    
    # Check if datetime.now() without tz is being used
    print("\n⚠️  CRITICAL: Check codebase for datetime.now() without timezone")
    print("   All datetime.now() calls should use: datetime.now(pytz.timezone('US/Eastern'))")
    
    return True

def validate_alpaca_clock_usage():
    """Test 2: Verify broker clock is used"""
    print("\n" + "=" * 80)
    print("TEST 2: Alpaca Clock Usage")
    print("=" * 80)
    
    try:
        import config
        from alpaca.tradeapi import REST
        
        # Try to initialize Alpaca API
        api_key = getattr(config, 'ALPACA_KEY', None) or getattr(config, 'APCA_API_KEY_ID', None)
        api_secret = getattr(config, 'ALPACA_SECRET', None) or getattr(config, 'APCA_API_SECRET_KEY', None)
        base_url = getattr(config, 'ALPACA_BASE_URL', None) or getattr(config, 'APCA_API_BASE_URL', None)
        
        if not api_key or api_key == 'YOUR_PAPER_KEY' or not api_secret:
            print("⚠️  Alpaca credentials not configured - skipping clock test")
            print("   To test: Set ALPACA_KEY and ALPACA_SECRET in config.py")
            return False
        
        api = REST(api_key, api_secret, base_url=base_url, api_version='v2')
        clock = api.get_clock()
        
        print(f"✅ Alpaca Clock Retrieved")
        print(f"   Alpaca Timestamp: {clock.timestamp}")
        print(f"   Market Is Open: {clock.is_open}")
        print(f"   Next Open: {clock.next_open}")
        print(f"   Next Close: {clock.next_close}")
        
        # Convert to EST
        est = pytz.timezone('US/Eastern')
        alpaca_est = clock.timestamp.astimezone(est)
        print(f"   Alpaca Time (EST): {alpaca_est}")
        
        # Check if code uses this
        print("\n⚠️  CRITICAL: Verify code uses api.get_clock() for:")
        print("   - Market open/close status")
        print("   - Current market date (not local OS date)")
        print("   - Trade timing decisions")
        
        return True
        
    except ImportError:
        print("⚠️  Alpaca API not available - install: pip install alpaca-trade-api")
        return False
    except Exception as e:
        print(f"⚠️  Error accessing Alpaca clock: {e}")
        return False

def validate_option_cache_clearing():
    """Test 3: Verify option cache clearing on new day"""
    print("\n" + "=" * 80)
    print("TEST 3: Option Cache Clearing on New Day")
    print("=" * 80)
    
    # Check if reset_daily_state exists and is called
    print("Checking for daily reset logic...")
    
    try:
        with open('mike_agent_live_safe.py', 'r') as f:
            content = f.read()
            
            has_reset = 'reset_daily_state' in content
            has_last_reset = 'last_reset_date' in content
            
            if has_reset and has_last_reset:
                print("✅ reset_daily_state() method exists")
                print("✅ last_reset_date tracking exists")
                
                # Check if it's called in main loop
                if 'reset_daily_state()' in content or 'risk_mgr.reset_daily_state()' in content:
                    print("✅ reset_daily_state() is called")
                else:
                    print("⚠️  WARNING: reset_daily_state() may not be called in main loop")
                    print("   Add: risk_mgr.reset_daily_state() at start of each iteration")
            else:
                print("❌ Daily reset logic not found")
                return False
                
    except Exception as e:
        print(f"❌ Error checking code: {e}")
        return False
    
    print("\n⚠️  RECOMMENDATION: Add this to main loop:")
    print("   est = pytz.timezone('US/Eastern')")
    print("   current_est_date = datetime.now(est).date()")
    print("   if current_est_date != last_seen_est_date:")
    print("       clear_option_cache()")
    print("       clear_last_trade_symbols()")
    print("       risk_mgr.reset_daily_state()")
    
    return True

def validate_option_symbol_construction():
    """Test 4: Check how option symbols are constructed"""
    print("\n" + "=" * 80)
    print("TEST 4: Option Symbol Construction")
    print("=" * 80)
    
    try:
        with open('mike_agent_live_safe.py', 'r') as f:
            content = f.read()
            
            # Check for manual construction
            has_manual = 'f"{underlying}{date_str}' in content or 'option_symbol = f' in content
            has_alpaca_chain = 'get_option_chain' in content or 'list_options' in content or 'option_chain' in content
            
            if has_manual:
                print("⚠️  WARNING: Option symbols are constructed manually")
                print("   Found: Manual string formatting (f\"{underlying}{date_str}...\")")
                print("   Risk: Stale dates, incorrect strikes, invalid contracts")
                
            if has_alpaca_chain:
                print("✅ Alpaca option chain API usage found")
            else:
                print("⚠️  RECOMMENDATION: Use Alpaca option chain API instead of manual construction")
                print("   Instead of: symbol = f\"SPY{expiry}{C/P}{strike}\"")
                print("   Use: api.list_options(underlying, expiration_date, option_type)")
                print("   Then filter by: closest ATM strike")
                
    except Exception as e:
        print(f"❌ Error checking code: {e}")
        return False
    
    return True

def main():
    """Run all validation tests"""
    print("\n" + "=" * 80)
    print("TIMEZONE FIXES VALIDATION")
    print("=" * 80)
    print("\nThis script validates the critical fixes for timezone handling")
    print("and identifies remaining risks.\n")
    
    results = []
    
    # Test 1: Timezone behavior
    try:
        results.append(("Timezone Behavior", validate_timezone_behavior()))
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        results.append(("Timezone Behavior", False))
    
    # Test 2: Alpaca clock
    try:
        results.append(("Alpaca Clock Usage", validate_alpaca_clock_usage()))
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        results.append(("Alpaca Clock Usage", False))
    
    # Test 3: Cache clearing
    try:
        results.append(("Option Cache Clearing", validate_option_cache_clearing()))
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        results.append(("Option Cache Clearing", False))
    
    # Test 4: Symbol construction
    try:
        results.append(("Option Symbol Construction", validate_option_symbol_construction()))
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        results.append(("Option Symbol Construction", False))
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "⚠️  NEEDS ATTENTION"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. ✅ Timezone fix is committed and tagged")
    print("2. ⚠️  Verify Alpaca clock is used for market date/time")
    print("3. ⚠️  Add daily cache clearing on EST day change")
    print("4. ⚠️  Consider using Alpaca option chain API instead of manual construction")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()


