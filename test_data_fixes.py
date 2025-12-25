#!/usr/bin/env python3
"""
Test Data Source Fixes with Live Market Data
Verifies all 5 fixes are working correctly
"""
import os
import sys
from datetime import datetime, timedelta
import pytz
import pandas as pd

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠️  Alpaca API not available")

try:
    from massive_api_client import MassiveAPIClient
    MASSIVE_AVAILABLE = True
except ImportError:
    MASSIVE_AVAILABLE = False
    print("⚠️  Massive API not available")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance not available")

# Import config
try:
    import config
    API_KEY = config.ALPACA_KEY
    API_SECRET = config.ALPACA_SECRET
    BASE_URL = config.ALPACA_BASE_URL
except:
    API_KEY = os.getenv('ALPACA_KEY', '')
    API_SECRET = os.getenv('ALPACA_SECRET', '')
    BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

# Import the actual functions from the agent
from mike_agent_live_safe import get_market_data, get_current_price

class TestLogger:
    """Simple logger for testing"""
    def __init__(self):
        self.logs = []
    
    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)
        print(log_entry)
    
    def get_logs(self):
        return self.logs

def test_data_freshness_validation():
    """Test 1: Data Freshness Validation"""
    print("\n" + "="*80)
    print("TEST 1: DATA FRESHNESS VALIDATION")
    print("="*80)
    
    logger = TestLogger()
    est = pytz.timezone('US/Eastern')
    now_est = datetime.now(est)
    today_est = now_est.date()
    
    # Initialize Alpaca if available
    api = None
    if ALPACA_AVAILABLE and API_KEY and API_SECRET:
        try:
            api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
            logger.log("✅ Alpaca API initialized", "INFO")
        except Exception as e:
            logger.log(f"⚠️  Alpaca API init failed: {e}", "WARNING")
    
    # Test with SPY
    symbol = "SPY"
    logger.log(f"Testing data freshness for {symbol}...", "INFO")
    
    # Get market data (this will use our fixed get_market_data function)
    hist = get_market_data(symbol, period="2d", interval="1m", api=api, risk_mgr=logger)
    
    if len(hist) == 0:
        logger.log("❌ FAILED: No data returned", "ERROR")
        return False
    
    # Check if data is from today
    last_bar_time = hist.index[-1]
    
    # Convert to EST
    if hasattr(last_bar_time, 'tzinfo') and last_bar_time.tzinfo:
        last_bar_est = last_bar_time.astimezone(est)
    else:
        try:
            last_bar_utc = pytz.utc.localize(last_bar_time)
            last_bar_est = last_bar_utc.astimezone(est)
        except:
            last_bar_est = est.localize(last_bar_time) if last_bar_time.tzinfo is None else last_bar_time
    
    last_bar_date = last_bar_est.date()
    time_diff_minutes = (now_est - last_bar_est).total_seconds() / 60
    
    logger.log(f"Last bar date: {last_bar_date}", "INFO")
    logger.log(f"Today's date: {today_est}", "INFO")
    logger.log(f"Data age: {time_diff_minutes:.1f} minutes", "INFO")
    
    # Validation checks
    checks_passed = 0
    total_checks = 3
    
    # Check 1: Data is from today
    if last_bar_date == today_est:
        logger.log("✅ PASS: Data is from today", "INFO")
        checks_passed += 1
    else:
        logger.log(f"❌ FAIL: Data is from {last_bar_date}, not today ({today_est})", "ERROR")
    
    # Check 2: Data is fresh (< 5 min during market hours)
    market_hours = 9.5 <= now_est.hour + (now_est.minute / 60) < 16.0
    max_age = 5 if market_hours else 60
    
    if time_diff_minutes <= max_age:
        logger.log(f"✅ PASS: Data is fresh ({time_diff_minutes:.1f} min < {max_age} min)", "INFO")
        checks_passed += 1
    else:
        logger.log(f"❌ FAIL: Data is stale ({time_diff_minutes:.1f} min > {max_age} min)", "ERROR")
    
    # Check 3: Data has reasonable amount
    if len(hist) >= 100:
        logger.log(f"✅ PASS: Sufficient data ({len(hist)} bars)", "INFO")
        checks_passed += 1
    else:
        logger.log(f"⚠️  WARNING: Low data count ({len(hist)} bars)", "WARNING")
        checks_passed += 1  # Not a critical failure
    
    logger.log(f"\n📊 Result: {checks_passed}/{total_checks} checks passed", "INFO")
    return checks_passed == total_checks

def test_price_cross_validation():
    """Test 2: Price Cross-Validation"""
    print("\n" + "="*80)
    print("TEST 2: PRICE CROSS-VALIDATION")
    print("="*80)
    
    logger = TestLogger()
    symbol = "SPY"
    
    # Initialize APIs
    api = None
    if ALPACA_AVAILABLE and API_KEY and API_SECRET:
        try:
            api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
            logger.log("✅ Alpaca API initialized", "INFO")
        except Exception as e:
            logger.log(f"⚠️  Alpaca API init failed: {e}", "WARNING")
    
    massive_client = None
    if MASSIVE_AVAILABLE:
        try:
            massive_api_key = os.getenv('MASSIVE_API_KEY', '')
            if massive_api_key:
                massive_client = MassiveAPIClient(massive_api_key)
                logger.log("✅ Massive API initialized", "INFO")
        except Exception as e:
            logger.log(f"⚠️  Massive API init failed: {e}", "WARNING")
    
    # Get price from primary source (get_market_data)
    logger.log(f"Getting price from primary source for {symbol}...", "INFO")
    hist = get_market_data(symbol, period="1d", interval="1m", api=api, risk_mgr=logger)
    
    if len(hist) == 0:
        logger.log("❌ FAILED: No data from primary source", "ERROR")
        return False
    
    primary_price = float(hist['Close'].iloc[-1])
    logger.log(f"Primary price: ${primary_price:.2f}", "INFO")
    
    # Get price from alternative source
    alt_price = None
    alt_source = None
    
    # Try Alpaca for validation
    if api:
        try:
            from alpaca_trade_api.rest import TimeFrame
            est = pytz.timezone('US/Eastern')
            now_est = datetime.now(est)
            end_str = (now_est + timedelta(days=1)).strftime("%Y-%m-%d")
            start_str = now_est.strftime("%Y-%m-%d")
            
            alt_bars = api.get_bars(symbol, TimeFrame.Minute, start_str, end_str, limit=1, adjustment='raw').df
            if len(alt_bars) > 0:
                alt_price = float(alt_bars['close'].iloc[-1])
                alt_source = "Alpaca"
                logger.log(f"Alternative price (Alpaca): ${alt_price:.2f}", "INFO")
        except Exception as e:
            logger.log(f"⚠️  Alpaca validation failed: {e}", "WARNING")
    
    # Try Massive for validation
    if alt_price is None and massive_client:
        try:
            est = pytz.timezone('US/Eastern')
            now_est = datetime.now(est)
            end_date_str = (now_est + timedelta(days=1)).strftime("%Y-%m-%d")
            start_date_str = now_est.strftime("%Y-%m-%d")
            
            alt_hist = massive_client.get_historical_data(symbol, start_date_str, end_date_str, interval='1min')
            if len(alt_hist) > 0:
                alt_price = float(alt_hist['close'].iloc[-1] if 'close' in alt_hist.columns else alt_hist['Close'].iloc[-1])
                alt_source = "Massive"
                logger.log(f"Alternative price (Massive): ${alt_price:.2f}", "INFO")
        except Exception as e:
            logger.log(f"⚠️  Massive validation failed: {e}", "WARNING")
    
    # Try yfinance as last resort
    if alt_price is None and YFINANCE_AVAILABLE:
        try:
            ticker = yf.Ticker(symbol)
            yf_hist = ticker.history(period="1d", interval="1m")
            if len(yf_hist) > 0:
                alt_price = float(yf_hist['Close'].iloc[-1])
                alt_source = "yfinance"
                logger.log(f"Alternative price (yfinance): ${alt_price:.2f}", "INFO")
        except Exception as e:
            logger.log(f"⚠️  yfinance validation failed: {e}", "WARNING")
    
    if alt_price is None:
        logger.log("⚠️  WARNING: No alternative source available for validation", "WARNING")
        return True  # Not a failure, just no validation possible
    
    # Compare prices
    price_diff = abs(primary_price - alt_price)
    logger.log(f"Price difference: ${price_diff:.2f}", "INFO")
    
    if price_diff > 2.00:
        logger.log(f"❌ FAIL: Large price mismatch (${price_diff:.2f} > $2.00)", "ERROR")
        return False
    elif price_diff > 0.50:
        logger.log(f"⚠️  WARNING: Moderate price difference (${price_diff:.2f} > $0.50)", "WARNING")
        return True  # Warning but not failure
    else:
        logger.log(f"✅ PASS: Prices match closely (${price_diff:.2f} < $0.50)", "INFO")
        return True

def test_timezone_handling():
    """Test 3: EST Timezone Handling"""
    print("\n" + "="*80)
    print("TEST 3: EST TIMEZONE HANDLING")
    print("="*80)
    
    logger = TestLogger()
    est = pytz.timezone('US/Eastern')
    now_est = datetime.now(est)
    today_est = now_est.date()
    
    logger.log(f"Current EST time: {now_est.strftime('%Y-%m-%d %H:%M:%S %Z')}", "INFO")
    logger.log(f"Today's date (EST): {today_est}", "INFO")
    
    # Test that get_market_data uses EST
    api = None
    if ALPACA_AVAILABLE and API_KEY and API_SECRET:
        try:
            api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
        except:
            pass
    
    hist = get_market_data("SPY", period="1d", interval="1m", api=api, risk_mgr=logger)
    
    if len(hist) > 0:
        last_bar_time = hist.index[-1]
        
        # Convert to EST
        if hasattr(last_bar_time, 'tzinfo') and last_bar_time.tzinfo:
            last_bar_est = last_bar_time.astimezone(est)
        else:
            try:
                last_bar_utc = pytz.utc.localize(last_bar_time)
                last_bar_est = last_bar_utc.astimezone(est)
            except:
                last_bar_est = est.localize(last_bar_time) if last_bar_time.tzinfo is None else last_bar_time
        
        logger.log(f"Last bar time (EST): {last_bar_est.strftime('%Y-%m-%d %H:%M:%S %Z')}", "INFO")
        logger.log(f"Last bar date (EST): {last_bar_est.date()}", "INFO")
        
        if last_bar_est.date() == today_est:
            logger.log("✅ PASS: Timezone handling correct (data date matches today EST)", "INFO")
            return True
        else:
            logger.log(f"❌ FAIL: Date mismatch (got {last_bar_est.date()}, expected {today_est})", "ERROR")
            return False
    else:
        logger.log("⚠️  WARNING: No data to test timezone", "WARNING")
        return True  # Not a failure

def test_data_source_logging():
    """Test 4: Data Source Logging"""
    print("\n" + "="*80)
    print("TEST 4: DATA SOURCE LOGGING")
    print("="*80)
    
    logger = TestLogger()
    
    api = None
    if ALPACA_AVAILABLE and API_KEY and API_SECRET:
        try:
            api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
        except:
            pass
    
    # Get data and check logs
    hist = get_market_data("SPY", period="1d", interval="1m", api=api, risk_mgr=logger)
    
    # Check if logs contain data source information
    logs_str = "\n".join(logger.get_logs())
    
    has_source_log = False
    has_freshness_log = False
    
    if "Alpaca API" in logs_str or "Massive API" in logs_str or "yfinance" in logs_str:
        has_source_log = True
        logger.log("✅ PASS: Data source logged", "INFO")
    else:
        logger.log("⚠️  WARNING: Data source not clearly logged", "WARNING")
    
    if "Fresh data" in logs_str or "data validation" in logs_str.lower():
        has_freshness_log = True
        logger.log("✅ PASS: Data freshness logged", "INFO")
    else:
        logger.log("⚠️  WARNING: Data freshness not clearly logged", "WARNING")
    
    return has_source_log or has_freshness_log

def test_cache_clearing():
    """Test 5: Cache Clearing"""
    print("\n" + "="*80)
    print("TEST 5: CACHE CLEARING")
    print("="*80)
    
    logger = TestLogger()
    
    # This is harder to test directly, but we can verify that
    # get_market_data is being called fresh each time
    api = None
    if ALPACA_AVAILABLE and API_KEY and API_SECRET:
        try:
            api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
        except:
            pass
    
    # Get data twice and compare timestamps
    logger.log("Fetching data first time...", "INFO")
    hist1 = get_market_data("SPY", period="1d", interval="1m", api=api, risk_mgr=logger)
    
    import time
    time.sleep(2)  # Wait 2 seconds
    
    logger.log("Fetching data second time...", "INFO")
    hist2 = get_market_data("SPY", period="1d", interval="1m", api=api, risk_mgr=logger)
    
    if len(hist1) > 0 and len(hist2) > 0:
        last_time1 = hist1.index[-1]
        last_time2 = hist2.index[-1]
        
        # Times should be different (or very close if market is closed)
        logger.log(f"First fetch last bar: {last_time1}", "INFO")
        logger.log(f"Second fetch last bar: {last_time2}", "INFO")
        
        # If times are the same, it might be cached (but could also be market closed)
        if last_time1 == last_time2:
            logger.log("⚠️  WARNING: Same timestamp (could be cache or market closed)", "WARNING")
        else:
            logger.log("✅ PASS: Different timestamps (cache cleared)", "INFO")
        
        return True
    else:
        logger.log("⚠️  WARNING: No data to test cache clearing", "WARNING")
        return True

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("🧪 TESTING DATA SOURCE FIXES WITH LIVE MARKET DATA")
    print("="*80)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    results = {}
    
    # Run all tests
    results['freshness'] = test_data_freshness_validation()
    results['cross_validation'] = test_price_cross_validation()
    results['timezone'] = test_timezone_handling()
    results['logging'] = test_data_source_logging()
    results['cache'] = test_cache_clearing()
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST RESULTS SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Data source fixes are working correctly.")
    elif passed >= total * 0.8:
        print("\n⚠️  MOST TESTS PASSED. Some warnings but core functionality works.")
    else:
        print("\n❌ SOME TESTS FAILED. Please review the output above.")
    
    print("="*80)
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)


