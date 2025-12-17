#!/usr/bin/env python3
"""
Data Collection Verification Script
Shows exactly what data is being collected and verifies it's real
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the actual function
from mike_agent_live_safe import get_market_data, get_current_price, init_alpaca

# Try to initialize Alpaca API if credentials are available
try:
    import config
    if hasattr(config, 'ALPACA_KEY') and hasattr(config, 'ALPACA_SECRET'):
        if config.ALPACA_KEY != 'YOUR_PAPER_KEY' and config.ALPACA_SECRET != 'YOUR_PAPER_SECRET':
            try:
                api = init_alpaca()
                print("✅ Alpaca API initialized for verification")
            except Exception as e:
                print(f"⚠️  Could not initialize Alpaca API: {e}")
                api = None
        else:
            api = None
    else:
        api = None
except (ImportError, AttributeError):
    # Try environment variables
    api_key = os.getenv('ALPACA_KEY')
    api_secret = os.getenv('ALPACA_SECRET')
    if api_key and api_secret:
        try:
            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(api_key, api_secret, 'https://paper-api.alpaca.markets', api_version='v2')
            api.get_account()  # Test connection
            print("✅ Alpaca API initialized from environment variables")
        except Exception as e:
            print(f"⚠️  Could not initialize Alpaca API: {e}")
            api = None
    else:
        api = None
        print("ℹ️  Alpaca API not configured - will use yfinance (limited to market hours data)")

def verify_data_collection(symbol="SPY"):
    """Verify data collection for a symbol"""
    print("=" * 80)
    print(f"🔍 DATA COLLECTION VERIFICATION: {symbol}")
    print("=" * 80)
    print()
    
    # Test 1: Get 2 days of 1-minute data
    print("📥 TEST 1: Requesting 2 days of 1-minute data")
    print("-" * 80)
    print(f"Function: get_market_data('{symbol}', period='2d', interval='1m')")
    print()
    
    try:
        # Pass API instance if available
        hist = get_market_data(symbol, period="2d", interval="1m", api=api if 'api' in globals() else None)
        
        if hist.empty:
            print("❌ ERROR: No data returned!")
            return
        
        print(f"✅ Data received")
        print(f"   Shape: {hist.shape}")
        print(f"   Columns: {list(hist.columns)}")
        print()
        
        # Expected: 2 days × 1,440 minutes/day = 2,880 bars (full 24/7)
        # Market hours only: 2 days × 390 minutes/day = 780 bars (9:30 AM - 4:00 PM EST)
        # yfinance limitation: Only returns market hours data (~780 bars for 2 days)
        # Alpaca API: Returns full 2 days of data (~2,880 bars)
        expected_bars_full = 2 * 1440  # 2,880 bars (full 24/7)
        expected_bars_market_hours = 2 * 390  # 780 bars (market hours only)
        actual_bars = len(hist)
        
        print(f"📊 DATA ANALYSIS:")
        print(f"   Expected bars (full 2 days × 1,440 min/day): {expected_bars_full:,}")
        print(f"   Expected bars (market hours only): {expected_bars_market_hours:,}")
        print(f"   Actual bars received: {actual_bars:,}")
        print()
        
        if actual_bars < expected_bars_market_hours:
            print(f"⚠️  WARNING: Only {actual_bars} bars received, expected at least {expected_bars_market_hours:,}")
            print(f"   This indicates a data source limitation or bug.")
            print()
        elif actual_bars < expected_bars_full:
            print(f"ℹ️  INFO: Received {actual_bars} bars (market hours only)")
            if api and 'api' in globals():
                print(f"   ⚠️  Alpaca API is configured but returned market hours only")
                print(f"   This may be normal if requesting market hours data")
            else:
                print(f"   ℹ️  Using yfinance (free, delayed) - limited to market hours data")
                print(f"   💡 To get full 2 days: Configure Alpaca API (you're paying for it!)")
            print()
        
        # Show data sample
        print(f"📋 DATA SAMPLE (First 5 bars):")
        print(hist.head().to_string())
        print()
        
        print(f"📋 DATA SAMPLE (Last 5 bars):")
        print(hist.tail().to_string())
        print()
        
        # Verify data quality
        print(f"🔍 DATA QUALITY CHECKS:")
        print(f"   Missing values: {hist.isnull().sum().sum()}")
        print(f"   Duplicate timestamps: {hist.index.duplicated().sum()}")
        print(f"   Date range: {hist.index.min()} to {hist.index.max()}")
        
        # Calculate time span
        if len(hist) > 1:
            time_span = hist.index.max() - hist.index.min()
            print(f"   Time span: {time_span}")
            print(f"   Expected: 2 days")
            
            if time_span < timedelta(days=1):
                print(f"   ⚠️  WARNING: Time span is less than 1 day!")
        
        print()
        
        # Verify data is not random
        print(f"🔍 DATA VALIDATION (Is it real data?):")
        
        # Check 1: Prices should be reasonable
        if 'Close' in hist.columns:
            close_prices = hist['Close']
            price_range = close_prices.max() - close_prices.min()
            price_change_pct = (price_range / close_prices.min()) * 100
            
            print(f"   Price range: ${close_prices.min():.2f} - ${close_prices.max():.2f}")
            print(f"   Price change: {price_change_pct:.2f}%")
            
            # SPY should be ~$400-700, QQQ ~$300-700
            if symbol == 'SPY':
                if 300 < close_prices.mean() < 800:
                    print(f"   ✅ Prices look reasonable for SPY")
                else:
                    print(f"   ⚠️  Prices seem unusual: ${close_prices.mean():.2f}")
            elif symbol == 'QQQ':
                if 200 < close_prices.mean() < 800:
                    print(f"   ✅ Prices look reasonable for QQQ")
                else:
                    print(f"   ⚠️  Prices seem unusual: ${close_prices.mean():.2f}")
        
        # Check 2: Volume should be present
        if 'Volume' in hist.columns:
            volumes = hist['Volume']
            print(f"   Volume range: {volumes.min():,.0f} - {volumes.max():,.0f}")
            if volumes.sum() > 0:
                print(f"   ✅ Volume data present")
            else:
                print(f"   ⚠️  No volume data")
        
        # Check 3: Prices should change (not constant)
        if 'Close' in hist.columns:
            unique_prices = close_prices.nunique()
            print(f"   Unique prices: {unique_prices} out of {len(hist)} bars")
            if unique_prices > len(hist) * 0.1:  # At least 10% unique
                print(f"   ✅ Prices are changing (not constant)")
            else:
                print(f"   ⚠️  Too many duplicate prices (possible data issue)")
        
        # Check 4: Timestamps should be sequential
        if len(hist) > 1:
            time_diffs = hist.index.to_series().diff().dropna()
            expected_interval = pd.Timedelta(minutes=1)
            avg_interval = time_diffs.mean()
            
            print(f"   Average interval: {avg_interval}")
            print(f"   Expected: 1 minute")
            
            if abs((avg_interval - expected_interval).total_seconds()) < 60:
                print(f"   ✅ Timestamps are approximately 1-minute intervals")
            else:
                print(f"   ⚠️  Timestamps don't match 1-minute interval")
        
        print()
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Check current price
    print("📥 TEST 2: Getting current price")
    print("-" * 80)
    try:
        current_price = get_current_price(symbol)
        if current_price:
            print(f"✅ Current price: ${current_price:.2f}")
        else:
            print(f"❌ No current price returned")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print()
    print("=" * 80)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 80)

def main():
    """Main function"""
    # Check if API is available
    if 'api' not in globals() or api is None:
        print("=" * 80)
        print("ℹ️  NOTE: Alpaca API not configured for this test")
        print("   The verification will use yfinance (free, delayed)")
        print("   yfinance only returns market hours data (~780 bars for 2 days)")
        print("   To test with Alpaca API, set ALPACA_KEY and ALPACA_SECRET")
        print("=" * 80)
        print()
    
    symbols = ['SPY', 'QQQ']
    
    for symbol in symbols:
        verify_data_collection(symbol)
        print("\n" * 2)

if __name__ == "__main__":
    main()

