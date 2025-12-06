#!/usr/bin/env python3
"""
Quick Alpaca connection test
Tests if API keys are configured correctly
"""
import sys
import os

try:
    import config
except ImportError:
    print("❌ Error: config.py not found")
    print("   Please copy config.py.example to config.py and add your API keys")
    sys.exit(1)

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    print("❌ Error: alpaca-trade-api not installed")
    print("   Install with: pip install alpaca-trade-api")
    sys.exit(1)

print("=" * 60)
print("Alpaca Connection Test")
print("=" * 60)

# Check if keys are set
if not hasattr(config, 'ALPACA_KEY') or config.ALPACA_KEY in ['YOUR_PAPER_KEY_HERE', 'YOUR_PAPER_KEY', 'PKXX2KTB6QGJ7EW4CG7YFX4XUF']:
    print("⚠️  Warning: ALPACA_KEY appears to be a placeholder")
    print("   Please update config.py with your actual Alpaca API key")
    
if not hasattr(config, 'ALPACA_SECRET') or config.ALPACA_SECRET in ['YOUR_PAPER_SECRET_HERE', 'YOUR_PAPER_SECRET']:
    print("⚠️  Warning: ALPACA_SECRET appears to be a placeholder")
    print("   Please update config.py with your actual Alpaca secret key")

# Get base URL
base_url = getattr(config, 'ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
is_paper = 'paper' in base_url.lower()

print(f"\n📡 Connecting to: {base_url}")
print(f"   Mode: {'PAPER TRADING' if is_paper else 'LIVE TRADING'}")

try:
    api = tradeapi.REST(
        config.ALPACA_KEY,
        config.ALPACA_SECRET,
        base_url,
        api_version='v2'
    )
    
    # Test connection
    account = api.get_account()
    
    print("\n✅ Connection Successful!")
    print(f"\n📊 Account Information:")
    print(f"   Status: {account.status}")
    print(f"   Equity: ${float(account.equity):,.2f}")
    print(f"   Buying Power: ${float(account.buying_power):,.2f}")
    print(f"   Cash: ${float(account.cash):,.2f}")
    print(f"   Pattern Day Trader: {account.pattern_day_trader}")
    
    # Check if options trading is enabled (if possible)
    try:
        positions = api.list_positions()
        print(f"\n📈 Current Positions: {len(positions)}")
        if positions:
            print("   Active positions found")
            for pos in positions[:3]:  # Show first 3
                print(f"   - {pos.symbol}: {pos.qty} @ ${float(pos.avg_entry_price):.2f}")
    except Exception as e:
        print(f"\n⚠️  Could not check positions: {e}")
    
    print("\n✅ Ready to start paper trading!")
    print("\nNext steps:")
    print("   1. Start agent: python mike_agent_live_safe.py")
    print("   2. Start dashboard: streamlit run app.py")
    
except Exception as e:
    print(f"\n❌ Connection Failed!")
    print(f"   Error: {e}")
    print("\nTroubleshooting:")
    print("   1. Verify API keys in config.py are correct")
    print("   2. Check internet connection")
    print("   3. Verify Alpaca service status: https://status.alpaca.markets")
    print("   4. Make sure you're using PAPER trading keys (not live)")
    sys.exit(1)


