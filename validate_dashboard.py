#!/usr/bin/env python3
"""
Validate Dashboard Configuration and Display
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("🔍 DASHBOARD VALIDATION")
print("=" * 80)
print()

# 1. Check config.py
print("1️⃣ CONFIG.PY VALIDATION")
print("-" * 80)
try:
    import config
    print(f"✅ config.py loaded successfully")
    print(f"   ALPACA_KEY: {config.ALPACA_KEY[:20]}... (length: {len(config.ALPACA_KEY)})")
    print(f"   ALPACA_SECRET: {config.ALPACA_SECRET[:20]}... (length: {len(config.ALPACA_SECRET)})")
    print(f"   ALPACA_BASE_URL: {config.ALPACA_BASE_URL}")
    
    # Check if keys look valid
    if len(config.ALPACA_KEY) < 20 or 'XX' in config.ALPACA_KEY:
        print("   ⚠️  ALPACA_KEY appears to be a placeholder")
    else:
        print("   ✅ ALPACA_KEY looks valid")
    
    if len(config.ALPACA_SECRET) < 20:
        print("   ⚠️  ALPACA_SECRET appears to be invalid")
    else:
        print("   ✅ ALPACA_SECRET looks valid")
except Exception as e:
    print(f"❌ Error loading config.py: {e}")
    sys.exit(1)

print()

# 2. Check environment variables
print("2️⃣ ENVIRONMENT VARIABLES")
print("-" * 80)
env_vars = {
    'APCA_API_KEY_ID': os.getenv('APCA_API_KEY_ID'),
    'APCA_API_SECRET_KEY': os.getenv('APCA_API_SECRET_KEY'),
    'ALPACA_KEY': os.getenv('ALPACA_KEY'),
    'ALPACA_SECRET': os.getenv('ALPACA_SECRET'),
}

for key, value in env_vars.items():
    if value:
        print(f"✅ {key}: {value[:20]}... (set)")
    else:
        print(f"   {key}: NOT SET")

print()

# 3. Test API connection
print("3️⃣ ALPACA API CONNECTION TEST")
print("-" * 80)
try:
    import alpaca_trade_api as tradeapi
    
    api = tradeapi.REST(
        config.ALPACA_KEY,
        config.ALPACA_SECRET,
        config.ALPACA_BASE_URL,
        api_version='v2'
    )
    
    account = api.get_account()
    print("✅ API Connection: SUCCESS")
    print(f"   Account Status: {account.status}")
    print(f"   Equity: ${float(account.equity):,.2f}")
    print(f"   Buying Power: ${float(account.buying_power):,.2f}")
    
    # Test getting positions
    try:
        positions = api.list_positions()
        print(f"   Open Positions: {len(positions)}")
    except Exception as e:
        print(f"   ⚠️  Could not fetch positions: {e}")
    
except Exception as e:
    error_msg = str(e)
    print(f"❌ API Connection: FAILED")
    print(f"   Error: {error_msg}")
    
    if 'Key ID' in error_msg or 'API key' in error_msg or 'authentication' in error_msg.lower():
        print()
        print("   🔧 FIX: Set environment variables:")
        print("      export APCA_API_KEY_ID=your_key")
        print("      export APCA_API_SECRET_KEY=your_secret")
    elif 'connection' in error_msg.lower() or 'timeout' in error_msg.lower():
        print("   ⚠️  Network/connection issue")
    else:
        print("   ⚠️  Unknown error")

print()

# 4. Check app.py imports
print("4️⃣ APP.PY IMPORTS")
print("-" * 80)
try:
    import streamlit as st
    print("✅ streamlit: OK")
except:
    print("❌ streamlit: NOT INSTALLED")

try:
    import alpaca_trade_api as tradeapi
    print("✅ alpaca_trade_api: OK")
except:
    print("❌ alpaca_trade_api: NOT INSTALLED")

try:
    import pandas as pd
    print("✅ pandas: OK")
except:
    print("❌ pandas: NOT INSTALLED")

try:
    import yfinance as yf
    print("✅ yfinance: OK")
except:
    print("❌ yfinance: NOT INSTALLED")

print()

# 5. Check app.py syntax
print("5️⃣ APP.PY SYNTAX CHECK")
print("-" * 80)
import py_compile
try:
    py_compile.compile('app.py', doraise=True)
    print("✅ app.py: No syntax errors")
except py_compile.PyCompileError as e:
    print(f"❌ app.py: Syntax error - {e}")
except Exception as e:
    print(f"❌ app.py: Error - {e}")

print()
print("=" * 80)
print("✅ VALIDATION COMPLETE")
print("=" * 80)

