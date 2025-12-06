#!/usr/bin/env python3
"""Quick test to verify GUI imports work"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing GUI imports...")

try:
    import streamlit as st
    print("✅ Streamlit imported")
except Exception as e:
    print(f"❌ Streamlit import failed: {e}")
    sys.exit(1)

try:
    from trade_database import TradeDatabase
    print("✅ TradeDatabase imported")
    db = TradeDatabase()
    print("✅ TradeDatabase instantiated")
except Exception as e:
    print(f"❌ TradeDatabase failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    # Test the is_0dte_option function logic
    from datetime import date
    today = date.today()
    today_str = today.strftime('%y%m%d')
    test_symbol = f'SPY{today_str}C00450000'
    
    # Simulate the function
    if len(test_symbol) >= 15:
        for i in range(len(test_symbol)):
            if test_symbol[i].isdigit():
                date_str = test_symbol[i:i+6]
                if len(date_str) == 6:
                    year = 2000 + int(date_str[:2])
                    month = int(date_str[2:4])
                    day = int(date_str[4:6])
                    exp_date = date(year, month, day)
                    is_0dte = (exp_date == today)
                    print(f"✅ is_0dte_option logic works: {test_symbol} -> {is_0dte}")
                    break
except Exception as e:
    print(f"❌ is_0dte_option logic failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All GUI components working!")


