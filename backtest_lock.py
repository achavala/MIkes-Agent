#!/usr/bin/env python3
"""
Lock mechanism to prevent backtest from running during market hours
Ensures live agent has priority during trading hours
"""

import os
import sys
import time
from datetime import datetime
import pytz

LIVE_AGENT_LOCK_FILE = "/tmp/mike_agent_live.lock"
BACKTEST_LOCK_FILE = "/tmp/mike_agent_backtest.lock"

def is_market_open():
    """Check if market is currently open"""
    est = pytz.timezone('US/Eastern')
    now_est = datetime.now(est)
    
    # Check if weekday
    if now_est.weekday() >= 5:
        return False
    
    # Check market hours (9:30 AM - 4:00 PM EST)
    market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now_est < market_close

def check_live_agent_lock():
    """Check if live agent lock exists (live agent is running)"""
    return os.path.exists(LIVE_AGENT_LOCK_FILE)

def prevent_backtest_during_market_hours():
    """Prevent backtest from running during market hours"""
    if is_market_open():
        if check_live_agent_lock():
            print("="*80)
            print("🚨 BACKTEST BLOCKED: Live agent is running during market hours")
            print("="*80)
            print()
            print("The live trading agent is currently running and has priority.")
            print("Backtests should only run when:")
            print("  1. Market is closed (before 9:30 AM or after 4:00 PM EST)")
            print("  2. Live agent is not running")
            print()
            print("To run backtest:")
            print("  1. Wait for market to close (after 4:00 PM EST)")
            print("  2. Or stop the live agent first")
            print()
            sys.exit(1)
        else:
            print("⚠️  WARNING: Market is open but live agent is not running!")
            print("   Starting backtest during market hours is not recommended.")
            print("   Consider starting the live agent instead.")
            print()
            response = input("Continue with backtest anyway? (yes/no): ")
            if response.lower() != 'yes':
                print("Backtest cancelled.")
                sys.exit(0)

if __name__ == "__main__":
    prevent_backtest_during_market_hours()


