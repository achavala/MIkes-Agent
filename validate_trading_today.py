#!/usr/bin/env python3
"""
Validate that system will trade TODAY when market opens
Comprehensive check of all components
"""

import os
import sys
import subprocess
from datetime import datetime
import pytz

def check_watchdog():
    """Check if watchdog is running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'ensure_live_agent_running.py'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            if pids and pids[0]:
                return True, int(pids[0])
        return False, None
    except:
        return False, None

def check_live_agent():
    """Check if live agent is running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'mike_agent_live_safe.py'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            if pids and pids[0]:
                return True, int(pids[0])
        return False, None
    except:
        return False, None

def check_backtest():
    """Check if backtest is running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'run_phase0'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            if pids and pids[0]:
                return True, int(pids[0])
        return False, None
    except:
        return False, None

def check_market_status():
    """Check current market status"""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    is_weekday = now.weekday() < 5
    market_is_open = is_weekday and (market_open <= now < market_close)
    
    time_until_open = None
    if now < market_open and is_weekday:
        time_until_open = (market_open - now).total_seconds() / 60
    
    return {
        'now': now,
        'is_weekday': is_weekday,
        'market_open': market_open,
        'market_close': market_close,
        'market_is_open': market_is_open,
        'time_until_open': time_until_open
    }

def main():
    """Main validation"""
    print("="*80)
    print("VALIDATION: WILL SYSTEM TRADE TODAY?")
    print("="*80)
    print()
    
    # Check market status
    market = check_market_status()
    print(f"Current Time: {market['now'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Is Weekday: {market['is_weekday']}")
    print(f"Market Open: {market['market_open'].strftime('%H:%M:%S %Z')}")
    print(f"Market Close: {market['market_close'].strftime('%H:%M:%S %Z')}")
    print()
    
    if market['market_is_open']:
        print("✅ MARKET IS CURRENTLY OPEN")
        print("   System should be trading NOW")
    elif market['time_until_open']:
        print(f"⏰ MARKET OPENS IN {market['time_until_open']:.0f} MINUTES")
        print(f"   System will start trading at {market['market_open'].strftime('%H:%M:%S %Z')}")
    else:
        print("⏸️  MARKET IS CLOSED")
    print()
    
    # Check watchdog
    print("="*80)
    print("1. WATCHDOG STATUS")
    print("="*80)
    watchdog_running, watchdog_pid = check_watchdog()
    if watchdog_running:
        print(f"✅ Watchdog is RUNNING (PID: {watchdog_pid})")
        print("   ✅ Will ensure live agent starts when market opens")
    else:
        print("❌ Watchdog is NOT running")
        print("   ⚠️  CRITICAL: Start watchdog now!")
        print("   Command: ./start_live_agent_with_watchdog.sh")
    print()
    
    # Check live agent
    print("="*80)
    print("2. LIVE AGENT STATUS")
    print("="*80)
    live_agent_running, live_agent_pid = check_live_agent()
    if live_agent_running:
        print(f"✅ Live agent is RUNNING (PID: {live_agent_pid})")
        print("   ✅ System is trading NOW")
    else:
        print("ℹ️  Live agent is NOT running")
        if market['market_is_open']:
            print("   ⚠️  MARKET IS OPEN - Agent should be running!")
            if watchdog_running:
                print("   ⚠️  Watchdog should have started it - check logs")
            else:
                print("   ⚠️  Watchdog is not running - start it now!")
        else:
            print("   ✅ Will start automatically when market opens")
    print()
    
    # Check backtest
    print("="*80)
    print("3. BACKTEST STATUS")
    print("="*80)
    backtest_running, backtest_pid = check_backtest()
    if backtest_running:
        if market['market_is_open']:
            print(f"⚠️  Backtest is running (PID: {backtest_pid})")
            print("   ⚠️  MARKET IS OPEN - Backtest should be killed!")
            print("   ⚠️  Watchdog should kill this - check if watchdog is working")
        else:
            print(f"ℹ️  Backtest is running (PID: {backtest_pid})")
            print("   ✅ OK - market is closed")
    else:
        print("✅ No backtest running (good)")
    print()
    
    # Check lock files
    print("="*80)
    print("4. LOCK FILES")
    print("="*80)
    lock_file = "/tmp/mike_agent_live.lock"
    pid_file = "/tmp/mike_agent_live.pid"
    
    lock_exists = os.path.exists(lock_file)
    pid_exists = os.path.exists(pid_file)
    
    if lock_exists:
        print(f"✅ Lock file exists: {lock_file}")
        if live_agent_running:
            print("   ✅ Matches running live agent")
        else:
            print("   ⚠️  Lock exists but agent not running (stale lock?)")
    else:
        print(f"ℹ️  Lock file does not exist")
        if market['market_is_open']:
            print("   ⚠️  Should exist if agent is running")
        else:
            print("   ✅ Will be created when agent starts")
    
    if pid_exists:
        print(f"✅ PID file exists: {pid_file}")
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            print(f"   PID: {pid}")
            if live_agent_running and pid == live_agent_pid:
                print("   ✅ Matches running agent")
        except:
            pass
    print()
    
    # Summary
    print("="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print()
    
    if market['market_is_open']:
        if live_agent_running:
            print("✅ SYSTEM IS TRADING NOW")
            print("   Live agent is running and market is open")
        elif watchdog_running:
            print("⚠️  MARKET IS OPEN BUT AGENT NOT RUNNING")
            print("   Watchdog should have started it")
            print("   Check: tail -f logs/watchdog.log")
        else:
            print("❌ MARKET IS OPEN BUT SYSTEM NOT RUNNING")
            print("   Start watchdog: ./start_live_agent_with_watchdog.sh")
    else:
        if watchdog_running:
            print("✅ SYSTEM IS READY FOR MARKET OPEN")
            print(f"   Watchdog will start agent at {market['market_open'].strftime('%H:%M:%S %Z')}")
        else:
            print("❌ WATCHDOG NOT RUNNING")
            print("   Start watchdog: ./start_live_agent_with_watchdog.sh")
    
    print()
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print()
    
    if not watchdog_running:
        print("1. START WATCHDOG NOW:")
        print("   ./start_live_agent_with_watchdog.sh")
        print()
    
    if market['market_is_open'] and not live_agent_running and watchdog_running:
        print("2. CHECK WATCHDOG LOGS:")
        print("   tail -f logs/watchdog.log")
        print()
    
    if backtest_running and market['market_is_open']:
        print("3. KILL BACKTEST (if watchdog doesn't):")
        print(f"   kill {backtest_pid}")
        print()
    
    print("4. MONITOR DURING MARKET HOURS:")
    print("   tail -f logs/watchdog.log")
    print("   tail -f logs/live_agent_$(date +%Y%m%d).log")
    print()
    
    print("="*80)

if __name__ == "__main__":
    main()


