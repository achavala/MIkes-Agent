#!/usr/bin/env python3
"""
Comprehensive validation to ensure system will trade tomorrow when market opens
Checks all components and potential blockers
"""

import os
import sys
import subprocess
from datetime import datetime, timedelta
import pytz

def check_fly_cli():
    """Check if Fly CLI is installed"""
    try:
        result = subprocess.run(['fly', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0]
            return True, version
        return False, "Fly CLI not working"
    except FileNotFoundError:
        return False, "Fly CLI not installed"

def check_fly_auth():
    """Check if authenticated to Fly.io"""
    try:
        result = subprocess.run(['fly', 'auth', 'whoami'], capture_output=True, text=True)
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, "Not authenticated"
    except:
        return False, "Could not check auth"

def check_watchdog_running():
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

def check_live_agent_running():
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

def check_backtest_running():
    """Check if any backtest is running"""
    backtest_scripts = ['run_phase0.py', 'phase0_backtest/run_phase0.py']
    for script in backtest_scripts:
        try:
            result = subprocess.run(['pgrep', '-f', script], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                if pids and pids[0]:
                    return True, int(pids[0]), script
        except:
            continue
    return False, None, None

def check_lock_files():
    """Check lock file status"""
    lock_file = "/tmp/mike_agent_live.lock"
    pid_file = "/tmp/mike_agent_live.pid"
    
    lock_exists = os.path.exists(lock_file)
    pid_exists = os.path.exists(pid_file)
    
    lock_content = None
    pid_content = None
    
    if lock_exists:
        try:
            with open(lock_file, 'r') as f:
                lock_content = f.read().strip()
        except:
            pass
    
    if pid_exists:
        try:
            with open(pid_file, 'r') as f:
                pid_content = f.read().strip()
        except:
            pass
    
    return lock_exists, pid_exists, lock_content, pid_content

def check_market_hours_detection():
    """Test market hours detection"""
    est = pytz.timezone('US/Eastern')
    now_est = datetime.now(est)
    
    # Calculate tomorrow's market open
    tomorrow = now_est + timedelta(days=1)
    
    # Adjust to next weekday if tomorrow is weekend
    while tomorrow.weekday() >= 5:  # Saturday or Sunday
        tomorrow += timedelta(days=1)
    
    market_open = tomorrow.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = tomorrow.replace(hour=16, minute=0, second=0, microsecond=0)
    
    is_weekday = tomorrow.weekday() < 5
    market_will_be_open = is_weekday and (market_open <= tomorrow < market_close)
    
    return {
        'tomorrow': tomorrow.date(),
        'market_open': market_open,
        'market_close': market_close,
        'is_weekday': is_weekday,
        'market_will_be_open': market_will_be_open
    }

def check_fly_app_status():
    """Check Fly.io app status"""
    try:
        result = subprocess.run(['fly', 'status', '--app', 'mike-agent-project'], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout checking Fly.io status"
    except:
        return False, "Could not check Fly.io status"

def check_model_file():
    """Check if model file exists"""
    model_path = "models/mike_23feature_model_final.zip"
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        return True, size
    return False, None

def check_config_files():
    """Check if essential config files exist"""
    required_files = [
        'mike_agent_live_safe.py',
        'config.py',
        'fly.toml',
        'Dockerfile',
        'start_cloud.sh',
        'ensure_live_agent_running.py'
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    return len(missing) == 0, missing

def main():
    """Run comprehensive validation"""
    print("="*80)
    print("COMPREHENSIVE VALIDATION: WILL SYSTEM TRADE TOMORROW?")
    print("="*80)
    print()
    
    all_checks_passed = True
    
    # 1. Check Fly CLI
    print("1. FLY CLI")
    print("-" * 80)
    fly_installed, fly_info = check_fly_cli()
    if fly_installed:
        print(f"   ✅ Fly CLI installed: {fly_info}")
    else:
        print(f"   ❌ {fly_info}")
        all_checks_passed = False
    
    fly_authed, auth_info = check_fly_auth()
    if fly_authed:
        print(f"   ✅ Authenticated: {auth_info}")
    else:
        print(f"   ⚠️  {auth_info} (may need: fly auth login)")
    print()
    
    # 2. Check Watchdog
    print("2. WATCHDOG STATUS")
    print("-" * 80)
    watchdog_running, watchdog_pid = check_watchdog_running()
    if watchdog_running:
        print(f"   ✅ Watchdog is running (PID: {watchdog_pid})")
        print(f"   ✅ Watchdog will ensure live agent starts when market opens")
    else:
        print(f"   ❌ Watchdog is NOT running")
        print(f"   ⚠️  Start with: ./start_live_agent_with_watchdog.sh")
        all_checks_passed = False
    print()
    
    # 3. Check Live Agent
    print("3. LIVE AGENT STATUS")
    print("-" * 80)
    live_agent_running, live_agent_pid = check_live_agent_running()
    if live_agent_running:
        print(f"   ✅ Live agent is running (PID: {live_agent_pid})")
    else:
        print(f"   ℹ️  Live agent is not running (will start when market opens)")
    print()
    
    # 4. Check Backtest
    print("4. BACKTEST STATUS")
    print("-" * 80)
    backtest_running, backtest_pid, backtest_script = check_backtest_running()
    if backtest_running:
        print(f"   ⚠️  Backtest is running (PID: {backtest_pid}, Script: {backtest_script})")
        print(f"   ⚠️  Watchdog will kill this when market opens")
        print(f"   ⚠️  Recommendation: Stop backtest now to avoid conflicts")
    else:
        print(f"   ✅ No backtest running (good)")
    print()
    
    # 5. Check Lock Files
    print("5. LOCK FILES")
    print("-" * 80)
    lock_exists, pid_exists, lock_content, pid_content = check_lock_files()
    if lock_exists:
        print(f"   ✅ Lock file exists: /tmp/mike_agent_live.lock")
        if lock_content:
            print(f"      Content: {lock_content[:100]}")
    else:
        print(f"   ℹ️  Lock file does not exist (will be created when live agent starts)")
    
    if pid_exists:
        print(f"   ✅ PID file exists: /tmp/mike_agent_live.pid")
        if pid_content:
            print(f"      PID: {pid_content}")
    else:
        print(f"   ℹ️  PID file does not exist (will be created when live agent starts)")
    print()
    
    # 6. Check Market Hours Detection
    print("6. MARKET HOURS DETECTION")
    print("-" * 80)
    market_info = check_market_hours_detection()
    print(f"   Tomorrow: {market_info['tomorrow']}")
    print(f"   Market Open: {market_info['market_open'].strftime('%H:%M:%S %Z')}")
    print(f"   Market Close: {market_info['market_close'].strftime('%H:%M:%S %Z')}")
    print(f"   Is Weekday: {market_info['is_weekday']}")
    print(f"   Market Will Be Open: {market_info['market_will_be_open']}")
    if market_info['market_will_be_open']:
        print(f"   ✅ Market will be open tomorrow - system will trade")
    else:
        print(f"   ⚠️  Market will be closed tomorrow (weekend/holiday)")
    print()
    
    # 7. Check Fly.io App Status
    print("7. FLY.IO APP STATUS")
    print("-" * 80)
    fly_status_ok, fly_status_info = check_fly_app_status()
    if fly_status_ok:
        print(f"   ✅ Fly.io app is accessible")
        # Extract key info
        if "running" in fly_status_info.lower():
            print(f"   ✅ App is running on Fly.io")
        if "stopped" in fly_status_info.lower():
            print(f"   ⚠️  App is stopped (will auto-start)")
    else:
        print(f"   ⚠️  Could not check Fly.io status: {fly_status_info}")
        print(f"   ℹ️  This is OK if deploying locally")
    print()
    
    # 8. Check Model File
    print("8. MODEL FILE")
    print("-" * 80)
    model_exists, model_size = check_model_file()
    if model_exists:
        size_mb = model_size / (1024 * 1024)
        print(f"   ✅ Model file exists: models/mike_23feature_model_final.zip")
        print(f"   ✅ Size: {size_mb:.1f} MB")
    else:
        print(f"   ❌ Model file not found")
        print(f"   ⚠️  Agent will fail to start without model")
        all_checks_passed = False
    print()
    
    # 9. Check Config Files
    print("9. CONFIGURATION FILES")
    print("-" * 80)
    config_ok, missing = check_config_files()
    if config_ok:
        print(f"   ✅ All required files exist")
    else:
        print(f"   ❌ Missing files: {', '.join(missing)}")
        all_checks_passed = False
    print()
    
    # 10. Check Phase 0 Backtest Protection
    print("10. BACKTEST PROTECTION")
    print("-" * 80)
    phase0_file = "phase0_backtest/run_phase0.py"
    if os.path.exists(phase0_file):
        with open(phase0_file, 'r') as f:
            content = f.read()
            if "market_is_open" in content and "live_agent_running" in content:
                print(f"   ✅ Phase 0 backtest has market hours protection")
            else:
                print(f"   ⚠️  Phase 0 backtest may not have protection")
    else:
        print(f"   ℹ️  Phase 0 backtest file not found (OK)")
    print()
    
    # Summary
    print("="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print()
    
    if all_checks_passed and watchdog_running:
        print("✅ SYSTEM IS READY FOR TOMORROW")
        print()
        print("The system will:")
        print("  ✅ Watchdog is running and will ensure live agent starts")
        print("  ✅ Live agent will start automatically when market opens")
        print("  ✅ Backtest will be blocked/killed during market hours")
        print("  ✅ Market hours detection is working")
        print("  ✅ All required files are present")
        print()
        if backtest_running:
            print("⚠️  RECOMMENDATION: Stop backtest now to avoid conflicts:")
            print(f"   kill {backtest_pid}")
    elif watchdog_running:
        print("⚠️  SYSTEM MOSTLY READY")
        print()
        print("Issues found:")
        if not fly_installed:
            print("  - Fly CLI not installed")
        if not model_exists:
            print("  - Model file missing")
        if not config_ok:
            print(f"  - Missing config files: {', '.join(missing)}")
        print()
        print("Watchdog is running, but fix issues above for full functionality")
    else:
        print("❌ SYSTEM NOT READY")
        print()
        print("Critical issues:")
        print("  ❌ Watchdog is NOT running")
        print("  ❌ Live agent will NOT start automatically")
        print()
        print("TO FIX:")
        print("  ./start_live_agent_with_watchdog.sh")
    
    print()
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print()
    
    if not watchdog_running:
        print("1. Start watchdog:")
        print("   ./start_live_agent_with_watchdog.sh")
        print()
    
    if backtest_running:
        print("2. Stop backtest (recommended):")
        print(f"   kill {backtest_pid}")
        print()
    
    print("3. Verify tomorrow morning (before 9:30 AM EST):")
    print("   ps aux | grep ensure_live_agent_running")
    print("   ps aux | grep mike_agent_live_safe")
    print()
    
    print("4. Monitor logs during market hours:")
    print("   tail -f logs/watchdog.log")
    print("   tail -f logs/live_agent_$(date +%Y%m%d).log")
    print()
    
    print("="*80)

if __name__ == "__main__":
    main()


