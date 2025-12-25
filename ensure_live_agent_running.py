#!/usr/bin/env python3
"""
Watchdog script to ensure live trading agent is always running during market hours
Prevents backtest from interfering with live trading
"""

import os
import sys
import time
import subprocess
import signal
from datetime import datetime
import pytz

# Configuration
LIVE_AGENT_SCRIPT = "mike_agent_live_safe.py"
BACKTEST_SCRIPTS = ["run_phase0.py", "phase0_backtest/run_phase0.py"]
LOCK_FILE = "/tmp/mike_agent_live.lock"
PID_FILE = "/tmp/mike_agent_live.pid"
CHECK_INTERVAL = 60  # Check every 60 seconds

def is_market_open():
    """Check if market is currently open (9:30 AM - 4:00 PM EST, Mon-Fri)"""
    est = pytz.timezone('US/Eastern')
    now_est = datetime.now(est)
    
    # Check if weekday (Monday=0, Friday=4)
    if now_est.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check market hours (9:30 AM - 4:00 PM EST)
    market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now_est < market_close

def is_live_agent_running():
    """Check if live agent process is running"""
    # Check if lock file exists (indicates live agent is running)
    if os.path.exists(LOCK_FILE):
        # Verify PID file exists and process is actually running
        if os.path.exists(PID_FILE):
            try:
                with open(PID_FILE, 'r') as f:
                    pid = int(f.read().strip())
                
                # Check if process exists using kill -0 (doesn't actually kill)
                # This is a standard Unix way to check if a process exists
                try:
                    os.kill(pid, 0)  # Signal 0 doesn't kill, just checks existence
                    return True
                except (OSError, ProcessLookupError):
                    # Process doesn't exist, clean up stale files
                    try:
                        os.remove(LOCK_FILE)
                        os.remove(PID_FILE)
                    except:
                        pass
                    return False
            except (ValueError, FileNotFoundError):
                return False
        return True  # Lock file exists, assume running
    
    return False

def is_backtest_running():
    """Check if any backtest script is running"""
    try:
        # Use pgrep to find backtest processes (works on Unix/Mac)
        for backtest_script in BACKTEST_SCRIPTS:
            script_name = os.path.basename(backtest_script)
            result = subprocess.run(
                ['pgrep', '-f', script_name],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                # Found process, get PID
                pids = result.stdout.strip().split('\n')
                if pids and pids[0]:
                    return True, int(pids[0])
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        # pgrep not available or error, try alternative method
        try:
            # Alternative: use ps and grep
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    for backtest_script in BACKTEST_SCRIPTS:
                        script_name = os.path.basename(backtest_script)
                        if script_name in line and 'python' in line.lower():
                            # Extract PID (second column in ps aux output)
                            parts = line.split()
                            if len(parts) > 1:
                                try:
                                    return True, int(parts[1])
                                except (ValueError, IndexError):
                                    continue
        except (subprocess.SubprocessError, ValueError):
            pass
    
    return False, None

def kill_backtest(pid):
    """Kill backtest process to free up resources for live agent"""
    try:
        print(f"⚠️  Killing backtest process {pid} to allow live agent to run")
        # Try graceful termination first
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        
        # Check if process still exists
        try:
            os.kill(pid, 0)  # Check if process exists
            # Process still running, force kill
            print(f"⚠️  Process {pid} still running, force killing...")
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
        except (OSError, ProcessLookupError):
            # Process already terminated
            pass
        
        return True
    except (OSError, ProcessLookupError) as e:
        print(f"⚠️  Could not kill backtest process {pid}: {e}")
        return False

def start_live_agent():
    """Start the live trading agent"""
    if os.path.exists(LOCK_FILE):
        print(f"⚠️  Lock file exists: {LOCK_FILE}")
        return False
    
    # Create lock file
    with open(LOCK_FILE, 'w') as f:
        f.write(f"Live agent lock - {datetime.now().isoformat()}\n")
    
    try:
        # Start live agent in background
        log_file = f"logs/live_agent_{datetime.now().strftime('%Y%m%d')}.log"
        os.makedirs("logs", exist_ok=True)
        
        with open(log_file, 'a') as log:
            proc = subprocess.Popen(
                [sys.executable, LIVE_AGENT_SCRIPT],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd()
            )
        
        # Save PID
        with open(PID_FILE, 'w') as f:
            f.write(str(proc.pid))
        
        print(f"✅ Started live agent (PID: {proc.pid})")
        print(f"   Logs: {log_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to start live agent: {e}")
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
        return False

def main():
    """Main watchdog loop"""
    print("="*80)
    print("LIVE AGENT WATCHDOG - Ensuring Live Agent Runs During Market Hours")
    print("="*80)
    print()
    print(f"Live Agent Script: {LIVE_AGENT_SCRIPT}")
    print(f"Check Interval: {CHECK_INTERVAL} seconds")
    print(f"Lock File: {LOCK_FILE}")
    print(f"PID File: {PID_FILE}")
    print()
    
    # Handle SIGTERM gracefully
    def signal_handler(sig, frame):
        print("\n🛑 Watchdog shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    while True:
        try:
            est = pytz.timezone('US/Eastern')
            now_est = datetime.now(est)
            
            # Check if market is open
            market_open = is_market_open()
            
            if market_open:
                # Market is open - ensure live agent is running
                backtest_running, backtest_pid = is_backtest_running()
                
                if backtest_running:
                    print(f"⚠️  [{now_est.strftime('%H:%M:%S')}] Backtest detected (PID: {backtest_pid})")
                    kill_backtest(backtest_pid)
                    time.sleep(2)  # Wait for backtest to terminate
                
                if not is_live_agent_running():
                    print(f"🚨 [{now_est.strftime('%H:%M:%S')}] Live agent NOT running during market hours!")
                    print(f"    Starting live agent...")
                    start_live_agent()
                else:
                    # Live agent is running - all good
                    if int(time.time()) % 300 == 0:  # Log every 5 minutes
                        print(f"✅ [{now_est.strftime('%H:%M:%S')}] Live agent is running (market open)")
            else:
                # Market is closed - log status but don't interfere
                if int(time.time()) % 600 == 0:  # Log every 10 minutes when closed
                    if is_live_agent_running():
                        print(f"ℹ️  [{now_est.strftime('%H:%M:%S')}] Live agent running (market closed)")
                    else:
                        print(f"ℹ️  [{now_est.strftime('%H:%M:%S')}] Market closed - live agent not required")
            
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n🛑 Watchdog stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in watchdog loop: {e}")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

