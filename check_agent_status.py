#!/usr/bin/env python3
"""
🔍 AGENT STATUS CHECKER

Quick diagnostic to check if agent is running and making decisions
"""

import os
import sys
from datetime import datetime
import pytz

# Check market status
est = pytz.timezone('US/Eastern')
now_est = datetime.now(est)
market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)
is_open = market_open <= now_est <= market_close

print("=" * 70)
print("🔍 AGENT STATUS CHECK")
print("=" * 70)
print(f"Current Time (EST): {now_est.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Market Status: {'✅ OPEN' if is_open else '❌ CLOSED'}")
print()

# Check if agent is running on Fly.io
print("📊 Checking Fly.io deployment...")
try:
    import subprocess
    result = subprocess.run(['fly', 'status', '--app', 'mike-agent-project'], 
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print("✅ Fly.io app is running")
        # Check for agent process
        if 'started' in result.stdout:
            print("✅ Machines are started")
    else:
        print("⚠️  Could not check Fly.io status")
except Exception as e:
    print(f"⚠️  Error checking Fly.io: {e}")

print()
print("📋 To check agent logs, run:")
print("   fly logs --app mike-agent-project | grep -E 'Agent|RL|Action|BLOCKED|EXECUTED'")
print()
print("📋 To check agent activity in real-time:")
print("   fly logs --app mike-agent-project")





