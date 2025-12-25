#!/usr/bin/env python3
"""
🔍 VALIDATE AGENT IS ACTIVE AND TRADING

Checks if agent is running, making decisions, and why trades might not be firing
"""

import subprocess
import sys
from datetime import datetime
import pytz

def check_fly_logs():
    """Check Fly.io logs for agent activity"""
    print("📊 Checking Fly.io logs for agent activity...")
    print()
    
    try:
        # Get recent logs
        result = subprocess.run(
            ['fly', 'logs', '--app', 'mike-agent-project', '--no-tail'],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode != 0:
            print("⚠️  Could not fetch logs")
            return
        
        logs = result.stdout
        
        # Check for agent startup
        if "Starting trading agent" in logs or "Agent started" in logs:
            print("✅ Agent startup detected in logs")
        else:
            print("⚠️  No agent startup message found")
        
        # Check for model loading
        if "Model loaded" in logs or "Loading RL model" in logs:
            print("✅ Model loading detected")
        else:
            print("⚠️  No model loading message found")
        
        # Check for Alpaca connection
        if "Connected to Alpaca" in logs:
            print("✅ Alpaca connection detected")
        else:
            print("⚠️  No Alpaca connection message found")
        
        # Check for RL decisions
        if "RL Decision" in logs or "Action" in logs:
            print("✅ RL decisions detected")
        else:
            print("⚠️  No RL decision messages found")
        
        # Check for blocks
        block_count = logs.count("BLOCKED")
        if block_count > 0:
            print(f"⚠️  Found {block_count} BLOCKED messages")
            # Show recent blocks
            lines = logs.split('\n')
            recent_blocks = [l for l in lines if 'BLOCKED' in l][-5:]
            if recent_blocks:
                print("   Recent blocks:")
                for block in recent_blocks:
                    print(f"   - {block[:100]}")
        else:
            print("ℹ️  No BLOCKED messages found (agent may not be making decisions)")
        
        # Check for executions
        exec_count = logs.count("EXECUTED") + logs.count("Order submitted")
        if exec_count > 0:
            print(f"✅ Found {exec_count} execution messages")
        else:
            print("⚠️  No execution messages found")
        
        # Check for errors
        error_count = logs.count("ERROR") + logs.count("Error") + logs.count("Failed")
        if error_count > 0:
            print(f"⚠️  Found {error_count} error messages")
            # Show recent errors
            lines = logs.split('\n')
            recent_errors = [l for l in lines if any(x in l for x in ['ERROR', 'Error', 'Failed'])][-5:]
            if recent_errors:
                print("   Recent errors:")
                for err in recent_errors:
                    print(f"   - {err[:100]}")
        
        print()
        print("=" * 70)
        print("📋 RECOMMENDATIONS:")
        print("=" * 70)
        
        if "No agent startup" in str(locals()):
            print("❌ Agent may not be running")
            print("   → Check: fly ssh console --app mike-agent-project")
            print("   → Check: fly logs --app mike-agent-project")
        
        if "No RL decision" in str(locals()):
            print("⚠️  Agent may not be making decisions")
            print("   → Check: Market hours (9:30 AM - 4:00 PM EST)")
            print("   → Check: Data collection working")
            print("   → Check: Model loaded correctly")
        
        if block_count > 0:
            print("⚠️  Trades are being blocked")
            print("   → Check: Safeguards (VIX, position limits, cooldowns)")
            print("   → Check: Confidence threshold (MIN_ACTION_STRENGTH_THRESHOLD = 0.65)")
        
        if exec_count == 0:
            print("⚠️  No trades executed today")
            print("   → This could be normal if:")
            print("     - Market conditions don't meet criteria")
            print("     - Confidence threshold too high (0.65)")
            print("     - Safeguards blocking all signals")
            print("     - Agent waiting for better setups")
        
    except subprocess.TimeoutExpired:
        print("⚠️  Log fetch timed out")
    except Exception as e:
        print(f"❌ Error checking logs: {e}")

def check_market_status():
    """Check if market is open"""
    est = pytz.timezone('US/Eastern')
    now_est = datetime.now(est)
    market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)
    is_open = market_open <= now_est <= market_close
    
    print("=" * 70)
    print("🕐 MARKET STATUS")
    print("=" * 70)
    print(f"Current Time (EST): {now_est.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Market Hours: 9:30 AM - 4:00 PM EST")
    print(f"Market Status: {'✅ OPEN' if is_open else '❌ CLOSED'}")
    print()
    
    if not is_open:
        print("⚠️  Market is CLOSED - agent will wait for market open")
        print("   Agent automatically starts trading at 9:30 AM EST")
    else:
        print("✅ Market is OPEN - agent should be active")
    print()

if __name__ == "__main__":
    print("=" * 70)
    print("🔍 AGENT ACTIVITY VALIDATION")
    print("=" * 70)
    print()
    
    check_market_status()
    check_fly_logs()
    
    print("=" * 70)
    print("✅ Validation complete")
    print("=" * 70)





