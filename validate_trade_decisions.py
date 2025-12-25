#!/usr/bin/env python3
"""
Validate trade decisions - shows what trades are being picked or rejected
"""
import subprocess
import re
from datetime import datetime

def analyze_logs():
    """Analyze Fly.io logs for trade decisions"""
    print("=" * 80)
    print("🔍 TRADE DECISION VALIDATION")
    print("=" * 80)
    print()
    
    print("📥 Fetching recent logs from Fly.io...")
    print()
    
    try:
        # Get last 200 lines of logs
        result = subprocess.run(
            ["fly", "logs", "--app", "mike-agent-project", "-n", "200"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        logs = result.stdout
        
        # Analyze different patterns
        patterns = {
            "RL Actions": r"RL.*action=(\d+).*strength=([\d.]+)",
            "Trade Signals": r"(BUY CALL|BUY PUT|HOLD)",
            "Blocked Trades": r"⛔ BLOCKED.*?Reason: (.*?)(?:\n|$)",
            "Executed Trades": r"✅.*?TRADE_OPENED|✓ EXECUTED.*?BUY",
            "Safeguard Checks": r"SAFEGUARD.*?TRIGGERED|check_safeguards",
            "Symbol Actions": r"SYMBOL SELECTION|choose_best_symbol",
            "Market Status": r"Market.*?(open|closed|waiting)",
        }
        
        findings = {}
        
        for category, pattern in patterns.items():
            matches = re.findall(pattern, logs, re.IGNORECASE | re.MULTILINE)
            findings[category] = matches
        
        # Print analysis
        print("=" * 80)
        print("📊 ANALYSIS RESULTS")
        print("=" * 80)
        print()
        
        # Market Status
        if findings.get("Market Status"):
            print("🕐 MARKET STATUS:")
            for match in findings["Market Status"][-5:]:
                print(f"   - {match}")
        else:
            print("🕐 MARKET STATUS: No recent market status logs")
        print()
        
        # RL Actions
        if findings.get("RL Actions"):
            print("🤖 RL MODEL DECISIONS (Last 10):")
            for action, strength in findings["RL Actions"][-10:]:
                action_name = {0: "HOLD", 1: "BUY CALL", 2: "BUY PUT"}.get(int(action), f"Action {action}")
                print(f"   - {action_name} (confidence: {float(strength):.2%})")
        else:
            print("🤖 RL MODEL DECISIONS: No recent RL actions found")
        print()
        
        # Blocked Trades
        if findings.get("Blocked Trades"):
            print("⛔ BLOCKED TRADES (Last 10):")
            for reason in findings["Blocked Trades"][-10:]:
                print(f"   - {reason.strip()}")
        else:
            print("⛔ BLOCKED TRADES: None found (or market closed)")
        print()
        
        # Executed Trades
        if findings.get("Executed Trades"):
            print("✅ EXECUTED TRADES (Last 10):")
            for trade in findings["Executed Trades"][-10:]:
                print(f"   - {trade[:100]}...")
        else:
            print("✅ EXECUTED TRADES: None found (or market closed)")
        print()
        
        # Safeguard Checks
        if findings.get("Safeguard Checks"):
            print("🛡️ SAFEGUARD CHECKS:")
            for check in findings["Safeguard Checks"][-5:]:
                print(f"   - {check}")
        else:
            print("🛡️ SAFEGUARD CHECKS: No recent safeguard triggers")
        print()
        
        # Summary
        print("=" * 80)
        print("📋 SUMMARY")
        print("=" * 80)
        print()
        print(f"RL Decisions Found: {len(findings.get('RL Actions', []))}")
        print(f"Blocked Trades: {len(findings.get('Blocked Trades', []))}")
        print(f"Executed Trades: {len(findings.get('Executed Trades', []))}")
        print()
        
        # Check if agent is running
        if "Starting agent" in logs or "Agent started" in logs:
            print("✅ Agent is running")
        else:
            print("⚠️  Agent status unclear from logs")
        
        if "Market closed" in logs or "Waiting for market open" in logs:
            print("ℹ️  Market appears to be closed (no trades expected)")
        
        print()
        print("=" * 80)
        print("💡 EXPLANATION")
        print("=" * 80)
        print()
        print("TRADE DECISION FLOW:")
        print("1. RL Model analyzes market data → outputs action (0=HOLD, 1=BUY CALL, 2=BUY PUT)")
        print("2. Safeguards check if trade is allowed (risk limits, cooldowns, etc.)")
        print("3. If blocked → ⛔ BLOCKED message with reason")
        print("4. If allowed → ✅ Trade executed")
        print()
        print("COMMON BLOCK REASONS:")
        print("- Max concurrent positions reached")
        print("- Daily loss limit hit")
        print("- VIX too high (crash mode)")
        print("- Cooldown period active")
        print("- Position size exceeds risk limits")
        print()
        
    except subprocess.TimeoutExpired:
        print("❌ Timeout fetching logs")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_logs()





