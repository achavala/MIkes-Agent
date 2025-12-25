#!/usr/bin/env python3
"""
Detailed analysis of why no trades occurred on Dec 23, 2025
Analyzes both agent_output.log and logs/mike_agent_safe_20251223.log
"""

import re
from datetime import datetime
import pytz
from collections import defaultdict, Counter

def analyze_dec23():
    """Analyze Dec 23rd logs for trade activity"""
    
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).date()
    
    print("="*80)
    print(f"DETAILED ANALYSIS: WHY NO TRADES ON DEC 23, 2025")
    print("="*80)
    print()
    
    # Check agent_output.log
    print("="*80)
    print("1. CHECKING agent_output.log")
    print("="*80)
    
    agent_output_data = []
    try:
        with open('agent_output.log', 'r') as f:
            for line in f:
                if '2025-12-23' in line or 'Dec 23' in line:
                    agent_output_data.append(line.strip())
    except FileNotFoundError:
        print("  agent_output.log not found")
    
    if agent_output_data:
        print(f"  Found {len(agent_output_data)} lines from Dec 23 in agent_output.log")
        print(f"  First 10 lines:")
        for line in agent_output_data[:10]:
            print(f"    {line[:150]}")
    else:
        print("  ⚠️  No Dec 23 data found in agent_output.log")
        print("  Checking last 20 lines of agent_output.log:")
        try:
            with open('agent_output.log', 'r') as f:
                lines = f.readlines()
                for line in lines[-20:]:
                    print(f"    {line.strip()[:150]}")
        except:
            pass
    print()
    
    # Check logs/mike_agent_safe_20251223.log
    print("="*80)
    print("2. CHECKING logs/mike_agent_safe_20251223.log")
    print("="*80)
    
    logfile = f"logs/mike_agent_safe_{today.strftime('%Y%m%d')}.log"
    
    try:
        with open(logfile, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"  ❌ Log file not found: {logfile}")
        return
    
    print(f"  Total log lines: {len(lines)}")
    print()
    
    # Extract key information
    rl_inferences = []
    hold_decisions = []
    blocked_trades = []
    confidence_scores = []
    action_strengths = []
    symbol_actions = []
    price_logs = []
    market_status = []
    option_issues = []
    gatekeeper_blocks = []
    data_issues = []
    ensemble_outputs = []
    trades_executed = []
    
    # Process each line - look for Dec 23 entries
    for i, line in enumerate(lines, 1):
        # Only process lines from Dec 23 (09:00-16:00 EST market hours)
        if '2025-12-23' not in line:
            continue
        
        # Check if it's during market hours (09:00-16:00)
        time_match = re.search(r'(\d{2}):(\d{2}):', line)
        if time_match:
            hour = int(time_match.group(1))
            if hour < 9 or hour >= 16:
                continue  # Skip non-market hours
        
        # RL Inference
        if '🧠' in line and ('RL Inference' in line or 'RL Action' in line or 'RL Debug' in line):
            rl_inferences.append((i, line.strip()))
        
        # Hold decisions
        if '🤔' in line or ('HOLD' in line and ('Multi-Symbol' in line or 'All HOLD' in line)):
            hold_decisions.append((i, line.strip()))
        
        # Blocked trades
        if '⛔' in line or 'BLOCKED' in line.upper():
            blocked_trades.append((i, line.strip()))
        
        # Confidence scores
        conf_match = re.search(r'confidence[:\s]+(\d+\.\d+)|Strength[:\s]+(\d+\.\d+)|strength[:\s]+(\d+\.\d+)', line, re.IGNORECASE)
        if conf_match:
            try:
                conf = float(conf_match.group(1) or conf_match.group(2) or conf_match.group(3))
                confidence_scores.append((i, conf, line.strip()))
            except:
                pass
        
        # Action strength
        strength_match = re.search(r'action_strength[:\s]+(\d+\.\d+)|Strength[:\s]+(\d+\.\d+)', line, re.IGNORECASE)
        if strength_match:
            try:
                strength = float(strength_match.group(1) or strength_match.group(2))
                action_strengths.append((i, strength, line.strip()))
            except:
                pass
        
        # Symbol actions
        if ('SPY' in line or 'QQQ' in line) and ('Action' in line or 'action=' in line):
            symbol_actions.append((i, line.strip()))
        
        # Price logs (status updates)
        if ('SPY:' in line or 'QQQ:' in line) and '$' in line and ('VIX:' in line or 'Action:' in line):
            price_logs.append((i, line.strip()))
        
        # Market status
        if 'Market' in line and ('OPEN' in line or 'CLOSED' in line):
            market_status.append((i, line.strip()))
        if 'clock.is_open' in line or 'ALPACA_CLOCK' in line:
            market_status.append((i, line.strip()))
        
        # Option issues
        if 'option' in line.lower() and ('available' in line.lower() or 'tradeable' in line.lower() or 'no' in line.lower()):
            option_issues.append((i, line.strip()))
        
        # Gatekeeper blocks
        if 'spread' in line.lower() or 'expected move' in line.lower() or 'breakeven' in line.lower():
            gatekeeper_blocks.append((i, line.strip()))
        
        # Data issues
        if 'stale' in line.lower() or 'data.*failed' in line.lower() or 'insufficient' in line.lower():
            data_issues.append((i, line.strip()))
        
        # Ensemble outputs
        if 'Ensemble' in line or 'Meta-Router' in line or 'Combined Signal' in line:
            ensemble_outputs.append((i, line.strip()))
        
        # Trades executed
        if 'EXECUTED' in line or '✓ EXECUTED' in line or 'TRADE' in line and 'BUY' in line:
            trades_executed.append((i, line.strip()))
    
    # Analysis
    print("="*80)
    print("3. DETAILED FINDINGS")
    print("="*80)
    print()
    
    print(f"RL Inferences: {len(rl_inferences)}")
    if rl_inferences:
        print(f"  Last 10 RL inferences:")
        for line_num, line in rl_inferences[-10:]:
            print(f"    Line {line_num}: {line[:120]}")
    else:
        print("  ⚠️  NO RL INFERENCES FOUND")
    print()
    
    print(f"Hold Decisions: {len(hold_decisions)}")
    if hold_decisions:
        print(f"  Last 10 HOLD decisions:")
        for line_num, line in hold_decisions[-10:]:
            print(f"    Line {line_num}: {line[:120]}")
    print()
    
    print(f"Blocked Trades: {len(blocked_trades)}")
    if blocked_trades:
        # Categorize blocks
        block_reasons = Counter()
        for line_num, line in blocked_trades:
            if 'confidence' in line.lower() or 'strength' in line.lower():
                block_reasons['Confidence too low'] += 1
            elif 'spread' in line.lower():
                block_reasons['Spread too wide'] += 1
            elif 'expected move' in line.lower() or 'breakeven' in line.lower():
                block_reasons['Expected move < breakeven'] += 1
            elif 'option' in line.lower() and 'available' in line.lower():
                block_reasons['No tradeable options'] += 1
            elif 'cooldown' in line.lower():
                block_reasons['Cooldown active'] += 1
            elif 'max' in line.lower() and 'trade' in line.lower():
                block_reasons['Max trades reached'] += 1
            else:
                block_reasons['Other'] += 1
        
        print(f"  Block reasons breakdown:")
        for reason, count in block_reasons.most_common():
            print(f"    {reason}: {count}")
        print(f"  Last 10 blocked trades:")
        for line_num, line in blocked_trades[-10:]:
            print(f"    Line {line_num}: {line[:120]}")
    print()
    
    print(f"Confidence Scores: {len(confidence_scores)}")
    if confidence_scores:
        confs = [c for _, c, _ in confidence_scores]
        print(f"  Min: {min(confs):.3f}, Max: {max(confs):.3f}, Avg: {sum(confs)/len(confs):.3f}")
        print(f"  Below 0.60: {sum(1 for c in confs if c < 0.60)}")
        print(f"  At/Above 0.60: {sum(1 for c in confs if c >= 0.60)}")
    print()
    
    print(f"Price/Status Logs: {len(price_logs)}")
    if price_logs:
        print(f"  Last 10 price/status logs:")
        for line_num, line in price_logs[-10:]:
            print(f"    Line {line_num}: {line[:120]}")
    print()
    
    print(f"Market Status: {len(market_status)}")
    if market_status:
        print(f"  Last 10 market status checks:")
        for line_num, line in market_status[-10:]:
            print(f"    Line {line_num}: {line[:120]}")
    else:
        print("  ⚠️  NO MARKET STATUS CHECKS FOUND")
    print()
    
    print(f"Trades Executed: {len(trades_executed)}")
    if trades_executed:
        print(f"  Trades found:")
        for line_num, line in trades_executed:
            print(f"    Line {line_num}: {line[:120]}")
    else:
        print("  ⚠️  NO TRADES EXECUTED")
    print()
    
    # Summary
    print("="*80)
    print("4. SUMMARY & ROOT CAUSE")
    print("="*80)
    print()
    
    if len(trades_executed) == 0:
        print("🔴 NO TRADES EXECUTED TODAY")
        print()
        
        if len(rl_inferences) == 0:
            print("  ❌ PRIMARY ISSUE: NO RL INFERENCES")
            print("     - Agent is not running RL inference")
            print("     - Possible causes:")
            print("       1. Agent stuck in data fetching loop")
            print("       2. Agent not reaching inference code")
            print("       3. Market detected as closed")
            print()
        
        if len(hold_decisions) > 0:
            print(f"  🟡 SECONDARY ISSUE: {len(hold_decisions)} HOLD DECISIONS")
            print("     - RL model is outputting HOLD (action 0)")
            print("     - This is correct if no good setups")
            print()
        
        if len(blocked_trades) > 0:
            print(f"  🟡 TERTIARY ISSUE: {len(blocked_trades)} TRADES BLOCKED")
            print("     - Trades were attempted but blocked by gates")
            print()
        
        if len(option_issues) > 0:
            print(f"  🟡 OPTION UNIVERSE ISSUE: {len(option_issues)} instances")
            print("     - Option universe filter finding no tradeable options")
            print()
    
    print("="*80)

if __name__ == "__main__":
    analyze_dec23()


