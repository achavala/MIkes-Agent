#!/usr/bin/env python3
"""
Comprehensive analysis of why no trades occurred today
Extracts all setups, RL decisions, rejections, and reasons
"""

import re
from datetime import datetime
import pytz
from collections import defaultdict, Counter

def analyze_todays_log():
    """Comprehensive analysis of today's trading activity"""
    
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).date()
    logfile = f"logs/mike_agent_safe_{today.strftime('%Y%m%d')}.log"
    
    print("="*80)
    print(f"DETAILED ANALYSIS: WHY NO TRADES TODAY ({today})")
    print("="*80)
    print()
    
    try:
        with open(logfile, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Log file not found: {logfile}")
        return
    
    print(f"📊 Total log lines: {len(lines)}")
    print()
    
    # Extract all relevant information
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
    
    # Process each line
    for i, line in enumerate(lines, 1):
        # Only process lines from today (Dec 23)
        if '2025-12-23' not in line:
            continue
        
        # RL Inference
        if '🧠' in line and ('RL Inference' in line or 'RL Action' in line):
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
        if 'SPY' in line and ('Action' in line or 'action=' in line):
            symbol_actions.append((i, line.strip()))
        if 'QQQ' in line and ('Action' in line or 'action=' in line):
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
    
    # Analysis
    print("="*80)
    print("1. MARKET STATUS")
    print("="*80)
    if market_status:
        print(f"   Found {len(market_status)} market status checks")
        print(f"   Last 5 market status entries:")
        for line_num, line in market_status[-5:]:
            print(f"   Line {line_num}: {line[:120]}")
    else:
        print("   ⚠️  No market status found - agent may not be checking market open/close")
    print()
    
    print("="*80)
    print("2. RL MODEL INFERENCES")
    print("="*80)
    print(f"   Total RL inferences: {len(rl_inferences)}")
    if rl_inferences:
        print(f"   Last 10 RL inferences:")
        for line_num, line in rl_inferences[-10:]:
            print(f"   Line {line_num}: {line[:120]}")
    else:
        print("   ⚠️  NO RL INFERENCES FOUND - This is the problem!")
        print("   The agent is fetching data but not running RL inference")
    print()
    
    print("="*80)
    print("3. CONFIDENCE SCORES")
    print("="*80)
    if confidence_scores:
        confs = [c for _, c, _ in confidence_scores]
        print(f"   Total confidence scores: {len(confs)}")
        print(f"   Min: {min(confs):.3f}")
        print(f"   Max: {max(confs):.3f}")
        print(f"   Avg: {sum(confs)/len(confs):.3f}")
        print(f"   Scores < 0.60 (threshold): {sum(1 for c in confs if c < 0.60)} ({sum(1 for c in confs if c < 0.60)/len(confs)*100:.1f}%)")
        print(f"   Scores >= 0.60: {sum(1 for c in confs if c >= 0.60)} ({sum(1 for c in confs if c >= 0.60)/len(confs)*100:.1f}%)")
        print()
        print(f"   Last 10 confidence scores:")
        for line_num, conf, line in confidence_scores[-10:]:
            status = "✅ ABOVE" if conf >= 0.60 else "❌ BELOW"
            print(f"   Line {line_num}: {status} threshold ({conf:.3f}) - {line[:100]}")
    else:
        print("   ⚠️  No confidence scores found")
    print()
    
    print("="*80)
    print("4. ACTION STRENGTHS")
    print("="*80)
    if action_strengths:
        strengths = [s for _, s, _ in action_strengths]
        print(f"   Total action strengths: {len(strengths)}")
        print(f"   Min: {min(strengths):.3f}")
        print(f"   Max: {max(strengths):.3f}")
        print(f"   Avg: {sum(strengths)/len(strengths):.3f}")
        print(f"   Strengths < 0.60: {sum(1 for s in strengths if s < 0.60)}")
        print(f"   Strengths >= 0.60: {sum(1 for s in strengths if s >= 0.60)}")
    else:
        print("   ⚠️  No action strengths found")
    print()
    
    print("="*80)
    print("5. HOLD DECISIONS")
    print("="*80)
    print(f"   Total HOLD decisions: {len(hold_decisions)}")
    if hold_decisions:
        print(f"   Last 10 HOLD decisions:")
        for line_num, line in hold_decisions[-10:]:
            print(f"   Line {line_num}: {line[:120]}")
    print()
    
    print("="*80)
    print("6. BLOCKED TRADES")
    print("="*80)
    print(f"   Total blocked trades: {len(blocked_trades)}")
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
            elif 'symbol' in line.lower() and 'eligible' in line.lower():
                block_reasons['No eligible symbols'] += 1
            else:
                block_reasons['Other'] += 1
        
        print(f"   Block reasons breakdown:")
        for reason, count in block_reasons.most_common():
            print(f"     {reason}: {count}")
        print()
        print(f"   Last 10 blocked trades:")
        for line_num, line in blocked_trades[-10:]:
            print(f"   Line {line_num}: {line[:120]}")
    else:
        print("   ✅ No explicit blocks found (may be HOLD decisions)")
    print()
    
    print("="*80)
    print("7. SYMBOL ACTIONS")
    print("="*80)
    print(f"   Total symbol action logs: {len(symbol_actions)}")
    if symbol_actions:
        print(f"   Last 10 symbol actions:")
        for line_num, line in symbol_actions[-10:]:
            print(f"   Line {line_num}: {line[:120]}")
    else:
        print("   ⚠️  No symbol actions found")
    print()
    
    print("="*80)
    print("8. PRICE/STATUS LOGS")
    print("="*80)
    print(f"   Total price/status logs: {len(price_logs)}")
    if price_logs:
        print(f"   Last 10 price/status logs:")
        for line_num, line in price_logs[-10:]:
            print(f"   Line {line_num}: {line[:120]}")
    else:
        print("   ⚠️  No price/status logs found")
    print()
    
    print("="*80)
    print("9. OPTION UNIVERSE ISSUES")
    print("="*80)
    print(f"   Total option universe issues: {len(option_issues)}")
    if option_issues:
        print(f"   Last 10 option universe issues:")
        for line_num, line in option_issues[-10:]:
            print(f"   Line {line_num}: {line[:120]}")
    else:
        print("   ✅ No option universe issues found")
    print()
    
    print("="*80)
    print("10. GATEKEEPER BLOCKS")
    print("="*80)
    print(f"   Total gatekeeper blocks: {len(gatekeeper_blocks)}")
    if gatekeeper_blocks:
        print(f"   Last 10 gatekeeper blocks:")
        for line_num, line in gatekeeper_blocks[-10:]:
            print(f"   Line {line_num}: {line[:120]}")
    print()
    
    print("="*80)
    print("11. ENSEMBLE OUTPUTS")
    print("="*80)
    print(f"   Total ensemble outputs: {len(ensemble_outputs)}")
    if ensemble_outputs:
        print(f"   Last 10 ensemble outputs:")
        for line_num, line in ensemble_outputs[-10:]:
            print(f"   Line {line_num}: {line[:120]}")
    else:
        print("   ⚠️  No ensemble outputs found")
    print()
    
    print("="*80)
    print("12. DATA ISSUES")
    print("="*80)
    print(f"   Total data issues: {len(data_issues)}")
    if data_issues:
        print(f"   Last 10 data issues:")
        for line_num, line in data_issues[-10:]:
            print(f"   Line {line_num}: {line[:120]}")
    else:
        print("   ✅ No data issues found")
    print()
    
    # Summary
    print("="*80)
    print("SUMMARY & ROOT CAUSE ANALYSIS")
    print("="*80)
    print()
    
    if len(rl_inferences) == 0:
        print("🔴 PRIMARY ISSUE: NO RL INFERENCES FOUND")
        print("   - Agent is fetching data but not running RL inference")
        print("   - Possible causes:")
        print("     1. Agent stuck in data fetching loop")
        print("     2. RL model not loading")
        print("     3. Agent not reaching inference code")
        print("     4. Logging disabled for RL inference")
        print()
    
    if len(hold_decisions) > 0:
        print(f"🟡 SECONDARY ISSUE: {len(hold_decisions)} HOLD DECISIONS")
        print("   - RL model is outputting HOLD (action 0)")
        print("   - This is correct behavior if no good setups")
        print()
    
    if len(blocked_trades) > 0:
        print(f"🟡 TERTIARY ISSUE: {len(blocked_trades)} TRADES BLOCKED")
        print("   - Trades were attempted but blocked by gates")
        print("   - See breakdown above for reasons")
        print()
    
    if len(option_issues) > 0:
        print(f"🟡 OPTION UNIVERSE ISSUE: {len(option_issues)} instances")
        print("   - Option universe filter finding no tradeable options")
        print("   - This is likely the main blocker")
        print()
    
    if len(confidence_scores) > 0:
        low_conf_count = sum(1 for _, c, _ in confidence_scores if c < 0.60)
        if low_conf_count > 0:
            print(f"🟡 CONFIDENCE ISSUE: {low_conf_count}/{len(confidence_scores)} scores below 0.60 threshold")
            print("   - RL model confidence too low")
            print()
    
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()
    
    if len(rl_inferences) == 0:
        print("1. 🔴 CRITICAL: Check if agent is actually running RL inference")
        print("   - Check if model is loaded")
        print("   - Check if agent reaches inference code")
        print("   - Check logs for errors")
        print()
    
    if len(option_issues) > 0:
        print("2. 🟡 Check option universe filter")
        print("   - Verify option chain API is working")
        print("   - Check if options exist for today")
        print("   - Verify spread/liquidity filters aren't too strict")
        print()
    
    if len(hold_decisions) > 0:
        print("3. 🟡 RL model is outputting HOLD")
        print("   - This may be correct (no good setups)")
        print("   - Or model may need retraining")
        print()
    
    print("="*80)

if __name__ == "__main__":
    analyze_todays_log()


