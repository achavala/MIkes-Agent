#!/usr/bin/env python3
"""
Detailed analysis of why no trades occurred today
Extracts all setups, rejections, and reasons from today's log file
"""

import re
from datetime import datetime
import pytz
from collections import defaultdict, Counter

def analyze_todays_log():
    """Analyze today's log file for trade attempts and rejections"""
    
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
    
    # Extract key information
    rl_actions = []
    rejections = []
    gatekeeper_blocks = []
    option_universe_issues = []
    hold_decisions = []
    market_status = []
    confidence_scores = []
    data_issues = []
    
    # Patterns
    for i, line in enumerate(lines, 1):
        # RL Actions
        if re.search(r'RL Action.*\d+|action.*\d+.*strength|confidence.*0\.\d+', line, re.IGNORECASE):
            rl_actions.append((i, line.strip()))
        
        # Rejections/Blocks
        if re.search(r'BLOCKED|⛔|REJECT|not allowed|veto', line, re.IGNORECASE):
            rejections.append((i, line.strip()))
        
        # Gatekeeper specific
        if re.search(r'spread.*%|expected move|breakeven|confidence.*<|gate', line, re.IGNORECASE):
            gatekeeper_blocks.append((i, line.strip()))
        
        # Option universe
        if re.search(r'tradeable.*option|no.*option.*available|option.*filter|option.*universe', line, re.IGNORECASE):
            option_universe_issues.append((i, line.strip()))
        
        # Hold decisions
        if re.search(r'HOLD|hold.*decision|all.*hold', line, re.IGNORECASE):
            hold_decisions.append((i, line.strip()))
        
        # Market status
        if re.search(r'Market.*OPEN|Market.*CLOSED|clock\.is_open|market.*status', line, re.IGNORECASE):
            market_status.append((i, line.strip()))
        
        # Confidence scores
        conf_match = re.search(r'confidence.*?(\d+\.\d+)', line, re.IGNORECASE)
        if conf_match:
            try:
                conf = float(conf_match.group(1))
                confidence_scores.append((i, conf, line.strip()))
            except:
                pass
        
        # Data issues
        if re.search(r'stale|data.*failed|insufficient|no.*data|empty.*data', line, re.IGNORECASE):
            data_issues.append((i, line.strip()))
    
    # Analysis
    print("="*80)
    print("1. MARKET STATUS")
    print("="*80)
    if market_status:
        print(f"   Found {len(market_status)} market status checks")
        for line_num, line in market_status[-10:]:  # Last 10
            print(f"   Line {line_num}: {line[:120]}")
    else:
        print("   ⚠️  No market status found in logs")
    print()
    
    print("="*80)
    print("2. RL MODEL OUTPUTS")
    print("="*80)
    print(f"   Total RL action logs: {len(rl_actions)}")
    if rl_actions:
        print(f"   Last 10 RL actions:")
        for line_num, line in rl_actions[-10:]:
            print(f"   Line {line_num}: {line[:120]}")
    else:
        print("   ⚠️  No RL actions found in logs")
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
        print(f"   Scores < 0.60 (threshold): {sum(1 for c in confs if c < 0.60)}")
        print(f"   Scores >= 0.60: {sum(1 for c in confs if c >= 0.60)}")
        print()
        print(f"   Last 10 confidence scores:")
        for line_num, conf, line in confidence_scores[-10:]:
            status = "✅ ABOVE" if conf >= 0.60 else "❌ BELOW"
            print(f"   Line {line_num}: {status} threshold ({conf:.3f}) - {line[:100]}")
    else:
        print("   ⚠️  No confidence scores found")
    print()
    
    print("="*80)
    print("4. HOLD DECISIONS")
    print("="*80)
    print(f"   Total HOLD decisions: {len(hold_decisions)}")
    if hold_decisions:
        print(f"   Last 10 HOLD decisions:")
        for line_num, line in hold_decisions[-10:]:
            print(f"   Line {line_num}: {line[:120]}")
    print()
    
    print("="*80)
    print("5. REJECTIONS/BLOCKS")
    print("="*80)
    print(f"   Total rejections/blocks: {len(rejections)}")
    if rejections:
        # Categorize rejections
        rejection_reasons = Counter()
        for line_num, line in rejections:
            if 'spread' in line.lower():
                rejection_reasons['Spread too wide'] += 1
            elif 'confidence' in line.lower():
                rejection_reasons['Confidence too low'] += 1
            elif 'expected move' in line.lower() or 'breakeven' in line.lower():
                rejection_reasons['Expected move < breakeven'] += 1
            elif 'option' in line.lower() and 'available' in line.lower():
                rejection_reasons['No tradeable options'] += 1
            elif 'cooldown' in line.lower():
                rejection_reasons['Cooldown active'] += 1
            elif 'max' in line.lower() and 'trade' in line.lower():
                rejection_reasons['Max trades reached'] += 1
            else:
                rejection_reasons['Other'] += 1
        
        print(f"   Rejection breakdown:")
        for reason, count in rejection_reasons.most_common():
            print(f"     {reason}: {count}")
        print()
        print(f"   Last 10 rejections:")
        for line_num, line in rejections[-10:]:
            print(f"   Line {line_num}: {line[:120]}")
    else:
        print("   ✅ No explicit rejections found (may be HOLD decisions)")
    print()
    
    print("="*80)
    print("6. GATEKEEPER BLOCKS")
    print("="*80)
    print(f"   Total gatekeeper blocks: {len(gatekeeper_blocks)}")
    if gatekeeper_blocks:
        print(f"   Last 10 gatekeeper blocks:")
        for line_num, line in gatekeeper_blocks[-10:]:
            print(f"   Line {line_num}: {line[:120]}")
    print()
    
    print("="*80)
    print("7. OPTION UNIVERSE ISSUES")
    print("="*80)
    print(f"   Total option universe issues: {len(option_universe_issues)}")
    if option_universe_issues:
        print(f"   Last 10 option universe issues:")
        for line_num, line in option_universe_issues[-10:]:
            print(f"   Line {line_num}: {line[:120]}")
    print()
    
    print("="*80)
    print("8. DATA ISSUES")
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
    print("SUMMARY")
    print("="*80)
    print()
    print("Key Findings:")
    print(f"  - RL Actions logged: {len(rl_actions)}")
    print(f"  - Confidence scores: {len(confidence_scores)}")
    print(f"  - HOLD decisions: {len(hold_decisions)}")
    print(f"  - Explicit rejections: {len(rejections)}")
    print(f"  - Gatekeeper blocks: {len(gatekeeper_blocks)}")
    print(f"  - Option universe issues: {len(option_universe_issues)}")
    print(f"  - Data issues: {len(data_issues)}")
    print()
    
    if confidence_scores:
        low_conf = sum(1 for _, c, _ in confidence_scores if c < 0.60)
        high_conf = sum(1 for _, c, _ in confidence_scores if c >= 0.60)
        print(f"Confidence Analysis:")
        print(f"  - Scores below 0.60 threshold: {low_conf} ({low_conf/len(confidence_scores)*100:.1f}%)")
        print(f"  - Scores at/above 0.60 threshold: {high_conf} ({high_conf/len(confidence_scores)*100:.1f}%)")
        print()
    
    if option_universe_issues:
        print(f"⚠️  OPTION UNIVERSE ISSUES DETECTED:")
        print(f"   {len(option_universe_issues)} instances of 'no tradeable options'")
        print(f"   This is likely the primary reason for no trades")
        print()
    
    print("="*80)

if __name__ == "__main__":
    analyze_todays_log()
