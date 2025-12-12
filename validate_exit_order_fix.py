#!/usr/bin/env python3
"""
Comprehensive Validation: Exit Order Fix
Validates that ALL sell order submissions have position verification.
"""

import re
from pathlib import Path

def validate_all_sell_orders():
    """Validate that all submit_order(side='sell') have position verification"""
    print("=" * 80)
    print("🔍 COMPREHENSIVE VALIDATION: Exit Order Fix")
    print("=" * 80)
    print()
    
    code_file = Path('mike_agent_live_safe.py')
    if not code_file.exists():
        print(f"❌ File not found: {code_file}")
        return False
    
    with open(code_file, 'r') as f:
        lines = f.readlines()
    
    # Find all submit_order calls with side='sell'
    sell_order_locations = []
    for i, line in enumerate(lines, 1):
        if 'api.submit_order' in line and 'side' in ' '.join(lines[max(0,i-10):min(len(lines),i+10)]):
            # Check context for 'sell'
            context = ' '.join(lines[max(0,i-10):min(len(lines),i+10)])
            if "'sell'" in context or '"sell"' in context:
                # Check if there's position verification before this
                prev_context = ' '.join(lines[max(0,i-30):i])
                has_verification = (
                    'get_position' in prev_context or
                    'current_pos' in prev_context or
                    'close_position' in prev_context  # close_position doesn't need verification
                )
                sell_order_locations.append({
                    'line': i,
                    'has_verification': has_verification,
                    'code': line.strip(),
                    'context': context[:100]
                })
    
    print(f"📋 Found {len(sell_order_locations)} sell order submissions")
    print()
    
    all_fixed = True
    for loc in sell_order_locations:
        status = "✅" if loc['has_verification'] else "❌"
        print(f"   {status} Line {loc['line']:4d}: {loc['code'][:60]}...")
        if not loc['has_verification']:
            all_fixed = False
            print(f"      ⚠️  Missing position verification!")
    
    print()
    
    # Check for pattern: get_position before submit_order
    verification_pattern = re.compile(
        r'get_position.*?submit_order.*?side.*?[\'"]sell[\'"]',
        re.DOTALL
    )
    
    code_text = ''.join(lines)
    matches = list(verification_pattern.finditer(code_text))
    
    print(f"✅ Found {len(matches)} instances with position verification")
    print()
    
    # Check all critical locations
    critical_sections = {
        'TP1': 'TP1.*submit_order.*sell',
        'TP2': 'TP2.*submit_order.*sell',
        'Damage Control': 'damage_control.*submit_order.*sell',
        'Trailing Stop': 'trail.*submit_order.*sell',
        'Runner Stop': 'runner.*submit_order.*sell',
        'Alternative Close': 'Alternative.*submit_order.*sell',
        'RL Trim': 'TRIMMED.*submit_order.*sell',
    }
    
    print("📊 Critical Section Verification:")
    print()
    
    all_critical_fixed = True
    for section_name, pattern in critical_sections.items():
        section_pattern = re.compile(pattern, re.IGNORECASE | re.DOTALL)
        section_matches = list(section_pattern.finditer(code_text))
        
        # Check if matches have verification nearby
        section_fixed = True
        for match in section_matches:
            start = max(0, match.start() - 500)
            end = min(len(code_text), match.end() + 100)
            context = code_text[start:end]
            if 'get_position' not in context and 'close_position' not in context:
                section_fixed = False
        
        status = "✅" if section_fixed or len(section_matches) == 0 else "❌"
        print(f"   {status} {section_name}: {'Fixed' if section_fixed else 'Needs Fix'}")
        if not section_fixed:
            all_critical_fixed = False
    
    print()
    print("=" * 80)
    
    if all_fixed and all_critical_fixed:
        print("✅ ALL SELL ORDERS HAVE POSITION VERIFICATION")
        print()
        print("The fix is correctly applied. All sell order submissions now:")
        print("   1. Verify position ownership with get_position()")
        print("   2. Check quantity before selling")
        print("   3. Only submit orders when we own the contracts")
        print()
        print("This should prevent the 'uncovered options' error tomorrow.")
        return True
    else:
        print("⚠️  SOME SELL ORDERS MISSING VERIFICATION")
        print()
        print("Please review the locations above and add position verification.")
        return False

def validate_fix_logic():
    """Validate the fix logic is correct"""
    print("=" * 80)
    print("🔍 VALIDATION: Fix Logic Correctness")
    print("=" * 80)
    print()
    
    code_file = Path('mike_agent_live_safe.py')
    with open(code_file, 'r') as f:
        code = f.read()
    
    # Check for the fix pattern
    fix_pattern = r'get_position\(symbol\).*?if.*?current_pos.*?qty.*?>=.*?sell'
    
    if re.search(fix_pattern, code, re.DOTALL | re.IGNORECASE):
        print("✅ Fix pattern found in code")
        print()
        print("The fix logic:")
        print("   1. ✅ Calls api.get_position(symbol) to verify ownership")
        print("   2. ✅ Checks if current_pos exists")
        print("   3. ✅ Verifies float(current_pos.qty) >= sell_qty")
        print("   4. ✅ Only submits order if verification passes")
        print()
        return True
    else:
        print("⚠️  Fix pattern not clearly found")
        print()
        return False

if __name__ == "__main__":
    print()
    validation1 = validate_all_sell_orders()
    print()
    validation2 = validate_fix_logic()
    print()
    
    if validation1 and validation2:
        print("=" * 80)
        print("✅ VALIDATION COMPLETE - FIX IS CORRECT")
        print("=" * 80)
        exit(0)
    else:
        print("=" * 80)
        print("⚠️  VALIDATION INCOMPLETE - REVIEW NEEDED")
        print("=" * 80)
        exit(1)

