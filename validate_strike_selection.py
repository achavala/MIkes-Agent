#!/usr/bin/env python3
"""
Validate Strike Selection Logic
Tests the new strike selection function against your successful trades
"""

import sys
sys.path.insert(0, '.')

def find_atm_strike(price: float, option_type: str = 'call', target_delta: float = 0.50) -> float:
    """
    Find optimal strike for 0DTE options trading
    
    Strategy (based on successful manual trades):
    - CALLS: Strike = current_price + $1-3 (slightly OTM, ~$0.50 premium)
    - PUTS: Strike = current_price - $1-5 (slightly OTM, ~$0.40-$0.60 premium)
    """
    if option_type.lower() == 'call':
        strike_offset = 2.0  # $2 above price (slightly OTM)
        strike = price + strike_offset
    else:  # put
        strike_offset = -3.0  # $3 below price (slightly OTM)
        strike = price + strike_offset
    
    # Round to nearest $0.50 increment
    if price >= 100:
        strike = round(strike)  # Round to nearest $1.00
    else:
        strike = round(strike * 2) / 2  # Round to nearest $0.50
    
    # Validation: Ensure strike is within reasonable range
    if abs(strike - price) > 10:
        if price >= 100:
            strike = round(price)
        else:
            strike = round(price * 2) / 2
    
    return strike

def validate_strike_selection():
    """Validate strike selection against your successful trades"""
    
    print("=" * 80)
    print("🔍 STRIKE SELECTION VALIDATION")
    print("=" * 80)
    print()
    
    # Test cases based on your successful trades
    test_cases = [
        # (Symbol, Price, Option Type, Expected Strike Range, Your Actual Strike, Description)
        ("SPY", 675.0, "put", (672, 674), 672, "SPY $672 PUTS at $0.50 entry (80% profit)"),
        ("SPY", 680.0, "call", (681, 683), 681, "SPY $681 CALLS at $0.50 entry (110% profit)"),
        ("QQQ", 609.0, "put", (603, 607), 603, "QQQ $603 PUTS at $0.60 entry (40% profit)"),
        ("SPY", 678.0, "put", (672, 676), 672, "SPY $672 PUTS at $0.40 entry (110% profit)"),
    ]
    
    print("📊 TEST CASES (Based on Your Successful Trades)")
    print("-" * 80)
    
    all_passed = True
    
    for symbol, price, option_type, expected_range, your_strike, description in test_cases:
        calculated_strike = find_atm_strike(price, option_type)
        min_expected, max_expected = expected_range
        
        # Check if calculated strike is within expected range
        within_range = min_expected <= calculated_strike <= max_expected
        
        # Check distance from your actual strike
        distance_from_your = abs(calculated_strike - your_strike)
        
        status = "✅ PASS" if within_range else "❌ FAIL"
        if not within_range:
            all_passed = False
        
        print(f"\n{status} | {symbol} {option_type.upper()}")
        print(f"  Price: ${price:.2f}")
        print(f"  Calculated Strike: ${calculated_strike:.2f}")
        print(f"  Your Strike: ${your_strike:.2f}")
        print(f"  Expected Range: ${min_expected:.2f} - ${max_expected:.2f}")
        print(f"  Distance from Your Strike: ${distance_from_your:.2f}")
        print(f"  Description: {description}")
        
        if not within_range:
            print(f"  ⚠️  WARNING: Calculated strike outside expected range!")
    
    print()
    print("=" * 80)
    
    # Additional edge cases
    print("\n🧪 EDGE CASE TESTS")
    print("-" * 80)
    
    edge_cases = [
        ("SPY", 676.66, "call", "SPY at current market price"),
        ("QQQ", 609.18, "put", "QQQ at current market price"),
        ("IWM", 200.0, "call", "IWM example"),
        ("SPY", 100.0, "put", "Low price test"),
    ]
    
    for symbol, price, option_type, description in edge_cases:
        strike = find_atm_strike(price, option_type)
        distance = abs(strike - price)
        
        print(f"\n{symbol} {option_type.upper()} @ ${price:.2f}")
        print(f"  Calculated Strike: ${strike:.2f}")
        print(f"  Distance from Price: ${distance:.2f}")
        print(f"  Description: {description}")
        
        # Validate distance is reasonable
        if distance > 10:
            print(f"  ⚠️  WARNING: Strike is >$10 from price (may be too far OTM)")
            all_passed = False
        elif distance < 0.5:
            print(f"  ⚠️  WARNING: Strike is <$0.50 from price (may be too close to ATM)")
        else:
            print(f"  ✅ Distance is reasonable (${distance:.2f} from price)")
    
    print()
    print("=" * 80)
    print(f"\n{'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    success = validate_strike_selection()
    sys.exit(0 if success else 1)





