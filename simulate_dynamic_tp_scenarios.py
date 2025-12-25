#!/usr/bin/env python3
"""
Dynamic TP Scenario Simulator
Tests dynamic TP calculations across extreme market conditions
"""

import pandas as pd
import numpy as np
from dynamic_take_profit import (
    compute_dynamic_tp_factors,
    compute_dynamic_takeprofits,
    get_ticker_personality_factor
)

# Base TP levels (from normal regime)
BASE_TP1 = 0.40
BASE_TP2 = 0.80
BASE_TP3 = 1.50


def create_mock_hist_data(volatility_level="normal"):
    """Create mock historical data with different volatility levels"""
    np.random.seed(42)
    n_bars = 100
    
    if volatility_level == "high":
        # High volatility: large swings
        returns = np.random.normal(0, 0.02, n_bars)  # 2% std dev
    elif volatility_level == "low":
        # Low volatility: small moves
        returns = np.random.normal(0, 0.005, n_bars)  # 0.5% std dev
    else:
        # Normal volatility
        returns = np.random.normal(0, 0.01, n_bars)  # 1% std dev
    
    # Generate price series
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLC DataFrame
    hist_data = pd.DataFrame({
        'Open': prices,
        'High': prices * (1 + abs(returns) * 0.5),
        'Low': prices * (1 - abs(returns) * 0.5),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_bars)
    })
    
    return hist_data


def test_scenario(name, ticker, vix, volatility_level, confidence=None, rl_action_raw=None):
    """Test a specific scenario"""
    print(f"\n{'='*70}")
    print(f"SCENARIO: {name}")
    print(f"{'='*70}")
    print(f"Ticker: {ticker}")
    print(f"VIX: {vix}")
    print(f"Volatility Level: {volatility_level}")
    print(f"Confidence: {confidence if confidence else 'N/A'}")
    
    # Create mock historical data
    hist_data = create_mock_hist_data(volatility_level)
    
    # Compute dynamic TP factors
    try:
        tp_factors = compute_dynamic_tp_factors(
            hist_data=hist_data,
            ticker=ticker,
            vix=vix,
            confidence=confidence,
            rl_action_raw=rl_action_raw
        )
        
        # Compute dynamic TPs
        dynamic_tp1, dynamic_tp2, dynamic_tp3 = compute_dynamic_takeprofits(
            base_tp1=BASE_TP1,
            base_tp2=BASE_TP2,
            base_tp3=BASE_TP3,
            adjustment_factors=tp_factors
        )
        
        # Display results
        print(f"\n📊 FACTORS:")
        print(f"   ATR Factor: {tp_factors['atr']:.2f}x")
        print(f"   Trend Factor: {tp_factors['trend']:.2f}x")
        print(f"   VIX Factor: {tp_factors['vix']:.2f}x")
        print(f"   Personality Factor: {tp_factors['personality']:.2f}x")
        print(f"   Confidence Factor: {tp_factors['confidence']:.2f}x")
        print(f"   Total Factor: {tp_factors['total']:.2f}x")
        print(f"   Trend Strength: {tp_factors['trend_strength']:.2f}")
        
        print(f"\n🎯 TAKE-PROFIT LEVELS:")
        print(f"   Base:  TP1={BASE_TP1:.0%}  TP2={BASE_TP2:.0%}  TP3={BASE_TP3:.0%}")
        print(f"   Dynamic: TP1={dynamic_tp1:.0%}  TP2={dynamic_tp2:.0%}  TP3={dynamic_tp3:.0%}")
        print(f"   Adjustment: TP1={dynamic_tp1/BASE_TP1:.1f}x  TP2={dynamic_tp2/BASE_TP2:.1f}x  TP3={dynamic_tp3/BASE_TP3:.1f}x")
        
        # Validate caps
        assert 0.20 <= dynamic_tp1 <= 0.80, f"TP1 out of range: {dynamic_tp1}"
        assert 0.40 <= dynamic_tp2 <= 1.20, f"TP2 out of range: {dynamic_tp2}"
        assert 0.80 <= dynamic_tp3 <= 2.00, f"TP3 out of range: {dynamic_tp3}"
        
        print(f"   ✅ Caps validated: TP1∈[20%,80%], TP2∈[40%,120%], TP3∈[80%,200%]")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all test scenarios"""
    print("="*70)
    print("DYNAMIC TAKE-PROFIT SCENARIO SIMULATOR")
    print("="*70)
    
    results = []
    
    # Scenario 1: NVDA High Volatility Day (FOMC-like)
    results.append(("NVDA High Vol Day", test_scenario(
        "NVDA High Volatility Day (FOMC-like)",
        ticker="NVDA",
        vix=36.0,
        volatility_level="high",
        confidence=0.75,
        rl_action_raw=0.8
    )))
    
    # Scenario 2: AAPL Low Volatility Day (Calm)
    results.append(("AAPL Low Vol Day", test_scenario(
        "AAPL Low Volatility Day (Calm)",
        ticker="AAPL",
        vix=12.0,
        volatility_level="low",
        confidence=0.35,
        rl_action_raw=0.3
    )))
    
    # Scenario 3: TSLA Strong Trend Day
    results.append(("TSLA Strong Trend", test_scenario(
        "TSLA Strong Trend Day",
        ticker="TSLA",
        vix=22.0,
        volatility_level="high",
        confidence=0.65,
        rl_action_raw=0.7
    )))
    
    # Scenario 4: SPY Normal Day
    results.append(("SPY Normal Day", test_scenario(
        "SPY Normal Market Day",
        ticker="SPY",
        vix=18.0,
        volatility_level="normal",
        confidence=0.50,
        rl_action_raw=0.5
    )))
    
    # Scenario 5: MSTR Extreme Volatility (BTC correlation)
    results.append(("MSTR Extreme Vol", test_scenario(
        "MSTR Extreme Volatility (BTC correlation)",
        ticker="MSTR",
        vix=30.0,
        volatility_level="high",
        confidence=0.80,
        rl_action_raw=0.9
    )))
    
    # Scenario 6: GOOG Slow Mover
    results.append(("GOOG Slow Mover", test_scenario(
        "GOOG Slow Mover (Low ATR)",
        ticker="GOOG",
        vix=14.0,
        volatility_level="low",
        confidence=0.40,
        rl_action_raw=0.4
    )))
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL SCENARIOS PASSED")
    else:
        print(f"❌ {total - passed} SCENARIOS FAILED")
    
    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()







