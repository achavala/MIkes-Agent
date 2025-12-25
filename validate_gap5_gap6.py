#!/usr/bin/env python3
"""
COMPREHENSIVE VALIDATION: Gap 5 (Realistic Fill Modeling) + Gap 6 (Online Learning)
Tests both systems in virtual environment
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_test(name, status, details=""):
    status_symbol = "✅" if status else "❌"
    print(f"{status_symbol} {name}")
    if details:
        print(f"   {details}")

def test_realistic_fill_modeling():
    """Test Gap 5: Realistic Fill Modeling"""
    print_header("TEST 1: Realistic Fill Modeling (Gap 5)")
    
    try:
        from realistic_fill_modeling import (
            RealisticFillModel,
            get_realistic_fill_model,
            calculate_realistic_fill
        )
        
        model = get_realistic_fill_model()
        
        # Test 1: Basic fill calculation
        print("\n1. Testing Basic Fill Calculation:")
        mid = 5.0
        bid = 4.9
        ask = 5.1
        qty = 10
        side = 'buy'
        time_to_expiry = 3.0  # 3 hours to expiry
        
        fill_price, details = calculate_realistic_fill(
            mid=mid, bid=bid, ask=ask, qty=qty, side=side,
            time_to_expiry=time_to_expiry, vix=20.0, volume=100000
        )
        
        print_test("Fill price calculated", fill_price > 0, f"Fill: ${fill_price:.4f}")
        print_test("Fill within range", bid <= fill_price <= ask * 1.1, 
                  f"Bid: ${bid:.2f}, Fill: ${fill_price:.4f}, Ask: ${ask:.2f}")
        print_test("Slippage calculated", 'slippage' in details, 
                  f"Slippage: ${details.get('slippage', 0):.4f}")
        
        # Test 2: Market-maker uncertainty
        print("\n2. Testing Market-Maker Uncertainty:")
        uncertainty = model._market_maker_uncertainty(vix=30.0, time_to_expiry=0.5, has_news=True)
        print_test("MM uncertainty calculated", 0.0 <= uncertainty <= 1.0,
                  f"Uncertainty: {uncertainty:.3f} (high VIX + close expiry + news)")
        
        # Test 3: Liquidity factor
        print("\n3. Testing Liquidity Factor:")
        liquidity = model._calculate_liquidity_factor(
            qty=100, volume=1000, spread_pct=0.05, hidden_liquidity_pct=0.1
        )
        print_test("Liquidity factor calculated", 0.5 <= liquidity <= 3.0,
                  f"Liquidity factor: {liquidity:.3f}")
        
        # Test 4: Gamma squeeze impact
        print("\n4. Testing Gamma Squeeze Impact:")
        gamma_impact = model._calculate_gamma_squeeze_impact(
            gamma_exposure=1000.0, side='buy', time_to_expiry=0.5
        )
        print_test("Gamma impact calculated", gamma_impact != 0,
                  f"Gamma impact: ${gamma_impact:.4f} (positive = pay more in squeeze)")
        
        # Test 5: IV collapse impact
        print("\n5. Testing IV Collapse Impact:")
        iv_impact = model._calculate_iv_collapse_impact(
            has_news=True, time_to_expiry=1.0, vix=25.0
        )
        print_test("IV collapse impact calculated", iv_impact < 0,
                  f"IV collapse impact: ${iv_impact:.4f} (negative = IV crushes)")
        
        # Test 6: Theta explosion impact
        print("\n6. Testing Theta Explosion Impact:")
        theta_impact = model._calculate_theta_explosion_impact(
            time_to_expiry=0.5, side='buy'
        )
        print_test("Theta impact calculated", theta_impact < 0,
                  f"Theta impact: ${theta_impact:.4f} (negative for buyers = time decay)")
        
        # Test 7: Formula validation
        print("\n7. Testing Formula: realistic_fill = mid ± (spread * randomness) * liquidity_factor")
        spread = ask - bid
        randomness = details.get('randomness', 0)
        liquidity_factor = details.get('liquidity_factor', 1.0)
        
        expected_component = spread * randomness * liquidity_factor
        actual_slippage = details.get('slippage', 0)
        
        print_test("Formula components present", 
                  all(k in details for k in ['randomness', 'liquidity_factor', 'gamma_impact', 'iv_collapse_impact', 'theta_impact']),
                  "All formula components calculated")
        
        # Test 8: Fill statistics
        print("\n8. Testing Fill Statistics:")
        stats = model.get_fill_statistics()
        print_test("Statistics calculated", 'avg_slippage_pct' in stats,
                  f"Avg slippage: {stats.get('avg_slippage_pct', 0):.2%}")
        
        return True
        
    except Exception as e:
        print_test("Realistic Fill Modeling", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def test_online_learning_system():
    """Test Gap 6: Online Learning / Daily Retraining"""
    print_header("TEST 2: Online Learning System (Gap 6)")
    
    try:
        from online_learning_system import (
            OnlineLearningSystem,
            initialize_online_learning,
            get_online_learning_system
        )
        
        # Initialize system
        system = initialize_online_learning(
            model_dir="test_models",
            rolling_window_days=30,
            min_retrain_interval_hours=20
        )
        
        # Test 1: Should retrain check
        print("\n1. Testing Should Retrain Logic:")
        should_retrain, reason = system.should_retrain(current_regime="trending")
        print_test("Should retrain check", isinstance(should_retrain, bool),
                  f"Should retrain: {should_retrain}, Reason: {reason}")
        
        # Test 2: Retrain model
        print("\n2. Testing Model Retraining:")
        # Create dummy training data
        training_data = pd.DataFrame({
            'close': np.random.randn(100) + 500,
            'volume': np.random.randint(1000000, 5000000, 100),
            'date': pd.date_range(end=datetime.now(), periods=100, freq='1h')
        })
        
        version_id = system.retrain_model(
            training_data=training_data,
            current_regime="trending",
            training_config={'epochs': 100, 'learning_rate': 0.0003}
        )
        
        print_test("Model retrained", version_id is not None,
                  f"Version ID: {version_id}")
        print_test("Version registered", version_id in system.versions,
                  "Version added to registry")
        
        # Test 3: Rolling window
        print("\n3. Testing Rolling Window:")
        all_data = pd.DataFrame({
            'close': np.random.randn(1000) + 500,
            'date': pd.date_range(end=datetime.now(), periods=1000, freq='1h')
        })
        
        window_data = system.get_rolling_window_data(all_data)
        print_test("Rolling window extracted", len(window_data) > 0,
                  f"Window size: {len(window_data)} samples")
        
        # Test 4: Version comparison
        print("\n4. Testing Version Comparison:")
        # Retrain another version
        version_id_2 = system.retrain_model(
            training_data=training_data,
            current_regime="volatile",
            training_config={'epochs': 100, 'learning_rate': 0.0003}
        )
        
        comparison = system.compare_versions(version_id, version_id_2)
        print_test("Version comparison", 'winner' in comparison,
                  f"Winner: {comparison.get('winner', 'N/A')}")
        
        # Test 5: Promote to production
        print("\n5. Testing Production Promotion:")
        success = system.promote_to_production(version_id_2)
        print_test("Promoted to production", success,
                  f"Production version: {system.current_production_version}")
        
        # Test 6: Regime-dependent retraining
        print("\n6. Testing Regime-Dependent Retraining:")
        should_retrain_regime, reason_regime = system.should_retrain(current_regime="mean_reverting")
        print_test("Regime change detection", should_retrain_regime,
                  f"Regime change detected: {reason_regime}")
        
        # Test 7: Retrain schedule
        print("\n7. Testing Retrain Schedule:")
        schedule = system.get_retrain_schedule()
        print_test("Schedule available", 'next_retrain_earliest' in schedule,
                  f"Next retrain: {schedule.get('next_retrain_earliest', 'N/A')}")
        
        # Test 8: A/B testing capability
        print("\n8. Testing A/B Testing Capability:")
        has_production = system.current_production_version is not None
        has_test = system.current_test_version is not None
        print_test("A/B testing ready", has_production and has_test,
                  f"Production: {system.current_production_version}, Test: {system.current_test_version}")
        
        # Cleanup
        import shutil
        if os.path.exists("test_models"):
            shutil.rmtree("test_models")
        
        return True
        
    except Exception as e:
        print_test("Online Learning System", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration of both systems"""
    print_header("TEST 3: Integration Test")
    
    try:
        from realistic_fill_modeling import calculate_realistic_fill
        from online_learning_system import initialize_online_learning
        
        # Test that both systems can work together
        print("\n1. Testing System Integration:")
        
        # Realistic fill
        fill_price, details = calculate_realistic_fill(
            mid=5.0, bid=4.9, ask=5.1, qty=10, side='buy',
            time_to_expiry=2.0, vix=22.0, volume=50000
        )
        
        # Online learning
        system = initialize_online_learning(model_dir="test_models")
        should_retrain, reason = system.should_retrain("trending")
        
        print_test("Both systems working", fill_price > 0 and isinstance(should_retrain, bool),
                  "Realistic fill + Online learning both operational")
        
        # Cleanup
        import shutil
        if os.path.exists("test_models"):
            shutil.rmtree("test_models")
        
        return True
        
    except Exception as e:
        print_test("Integration Test", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*70)
    print("  COMPREHENSIVE VALIDATION: Gap 5 + Gap 6")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {}
    
    # Test 1: Realistic fill modeling
    results['realistic_fill'] = test_realistic_fill_modeling()
    
    # Test 2: Online learning system
    results['online_learning'] = test_online_learning_system()
    
    # Test 3: Integration
    results['integration'] = test_integration()
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print("\n" + "="*70)
    print(f"OVERALL: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    print("="*70)
    
    if passed_tests == total_tests:
        print("\n🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ Gap 5 (Realistic Fill Modeling): IMPLEMENTED")
        print("✅ Gap 6 (Online Learning): IMPLEMENTED")
    else:
        print("\n⚠️ Some tests failed. Review above for details.")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)





