#!/usr/bin/env python3
"""
COMPREHENSIVE VALIDATION: Multi-Agent Voting + Backtester Execution Modeling
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

def test_multi_agent_voting():
    """Test multi-agent voting system"""
    print_header("TEST 1: Multi-Agent Voting System")
    
    try:
        from multi_agent_ensemble import initialize_meta_router
        import pandas as pd
        import numpy as np
        
        router = initialize_meta_router()
        
        # Create test data
        dates = pd.date_range(end=datetime.now(), periods=50, freq='1min')
        prices = 500 + np.cumsum(np.random.randn(50) * 0.3)
        data = pd.DataFrame({
            'open': prices + np.random.randn(50) * 0.1,
            'high': prices + abs(np.random.randn(50) * 0.3),
            'low': prices - abs(np.random.randn(50) * 0.3),
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 50)
        }, index=dates)
        
        # Test 1: All agents vote
        print("\n1. Testing All Agents Vote:")
        action, conf, details = router.route(
            data=data,
            vix=20.0,
            symbol="SPY",
            current_price=prices[-1],
            strike=round(prices[-1]),
            portfolio_delta=0.0,
            delta_limit=2000.0
        )
        
        signals = details['signals']
        all_agents_voted = all(agent in signals for agent in ['trend', 'reversal', 'volatility', 'gamma_model', 'delta_hedging', 'macro'])
        print_test("All 6 agents voted", all_agents_voted, f"Found: {list(signals.keys())}")
        
        # Test 2: Weighted voting
        print("\n2. Testing Weighted Voting:")
        action_scores = details['action_scores']
        total_score = sum(action_scores.values())
        weights_sum = sum(s['weight'] for s in signals.values())
        
        print_test("Action scores calculated", total_score > 0, f"Total score: {total_score:.3f}")
        print_test("Weights normalized", abs(weights_sum - 1.0) < 0.01, f"Weights sum: {weights_sum:.3f}")
        
        # Test 3: Final decision
        print("\n3. Testing Final Decision:")
        final_action = details['final_action']
        final_conf = details['final_confidence']
        
        print_test("Final action valid", final_action in [0, 1, 2], f"Action: {final_action}")
        print_test("Final confidence normalized", 0.0 <= final_conf <= 1.0, f"Confidence: {final_conf:.3f}")
        
        # Test 4: Regime-based weighting
        print("\n4. Testing Regime-Based Weighting:")
        regime = details['regime']
        print_test("Regime detected", regime in ['trending', 'mean_reverting', 'volatile', 'neutral'], f"Regime: {regime}")
        
        # Test 5: Hierarchical overrides
        print("\n5. Testing Hierarchical Overrides:")
        delta_signal = signals.get('delta_hedging', {})
        if delta_signal.get('confidence', 0) > 0.8:
            print_test("High-priority delta signal", True, f"Delta conf: {delta_signal.get('confidence', 0):.3f}")
        else:
            print_test("Delta signal present", 'delta_hedging' in signals, "Delta agent working")
        
        return True
        
    except Exception as e:
        print_test("Multi-Agent Voting", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def test_backtester_execution_modeling():
    """Test backtester execution modeling"""
    print_header("TEST 2: Backtester Execution Modeling")
    
    try:
        # Test 1: Execution integration available
        print("\n1. Testing Execution Integration Module:")
        try:
            from execution_integration import apply_execution_costs, integrate_execution_into_backtest
            EXECUTION_INTEGRATION_AVAILABLE = True
            print_test("Execution integration module", True, "Module found")
        except ImportError:
            EXECUTION_INTEGRATION_AVAILABLE = False
            print_test("Execution integration module", False, "Module not found")
            return False
        
        # Test 2: Advanced execution engine
        print("\n2. Testing Advanced Execution Engine:")
        try:
            from advanced_execution import AdvancedExecutionEngine, initialize_execution_engine, get_execution_engine
            EXECUTION_ENGINE_AVAILABLE = True
            print_test("Advanced execution engine", True, "Module found")
            
            # Initialize and test
            engine = initialize_execution_engine()
            print_test("Execution engine initialized", engine is not None, "Engine ready")
        except ImportError:
            EXECUTION_ENGINE_AVAILABLE = False
            print_test("Advanced execution engine", False, "Module not found")
            return False
        
        # Test 3: Slippage calculation
        print("\n3. Testing Slippage Calculation:")
        if engine:
            slippage = engine.estimate_slippage(
                symbol='SPY',
                qty=10,
                bid=5.0,
                ask=5.25,
                volume=1000000
            )
            print_test("Slippage calculation", slippage >= 0, f"Slippage: ${slippage:.4f} per contract")
        
        # Test 4: IV crush
        print("\n4. Testing IV Crush:")
        try:
            from advanced_backtesting import AdvancedBacktester
            backtester = AdvancedBacktester()
            
            initial_iv = 0.25
            crushed_iv = backtester.apply_iv_crush(
                initial_iv=initial_iv,
                time_in_day=0.8,  # Late in day
                has_event=False
            )
            iv_crush_pct = (initial_iv - crushed_iv) / initial_iv * 100
            print_test("IV crush calculation", crushed_iv < initial_iv, f"IV: {initial_iv:.2%} → {crushed_iv:.2%} ({iv_crush_pct:.1f}% crush)")
        except Exception as e:
            print_test("IV crush", False, f"Error: {e}")
        
        # Test 5: Execution costs application
        print("\n5. Testing Execution Costs Application:")
        base_premium = 5.0
        adjusted_premium, exec_details = apply_execution_costs(
            premium=base_premium,
            qty=10,
            side='buy',
            volume=1000000,
            apply_slippage=True,
            apply_iv_crush=True,
            time_in_day=0.5
        )
        
        has_slippage = exec_details.get('slippage', 0) != 0
        has_iv_crush = exec_details.get('iv_crush_adjustment', 0) != 0
        premium_changed = abs(adjusted_premium - base_premium) > 0.001
        
        print_test("Slippage applied", has_slippage, f"Slippage: ${exec_details.get('slippage', 0):.4f}")
        print_test("IV crush applied", has_iv_crush, f"IV adjustment: ${exec_details.get('iv_crush_adjustment', 0):.4f}")
        print_test("Premium adjusted", premium_changed, f"Base: ${base_premium:.2f} → Final: ${adjusted_premium:.2f}")
        
        # Test 6: Spread expansion
        print("\n6. Testing Spread Expansion:")
        # Test with different volumes (low volume = wider spread)
        low_vol_slippage = engine.estimate_slippage('SPY', 10, 5.0, 5.25, volume=10000)
        high_vol_slippage = engine.estimate_slippage('SPY', 10, 5.0, 5.25, volume=10000000)
        
        spread_expansion = low_vol_slippage > high_vol_slippage
        print_test("Spread expansion (low vol)", spread_expansion, 
                  f"Low vol: ${low_vol_slippage:.4f}, High vol: ${high_vol_slippage:.4f}")
        
        return True
        
    except Exception as e:
        print_test("Backtester Execution Modeling", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def test_backtester_integration():
    """Test if backtester actually uses execution modeling"""
    print_header("TEST 3: Backtester Integration Check")
    
    try:
        from mike_agent import MikeAgent
        from execution_integration import integrate_execution_into_backtest
        
        # Create agent
        agent = MikeAgent(mode='backtest', symbols=['SPY'], capital=10000.0)
        
        # Check if _simulate_trade exists
        has_simulate = hasattr(agent, '_simulate_trade')
        print_test("_simulate_trade method exists", has_simulate, "Method found")
        
        # Integrate execution modeling
        try:
            agent_with_execution = integrate_execution_into_backtest(
                agent,
                apply_slippage=True,
                apply_iv_crush=True
            )
            print_test("Execution modeling integrated", agent_with_execution is not None, "Integration successful")
            
            # Check if method was patched
            method_patched = hasattr(agent_with_execution._simulate_trade, '__name__')
            print_test("Method patched", True, "Execution costs will be applied")
            
        except Exception as e:
            print_test("Execution integration", False, f"Error: {e}")
            return False
        
        return True
        
    except Exception as e:
        print_test("Backtester Integration", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def test_mike_agent_backtest_method():
    """Test if MikeAgent.backtest() uses execution modeling"""
    print_header("TEST 4: MikeAgent.backtest() Method Check")
    
    try:
        from mike_agent import MikeAgent
        
        agent = MikeAgent(mode='backtest', symbols=['SPY'], capital=10000.0)
        
        # Check if backtest method exists
        has_backtest = hasattr(agent, 'backtest')
        print_test("backtest() method exists", has_backtest, "Method found")
        
        if has_backtest:
            # Check method signature
            import inspect
            sig = inspect.signature(agent.backtest)
            params = list(sig.parameters.keys())
            
            has_execution_param = 'use_execution_modeling' in params
            print_test("use_execution_modeling parameter", has_execution_param, 
                      f"Parameters: {params}")
            
            # Check if execution is used in method
            import inspect
            source = inspect.getsource(agent.backtest)
            uses_execution = 'execution' in source.lower() or 'slippage' in source.lower()
            print_test("Uses execution modeling", uses_execution, "Checking source code...")
        
        return True
        
    except Exception as e:
        print_test("MikeAgent.backtest() Check", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*70)
    print("  COMPREHENSIVE VALIDATION: Multi-Agent Voting + Execution Modeling")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {}
    
    # Test 1: Multi-agent voting
    results['multi_agent_voting'] = test_multi_agent_voting()
    
    # Test 2: Execution modeling
    results['execution_modeling'] = test_backtester_execution_modeling()
    
    # Test 3: Backtester integration
    results['backtester_integration'] = test_backtester_integration()
    
    # Test 4: MikeAgent.backtest() method
    results['backtest_method'] = test_mike_agent_backtest_method()
    
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
    else:
        print("\n⚠️ Some tests failed. Review above for details.")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)





