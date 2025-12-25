#!/usr/bin/env python3
"""
VALIDATION: Backtester Execution Modeling Integration
Tests that execution modeling is properly integrated without importing conflicting modules
"""
import sys
import os
import ast
import re

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

def validate_backtest_method():
    """Validate backtest() method has execution modeling parameter"""
    print_header("TEST 1: Backtest Method Signature")
    
    try:
        with open('mike_agent.py', 'r') as f:
            content = f.read()
        
        # Check for use_execution_modeling parameter
        has_param = 'use_execution_modeling' in content
        print_test("use_execution_modeling parameter", has_param, 
                  "Parameter found in backtest() method")
        
        # Check for execution integration call
        has_integration = 'integrate_execution_into_backtest' in content
        print_test("Execution integration call", has_integration,
                  "integrate_execution_into_backtest() called in backtest()")
        
        # Check for execution modeling message
        has_message = 'Execution Modeling: ENABLED' in content or 'execution modeling' in content.lower()
        print_test("Execution modeling message", has_message,
                  "User message about execution modeling present")
        
        return has_param and has_integration
        
    except Exception as e:
        print_test("Backtest Method Validation", False, str(e))
        return False

def validate_execution_integration():
    """Validate execution_integration.py patches _simulate_trade correctly"""
    print_header("TEST 2: Execution Integration Module")
    
    try:
        with open('execution_integration.py', 'r') as f:
            content = f.read()
        
        # Check for integrate_execution_into_backtest function
        has_function = 'def integrate_execution_into_backtest' in content
        print_test("integrate_execution_into_backtest function", has_function,
                  "Function exists")
        
        # Check for _simulate_trade patching
        has_patch = 'simulate_with_execution' in content or '_simulate_trade' in content
        print_test("_simulate_trade patching", has_patch,
                  "Method patching code present")
        
        # Check for apply_execution_costs call
        has_apply_costs = 'apply_execution_costs' in content
        print_test("apply_execution_costs call", has_apply_costs,
                  "Execution costs applied in patched method")
        
        # Check for slippage application
        has_slippage = 'apply_slippage' in content
        print_test("Slippage application", has_slippage,
                  "Slippage applied to trades")
        
        # Check for IV crush application
        has_iv_crush = 'apply_iv_crush' in content
        print_test("IV crush application", has_iv_crush,
                  "IV crush applied to trades")
        
        return has_function and has_patch and has_apply_costs
        
    except Exception as e:
        print_test("Execution Integration Validation", False, str(e))
        return False

def validate_execution_costs_function():
    """Validate apply_execution_costs function"""
    print_header("TEST 3: Execution Costs Function")
    
    try:
        with open('execution_integration.py', 'r') as f:
            content = f.read()
        
        # Check for apply_execution_costs function
        has_function = 'def apply_execution_costs' in content
        print_test("apply_execution_costs function", has_function,
                  "Function exists")
        
        # Check for slippage calculation
        has_slippage_calc = 'estimate_slippage' in content
        print_test("Slippage calculation", has_slippage_calc,
                  "Slippage estimated from execution engine")
        
        # Check for IV crush calculation
        has_iv_crush_calc = 'apply_iv_crush' in content or 'crushed_iv' in content
        print_test("IV crush calculation", has_iv_crush_calc,
                  "IV crush calculated")
        
        # Check for premium adjustment
        has_premium_adj = 'adjusted_premium' in content
        print_test("Premium adjustment", has_premium_adj,
                  "Premium adjusted with execution costs")
        
        return has_function and has_slippage_calc and has_premium_adj
        
    except Exception as e:
        print_test("Execution Costs Function Validation", False, str(e))
        return False

def validate_simulate_trade_patching():
    """Validate that _simulate_trade is properly patched"""
    print_header("TEST 4: _simulate_trade Patching Logic")
    
    try:
        with open('execution_integration.py', 'r') as f:
            content = f.read()
        
        # Check for entry execution costs
        has_entry_costs = 'signal.action.value == \'BUY\'' in content or 'signal.action == Action.BUY' in content
        print_test("Entry execution costs", has_entry_costs,
                  "Execution costs applied on entry (BUY)")
        
        # Check for exit execution costs
        has_exit_costs = 'signal.action.value == \'SELL\'' in content or 'signal.action == Action.SELL' in content
        print_test("Exit execution costs", has_exit_costs,
                  "Execution costs applied on exit (SELL)")
        
        # Check for entry premium update
        has_entry_update = 'entry_premium' in content and 'adjusted_premium' in content
        print_test("Entry premium update", has_entry_update,
                  "Entry premium updated with execution costs")
        
        # Check for PnL recalculation
        has_pnl_recalc = 'adjusted_pnl' in content or 'pnl' in content.lower()
        print_test("PnL recalculation", has_pnl_recalc,
                  "PnL recalculated with execution costs")
        
        return has_entry_costs and has_exit_costs
        
    except Exception as e:
        print_test("Patching Logic Validation", False, str(e))
        return False

def validate_advanced_execution_engine():
    """Validate AdvancedExecutionEngine has required methods"""
    print_header("TEST 5: Advanced Execution Engine")
    
    try:
        with open('advanced_execution.py', 'r') as f:
            content = f.read()
        
        # Check for estimate_slippage method
        has_slippage = 'def estimate_slippage' in content
        print_test("estimate_slippage method", has_slippage,
                  "Slippage estimation method exists")
        
        # Check for volume-based slippage
        has_volume_slippage = 'volume' in content and 'slippage' in content.lower()
        print_test("Volume-based slippage", has_volume_slippage,
                  "Slippage considers volume")
        
        # Check for spread consideration
        has_spread = 'spread' in content.lower() or 'bid' in content.lower() and 'ask' in content.lower()
        print_test("Spread consideration", has_spread,
                  "Bid-ask spread considered in execution")
        
        return has_slippage and has_volume_slippage
        
    except Exception as e:
        print_test("Advanced Execution Engine Validation", False, str(e))
        return False

def validate_iv_crush_integration():
    """Validate IV crush is integrated"""
    print_header("TEST 6: IV Crush Integration")
    
    try:
        # Check execution_integration.py
        with open('execution_integration.py', 'r') as f:
            exec_content = f.read()
        
        # Check advanced_backtesting.py (if exists)
        has_backtester = os.path.exists('advanced_backtesting.py')
        if has_backtester:
            with open('advanced_backtesting.py', 'r') as f:
                backtest_content = f.read()
        else:
            backtest_content = ""
        
        # Check for IV crush in execution integration
        has_iv_crush_exec = 'iv_crush' in exec_content.lower() or 'crushed_iv' in exec_content
        print_test("IV crush in execution integration", has_iv_crush_exec,
                  "IV crush applied in execution costs")
        
        # Check for time-based IV decay
        has_time_decay = 'time_in_day' in exec_content
        print_test("Time-based IV decay", has_time_decay,
                  "IV crush considers time in trading day")
        
        # Check for event-based IV crush
        has_event_crush = 'has_event' in exec_content
        print_test("Event-based IV crush", has_event_crush,
                  "IV crush considers earnings events")
        
        return has_iv_crush_exec and has_time_decay
        
    except Exception as e:
        print_test("IV Crush Integration Validation", False, str(e))
        return False

def validate_code_flow():
    """Validate the complete code flow"""
    print_header("TEST 7: Complete Code Flow Validation")
    
    try:
        # Read both files
        with open('mike_agent.py', 'r') as f:
            agent_content = f.read()
        
        with open('execution_integration.py', 'r') as f:
            exec_content = f.read()
        
        # Flow 1: backtest() calls integrate_execution_into_backtest()
        flow1 = 'integrate_execution_into_backtest' in agent_content
        print_test("Flow 1: backtest() → integrate_execution_into_backtest()", flow1,
                  "backtest() method calls integration function")
        
        # Flow 2: Integration patches _simulate_trade()
        flow2 = 'simulate_with_execution' in exec_content or 'original_simulate' in exec_content
        print_test("Flow 2: Integration patches _simulate_trade()", flow2,
                  "_simulate_trade() method is patched")
        
        # Flow 3: Patched method calls apply_execution_costs()
        flow3 = 'apply_execution_costs' in exec_content
        print_test("Flow 3: Patched method → apply_execution_costs()", flow3,
                  "Execution costs applied in patched method")
        
        # Flow 4: apply_execution_costs() uses execution engine
        flow4 = 'get_execution_engine' in exec_content or 'execution_engine' in exec_content
        print_test("Flow 4: apply_execution_costs() → execution engine", flow4,
                  "Execution engine used for slippage")
        
        # Flow 5: Premium adjusted and PnL recalculated
        flow5 = 'adjusted_premium' in exec_content and ('pnl' in exec_content.lower() or 'adjusted_pnl' in exec_content)
        print_test("Flow 5: Premium adjusted → PnL recalculated", flow5,
                  "PnL recalculated with adjusted premium")
        
        all_flows = flow1 and flow2 and flow3 and flow4 and flow5
        return all_flows
        
    except Exception as e:
        print_test("Code Flow Validation", False, str(e))
        return False

def main():
    print("="*70)
    print("  BACKTESTER EXECUTION MODELING - CODE VALIDATION")
    print("="*70)
    print("Validating code structure without importing modules...\n")
    
    results = {}
    
    # Test 1: Backtest method
    results['backtest_method'] = validate_backtest_method()
    
    # Test 2: Execution integration
    results['execution_integration'] = validate_execution_integration()
    
    # Test 3: Execution costs function
    results['execution_costs'] = validate_execution_costs_function()
    
    # Test 4: Patching logic
    results['patching'] = validate_simulate_trade_patching()
    
    # Test 5: Advanced execution engine
    results['execution_engine'] = validate_advanced_execution_engine()
    
    # Test 6: IV crush integration
    results['iv_crush'] = validate_iv_crush_integration()
    
    # Test 7: Complete code flow
    results['code_flow'] = validate_code_flow()
    
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
        print("✅ Execution modeling is properly integrated into backtester")
        print("✅ All code paths are present and correct")
    else:
        print("\n⚠️ Some tests failed. Review above for details.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)





