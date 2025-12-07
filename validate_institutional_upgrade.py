#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE VALIDATION SCRIPT
Validates all institutional upgrade components
"""

import sys
import os

def validate_imports():
    """Validate all required imports"""
    print("\n" + "="*70)
    print("STEP 1: VALIDATING IMPORTS")
    print("="*70)
    
    results = []
    
    # Core dependencies
    try:
        import pandas as pd
        results.append(("✅", "pandas", "Core library"))
    except ImportError as e:
        results.append(("❌", "pandas", f"Missing: {e}"))
    
    try:
        import numpy as np
        results.append(("✅", "numpy", "Core library"))
    except ImportError as e:
        results.append(("❌", "numpy", f"Missing: {e}"))
    
    try:
        import yfinance as yf
        results.append(("✅", "yfinance", "Core library"))
    except ImportError as e:
        results.append(("❌", "yfinance", f"Missing: {e}"))
    
    # Institutional features module
    try:
        from institutional_features import InstitutionalFeatureEngine, create_feature_engine
        results.append(("✅", "institutional_features", "Main module"))
    except ImportError as e:
        results.append(("❌", "institutional_features", f"Missing: {e}"))
    
    # Optional dependencies
    try:
        from ta import add_all_ta_features
        results.append(("✅", "ta", "Optional - technical analysis"))
    except ImportError:
        results.append(("⚠️", "ta", "Optional - install with: pip install ta"))
    
    try:
        import talib
        results.append(("✅", "talib", "Optional - advanced technical analysis"))
    except ImportError:
        results.append(("⚠️", "talib", "Optional - requires system installation"))
    
    # Stable-Baselines3
    try:
        from stable_baselines3 import PPO
        results.append(("✅", "stable_baselines3", "RL framework"))
    except ImportError:
        results.append(("❌", "stable_baselines3", "Required for RL - install with: pip install stable-baselines3"))
    
    # Print results
    for status, module, note in results:
        print(f"{status} {module:30s} - {note}")
    
    return all(r[0] == "✅" for r in results[:4])  # Core modules must be available


def validate_feature_engine():
    """Validate feature engine functionality"""
    print("\n" + "="*70)
    print("STEP 2: VALIDATING FEATURE ENGINE")
    print("="*70)
    
    try:
        from institutional_features import InstitutionalFeatureEngine
        import pandas as pd
        import numpy as np
        import yfinance as yf
        
        # Create engine
        engine = InstitutionalFeatureEngine(lookback_minutes=20)
        print("✅ Feature engine instantiated successfully")
        
        # Get test data
        print("📥 Fetching test data (SPY, 1-minute bars)...")
        spy = yf.Ticker("SPY")
        data = spy.history(period="1d", interval="1m")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if len(data) < 20:
            print("⚠️ Not enough data for full validation (need 20+ bars)")
            return False
        
        # Ensure lowercase columns
        data.columns = [col.lower() for col in data.columns]
        
        # Extract features
        print("🔧 Extracting features...")
        all_features, feature_groups = engine.extract_all_features(
            data.tail(30),  # Use last 30 bars
            symbol='SPY',
            risk_mgr=None,
            include_microstructure=True
        )
        
        print(f"✅ Features extracted: {all_features.shape[0]} bars × {all_features.shape[1]} features")
        print(f"✅ Feature groups: {list(feature_groups.keys())}")
        
        # Verify feature counts
        total_features = sum(g.shape[1] for g in feature_groups.values())
        print(f"✅ Total features from groups: {total_features}")
        
        if all_features.shape[1] >= 500:
            print(f"✅ Feature count meets target (500+): {all_features.shape[1]} features")
        else:
            print(f"⚠️ Feature count below target: {all_features.shape[1]} (expected 500+)")
        
        # Check for NaN/Inf
        has_nan = np.isnan(all_features).any()
        has_inf = np.isinf(all_features).any()
        
        if has_nan or has_inf:
            print(f"⚠️ Found NaN or Inf values in features")
            if has_nan:
                nan_count = np.isnan(all_features).sum()
                print(f"   NaN count: {nan_count}")
            if has_inf:
                inf_count = np.isinf(all_features).sum()
                print(f"   Inf count: {inf_count}")
        else:
            print("✅ No NaN or Inf values detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating feature engine: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_integration():
    """Validate integration in mike_agent_live_safe.py"""
    print("\n" + "="*70)
    print("STEP 3: VALIDATING INTEGRATION")
    print("="*70)
    
    try:
        # Read the file
        with open('mike_agent_live_safe.py', 'r') as f:
            content = f.read()
        
        checks = []
        
        # Check 1: Import statement
        if 'from institutional_features import' in content:
            checks.append(("✅", "Import statement present"))
        else:
            checks.append(("❌", "Import statement missing"))
        
        # Check 2: Configuration flag
        if 'USE_INSTITUTIONAL_FEATURES' in content:
            checks.append(("✅", "Configuration flag present"))
        else:
            checks.append(("❌", "Configuration flag missing"))
        
        # Check 3: Feature engine initialization
        if 'feature_engine' in content and 'create_feature_engine' in content:
            checks.append(("✅", "Feature engine initialization present"))
        else:
            checks.append(("❌", "Feature engine initialization missing"))
        
        # Check 4: Enhanced prepare_observation function
        if 'prepare_observation_institutional' in content:
            checks.append(("✅", "Institutional observation function present"))
        else:
            checks.append(("❌", "Institutional observation function missing"))
        
        # Check 5: Function call in main loop
        if 'prepare_observation(hist, risk_mgr' in content:
            checks.append(("✅", "Function called in main loop"))
            # Check if symbol parameter is passed
            if 'prepare_observation(hist, risk_mgr, symbol=' in content:
                checks.append(("✅", "Symbol parameter passed"))
            else:
                checks.append(("⚠️", "Symbol parameter not passed (uses default)"))
        else:
            checks.append(("❌", "Function not called in main loop"))
        
        # Print results
        for status, check in checks:
            print(f"{status} {check}")
        
        return all(c[0] == "✅" for c in checks[:5])
        
    except Exception as e:
        print(f"❌ Error validating integration: {e}")
        return False


def validate_syntax():
    """Validate Python syntax"""
    print("\n" + "="*70)
    print("STEP 4: VALIDATING SYNTAX")
    print("="*70)
    
    import py_compile
    import ast
    
    files_to_check = [
        'institutional_features.py',
        'mike_agent_live_safe.py'
    ]
    
    results = []
    
    for filename in files_to_check:
        if not os.path.exists(filename):
            results.append(("❌", filename, "File not found"))
            continue
        
        try:
            # Check syntax
            with open(filename, 'r') as f:
                code = f.read()
            ast.parse(code)
            results.append(("✅", filename, "Syntax valid"))
        except SyntaxError as e:
            results.append(("❌", filename, f"Syntax error: {e}"))
        except Exception as e:
            results.append(("⚠️", filename, f"Error: {e}"))
    
    for status, filename, note in results:
        print(f"{status} {filename:40s} - {note}")
    
    return all(r[0] == "✅" for r in results)


def check_missing_components():
    """Check what's missing from the full implementation"""
    print("\n" + "="*70)
    print("STEP 5: CHECKING MISSING COMPONENTS")
    print("="*70)
    
    expected_files = {
        'institutional_features.py': 'Phase 1.1 - Feature Engineering',
        'institutional_rl_model.py': 'Phase 1.3 - LSTM Backbone',
        'institutional_risk.py': 'Phase 1.4 - Advanced Risk Metrics',
        'multi_agent_system.py': 'Phase 2 - Multi-Agent Framework',
        'execution_engine.py': 'Phase 3 - Execution Optimization',
        'institutional_backtest.py': 'Phase 4 - Advanced Backtesting',
        'automation_pipeline.py': 'Phase 5 - Automation'
    }
    
    results = []
    
    for filename, description in expected_files.items():
        if os.path.exists(filename):
            results.append(("✅", filename, description))
        else:
            results.append(("⏳", filename, f"Not yet created - {description}"))
    
    for status, filename, note in results:
        print(f"{status} {filename:40s} - {note}")
    
    return results


def main():
    """Run all validations"""
    print("\n" + "="*70)
    print("🏦 INSTITUTIONAL UPGRADE - COMPREHENSIVE VALIDATION")
    print("="*70)
    
    results = {}
    
    # Step 1: Validate imports
    results['imports'] = validate_imports()
    
    # Step 2: Validate feature engine
    if results['imports']:
        results['feature_engine'] = validate_feature_engine()
    else:
        print("\n⚠️ Skipping feature engine validation (missing imports)")
        results['feature_engine'] = False
    
    # Step 3: Validate integration
    results['integration'] = validate_integration()
    
    # Step 4: Validate syntax
    results['syntax'] = validate_syntax()
    
    # Step 5: Check missing components
    missing = check_missing_components()
    
    # Final summary
    print("\n" + "="*70)
    print("📊 VALIDATION SUMMARY")
    print("="*70)
    
    print(f"\n✅ Imports: {'PASS' if results.get('imports') else 'FAIL'}")
    print(f"{'✅' if results.get('feature_engine') else '❌'} Feature Engine: {'PASS' if results.get('feature_engine') else 'FAIL'}")
    print(f"{'✅' if results.get('integration') else '❌'} Integration: {'PASS' if results.get('integration') else 'FAIL'}")
    print(f"{'✅' if results.get('syntax') else '❌'} Syntax: {'PASS' if results.get('syntax') else 'FAIL'}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL CORE VALIDATIONS PASSED")
        print("="*70)
        print("\nPhase 1.1-1.2 is correctly implemented and ready for use!")
    else:
        print("⚠️ SOME VALIDATIONS FAILED")
        print("="*70)
        print("\nPlease review the errors above and fix before proceeding.")
    
    print("\nMissing Components (Future Phases):")
    missing_count = sum(1 for r in missing if r[0] == "⏳")
    if missing_count > 0:
        print(f"  {missing_count} modules not yet created (expected for future phases)")
    else:
        print("  All expected modules exist!")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

