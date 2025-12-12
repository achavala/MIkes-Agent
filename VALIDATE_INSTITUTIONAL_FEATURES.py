#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation Script: Check if institutional-grade features are implemented
"""

import sys
import os
sys.path.insert(0, '.')

print("=" * 100)
print("INSTITUTIONAL-GRADE FEATURES VALIDATION")
print("=" * 100)
print()

features_status = {}

# Read files
try:
    with open('mike_agent_live_safe.py', 'r', encoding='utf-8') as f:
        content = f.read()
except Exception as e:
    print(f"Error reading mike_agent_live_safe.py: {e}")
    sys.exit(1)

inst_content = ""
if os.path.exists('institutional_features.py'):
    try:
        with open('institutional_features.py', 'r', encoding='utf-8') as f:
            inst_content = f.read()
    except Exception as e:
        print(f"Warning: Could not read institutional_features.py: {e}")

# 1. Check IV (Implied Volatility)
print("1. IV (Implied Volatility)")
print("-" * 100)
try:
    if 'vix' in content.lower() and ('sigma' in content.lower() or 'volatility' in content.lower()):
        if 'vix / 100' in content or 'vix/100' in content or 'sigma_estimated' in content:
            print("   [OK] IV Estimation: Found (estimated from VIX)")
            features_status['IV'] = 'PARTIAL - Estimated from VIX (not real-time IV surface)'
        else:
            print("   [WARN] IV Estimation: VIX found but unclear if used for IV")
            features_status['IV'] = 'PARTIAL'
    else:
        print("   [FAIL] IV Estimation: NOT FOUND")
        features_status['IV'] = 'MISSING'
except Exception as e:
    print(f"   [FAIL] Error checking IV: {e}")
    features_status['IV'] = 'ERROR'
print()

# 2. Check Greeks (Delta, Gamma, Vega, Theta)
print("2. Greeks (Delta, Gamma, Vega, Theta)")
print("-" * 100)
try:
    if 'GreeksCalculator' in content or 'calculate_greeks' in content or 'greeks_calc' in content:
        print("   [OK] Greeks Calculator: Found")
        if 'delta' in content.lower() and 'gamma' in content.lower() and 'vega' in content.lower() and 'theta' in content.lower():
            print("   [OK] All Greeks: Delta, Gamma, Vega, Theta - FOUND")
            features_status['Greeks'] = 'FULL - All 4 Greeks calculated'
        else:
            print("   [WARN] Greeks: Partial implementation")
            features_status['Greeks'] = 'PARTIAL'
    else:
        print("   [FAIL] Greeks Calculator: NOT FOUND")
        features_status['Greeks'] = 'MISSING'
except Exception as e:
    print(f"   [FAIL] Error checking Greeks: {e}")
    features_status['Greeks'] = 'ERROR'
print()

# 3. Check Theta Decay Model
print("3. Theta Decay Model")
print("-" * 100)
try:
    if 'theta' in content.lower() and ('decay' in content.lower() or 'time' in content.lower()):
        if 'T_0DTE' in content or 'time.*expiration' in content.lower():
            print("   [OK] Theta Decay: Found (0DTE time decay calculation)")
            features_status['Theta_Decay'] = 'FULL - 0DTE decay model'
        else:
            print("   [WARN] Theta Decay: Partial (theta calculated but decay model unclear)")
            features_status['Theta_Decay'] = 'PARTIAL'
    else:
        print("   [FAIL] Theta Decay Model: NOT FOUND")
        features_status['Theta_Decay'] = 'MISSING'
except Exception as e:
    print(f"   [FAIL] Error checking Theta Decay: {e}")
    features_status['Theta_Decay'] = 'ERROR'
print()

# 4. Check Market Microstructure (Order Flow Imbalance)
print("4. Market Microstructure (Order Flow Imbalance)")
print("-" * 100)
try:
    if inst_content and ('order.*flow' in inst_content.lower() or 'ofi' in inst_content.lower() or 'imbalance' in inst_content.lower() or 'microstructure' in inst_content.lower()):
        print("   [OK] Market Microstructure: Found in institutional_features.py")
        features_status['Microstructure'] = 'FOUND'
    else:
        print("   [FAIL] Market Microstructure: NOT FOUND")
        features_status['Microstructure'] = 'MISSING'
except Exception as e:
    print(f"   [FAIL] Error checking Microstructure: {e}")
    features_status['Microstructure'] = 'ERROR'
print()

# 5. Check Correlations (SPY-QQQ-VIX-SPX)
print("5. Correlations (SPY-QQQ-VIX-SPX)")
print("-" * 100)
try:
    if inst_content and ('correlation' in inst_content.lower() or 'spy.*qqq' in inst_content.lower() or 'cross.*asset' in inst_content.lower() or 'corr_spy' in inst_content or 'corr_qqq' in inst_content or 'corr_spx' in inst_content):
        print("   [OK] Correlations: Found in institutional_features.py")
        features_status['Correlations'] = 'FOUND'
    else:
        print("   [FAIL] Correlations: NOT FOUND")
        features_status['Correlations'] = 'MISSING'
except Exception as e:
    print(f"   [FAIL] Error checking Correlations: {e}")
    features_status['Correlations'] = 'ERROR'
print()

# 6. Check Volatility Regime Classification
print("6. Volatility Regime Classification")
print("-" * 100)
try:
    if 'vol_regime' in content or 'volatility.*regime' in content.lower() or 'get_vol_regime' in content:
        print("   [OK] Volatility Regime: Found")
        if 'calm' in content.lower() and 'normal' in content.lower() and 'storm' in content.lower() and 'crash' in content.lower():
            print("   [OK] Regime Classification: Full (Calm, Normal, Storm, Crash)")
            features_status['Vol_Regime'] = 'FULL - 4 regimes'
        else:
            print("   [WARN] Regime Classification: Partial")
            features_status['Vol_Regime'] = 'PARTIAL'
    else:
        print("   [FAIL] Volatility Regime: NOT FOUND")
        features_status['Vol_Regime'] = 'MISSING'
except Exception as e:
    print(f"   [FAIL] Error checking Volatility Regime: {e}")
    features_status['Vol_Regime'] = 'ERROR'
print()

# 7. Check TPO/Market Profile Signals
print("7. TPO/Market Profile Signals")
print("-" * 100)
try:
    if inst_content and ('tpo' in inst_content.lower() or 'market.*profile' in inst_content.lower() or 'time.*price' in inst_content.lower() or 'value_area' in inst_content.lower() or 'poc' in inst_content.lower() or '_extract_market_profile' in inst_content):
        print("   [OK] TPO/Market Profile: Found in institutional_features.py")
        features_status['TPO'] = 'FOUND'
    else:
        print("   [FAIL] TPO/Market Profile: NOT FOUND")
        features_status['TPO'] = 'MISSING'
except Exception as e:
    print(f"   [FAIL] Error checking TPO: {e}")
    features_status['TPO'] = 'ERROR'
print()

# 8. Check if Institutional Features are Enabled
print("8. Institutional Features Integration")
print("-" * 100)
try:
    if 'USE_INSTITUTIONAL_FEATURES' in content:
        if 'USE_INSTITUTIONAL_FEATURES = True' in content:
            print("   [OK] Institutional Features: ENABLED by default")
        elif 'USE_INSTITUTIONAL_FEATURES = False' in content:
            print("   [WARN] Institutional Features: DISABLED (set to False)")
        else:
            print("   [WARN] Institutional Features: Found but status unclear")
        features_status['Integration'] = 'FOUND'
    else:
        print("   [FAIL] Institutional Features Integration: NOT FOUND")
        features_status['Integration'] = 'MISSING'
except Exception as e:
    print(f"   [FAIL] Error checking Integration: {e}")
    features_status['Integration'] = 'ERROR'
print()

# 9. Check Current Observation Shape
print("9. Current Observation Space")
print("-" * 100)
try:
    if '(20, 10)' in content:
        print("   [OK] Observation Shape: (20, 10) - 20 timesteps, 10 features")
        print("        Breakdown: 5 OHLCV + 1 VIX + 4 Greeks (Delta, Gamma, Theta, Vega)")
        features_status['Observation'] = '(20, 10) - OHLCV + VIX + Greeks'
    elif '(20, 5)' in content:
        print("   [WARN] Observation Shape: (20, 5) - 20 timesteps, 5 features (OHLCV only)")
        features_status['Observation'] = '(20, 5) - OHLCV only'
    else:
        print("   [FAIL] Observation Shape: Unknown")
        features_status['Observation'] = 'UNKNOWN'
except Exception as e:
    print(f"   [FAIL] Error checking Observation: {e}")
    features_status['Observation'] = 'ERROR'
print()

# Summary
print("=" * 100)
print("SUMMARY - INSTITUTIONAL FEATURES STATUS")
print("=" * 100)
print()

full_count = sum(1 for v in features_status.values() if 'FULL' in str(v) or v == 'FOUND')
partial_count = sum(1 for v in features_status.values() if 'PARTIAL' in str(v))
missing_count = sum(1 for v in features_status.values() if 'MISSING' in str(v))

for feature, status in features_status.items():
    if "FULL" in str(status) or status == "FOUND" or "(20, 10)" in str(status):
        icon = "[OK]"
    elif "PARTIAL" in str(status):
        icon = "[WARN]"
    else:
        icon = "[FAIL]"
    print(f"{icon} {feature:25s}: {status}")

print()
print(f"[OK] Fully Implemented: {full_count}/8")
print(f"[WARN] Partially Implemented: {partial_count}/8")
print(f"[FAIL] Missing: {missing_count}/8")
print()

if full_count >= 6:
    print("STATUS: INSTITUTIONAL-GRADE (Most features implemented)")
elif full_count >= 4:
    print("STATUS: PARTIAL (Some features missing)")
else:
    print("STATUS: BASIC (Most features missing)")
print()
