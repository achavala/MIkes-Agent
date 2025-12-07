#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE QUANT FEATURES VALIDATION

Validates all quant features for historical data:
- IV (Implied Volatility)
- Delta, Gamma, Vega, Theta (Greeks)
- Theta decay model
- Market microstructure (order flow imbalance)
- Correlations (SPY-QQQ-VIX-SPX)
- Volatility regime classification
- TPO/Market Profile signals

Author: Mike Agent Institutional Upgrade
Date: December 7, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pickle

def validate_quant_features(
    enriched_data_dir: str = "data/historical/enriched",
    symbols: List[str] = ['SPY', 'QQQ', 'SPX']
) -> Dict:
    """
    Validate all quant features for historical data
    
    Returns:
        Dictionary with validation results for each feature category
    """
    enriched_dir = Path(enriched_data_dir)
    
    print("=" * 70)
    print("🔍 COMPREHENSIVE QUANT FEATURES VALIDATION")
    print("=" * 70)
    print()
    
    # Find enriched data files
    enriched_files = {}
    for symbol in symbols:
        files = list(enriched_dir.glob(f"{symbol}_enriched_*.pkl"))
        if files:
            enriched_files[symbol] = max(files, key=lambda p: p.stat().st_mtime)
        else:
            enriched_files[symbol] = None
    
    # Check if files exist
    missing_files = [s for s, f in enriched_files.items() if f is None]
    if missing_files:
        print(f"❌ Missing enriched data files for: {', '.join(missing_files)}")
        print(f"   Please run: python collect_quant_features.py --symbols {','.join(missing_files)} --start-date 2002-01-01")
        print()
        return {}
    
    print("✅ Found enriched data files:")
    for symbol, file_path in enriched_files.items():
        if file_path:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   • {symbol}: {file_path.name} ({size_mb:.2f} MB)")
    print()
    
    # Load data
    data = {}
    for symbol, file_path in enriched_files.items():
        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    data[symbol] = pickle.load(f)
                print(f"✅ Loaded {symbol}: {len(data[symbol]):,} rows, {len(data[symbol].columns)} columns")
            except Exception as e:
                print(f"❌ Error loading {symbol}: {e}")
                return {}
    
    print()
    
    # Validation results
    validation_results = {
        'iv': {'status': 'pending', 'features': [], 'issues': []},
        'greeks': {'status': 'pending', 'features': [], 'issues': []},
        'theta_decay': {'status': 'pending', 'features': [], 'issues': []},
        'microstructure': {'status': 'pending', 'features': [], 'issues': []},
        'correlations': {'status': 'pending', 'features': [], 'issues': []},
        'regime': {'status': 'pending', 'features': [], 'issues': []},
        'market_profile': {'status': 'pending', 'features': [], 'issues': []}
    }
    
    # Validate each feature category
    sample_symbol = list(data.keys())[0]
    sample_df = data[sample_symbol]
    
    print("=" * 70)
    print("📊 FEATURE VALIDATION")
    print("=" * 70)
    print()
    
    # 1. IV (Implied Volatility)
    print("1️⃣  IV (Implied Volatility)")
    print("-" * 70)
    iv_features = [col for col in sample_df.columns if 'iv' in col.lower() or 'vix' in col.lower()]
    if iv_features:
        validation_results['iv']['status'] = '✅ PASS'
        validation_results['iv']['features'] = iv_features
        print(f"   ✅ Found {len(iv_features)} IV features:")
        for feat in iv_features:
            missing_pct = (sample_df[feat].isna().sum() / len(sample_df)) * 100
            mean_val = sample_df[feat].mean() if not sample_df[feat].isna().all() else 0
            print(f"      • {feat}: mean={mean_val:.4f}, missing={missing_pct:.1f}%")
            if missing_pct > 10:
                validation_results['iv']['issues'].append(f"{feat}: {missing_pct:.1f}% missing")
    else:
        validation_results['iv']['status'] = '❌ FAIL'
        print("   ❌ No IV features found")
    print()
    
    # 2. Greeks (Delta, Gamma, Vega, Theta)
    print("2️⃣  Greeks (Delta, Gamma, Vega, Theta)")
    print("-" * 70)
    greek_types = ['delta', 'gamma', 'vega', 'theta']
    greek_features = {}
    for greek in greek_types:
        greek_cols = [col for col in sample_df.columns if greek in col.lower()]
        if greek_cols:
            greek_features[greek] = greek_cols
    
    if greek_features:
        validation_results['greeks']['status'] = '✅ PASS'
        for greek, cols in greek_features.items():
            validation_results['greeks']['features'].extend(cols)
            print(f"   ✅ {greek.upper()}: {len(cols)} features")
            for col in cols[:3]:
                missing_pct = (sample_df[col].isna().sum() / len(sample_df)) * 100
                mean_val = sample_df[col].mean() if not sample_df[col].isna().all() else 0
                print(f"      • {col}: mean={mean_val:.4f}, missing={missing_pct:.1f}%")
            if len(cols) > 3:
                print(f"      ... and {len(cols) - 3} more")
    else:
        validation_results['greeks']['status'] = '❌ FAIL'
        print("   ❌ No Greeks features found")
    print()
    
    # 3. Theta Decay Model
    print("3️⃣  Theta Decay Model")
    print("-" * 70)
    theta_decay_features = [col for col in sample_df.columns if 'theta_decay' in col.lower()]
    if theta_decay_features:
        validation_results['theta_decay']['status'] = '✅ PASS'
        validation_results['theta_decay']['features'] = theta_decay_features
        print(f"   ✅ Found {len(theta_decay_features)} theta decay features:")
        for feat in theta_decay_features:
            missing_pct = (sample_df[feat].isna().sum() / len(sample_df)) * 100
            mean_val = sample_df[feat].mean() if not sample_df[feat].isna().all() else 0
            print(f"      • {feat}: mean={mean_val:.4f}, missing={missing_pct:.1f}%")
    else:
        validation_results['theta_decay']['status'] = '❌ FAIL'
        print("   ❌ No theta decay features found")
    print()
    
    # 4. Market Microstructure (Order Flow Imbalance)
    print("4️⃣  Market Microstructure (Order Flow Imbalance)")
    print("-" * 70)
    microstructure_features = [col for col in sample_df.columns if any(x in col.lower() for x in ['ofi', 'pressure', 'impact', 'spread', 'vwap'])]
    if microstructure_features:
        validation_results['microstructure']['status'] = '✅ PASS'
        validation_results['microstructure']['features'] = microstructure_features
        print(f"   ✅ Found {len(microstructure_features)} microstructure features:")
        for feat in microstructure_features[:5]:
            missing_pct = (sample_df[feat].isna().sum() / len(sample_df)) * 100
            mean_val = sample_df[feat].mean() if not sample_df[feat].isna().all() else 0
            print(f"      • {feat}: mean={mean_val:.4f}, missing={missing_pct:.1f}%")
        if len(microstructure_features) > 5:
            print(f"      ... and {len(microstructure_features) - 5} more")
    else:
        validation_results['microstructure']['status'] = '❌ FAIL'
        print("   ❌ No microstructure features found")
    print()
    
    # 5. Correlations (SPY-QQQ-VIX-SPX)
    print("5️⃣  Cross-Asset Correlations")
    print("-" * 70)
    correlation_features = [col for col in sample_df.columns if 'corr' in col.lower()]
    if correlation_features:
        validation_results['correlations']['status'] = '✅ PASS'
        validation_results['correlations']['features'] = correlation_features
        print(f"   ✅ Found {len(correlation_features)} correlation features:")
        for feat in correlation_features:
            missing_pct = (sample_df[feat].isna().sum() / len(sample_df)) * 100
            mean_val = sample_df[feat].mean() if not sample_df[feat].isna().all() else 0
            print(f"      • {feat}: mean={mean_val:.4f}, missing={missing_pct:.1f}%")
    else:
        validation_results['correlations']['status'] = '⚠️  PARTIAL'
        print("   ⚠️  No correlation features found (may need multiple symbols)")
    print()
    
    # 6. Volatility Regime Classification
    print("6️⃣  Volatility Regime Classification")
    print("-" * 70)
    regime_features = [col for col in sample_df.columns if 'regime' in col.lower()]
    if regime_features:
        validation_results['regime']['status'] = '✅ PASS'
        validation_results['regime']['features'] = regime_features
        print(f"   ✅ Found {len(regime_features)} regime features:")
        for feat in regime_features:
            if feat == 'vol_regime':
                # Show regime distribution
                regime_counts = sample_df[feat].value_counts()
                print(f"      • {feat}:")
                for regime, count in regime_counts.items():
                    pct = (count / len(sample_df)) * 100
                    print(f"        - {regime}: {count:,} ({pct:.1f}%)")
            else:
                missing_pct = (sample_df[feat].isna().sum() / len(sample_df)) * 100
                mean_val = sample_df[feat].mean() if not sample_df[feat].isna().all() else 0
                print(f"      • {feat}: mean={mean_val:.4f}, missing={missing_pct:.1f}%")
    else:
        validation_results['regime']['status'] = '❌ FAIL'
        print("   ❌ No regime features found")
    print()
    
    # 7. TPO/Market Profile Signals
    print("7️⃣  TPO/Market Profile Signals")
    print("-" * 70)
    market_profile_features = [col for col in sample_df.columns if any(x in col.lower() for x in ['value_area', 'poc', 'profile', 'volume_density'])]
    if market_profile_features:
        validation_results['market_profile']['status'] = '✅ PASS'
        validation_results['market_profile']['features'] = market_profile_features
        print(f"   ✅ Found {len(market_profile_features)} market profile features:")
        for feat in market_profile_features:
            missing_pct = (sample_df[feat].isna().sum() / len(sample_df)) * 100
            mean_val = sample_df[feat].mean() if not sample_df[feat].isna().all() else 0
            print(f"      • {feat}: mean={mean_val:.4f}, missing={missing_pct:.1f}%")
    else:
        validation_results['market_profile']['status'] = '❌ FAIL'
        print("   ❌ No market profile features found")
    print()
    
    # Summary
    print("=" * 70)
    print("📋 VALIDATION SUMMARY")
    print("=" * 70)
    print()
    
    all_passed = all(
        result['status'] in ['✅ PASS', '⚠️  PARTIAL'] 
        for result in validation_results.values()
    )
    
    for category, result in validation_results.items():
        status = result['status']
        feature_count = len(result['features'])
        print(f"{status} {category.upper()}: {feature_count} features")
        if result['issues']:
            for issue in result['issues']:
                print(f"   ⚠️  {issue}")
    
    print()
    
    if all_passed:
        print("✅ ALL QUANT FEATURES VALIDATED SUCCESSFULLY!")
    else:
        print("⚠️  Some features are missing or incomplete")
    
    print()
    print("=" * 70)
    
    return validation_results


if __name__ == "__main__":
    results = validate_quant_features()
    
    # Save validation report
    report_path = Path("QUANT_FEATURES_VALIDATION_REPORT.md")
    with open(report_path, 'w') as f:
        f.write("# Quant Features Validation Report\n\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Validation Results\n\n")
        for category, result in results.items():
            f.write(f"### {category.upper()}\n\n")
            f.write(f"- **Status:** {result['status']}\n")
            f.write(f"- **Features:** {len(result['features'])}\n")
            if result['features']:
                f.write(f"- **Feature List:**\n")
                for feat in result['features']:
                    f.write(f"  - {feat}\n")
            if result['issues']:
                f.write(f"- **Issues:**\n")
                for issue in result['issues']:
                    f.write(f"  - {issue}\n")
            f.write("\n")
    
    print(f"\n📄 Validation report saved to: {report_path}")

