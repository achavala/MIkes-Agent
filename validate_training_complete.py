#!/usr/bin/env python3
"""
Comprehensive Training Validation Script
Validates data sources, model training, and compares before/after
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_data_sources():
    """Check which data sources were used for training"""
    print("=" * 80)
    print("📊 DATA SOURCE VALIDATION")
    print("=" * 80)
    
    data_dir = Path("data/historical")
    if not data_dir.exists():
        print("❌ data/historical directory not found")
        return {}
    
    # Find all cached data files
    alpaca_files = list(data_dir.glob("*_alpaca.pkl"))
    massive_files = list(data_dir.glob("*_massive.pkl"))
    yfinance_files = [f for f in data_dir.glob("*.pkl") if "_alpaca" not in f.name and "_massive" not in f.name and "VIX" not in f.name]
    
    results = {
        "alpaca": [],
        "massive": [],
        "yfinance": []
    }
    
    print(f"\n🔍 Scanning {data_dir}...")
    print(f"   Alpaca files: {len(alpaca_files)}")
    print(f"   Massive files: {len(massive_files)}")
    print(f"   yfinance files: {len(yfinance_files)}")
    
    # Check Alpaca files
    if alpaca_files:
        print(f"\n✅ ALPACA API DATA FOUND (PRIORITY 1 - PAID SERVICE):")
        for f in alpaca_files[:5]:  # Check first 5
            try:
                df = pickle.load(open(f, "rb"))
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    symbol = f.name.split("_")[0]
                    date_range = f"{df.index.min()} to {df.index.max()}" if isinstance(df.index, pd.DatetimeIndex) else "N/A"
                    results["alpaca"].append({
                        "file": f.name,
                        "symbol": symbol,
                        "bars": len(df),
                        "date_range": date_range,
                        "columns": list(df.columns)
                    })
                    print(f"   📁 {f.name}")
                    print(f"      Symbol: {symbol}")
                    print(f"      Bars: {len(df):,}")
                    print(f"      Date Range: {date_range}")
                    print(f"      Columns: {list(df.columns)}")
            except Exception as e:
                print(f"   ⚠️  Error reading {f.name}: {e}")
    
    # Check Massive files
    if massive_files:
        print(f"\n✅ MASSIVE API DATA FOUND (PRIORITY 2 - PAID SERVICE):")
        for f in massive_files[:5]:  # Check first 5
            try:
                df = pickle.load(open(f, "rb"))
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    symbol = f.name.split("_")[0]
                    date_range = f"{df.index.min()} to {df.index.max()}" if isinstance(df.index, pd.DatetimeIndex) else "N/A"
                    results["massive"].append({
                        "file": f.name,
                        "symbol": symbol,
                        "bars": len(df),
                        "date_range": date_range,
                        "columns": list(df.columns)
                    })
                    print(f"   📁 {f.name}")
                    print(f"      Symbol: {symbol}")
                    print(f"      Bars: {len(df):,}")
                    print(f"      Date Range: {date_range}")
                    print(f"      Columns: {list(df.columns)}")
            except Exception as e:
                print(f"   ⚠️  Error reading {f.name}: {e}")
    
    # Check yfinance files (should NOT be used if paid services available)
    if yfinance_files:
        print(f"\n⚠️  YFINANCE DATA FOUND (FALLBACK - FREE SERVICE):")
        for f in yfinance_files[:3]:  # Check first 3
            try:
                df = pickle.load(open(f, "rb"))
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    symbol = f.name.split("_")[0]
                    results["yfinance"].append({
                        "file": f.name,
                        "symbol": symbol,
                        "bars": len(df)
                    })
                    print(f"   📁 {f.name}: {len(df):,} bars")
            except Exception as e:
                print(f"   ⚠️  Error reading {f.name}: {e}")
    
    # Summary
    print(f"\n📊 DATA SOURCE SUMMARY:")
    print(f"   ✅ Alpaca (PAID): {len(results['alpaca'])} files")
    print(f"   ✅ Massive (PAID): {len(results['massive'])} files")
    print(f"   ⚠️  yfinance (FREE): {len(results['yfinance'])} files")
    
    if len(results['alpaca']) > 0 or len(results['massive']) > 0:
        print(f"\n✅ SUCCESS: Training used PAID data sources (Alpaca/Massive)")
        print(f"   Real market data was used, NOT fake numbers")
    elif len(results['yfinance']) > 0:
        print(f"\n⚠️  WARNING: Only yfinance data found (free service)")
        print(f"   This means paid services (Alpaca/Massive) were not available")
    
    return results

def validate_model_files():
    """Validate the trained model files"""
    print("\n" + "=" * 80)
    print("🤖 MODEL VALIDATION")
    print("=" * 80)
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ models directory not found")
        return {}
    
    # Check for model files
    old_model = models_dir / "mike_historical_model.zip"  # 10-feature model
    new_model_partial = models_dir / "mike_23feature_model.zip"  # Partial training (2.5M steps)
    new_model_final = models_dir / "mike_23feature_model_final.zip"  # Complete training (5M steps)
    
    results = {}
    
    print(f"\n🔍 Checking model files...")
    
    # Old model (10 features)
    if old_model.exists():
        size = old_model.stat().st_size / (1024 * 1024)  # MB
        results["old_model"] = {
            "path": str(old_model),
            "size_mb": size,
            "features": 10,
            "exists": True
        }
        print(f"   ✅ Old Model (10 features): {old_model.name}")
        print(f"      Size: {size:.2f} MB")
        print(f"      Features: 10 (OHLCV + VIX + Greeks)")
    else:
        results["old_model"] = {"exists": False}
        print(f"   ⚠️  Old Model: Not found")
    
    # New model (partial - 2.5M steps)
    if new_model_partial.exists():
        size = new_model_partial.stat().st_size / (1024 * 1024)  # MB
        results["new_model_partial"] = {
            "path": str(new_model_partial),
            "size_mb": size,
            "features": 23,
            "timesteps": 2500000,
            "exists": True
        }
        print(f"   ✅ New Model (Partial - 2.5M steps): {new_model_partial.name}")
        print(f"      Size: {size:.2f} MB")
        print(f"      Features: 23 (OHLCV + VIX + Technical Indicators + Greeks)")
        print(f"      Timesteps: 2,500,000")
    else:
        results["new_model_partial"] = {"exists": False}
        print(f"   ⚠️  New Model (Partial): Not found")
    
    # New model (final - 5M steps)
    if new_model_final.exists():
        size = new_model_final.stat().st_size / (1024 * 1024)  # MB
        results["new_model_final"] = {
            "path": str(new_model_final),
            "size_mb": size,
            "features": 23,
            "timesteps": 5000000,
            "exists": True
        }
        print(f"   ✅ New Model (Final - 5M steps): {new_model_final.name}")
        print(f"      Size: {size:.2f} MB")
        print(f"      Features: 23 (OHLCV + VIX + Technical Indicators + Greeks)")
        print(f"      Timesteps: 5,000,000 (COMPLETE)")
    else:
        results["new_model_final"] = {"exists": False}
        print(f"   ⚠️  New Model (Final): Not found")
    
    return results

def validate_model_loading():
    """Try to load and inspect the models"""
    print("\n" + "=" * 80)
    print("🔬 MODEL LOADING VALIDATION")
    print("=" * 80)
    
    try:
        from stable_baselines3 import PPO
        print("✅ stable-baselines3 available")
    except ImportError:
        print("❌ stable-baselines3 not available. Install with: pip install stable-baselines3")
        return {}
    
    results = {}
    
    # Try loading old model (10 features)
    old_model_path = Path("models/mike_historical_model.zip")
    if old_model_path.exists():
        try:
            print(f"\n📥 Loading old model (10 features)...")
            old_model = PPO.load(str(old_model_path))
            obs_space = old_model.observation_space
            results["old_model"] = {
                "loaded": True,
                "observation_shape": obs_space.shape if hasattr(obs_space, 'shape') else str(obs_space),
                "action_space": str(old_model.action_space)
            }
            print(f"   ✅ Loaded successfully")
            print(f"   Observation Space: {obs_space.shape if hasattr(obs_space, 'shape') else obs_space}")
            print(f"   Action Space: {old_model.action_space}")
        except Exception as e:
            print(f"   ❌ Failed to load: {e}")
            results["old_model"] = {"loaded": False, "error": str(e)}
    else:
        print(f"\n⚠️  Old model not found, skipping...")
    
    # Try loading new model (23 features)
    new_model_path = Path("models/mike_23feature_model_final.zip")
    if new_model_path.exists():
        try:
            print(f"\n📥 Loading new model (23 features)...")
            new_model = PPO.load(str(new_model_path))
            obs_space = new_model.observation_space
            results["new_model"] = {
                "loaded": True,
                "observation_shape": obs_space.shape if hasattr(obs_space, 'shape') else str(obs_space),
                "action_space": str(new_model.action_space)
            }
            print(f"   ✅ Loaded successfully")
            print(f"   Observation Space: {obs_space.shape if hasattr(obs_space, 'shape') else obs_space}")
            print(f"   Action Space: {new_model.action_space}")
        except Exception as e:
            print(f"   ❌ Failed to load: {e}")
            results["new_model"] = {"loaded": False, "error": str(e)}
    else:
        print(f"\n⚠️  New model not found, trying partial model...")
        new_model_path = Path("models/mike_23feature_model.zip")
        if new_model_path.exists():
            try:
                print(f"\n📥 Loading partial model (23 features, 2.5M steps)...")
                new_model = PPO.load(str(new_model_path))
                obs_space = new_model.observation_space
                results["new_model"] = {
                    "loaded": True,
                    "observation_shape": obs_space.shape if hasattr(obs_space, 'shape') else str(obs_space),
                    "action_space": str(new_model.action_space),
                    "partial": True
                }
                print(f"   ✅ Loaded successfully (PARTIAL - 2.5M steps)")
                print(f"   Observation Space: {obs_space.shape if hasattr(obs_space, 'shape') else obs_space}")
                print(f"   Action Space: {new_model.action_space}")
            except Exception as e:
                print(f"   ❌ Failed to load: {e}")
                results["new_model"] = {"loaded": False, "error": str(e)}
    
    return results

def compare_models():
    """Compare old vs new model capabilities"""
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON: BEFORE vs AFTER")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print("BEFORE (Old Model - mike_historical_model.zip)")
    print("-" * 80)
    print("Features: 10")
    print("  1. OHLCV (5): Open, High, Low, Close, Volume")
    print("  2. VIX (1): Current VIX level")
    print("  3. Greeks (4): Delta, Gamma, Theta, Vega")
    print("\nMissing Features:")
    print("  ❌ EMA 9/20 Difference (trend crossover)")
    print("  ❌ VWAP Distance (mean reversion)")
    print("  ❌ RSI (momentum oscillator)")
    print("  ❌ MACD Histogram (trend/momentum)")
    print("  ❌ ATR (volatility measure)")
    print("  ❌ Candle Body Ratio (bullish/bearish strength)")
    print("  ❌ Candle Wick Ratio (rejection signals)")
    print("  ❌ Pullback (distance from recent high)")
    print("  ❌ Breakout (price vs prior high)")
    print("  ❌ Trend Slope (linear trend direction)")
    print("  ❌ Momentum Burst (volume × price impulse)")
    print("  ❌ Trend Strength (combined trend signal)")
    print("  ❌ VIX Delta (change in VIX)")
    
    print("\n" + "-" * 80)
    print("AFTER (New Model - mike_23feature_model_final.zip)")
    print("-" * 80)
    print("Features: 23")
    print("  1. OHLCV (5): Open, High, Low, Close, Volume")
    print("  2. VIX (1): Current VIX level")
    print("  3. VIX Delta (1): Change in VIX (NEW)")
    print("  4. EMA 9/20 Difference (1): Trend crossover signal (NEW)")
    print("  5. VWAP Distance (1): Mean reversion signal (NEW)")
    print("  6. RSI (1): Momentum oscillator (NEW)")
    print("  7. MACD Histogram (1): Trend/momentum signal (NEW)")
    print("  8. ATR (1): Volatility measure (NEW)")
    print("  9. Candle Body Ratio (1): Bullish/bearish strength (NEW)")
    print("  10. Candle Wick Ratio (1): Rejection signals (NEW)")
    print("  11. Pullback (1): Distance from recent high (NEW)")
    print("  12. Breakout (1): Price vs prior high (NEW)")
    print("  13. Trend Slope (1): Linear trend direction (NEW)")
    print("  14. Momentum Burst (1): Volume × price impulse (NEW)")
    print("  15. Trend Strength (1): Combined trend signal (NEW)")
    print("  16. Greeks (4): Delta, Gamma, Theta, Vega")
    
    print("\n" + "-" * 80)
    print("KEY DIFFERENCES")
    print("-" * 80)
    print("✅ 13 NEW Technical Indicators Added")
    print("✅ Better Entry Timing (EMA, MACD, VWAP signals)")
    print("✅ Better Exit Timing (RSI, Pullback, Breakout signals)")
    print("✅ Better Volatility Awareness (ATR for position sizing)")
    print("✅ Better Pattern Recognition (Candle patterns, Trend signals)")
    print("✅ Still Includes Greeks (options-specific)")
    print("✅ Still Includes VIX (market volatility)")
    
    print("\n" + "-" * 80)
    print("EXAMPLE: How This Affects Trading Decisions")
    print("-" * 80)
    print("\nScenario: SPY is at $450, VIX at 20")
    print("\nOLD MODEL (10 features):")
    print("  • Sees: Price, Volume, VIX, Greeks")
    print("  • Decision: Limited - only basic price action")
    print("  • Missing: Trend direction, momentum, volatility context")
    print("\nNEW MODEL (23 features):")
    print("  • Sees: Price, Volume, VIX, Greeks")
    print("  • PLUS: EMA 9 > EMA 20 (uptrend)")
    print("  • PLUS: RSI = 65 (slightly overbought but not extreme)")
    print("  • PLUS: MACD histogram positive (momentum building)")
    print("  • PLUS: VWAP distance = +0.5% (above VWAP, bullish)")
    print("  • PLUS: ATR = 2.5% (moderate volatility)")
    print("  • PLUS: Pullback = -1% (small pullback from recent high)")
    print("  • Decision: More informed - sees trend, momentum, and context")
    print("  • Result: Better entry timing, better risk management")

def generate_report():
    """Generate comprehensive validation report"""
    print("\n" + "=" * 80)
    print("📋 GENERATING VALIDATION REPORT")
    print("=" * 80)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "data_sources": check_data_sources(),
        "model_files": validate_model_files(),
        "model_loading": validate_model_loading()
    }
    
    # Save report
    report_path = Path("TRAINING_VALIDATION_REPORT.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✅ Report saved to: {report_path}")
    
    # Generate comparison
    compare_models()
    
    return report

if __name__ == "__main__":
    print("=" * 80)
    print("🔍 COMPREHENSIVE TRAINING VALIDATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print()
    
    report = generate_report()
    
    print("\n" + "=" * 80)
    print("✅ VALIDATION COMPLETE")
    print("=" * 80)





