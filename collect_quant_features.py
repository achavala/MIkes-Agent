#!/usr/bin/env python3
"""
📊 COMPREHENSIVE QUANT FEATURES COLLECTION

Collects all institutional-grade quant features for historical data:
- IV (Implied Volatility) from VIX
- Greeks (Delta, Gamma, Vega, Theta)
- Theta decay model
- Market microstructure (order flow imbalance)
- Cross-asset correlations (SPY-QQQ-VIX-SPX)
- Volatility regime classification
- Market Profile/TPO signals

Usage:
    python collect_quant_features.py --symbols SPY,QQQ,SPX --start-date 2002-01-01

Author: Mike Agent Institutional Upgrade
Date: December 7, 2025
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import time

from historical_training_system import HistoricalDataCollector
from quant_features_collector import QuantFeaturesCollector, collect_quant_features_for_symbol


def main():
    parser = argparse.ArgumentParser(
        description='Collect comprehensive quant features for historical data'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        default='SPY,QQQ,SPX',
        help='Comma-separated symbols (default: SPY,QQQ,SPX)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2002-01-01',
        help='Start date (YYYY-MM-DD, default: 2002-01-01)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD, default: today)'
    )
    parser.add_argument(
        '--interval',
        type=str,
        default='1d',
        help='Data interval (1d for daily, default: 1d)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/historical/enriched',
        help='Output directory for enriched data (default: data/historical/enriched)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip symbols that already have enriched data'
    )
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("📊 COMPREHENSIVE QUANT FEATURES COLLECTION")
    print("=" * 70)
    print()
    print(f"Symbols: {symbols}")
    print(f"Start Date: {args.start_date}")
    print(f"End Date: {args.end_date or 'today'}")
    print(f"Interval: {args.interval}")
    print(f"Output Directory: {output_dir}")
    print()
    print("Features to collect:")
    print("  ✅ IV (Implied Volatility) from VIX")
    print("  ✅ Greeks (Delta, Gamma, Vega, Theta)")
    print("  ✅ Theta decay model")
    print("  ✅ Market microstructure (order flow imbalance)")
    print("  ✅ Cross-asset correlations (SPY-QQQ-VIX-SPX)")
    print("  ✅ Volatility regime classification")
    print("  ✅ Market Profile/TPO signals")
    print()
    print("=" * 70)
    print()
    
    # Initialize collectors
    data_collector = HistoricalDataCollector(cache_dir="data/historical")
    quant_collector = QuantFeaturesCollector()
    
    start_time = time.time()
    
    # Collect VIX data first (needed for all symbols)
    print("📥 Collecting VIX data...")
    try:
        vix_data = data_collector.get_vix_data(
            start_date=args.start_date,
            end_date=args.end_date,
            use_cache=True
        )
        print(f"✅ VIX data collected: {len(vix_data):,} values")
    except Exception as e:
        print(f"❌ Error collecting VIX: {e}")
        return
    
    print()
    
    # Collect and enrich data for each symbol
    enriched_data = {}
    
    for symbol in symbols:
        print(f"{'='*70}")
        print(f"📊 Processing {symbol}")
        print(f"{'='*70}\n")
        
        # Check if already exists
        output_file = output_dir / f"{symbol}_enriched_{args.start_date}_{args.end_date or 'latest'}.pkl"
        if args.skip_existing and output_file.exists():
            print(f"⏭️  Skipping {symbol} - enriched data already exists")
            print(f"   File: {output_file.name}\n")
            continue
        
        try:
            # Collect base OHLCV data
            print(f"📥 Collecting base data for {symbol}...")
            price_data = data_collector.get_historical_data(
                symbol=symbol,
                start_date=args.start_date,
                end_date=args.end_date,
                interval=args.interval,
                use_cache=True
            )
            
            if len(price_data) == 0:
                print(f"⚠️  No data collected for {symbol}")
                continue
            
            print(f"   ✅ {len(price_data):,} bars collected")
            
            # Enrich with quant features
            print(f"🔧 Enriching {symbol} with quant features...")
            enriched = quant_collector.collect_all_features(
                price_data=price_data,
                vix_data=vix_data,
                symbol=symbol
            )
            
            print(f"   ✅ Features added:")
            new_columns = [col for col in enriched.columns if col not in price_data.columns]
            print(f"      • {len(new_columns)} new feature columns")
            
            # Show sample of new features
            if len(new_columns) > 0:
                print(f"\n   Sample features:")
                for col in new_columns[:10]:
                    sample_val = enriched[col].iloc[-1] if len(enriched) > 0 else 0
                    print(f"      • {col}: {sample_val:.4f}")
                if len(new_columns) > 10:
                    print(f"      ... and {len(new_columns) - 10} more")
            
            # Save enriched data
            enriched.to_pickle(output_file)
            enriched_data[symbol] = enriched
            
            print(f"\n   💾 Saved to: {output_file.name}")
            file_size = output_file.stat().st_size / 1024
            print(f"   📦 File size: {file_size:.2f} KB")
            
        except Exception as e:
            print(f"\n❌ Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Collect cross-asset features (correlations)
    if len(enriched_data) > 1:
        print(f"{'='*70}")
        print("📊 Adding Cross-Asset Correlations")
        print(f"{'='*70}\n")
        
        try:
            # Collect base data for all symbols
            symbol_base_data = {}
            for symbol in symbols:
                if symbol in enriched_data:
                    # Extract base columns from enriched data
                    base_cols = ['open', 'high', 'low', 'close', 'volume']
                    symbol_base_data[symbol] = enriched_data[symbol][base_cols].copy()
            
            # Add cross-asset correlations
            enriched_with_corr = quant_collector.collect_cross_asset_features(
                symbol_data=symbol_base_data,
                vix_data=vix_data
            )
            
            # Merge correlations back into enriched data
            for symbol in enriched_with_corr:
                if symbol in enriched_data:
                    corr_cols = [col for col in enriched_with_corr[symbol].columns 
                                if col.startswith('corr_')]
                    for col in corr_cols:
                        enriched_data[symbol][col] = enriched_with_corr[symbol][col]
            
            print(f"✅ Cross-asset correlations added")
            print(f"   • Correlations between: {', '.join(symbols)}")
            print(f"   • VIX correlations included")
            
            # Re-save with correlations
            for symbol in enriched_data:
                output_file = output_dir / f"{symbol}_enriched_{args.start_date}_{args.end_date or 'latest'}.pkl"
                enriched_data[symbol].to_pickle(output_file)
                print(f"   💾 Updated: {output_file.name}")
            
        except Exception as e:
            print(f"⚠️  Error adding cross-asset features: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Summary
    elapsed = time.time() - start_time
    
    print("=" * 70)
    print("✅ QUANT FEATURES COLLECTION COMPLETE")
    print("=" * 70)
    print()
    print(f"Total time: {elapsed/60:.2f} minutes")
    print(f"Symbols processed: {len(enriched_data)}")
    print()
    
    print("Enriched data files:")
    for symbol in enriched_data:
        output_file = output_dir / f"{symbol}_enriched_{args.start_date}_{args.end_date or 'latest'}.pkl"
        if output_file.exists():
            file_size = output_file.stat().st_size / 1024
            cols = len(enriched_data[symbol].columns)
            rows = len(enriched_data[symbol])
            print(f"  ✅ {symbol}: {output_file.name}")
            print(f"     Size: {file_size:.2f} KB | Rows: {rows:,} | Columns: {cols}")
    
    print()
    print("=" * 70)
    print()
    
    # Feature summary
    if len(enriched_data) > 0:
        print("📊 FEATURE SUMMARY")
        print("=" * 70)
        print()
        
        sample_symbol = list(enriched_data.keys())[0]
        sample_df = enriched_data[sample_symbol]
        
        feature_groups = {
            'IV Features': [col for col in sample_df.columns if 'iv' in col.lower() or 'vix' in col.lower()],
            'Greeks': [col for col in sample_df.columns if any(g in col.lower() for g in ['delta', 'gamma', 'vega', 'theta'])],
            'Theta Decay': [col for col in sample_df.columns if 'theta_decay' in col.lower()],
            'Microstructure': [col for col in sample_df.columns if any(x in col.lower() for x in ['ofi', 'pressure', 'impact', 'spread'])],
            'Regime': [col for col in sample_df.columns if 'regime' in col.lower()],
            'Correlations': [col for col in sample_df.columns if 'corr' in col.lower()],
            'Market Profile': [col for col in sample_df.columns if any(x in col.lower() for x in ['value_area', 'poc', 'profile'])],
        }
        
        for group_name, features in feature_groups.items():
            if features:
                print(f"✅ {group_name}: {len(features)} features")
                for feat in features[:5]:
                    print(f"   • {feat}")
                if len(features) > 5:
                    print(f"   ... and {len(features) - 5} more")
                print()
        
        print(f"📊 Total features: {len(sample_df.columns)}")
        print()
        print("=" * 70)


if __name__ == "__main__":
    main()

