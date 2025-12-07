#!/usr/bin/env python3
"""
📥 HISTORICAL DATA COLLECTION SCRIPT

Collects historical data for SPX, SPY, QQQ from 2002-present
Saves to cache for use in training

Usage:
    python collect_historical_data.py --symbols SPY,QQQ --start-date 2002-01-01
"""

import argparse
from historical_training_system import HistoricalDataCollector
from datetime import datetime
import time


def main():
    parser = argparse.ArgumentParser(description='Collect historical data for training')
    parser.add_argument('--symbols', type=str, default='SPY,QQQ', help='Comma-separated symbols')
    parser.add_argument('--start-date', type=str, default='2002-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD), default=today')
    parser.add_argument('--interval', type=str, default='1m', help='Data interval (1m, 5m, 1h, 1d)')
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    print("=" * 70)
    print("📥 HISTORICAL DATA COLLECTION")
    print("=" * 70)
    print(f"Symbols: {symbols}")
    print(f"Start Date: {args.start_date}")
    print(f"End Date: {args.end_date or 'today'}")
    print(f"Interval: {args.interval}")
    print("=" * 70)
    print()
    print("⚠️  NOTE: This will take several hours for 20+ years of minute data")
    print("   Data will be cached - you can stop and resume anytime")
    print()
    
    collector = HistoricalDataCollector(cache_dir="data/historical")
    
    start_time = time.time()
    
    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"📊 Collecting {symbol} data...")
        print(f"{'='*70}\n")
        
        try:
            data = collector.get_historical_data(
                symbol=symbol,
                start_date=args.start_date,
                end_date=args.end_date,
                interval=args.interval,
                use_cache=True
            )
            
            if len(data) > 0:
                print(f"\n✅ {symbol}: {len(data):,} bars collected")
                
                # Show date range
                if hasattr(data.index, 'min') and hasattr(data.index, 'max'):
                    print(f"   Date Range: {data.index.min()} to {data.index.max()}")
            else:
                print(f"\n⚠️  {symbol}: No data collected")
        
        except Exception as e:
            print(f"\n❌ Error collecting {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    # Collect VIX data
    print(f"\n{'='*70}")
    print("📊 Collecting VIX data...")
    print(f"{'='*70}\n")
    
    try:
        vix_data = collector.get_vix_data(
            start_date=args.start_date,
            end_date=args.end_date,
            use_cache=True
        )
        
        if len(vix_data) > 0:
            print(f"\n✅ VIX: {len(vix_data):,} values collected")
            print(f"   Date Range: {vix_data.index.min()} to {vix_data.index.max()}")
        else:
            print(f"\n⚠️  VIX: No data collected")
    
    except Exception as e:
        print(f"\n❌ Error collecting VIX: {e}")
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("✅ DATA COLLECTION COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed/3600:.2f} hours")
    print(f"Data cached in: data/historical/")
    print("=" * 70)


if __name__ == "__main__":
    main()

