#!/usr/bin/env python3
"""
Validate Historical Data Collection and Training Status
"""

import os
from pathlib import Path
from datetime import datetime

def validate_data_collection():
    """Validate historical data collection status"""
    print("=" * 70)
    print("📊 DATA COLLECTION STATUS")
    print("=" * 70)
    print()
    
    cache_dir = Path("data/historical")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    from historical_training_system import HistoricalDataCollector
    collector = HistoricalDataCollector(cache_dir=str(cache_dir))
    
    status = {
        'spy': {'found': False, 'bars': 0, 'date_range': None},
        'qqq': {'found': False, 'bars': 0, 'date_range': None},
        'spx': {'found': False, 'bars': 0, 'date_range': None},
        'vix': {'found': False, 'values': 0, 'date_range': None}
    }
    
    # Check SPY
    try:
        spy_data = collector.get_historical_data('SPY', '2002-01-01', None, '1d', use_cache=True)
        if len(spy_data) > 0:
            status['spy'] = {
                'found': True,
                'bars': len(spy_data),
                'date_range': (spy_data.index.min(), spy_data.index.max())
            }
    except Exception as e:
        print(f"⚠️  SPY data error: {e}")
    
    # Check QQQ
    try:
        qqq_data = collector.get_historical_data('QQQ', '2002-01-01', None, '1d', use_cache=True)
        if len(qqq_data) > 0:
            status['qqq'] = {
                'found': True,
                'bars': len(qqq_data),
                'date_range': (qqq_data.index.min(), qqq_data.index.max())
            }
    except Exception as e:
        print(f"⚠️  QQQ data error: {e}")
    
    # Check SPX
    try:
        spx_data = collector.get_historical_data('SPX', '2002-01-01', None, '1d', use_cache=True)
        if len(spx_data) > 0:
            status['spx'] = {
                'found': True,
                'bars': len(spx_data),
                'date_range': (spx_data.index.min(), spx_data.index.max())
            }
    except Exception as e:
        print(f"⚠️  SPX data error: {e}")
    
    # Check VIX
    try:
        vix_data = collector.get_vix_data('2002-01-01', None, use_cache=True)
        if len(vix_data) > 0:
            status['vix'] = {
                'found': True,
                'values': len(vix_data),
                'date_range': (vix_data.index.min(), vix_data.index.max())
            }
    except Exception as e:
        print(f"⚠️  VIX data error: {e}")
    
    # Print results
    print("SYMBOL STATUS:")
    print("-" * 70)
    
    for symbol, data in status.items():
        if data['found']:
            bars_or_values = 'bars' if symbol != 'vix' else 'values'
            count = data['bars'] if symbol != 'vix' else data['values']
            date_range = data['date_range']
            print(f"✅ {symbol.upper()}: {count:,} {bars_or_values}")
            if date_range:
                print(f"   Date range: {date_range[0]} to {date_range[1]}")
        else:
            print(f"❌ {symbol.upper()}: Not collected")
    
    print()
    
    # Overall status
    all_collected = all(s['found'] for s in status.values())
    if all_collected:
        print("✅ DATA COLLECTION: COMPLETE")
        print(f"   SPY: {status['spy']['bars']:,} bars")
        print(f"   QQQ: {status['qqq']['bars']:,} bars")
        print(f"   SPX: {status['spx']['bars']:,} bars")
        print(f"   VIX: {status['vix']['values']:,} values")
    else:
        print("⚠️  DATA COLLECTION: INCOMPLETE")
        missing = [k.upper() for k, v in status.items() if not v['found']]
        print(f"   Missing: {', '.join(missing)}")
    
    print()
    return status


def validate_training():
    """Validate training status"""
    print("=" * 70)
    print("🎓 TRAINING STATUS")
    print("=" * 70)
    print()
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Check for trained models
    model_files = list(models_dir.glob("*.zip"))
    checkpoint_dir = models_dir / "checkpoints"
    checkpoint_files = list(checkpoint_dir.glob("*.zip")) if checkpoint_dir.exists() else []
    
    print("MODEL FILES:")
    print("-" * 70)
    
    if model_files:
        print(f"✅ Found {len(model_files)} trained model(s):")
        for model_file in model_files:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(model_file.stat().st_mtime)
            print(f"   • {model_file.name}")
            print(f"     Size: {size_mb:.2f} MB")
            print(f"     Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("❌ No trained models found")
    
    if checkpoint_files:
        print(f"\n✅ Found {len(checkpoint_files)} checkpoint(s)")
        latest = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        print(f"   Latest: {latest.name}")
    
    print()
    
    # Overall status
    if model_files:
        print("✅ TRAINING: COMPLETE")
        print(f"   Models: {len(model_files)}")
        print(f"   Checkpoints: {len(checkpoint_files)}")
    else:
        print("⚠️  TRAINING: NOT STARTED")
        print("   No model files found")
    
    print()
    return len(model_files) > 0


def main():
    print("\n" + "=" * 70)
    print("🔍 HISTORICAL TRAINING VALIDATION")
    print("=" * 70)
    print()
    
    # Validate data collection
    data_status = validate_data_collection()
    
    # Validate training
    training_complete = validate_training()
    
    # Summary
    print("=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    print()
    
    data_complete = all(s['found'] for s in data_status.values())
    
    print(f"Data Collection: {'✅ COMPLETE' if data_complete else '⚠️  INCOMPLETE'}")
    print(f"Training: {'✅ COMPLETE' if training_complete else '⚠️  NOT STARTED'}")
    print()
    
    if data_complete and training_complete:
        print("🎉 ALL SYSTEMS READY!")
        print("   Data collected and models trained successfully.")
    elif data_complete and not training_complete:
        print("📊 DATA READY - READY TO TRAIN")
        print("   Data collection complete. You can now start training.")
    elif not data_complete:
        print("⏳ DATA COLLECTION IN PROGRESS OR NEEDED")
        print("   Complete data collection before training.")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

