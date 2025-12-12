#!/usr/bin/env python3
"""
Performance analysis of the trained model
Analyzes training metrics, checkpoints, and model characteristics
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

def analyze_training_performance():
    """Analyze training performance from logs and checkpoints"""
    print("=" * 80)
    print("📊 PERFORMANCE ANALYSIS")
    print("=" * 80)
    print()
    
    # 1. Checkpoint analysis
    print("1️⃣ CHECKPOINT ANALYSIS")
    print("-" * 80)
    checkpoint_dir = Path("models/checkpoints")
    
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("*.zip"))
        print(f"Total checkpoints: {len(checkpoints)}")
        
        if checkpoints:
            # Extract timesteps from filenames
            timesteps_list = []
            for cp in checkpoints:
                match = re.search(r'(\d+)_steps', cp.name)
                if match:
                    timesteps_list.append(int(match.group(1)))
            
            if timesteps_list:
                print(f"   First: {min(timesteps_list):,} timesteps")
                print(f"   Last: {max(timesteps_list):,} timesteps")
                print(f"   Increment: {timesteps_list[1] - timesteps_list[0]:,} timesteps" if len(timesteps_list) > 1 else "")
                print(f"   Total progress: {max(timesteps_list):,} / 5,000,000 ({max(timesteps_list)/5000000*100:.1f}%)")
        else:
            print("   No checkpoints found")
    else:
        print("   Checkpoint directory not found")
    print()
    
    # 2. Model file analysis
    print("2️⃣ MODEL FILE ANALYSIS")
    print("-" * 80)
    model_path = Path("models/mike_historical_model.zip")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        mod_time = datetime.fromtimestamp(model_path.stat().st_mtime)
        print(f"✅ Model file: {model_path}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"❌ Model file not found: {model_path}")
    print()
    
    # 3. Training log analysis
    print("3️⃣ TRAINING LOG ANALYSIS")
    print("-" * 80)
    log_files = sorted(Path(".").glob("training_*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if log_files:
        latest_log = log_files[0]
        size_mb = latest_log.stat().st_size / (1024 * 1024)
        lines = sum(1 for _ in open(latest_log))
        mod_time = datetime.fromtimestamp(latest_log.stat().st_mtime)
        
        print(f"Latest log: {latest_log.name}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Lines: {lines:,}")
        print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Extract key metrics from log
        with open(latest_log, 'r') as f:
            content = f.read()
            
            # Look for training completion
            if "TRAINING COMPLETE" in content:
                print("   ✅ Training completed successfully")
            
            # Look for training time
            time_match = re.search(r'Training time: ([\d.]+) hours', content)
            if time_match:
                print(f"   Training time: {time_match.group(1)} hours")
            
            # Look for timesteps
            ts_match = re.search(r'Total timesteps: ([\d,]+)', content)
            if ts_match:
                print(f"   Total timesteps: {ts_match.group(1)}")
    else:
        print("   No training logs found")
    print()
    
    # 4. Summary
    print("4️⃣ SUMMARY")
    print("-" * 80)
    print("✅ Performance analysis complete")
    print()
    print("Model Status:")
    if model_path.exists():
        print("   ✅ Final model saved")
    else:
        print("   ⚠️  Final model not found")
    
    if checkpoints:
        print(f"   ✅ {len(checkpoints)} checkpoints available")
    else:
        print("   ⚠️  No checkpoints found")
    
    if log_files:
        print(f"   ✅ Training logs available")
    else:
        print("   ⚠️  No training logs found")
    print()

if __name__ == "__main__":
    analyze_training_performance()

