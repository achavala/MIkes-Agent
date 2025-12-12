#!/usr/bin/env python3
"""
Detailed Training Progress Analysis
"""

import os
import re
from pathlib import Path
from datetime import datetime
import subprocess

print("=" * 80)
print("📊 DETAILED TRAINING PROGRESS ANALYSIS")
print("=" * 80)
print()

# 1. Check if training is running
print("1️⃣ PROCESS STATUS")
print("-" * 80)
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
training_processes = [line for line in result.stdout.split('\n') if 'train_historical_model' in line and 'grep' not in line]

if training_processes:
    proc_info = training_processes[0].split()
    pid = proc_info[1]
    cpu = proc_info[2]
    mem = proc_info[3]
    etime = subprocess.run(['ps', '-p', pid, '-o', 'etime='], capture_output=True, text=True).stdout.strip()
    print(f"✅ Training is RUNNING")
    print(f"   PID: {pid}")
    print(f"   CPU: {cpu}%")
    print(f"   Memory: {mem}%")
    print(f"   Runtime: {etime}")
else:
    print("❌ Training is NOT running")

print()

# 2. Check log files
print("2️⃣ LOG FILE ANALYSIS")
print("-" * 80)
log_files = sorted(Path('.').glob('training_*.log'), key=lambda x: x.stat().st_mtime, reverse=True)
if log_files:
    latest_log = log_files[0]
    log_size = latest_log.stat().st_size
    log_lines = sum(1 for _ in open(latest_log))
    log_mtime = datetime.fromtimestamp(latest_log.stat().st_mtime)
    
    print(f"✅ Latest log: {latest_log.name}")
    print(f"   Size: {log_size:,} bytes ({log_size/1024:.1f} KB)")
    print(f"   Lines: {log_lines:,}")
    print(f"   Last updated: {log_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Read last 20 lines
    with open(latest_log, 'r') as f:
        lines = f.readlines()
        print(f"\n   Last 10 lines:")
        for line in lines[-10:]:
            print(f"   {line.rstrip()}")
else:
    print("❌ No log files found")

print()

# 3. Check checkpoint files
print("3️⃣ CHECKPOINT ANALYSIS")
print("-" * 80)
checkpoint_dir = Path('models/checkpoints')
if checkpoint_dir.exists():
    checkpoints = list(checkpoint_dir.glob('*.zip'))
    print(f"✅ Found {len(checkpoints)} checkpoint files")
    
    if checkpoints:
        # Get latest checkpoint
        latest_cp = max(checkpoints, key=lambda x: x.stat().st_mtime)
        cp_size = latest_cp.stat().st_size
        cp_mtime = datetime.fromtimestamp(latest_cp.stat().st_mtime)
        
        print(f"   Latest: {latest_cp.name}")
        print(f"   Size: {cp_size:,} bytes ({cp_size/1024/1024:.1f} MB)")
        print(f"   Time: {cp_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Extract timesteps from filename
        match = re.search(r'(\d+)_steps', latest_cp.name)
        if match:
            cp_timesteps = int(match.group(1))
            total_timesteps = 5000000
            progress_pct = (cp_timesteps / total_timesteps) * 100
            remaining = total_timesteps - cp_timesteps
            
            print(f"\n   📈 Progress:")
            print(f"      Completed: {cp_timesteps:,} / {total_timesteps:,} timesteps")
            print(f"      Progress: {progress_pct:.2f}%")
            print(f"      Remaining: {remaining:,} timesteps")
            
            # Estimate time remaining
            if etime and training_processes:
                # Parse elapsed time
                etime_parts = etime.split(':')
                if len(etime_parts) == 3:  # HH:MM:SS
                    hours = int(etime_parts[0])
                    minutes = int(etime_parts[1])
                    seconds = int(etime_parts[2])
                    total_seconds = hours * 3600 + minutes * 60 + seconds
                    
                    if cp_timesteps > 0:
                        rate = cp_timesteps / total_seconds  # timesteps per second
                        remaining_seconds = remaining / rate if rate > 0 else 0
                        remaining_hours = remaining_seconds / 3600
                        
                        print(f"\n   ⏱️  Time Estimates:")
                        print(f"      Rate: {rate:.1f} timesteps/second")
                        print(f"      Estimated remaining: {remaining_hours:.1f} hours")
        else:
            print("   ⚠️  Could not extract timesteps from checkpoint filename")
    else:
        print("   ⚠️  No checkpoint files found yet")
else:
    print("❌ Checkpoint directory does not exist")

print()

# 4. Check final model
print("4️⃣ FINAL MODEL FILES")
print("-" * 80)
model_files = list(Path('models').glob('mike_rl_model_*.zip'))
if model_files:
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    model_size = latest_model.stat().st_size
    model_mtime = datetime.fromtimestamp(latest_model.stat().st_mtime)
    
    print(f"✅ Latest model: {latest_model.name}")
    print(f"   Size: {model_size:,} bytes ({model_size/1024/1024:.1f} MB)")
    print(f"   Time: {model_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
else:
    print("⚠️  No final model files found yet")

print()

# 5. Parse log for training metrics
print("5️⃣ TRAINING METRICS (from log)")
print("-" * 80)
if log_files:
    with open(latest_log, 'r') as f:
        content = f.read()
    
    # Look for common training metrics
    metrics_found = []
    
    # Timesteps
    timestep_matches = re.findall(r'(\d+)/(\d+)\s+timesteps', content)
    if timestep_matches:
        latest = timestep_matches[-1]
        metrics_found.append(f"Timesteps: {latest[0]:,} / {latest[1]:,}")
    
    # Episodes
    episode_matches = re.findall(r'episode[=:]?\s*(\d+)', content, re.IGNORECASE)
    if episode_matches:
        metrics_found.append(f"Episodes: {episode_matches[-1]}")
    
    # Rewards
    reward_matches = re.findall(r'(?:mean_)?reward[=:]?\s*([-]?\d+\.?\d*)', content, re.IGNORECASE)
    if reward_matches:
        metrics_found.append(f"Latest reward: {reward_matches[-1]}")
    
    # Loss
    loss_matches = re.findall(r'loss[=:]?\s*(\d+\.?\d*)', content, re.IGNORECASE)
    if loss_matches:
        metrics_found.append(f"Latest loss: {loss_matches[-1]}")
    
    if metrics_found:
        for metric in metrics_found:
            print(f"   {metric}")
    else:
        print("   ⚠️  No training metrics found in log")
        print("   (This is normal if training just started)")

print()

# 6. Summary
print("6️⃣ SUMMARY")
print("-" * 80)
if training_processes and checkpoints:
    if match:
        cp_timesteps = int(match.group(1))
        total_timesteps = 5000000
        progress_pct = (cp_timesteps / total_timesteps) * 100
        
        print(f"✅ Training Status: ACTIVE")
        print(f"📊 Progress: {progress_pct:.2f}% complete")
        print(f"📁 Checkpoints: {len(checkpoints)} saved")
        print(f"⏱️  Runtime: {etime}")
        
        if progress_pct < 10:
            print(f"\n⚠️  Early stage - training just started")
        elif progress_pct < 50:
            print(f"\n✅ Good progress - training is advancing")
        elif progress_pct < 90:
            print(f"\n✅ Excellent progress - more than halfway done")
        else:
            print(f"\n🎉 Almost complete - final stages")
    else:
        print("✅ Training is running")
        print("⚠️  Could not calculate exact progress")
else:
    print("❌ Training appears to be stopped or not started")

print()
print("=" * 80)

