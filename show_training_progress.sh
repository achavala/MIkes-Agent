#!/bin/bash
#
# Show Training Progress
# Displays current progress, percentage, and time estimates
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE=$(ls -t training_*.log 2>/dev/null | head -1)
TOTAL_TIMESTEPS=5000000

if [ -z "$LOG_FILE" ] || [ ! -f "$LOG_FILE" ]; then
    echo "❌ No training log file found"
    exit 1
fi

# Extract latest progress from log
python3 << PYEOF
import re
from datetime import datetime, timedelta

log_file = "$LOG_FILE"
total_timesteps_target = $TOTAL_TIMESTEPS

try:
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Find the latest metrics
    latest_timesteps = 0
    latest_time_elapsed = 0
    latest_iterations = 0
    latest_fps = 0
    
    for line in reversed(lines):  # Start from end for latest
        # Look for total_timesteps
        match = re.search(r'total_timesteps\s*\|\s*(\d+)', line)
        if match and latest_timesteps == 0:
            latest_timesteps = int(match.group(1))
        
        # Look for time_elapsed
        match = re.search(r'time_elapsed\s*\|\s*(\d+)', line)
        if match and latest_time_elapsed == 0:
            latest_time_elapsed = int(match.group(1))
        
        # Look for fps
        match = re.search(r'fps\s*\|\s*(\d+)', line)
        if match and latest_fps == 0:
            latest_fps = int(match.group(1))
        
        # Look for iterations
        match = re.search(r'iterations\s*\|\s*(\d+)', line)
        if match:
            latest_iterations = int(match.group(1))
        
        # Stop if we found timesteps
        if latest_timesteps > 0 and latest_time_elapsed > 0:
            break
    
    if latest_timesteps > 0:
        percentage = (latest_timesteps / total_timesteps_target) * 100
        remaining = total_timesteps_target - latest_timesteps
        remaining_pct = 100 - percentage
        
        # Calculate time estimates
        if latest_time_elapsed > 0 and latest_timesteps > 0:
            steps_per_second = latest_timesteps / latest_time_elapsed
            remaining_seconds = remaining / steps_per_second
            remaining_hours = remaining_seconds / 3600
            remaining_days = remaining_hours / 24
            
            elapsed_hours = latest_time_elapsed / 3600
            elapsed_days = elapsed_hours / 24
        else:
            steps_per_second = latest_fps if latest_fps > 0 else 400
            remaining_seconds = remaining / steps_per_second
            remaining_hours = remaining_seconds / 3600
            remaining_days = remaining_hours / 24
            elapsed_hours = 0
            elapsed_days = 0
        
        print("=" * 70)
        print("📊 TRAINING PROGRESS REPORT")
        print("=" * 70)
        print()
        print(f"Current: {latest_timesteps:,} / {total_timesteps_target:,} steps")
        print()
        print(f"✅ Completed: {percentage:.2f}%")
        print(f"⏳ Remaining:  {remaining_pct:.2f}% ({remaining:,} steps)")
        print()
        
        if elapsed_hours > 0:
            print(f"⏱️  Time Elapsed: {elapsed_hours:.1f} hours ({elapsed_days:.2f} days)")
        else:
            print(f"⏱️  Time Elapsed: {latest_time_elapsed} seconds ({latest_time_elapsed/60:.1f} minutes)")
        
        print(f"📈 Speed: ~{steps_per_second:.0f} steps/second")
        print(f"🔄 Iterations: {latest_iterations:,}")
        print()
        print(f"⏱️  Estimated Time Remaining:")
        print(f"   {remaining_hours:.1f} hours ({remaining_days:.2f} days)")
        print()
        
        # Progress bar
        bar_length = 50
        filled = int(bar_length * percentage / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"Progress: [{bar}] {percentage:.2f}%")
        print()
        
        # Milestone tracking
        print("=" * 70)
        print("🎯 MILESTONE TRACKING")
        print("=" * 70)
        print()
        
        milestones = [
            (100000, "100K steps", "~3.3 hours"),
            (500000, "500K steps", "~16.7 hours"),
            (1000000, "1M steps", "~33.3 hours"),
            (5000000, "5M steps", "~7 days")
        ]
        
        for steps, name, expected_time in milestones:
            if latest_timesteps >= steps:
                status = "✅ COMPLETED"
            else:
                remaining_steps = steps - latest_timesteps
                remaining_time = remaining_steps / steps_per_second / 3600
                status = f"⏳ {remaining_time:.1f}h remaining"
            
            pct = (steps / total_timesteps_target) * 100
            print(f"{name:12} ({pct:5.1f}%) → {status}")
        
        print()
        print("=" * 70)
        
    else:
        print("⚠️  Training still initializing...")
        print("   Check log file: $LOG_FILE")
        
except FileNotFoundError:
    print("❌ Log file not found: $LOG_FILE")
except Exception as e:
    print(f"❌ Error: {e}")
PYEOF

