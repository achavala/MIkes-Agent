#!/usr/bin/env python3
"""
Training Time Estimation Calculator

Estimates training time based on:
- Dataset size (rows, columns)
- Number of timesteps
- Hardware (CPU/GPU)
- Model complexity
"""

import sys

def estimate_training_time():
    """Estimate training time for the historical dataset"""
    
    # Dataset parameters
    symbols = ['SPY', 'QQQ', 'SPX']
    rows_per_symbol = 6022
    columns_per_symbol = 77
    total_rows = rows_per_symbol * len(symbols)
    
    # Training parameters
    timesteps = 5000000  # 5M timesteps (default)
    window_size = 20
    episodes_per_day = 10  # Approximate episodes per trading day
    
    print("=" * 70)
    print("📊 TRAINING TIME ESTIMATION")
    print("=" * 70)
    print()
    
    print("DATASET PARAMETERS:")
    print("-" * 70)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Rows per symbol: {rows_per_symbol:,}")
    print(f"Columns per symbol: {columns_per_symbol}")
    print(f"Total rows: {total_rows:,}")
    print(f"Window size: {window_size}")
    print(f"Timesteps: {timesteps:,}")
    print()
    
    # Calculate approximate iterations
    steps_per_episode = rows_per_symbol - window_size  # ~6000 steps per episode
    episodes_total = timesteps / steps_per_episode  # ~833 episodes for 5M steps
    
    print("TRAINING ITERATIONS:")
    print("-" * 70)
    print(f"Steps per episode: ~{steps_per_episode:,}")
    print(f"Total episodes: ~{episodes_total:.0f}")
    print(f"Episodes per symbol: ~{episodes_total / len(symbols):.0f}")
    print()
    
    # Time estimates (per 1000 steps)
    # These are conservative estimates based on typical RL training speeds
    
    print("TIME ESTIMATES (Conservative):")
    print("-" * 70)
    print()
    
    # CPU-only (Mac without GPU acceleration)
    cpu_per_1k_steps = 120  # seconds per 1000 steps (conservative)
    cpu_total_hours = (timesteps / 1000) * cpu_per_1k_steps / 3600
    cpu_total_days = cpu_total_hours / 24
    
    print("🖥️  CPU-Only (Mac without GPU):")
    print(f"   Speed: ~{cpu_per_1k_steps}s per 1,000 steps")
    print(f"   Total time: ~{cpu_total_hours:.1f} hours ({cpu_total_days:.1f} days)")
    print(f"   Per 1M steps: ~{cpu_total_hours / (timesteps / 1000000):.1f} hours")
    print()
    
    # GPU-accelerated (if available)
    gpu_per_1k_steps = 30  # seconds per 1000 steps (with GPU)
    gpu_total_hours = (timesteps / 1000) * gpu_per_1k_steps / 3600
    gpu_total_days = gpu_total_hours / 24
    
    print("🚀 GPU-Accelerated (if available):")
    print(f"   Speed: ~{gpu_per_1k_steps}s per 1,000 steps")
    print(f"   Total time: ~{gpu_total_hours:.1f} hours ({gpu_total_days:.1f} days)")
    print(f"   Per 1M steps: ~{gpu_total_hours / (timesteps / 1000000):.1f} hours")
    print()
    
    # Realistic estimate (some GPU, some CPU)
    realistic_per_1k_steps = 60  # seconds per 1000 steps (mixed)
    realistic_total_hours = (timesteps / 1000) * realistic_per_1k_steps / 3600
    realistic_total_days = realistic_total_hours / 24
    
    print("⚡ Realistic Estimate (Mixed CPU/GPU):")
    print(f"   Speed: ~{realistic_per_1k_steps}s per 1,000 steps")
    print(f"   Total time: ~{realistic_total_hours:.1f} hours ({realistic_total_days:.1f} days)")
    print(f"   Per 1M steps: ~{realistic_total_hours / (timesteps / 1000000):.1f} hours")
    print()
    
    print("=" * 70)
    print("💡 RECOMMENDATIONS")
    print("=" * 70)
    print()
    print(f"For {timesteps:,} timesteps:")
    print()
    print("1. START WITH SHORTER RUN:")
    print("   • Try 500,000 steps first (~{:.1f} hours CPU)".format((500000 / 1000) * cpu_per_1k_steps / 3600))
    print("   • Validate model is learning")
    print("   • Then scale up to full training")
    print()
    print("2. USE CHECKPOINTING:")
    print("   • Save model every 100,000 steps")
    print("   • Can resume if interrupted")
    print("   • Monitor progress incrementally")
    print()
    print("3. CONSIDER CLOUD TRAINING:")
    print("   • Google Colab (free GPU)")
    print("   • AWS/GCP (pay-per-use)")
    print("   • Much faster than local CPU")
    print()
    print("4. OPTIMIZE TRAINING:")
    print("   • Reduce timesteps if needed (2M instead of 5M)")
    print("   • Use smaller window size (15 instead of 20)")
    print("   • Train on subset of data first")
    print()
    
    return {
        'timesteps': timesteps,
        'cpu_hours': cpu_total_hours,
        'cpu_days': cpu_total_days,
        'gpu_hours': gpu_total_hours,
        'gpu_days': gpu_total_days,
        'realistic_hours': realistic_total_hours,
        'realistic_days': realistic_total_days
    }


if __name__ == "__main__":
    estimate_training_time()

