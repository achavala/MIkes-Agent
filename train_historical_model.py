#!/usr/bin/env python3
"""
🚀 HISTORICAL MODEL TRAINING SCRIPT

Trains RL model on historical data (2002-present) with regime-aware sampling
Ensures model learns from good, bad, and worst market days

Usage:
    python train_historical_model.py --symbols SPY,QQQ --start-date 2002-01-01 --timesteps 1000000

Author: Mike Agent Institutional Upgrade
Date: December 6, 2025
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# RL imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Import training system
from historical_training_system import (
    HistoricalDataCollector,
    HistoricalTradingEnv,
    create_regime_aware_training_data
)

# Suppress warnings
os.environ['GYM_NO_DEPRECATION_WARN'] = '1'


class RegimeAwareTrainingCallback:
    """
    Ensures training samples from all market regimes
    """
    
    def __init__(self, vix_data: pd.Series):
        self.vix_data = vix_data
        self.regime_counts = {
            'calm': 0,
            'normal': 0,
            'storm': 0,
            'crash': 0
        }
    
    def classify_regime(self, vix: float) -> str:
        """Classify VIX into regime"""
        if vix < 18:
            return 'calm'
        elif vix < 25:
            return 'normal'
        elif vix < 35:
            return 'storm'
        else:
            return 'crash'
    
    def get_regime_for_date(self, date) -> str:
        """Get regime for a specific date"""
        try:
            if isinstance(date, str):
                date = pd.to_datetime(date).date()
            elif hasattr(date, 'date'):
                date = date.date()
            
            # Find VIX for this date
            if isinstance(self.vix_data.index, pd.DatetimeIndex):
                vix_values = self.vix_data[self.vix_data.index.date == date]
                if len(vix_values) > 0:
                    vix = float(vix_values.iloc[-1])
                    regime = self.classify_regime(vix)
                    self.regime_counts[regime] += 1
                    return regime
            
            return 'normal'  # Default
        except:
            return 'normal'


def create_env_from_data(
    data: pd.DataFrame,
    vix_data: pd.Series,
    symbol: str,
    use_greeks: bool = True,
    use_features: bool = False
) -> HistoricalTradingEnv:
    """Create environment from historical data"""
    env = HistoricalTradingEnv(
        data=data,
        vix_data=vix_data,
        symbol=symbol,
        window_size=20,
        initial_capital=100000.0,
        use_greeks=use_greeks,
        use_features=use_features
    )
    return env


def split_data_by_regime(
    data: pd.DataFrame,
    vix_data: pd.Series
) -> Dict[str, pd.DataFrame]:
    """
    Split data by market regime to ensure balanced training
    """
    regimes = {
        'calm': [],
        'normal': [],
        'storm': [],
        'crash': []
    }
    
    if not isinstance(data.index, pd.DatetimeIndex):
        return {'all': data}
    
    # Group by date
    data['date'] = data.index.date
    
    for date, day_data in data.groupby('date'):
        try:
            # Get VIX for this date
            vix_values = vix_data[vix_data.index.date == date]
            if len(vix_values) > 0:
                vix = float(vix_values.iloc[-1])
                
                if vix < 18:
                    regime = 'calm'
                elif vix < 25:
                    regime = 'normal'
                elif vix < 35:
                    regime = 'storm'
                else:
                    regime = 'crash'
                
                regimes[regime].append(day_data)
        except:
            regimes['normal'].append(day_data)
    
    # Combine days for each regime
    result = {}
    for regime, day_list in regimes.items():
        if len(day_list) > 0:
            combined = pd.concat(day_list, axis=0)
            combined = combined.drop('date', axis=1) if 'date' in combined.columns else combined
            result[regime] = combined
    
    return result


def train_on_historical_data(
    symbols: List[str],
    start_date: str = "2002-01-01",
    end_date: Optional[str] = None,
    total_timesteps: int = 1000000,
    model_name: str = "mike_historical_model",
    use_greeks: bool = True,
    use_features: bool = False,
    regime_balanced: bool = True
):
    """
    Train RL model on historical data with regime-aware sampling
    """
    print("=" * 70)
    print("🚀 HISTORICAL MODEL TRAINING")
    print("=" * 70)
    print(f"Symbols: {symbols}")
    print(f"Date Range: {start_date} to {end_date or 'today'}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Use Greeks: {use_greeks}")
    print(f"Use Features: {use_features}")
    print(f"Regime Balanced: {regime_balanced}")
    print("=" * 70)
    print()
    
    # Initialize data collector
    collector = HistoricalDataCollector(cache_dir="data/historical")
    
    # Collect all data
    print("📥 COLLECTING HISTORICAL DATA...")
    all_data_dict = {}
    vix_data = collector.get_vix_data(start_date, end_date)
    
    for symbol in symbols:
        print(f"\n📊 Collecting {symbol} data...")
        data = collector.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval='1m',
            use_cache=True
        )
        
        if len(data) == 0:
            print(f"⚠️ No data for {symbol}, skipping")
            continue
        
        # Filter to trading hours
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.between_time('09:30', '16:00')
        
        print(f"✅ {symbol}: {len(data):,} bars")
        all_data_dict[symbol] = data
    
    if len(all_data_dict) == 0:
        print("❌ No data collected! Exiting.")
        return
    
    print("\n" + "=" * 70)
    print("📚 PREPARING TRAINING ENVIRONMENTS...")
    print("=" * 70)
    
    # Create environments for each symbol/regime
    envs = []
    
    if regime_balanced:
        # Split by regime for balanced training
        for symbol, data in all_data_dict.items():
            regime_data = split_data_by_regime(data, vix_data)
            
            for regime, regime_df in regime_data.items():
                if len(regime_df) < 100:  # Skip if too little data
                    continue
                
                print(f"  Creating env: {symbol} ({regime}) - {len(regime_df):,} bars")
                env = create_env_from_data(
                    data=regime_df,
                    vix_data=vix_data,
                    symbol=symbol,
                    use_greeks=use_greeks,
                    use_features=use_features
                )
                envs.append(env)
    else:
        # Simple: one env per symbol
        for symbol, data in all_data_dict.items():
            print(f"  Creating env: {symbol} - {len(data):,} bars")
            env = create_env_from_data(
                data=data,
                vix_data=vix_data,
                symbol=symbol,
                use_greeks=use_greeks,
                use_features=use_features
            )
            envs.append(env)
    
    if len(envs) == 0:
        print("❌ No environments created! Exiting.")
        return
    
    print(f"\n✅ Created {len(envs)} training environments")
    
    # Create vectorized environment (use first env as template)
    print("\n" + "=" * 70)
    print("🎓 STARTING TRAINING...")
    print("=" * 70)
    
    # For now, train on first environment (can be enhanced for multi-env)
    training_env = envs[0]
    
    # Wrap with Monitor for logging
    log_dir = "logs/training"
    os.makedirs(log_dir, exist_ok=True)
    training_env = Monitor(training_env, log_dir)
    
    # Create vectorized env
    vec_env = DummyVecEnv([lambda: training_env])
    
    # Model configuration
    policy_kwargs = {}
    if use_features:
        policy_kwargs['net_arch'] = [256, 256, 128]  # Larger network for more features
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="logs/tensorboard",
        policy_kwargs=policy_kwargs,
        device="cpu"
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='models/checkpoints',
        name_prefix=model_name
    )
    
    # Start training
    print(f"\n🏋️ Training for {total_timesteps:,} timesteps...")
    print("   (This will take a while - progress shown below)\n")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    
    training_time = time.time() - start_time
    
    # Save final model
    model_path = f"models/{model_name}.zip"
    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved: {model_path}")
    print(f"Training time: {training_time/3600:.2f} hours")
    print(f"Total timesteps: {total_timesteps:,}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Train RL model on historical data')
    parser.add_argument('--symbols', type=str, default='SPY,QQQ', help='Comma-separated symbols')
    parser.add_argument('--start-date', type=str, default='2002-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD), default=today')
    parser.add_argument('--timesteps', type=int, default=1000000, help='Total training timesteps')
    parser.add_argument('--model-name', type=str, default='mike_historical_model', help='Model name')
    parser.add_argument('--use-greeks', action='store_true', default=True, help='Use Greeks in observations')
    parser.add_argument('--use-features', action='store_true', default=False, help='Use institutional features')
    parser.add_argument('--regime-balanced', action='store_true', default=True, help='Balance training across regimes')
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    train_on_historical_data(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        total_timesteps=args.timesteps,
        model_name=args.model_name,
        use_greeks=args.use_greeks,
        use_features=args.use_features,
        regime_balanced=args.regime_balanced
    )


if __name__ == "__main__":
    main()

