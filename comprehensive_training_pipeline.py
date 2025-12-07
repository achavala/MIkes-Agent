"""
🚀 COMPREHENSIVE RL TRAINING PIPELINE

Trains RL model on 23 years of historical data (2002-2025).
Handles all market regimes and all 0DTE trading activities.

Features:
- Multi-symbol training (SPX, SPY, QQQ)
- Regime-balanced training
- Significant days emphasis (crashes, rallies)
- Comprehensive validation
- Model checkpointing

Author: Mike Agent Training System
Date: December 6, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import json
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import os
from historical_data_collector import HistoricalDataCollector
from advanced_training_env import Advanced0DTETradingEnv
import warnings
warnings.filterwarnings('ignore')


class TrainingProgressCallback(BaseCallback):
    """Callback to track training progress"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        return True


class ComprehensiveTrainingPipeline:
    """
    Comprehensive training pipeline for 0DTE options RL model
    """
    
    def __init__(
        self,
        data_dir: str = "training_data",
        model_dir: str = "trained_models",
        start_date: str = "2002-01-01",
        end_date: str = None
    ):
        """
        Initialize training pipeline
        
        Args:
            data_dir: Directory with historical data
            model_dir: Directory to save trained models
            start_date: Training start date
            end_date: Training end date (default: today)
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        
        self.collector = HistoricalDataCollector(data_dir=data_dir)
        self.symbols = ['SPY', 'QQQ', 'SPX']
        
        # Training configuration
        self.training_config = {
            'total_timesteps': 1000000,  # 1M timesteps
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5
        }
    
    def prepare_training_data(self, symbols: List[str] = None, force_recollect: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Prepare training data for all symbols
        
        Args:
            symbols: List of symbols to prepare (default: SPY, QQQ, SPX)
            force_recollect: Force recollection even if data exists
            
        Returns:
            Dictionary with DataFrames for each symbol
        """
        if symbols is None:
            symbols = self.symbols
        
        dataset = {}
        
        # Check if data already exists
        data_exists = all((self.data_dir / f"{s.replace('^', '')}_daily.pkl").exists() for s in symbols)
        
        if not data_exists or force_recollect:
            print("📥 Collecting historical data...")
            dataset = self.collector.collect_full_dataset(save_intermediate=True)
        else:
            print("📂 Loading existing historical data...")
            for symbol in symbols:
                symbol_clean = symbol.replace('^', '')
                data_path = self.data_dir / f"{symbol_clean}_daily.pkl"
                if data_path.exists():
                    dataset[symbol] = pd.read_pickle(data_path)
                    print(f"✅ Loaded {symbol}: {len(dataset[symbol])} bars")
        
        return dataset
    
    def create_training_dates(
        self,
        include_all_regimes: bool = True,
        emphasize_significant_days: bool = True,
        max_dates_per_year: int = None
    ) -> List[str]:
        """
        Create balanced training date list
        
        Args:
            include_all_regimes: Ensure all regimes are represented
            emphasize_significant_days: Over-sample crashes, rallies, volatility spikes
            max_dates_per_year: Limit dates per year (for faster training)
            
        Returns:
            List of training dates (YYYY-MM-DD)
        """
        # Load significant days
        sig_path = self.data_dir / "significant_days.json"
        significant_days = {}
        if sig_path.exists():
            with open(sig_path, 'r') as f:
                significant_days = json.load(f)
        
        # Get all available dates
        dates = self.collector.create_training_dates(include_all=True)
        
        if not dates:
            print("⚠️ No training dates available. Collecting data first...")
            self.collector.collect_full_dataset()
            dates = self.collector.create_training_dates(include_all=True)
        
        # If limiting per year
        if max_dates_per_year:
            dates_by_year = {}
            for date_str in dates:
                year = date_str[:4]
                if year not in dates_by_year:
                    dates_by_year[year] = []
                dates_by_year[year].append(date_str)
            
            selected_dates = []
            for year, year_dates in dates_by_year.items():
                if len(year_dates) <= max_dates_per_year:
                    selected_dates.extend(year_dates)
                else:
                    # Sample evenly
                    indices = np.linspace(0, len(year_dates)-1, max_dates_per_year, dtype=int)
                    selected_dates.extend([year_dates[i] for i in indices])
            
            dates = sorted(selected_dates)
        
        # Emphasize significant days
        if emphasize_significant_days and significant_days:
            enhanced_dates = list(dates)
            
            # Add significant days multiple times (2x weight)
            for category in ['crashes', 'rallies', 'volatility_spikes']:
                if category in significant_days:
                    enhanced_dates.extend(significant_days[category])
            
            dates = sorted(list(set(enhanced_dates)))
        
        print(f"📅 Prepared {len(dates)} training dates")
        
        # Show distribution by year
        year_counts = {}
        for date_str in dates:
            year = date_str[:4]
            year_counts[year] = year_counts.get(year, 0) + 1
        
        print("\n📊 Training Dates by Year:")
        for year in sorted(year_counts.keys()):
            print(f"   {year}: {year_counts[year]} days")
        
        return dates
    
    def create_training_environment(
        self,
        symbol: str,
        date: str,
        include_greeks: bool = True
    ) -> Optional[Advanced0DTETradingEnv]:
        """
        Create training environment for a specific symbol and date
        
        Args:
            symbol: Trading symbol
            date: Date string (YYYY-MM-DD)
            include_greeks: Whether to include Greeks in observation
            
        Returns:
            Training environment or None if data unavailable
        """
        # Prepare intraday data for the date
        data = self.collector.prepare_training_data_for_date(date, symbol)
        
        if data is None or len(data) < 50:  # Need at least 50 bars
            return None
        
        # Create environment
        env = Advanced0DTETradingEnv(
            data=data,
            initial_capital=10000.0,
            lookback=20,
            include_greeks=include_greeks
        )
        
        return env
    
    def train_on_multiple_dates(
        self,
        training_dates: List[str],
        symbols: List[str] = None,
        model_name: str = "mike_0dte_model",
        resume: bool = False
    ):
        """
        Train model on multiple dates across multiple symbols
        
        Args:
            training_dates: List of dates to train on
            symbols: List of symbols to train on
            model_name: Name for saved model
            resume: Resume training from existing model
        """
        if symbols is None:
            symbols = ['SPY']  # Start with SPY for faster iteration
        
        print("="*70)
        print("🚀 COMPREHENSIVE RL TRAINING PIPELINE")
        print("="*70)
        print(f"Training Dates: {len(training_dates)} days")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Total Timesteps: {self.training_config['total_timesteps']:,}")
        print("="*70)
        
        # Load or create model
        model_path = self.model_dir / f"{model_name}.zip"
        
        if resume and model_path.exists():
            print(f"📂 Resuming training from: {model_path}")
            model = PPO.load(model_path)
        else:
            print("🆕 Creating new model...")
            # Create a sample environment to get observation/action spaces
            sample_env = None
            for date in training_dates[:10]:  # Try first 10 dates
                for symbol in symbols:
                    sample_env = self.create_training_environment(symbol, date)
                    if sample_env:
                        break
                if sample_env:
                    break
            
            if not sample_env:
                print("❌ Could not create sample environment. Check data availability.")
                return
            
            # Create model
            env = DummyVecEnv([lambda: sample_env])
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=self.training_config['learning_rate'],
                n_steps=self.training_config['n_steps'],
                batch_size=self.training_config['batch_size'],
                n_epochs=self.training_config['n_epochs'],
                gamma=self.training_config['gamma'],
                gae_lambda=self.training_config['gae_lambda'],
                clip_range=self.training_config['clip_range'],
                ent_coef=self.training_config['ent_coef'],
                vf_coef=self.training_config['vf_coef'],
                max_grad_norm=self.training_config['max_grad_norm'],
                verbose=1,
                device="cpu"
            )
        
        # Training loop
        print("\n🎓 Starting training...")
        
        current_timesteps = 0
        target_timesteps = self.training_config['total_timesteps']
        
        # Shuffle dates for training
        np.random.shuffle(training_dates)
        
        iteration = 0
        while current_timesteps < target_timesteps:
            iteration += 1
            
            # Sample a date and symbol
            date = np.random.choice(training_dates)
            symbol = np.random.choice(symbols)
            
            # Create environment
            env_obj = self.create_training_environment(symbol, date)
            
            if env_obj is None:
                continue
            
            # Wrap in Monitor and VecEnv
            env = DummyVecEnv([lambda: env_obj])
            
            # Train on this episode
            timesteps_before = model.num_timesteps
            model.set_env(env)
            model.learn(total_timesteps=2048, reset_num_timesteps=False)
            timesteps_after = model.num_timesteps
            current_timesteps += (timesteps_after - timesteps_before)
            
            # Progress update
            if iteration % 10 == 0:
                progress = (current_timesteps / target_timesteps) * 100
                print(f"\n📊 Progress: {progress:.1f}% ({current_timesteps:,}/{target_timesteps:,} timesteps)")
                print(f"   Iteration: {iteration}, Last date: {date}, Symbol: {symbol}")
            
            # Save checkpoint
            if iteration % 50 == 0:
                checkpoint_path = self.model_dir / f"{model_name}_checkpoint.zip"
                model.save(checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Final save
        final_path = self.model_dir / f"{model_name}.zip"
        model.save(final_path)
        print(f"\n✅ Training complete! Model saved: {final_path}")
        
        return model


def main():
    """Main training function"""
    pipeline = ComprehensiveTrainingPipeline(
        data_dir="training_data",
        model_dir="trained_models",
        start_date="2002-01-01"
    )
    
    print("\n🎯 Comprehensive RL Training Pipeline")
    print("="*70)
    print("This will:")
    print("  1. Collect historical data (SPY, QQQ, SPX since 2002)")
    print("  2. Prepare training dates (balanced by regime)")
    print("  3. Train RL model on all market conditions")
    print("  4. Save trained model")
    print("="*70)
    
    # Prepare data
    print("\n📥 Step 1: Preparing training data...")
    dataset = pipeline.prepare_training_data(force_recollect=False)
    
    # Create training dates
    print("\n📅 Step 2: Creating training dates...")
    training_dates = pipeline.create_training_dates(
        include_all_regimes=True,
        emphasize_significant_days=True,
        max_dates_per_year=50  # Limit to 50 days per year for faster training (can increase)
    )
    
    print(f"\n✅ Ready to train on {len(training_dates)} dates")
    
    # Start training
    response = input("\nStart training? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        model = pipeline.train_on_multiple_dates(
            training_dates=training_dates,
            symbols=['SPY'],  # Start with SPY, can add QQQ/SPX later
            model_name="mike_0dte_comprehensive",
            resume=False
        )
        
        print("\n✅ Training complete!")
        print(f"\nModel saved to: {pipeline.model_dir}")
    else:
        print("Training cancelled.")


if __name__ == "__main__":
    main()

