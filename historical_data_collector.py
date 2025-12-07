"""
📊 HISTORICAL DATA COLLECTOR

Collects and organizes historical data for SPX, SPY, QQQ since 2002.
Prepares comprehensive dataset for RL model training across all market regimes.

Features:
- Daily and intraday data collection
- Market regime labeling
- 0DTE options simulation
- Data validation and cleaning
- Organized storage for training

Author: Mike Agent Training System
Date: December 6, 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class HistoricalDataCollector:
    """
    Comprehensive historical data collector for RL training
    """
    
    def __init__(self, data_dir: str = "training_data"):
        """
        Initialize data collector
        
        Args:
            data_dir: Directory to store collected data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.symbols = ['SPY', 'QQQ', '^SPX']  # SPX uses ^ prefix for yfinance
        self.start_date = "2002-01-01"
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Market regime thresholds
        self.regime_thresholds = {
            'calm': 18,
            'normal': 25,
            'storm': 35,
            'crash': float('inf')
        }
    
    def collect_daily_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Collect daily OHLCV data for a symbol
        
        Args:
            symbol: Trading symbol (SPY, QQQ, or ^SPX)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with daily OHLCV data
        """
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
        
        print(f"📥 Collecting daily data: {symbol} from {start_date} to {end_date}...")
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if len(data) == 0:
                print(f"⚠️ No data returned for {symbol}")
                return pd.DataFrame()
            
            # Handle MultiIndex columns (yfinance 2025+ compatibility)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Ensure lowercase columns
            data.columns = [col.lower() for col in data.columns]
            
            # Add symbol column
            clean_symbol = symbol.replace('^', '')
            data['symbol'] = clean_symbol
            
            # Add date column
            data['date'] = data.index.date
            
            print(f"✅ Collected {len(data)} daily bars for {symbol}")
            return data
            
        except Exception as e:
            print(f"❌ Error collecting daily data for {symbol}: {e}")
            return pd.DataFrame()
    
    def collect_intraday_data(self, symbol: str, date: str, interval: str = '1m') -> pd.DataFrame:
        """
        Collect intraday (minute) data for a specific date
        
        Args:
            symbol: Trading symbol
            date: Date string (YYYY-MM-DD)
            interval: Data interval ('1m', '5m', etc.)
            
        Returns:
            DataFrame with intraday data
        """
        try:
            ticker = yf.Ticker(symbol)
            # Get data for the specific date
            start = pd.Timestamp(date)
            end = start + timedelta(days=1)
            
            data = ticker.history(start=start, end=end, interval=interval)
            
            if len(data) == 0:
                return pd.DataFrame()
            
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Ensure lowercase columns
            data.columns = [col.lower() for col in data.columns]
            
            return data
            
        except Exception as e:
            print(f"⚠️ Error collecting intraday data for {symbol} on {date}: {e}")
            return pd.DataFrame()
    
    def get_vix_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Collect VIX data for regime classification
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with VIX data
        """
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
        
        print(f"📥 Collecting VIX data from {start_date} to {end_date}...")
        
        try:
            vix = yf.Ticker("^VIX")
            data = vix.history(start=start_date, end=end_date, interval='1d')
            
            if len(data) == 0:
                return pd.DataFrame()
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            data.columns = [col.lower() for col in data.columns]
            data['date'] = data.index.date
            
            print(f"✅ Collected {len(data)} VIX bars")
            return data
            
        except Exception as e:
            print(f"❌ Error collecting VIX data: {e}")
            return pd.DataFrame()
    
    def classify_market_regime(self, vix: float) -> str:
        """
        Classify market regime based on VIX
        
        Args:
            vix: VIX level
            
        Returns:
            Regime string: 'calm', 'normal', 'storm', 'crash'
        """
        if vix < self.regime_thresholds['calm']:
            return 'calm'
        elif vix < self.regime_thresholds['normal']:
            return 'normal'
        elif vix < self.regime_thresholds['storm']:
            return 'storm'
        else:
            return 'crash'
    
    def identify_significant_days(self, daily_data: pd.DataFrame, vix_data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify significant trading days (crashes, rallies, volatility spikes)
        
        Args:
            daily_data: Daily price data
            vix_data: VIX data
            
        Returns:
            Dictionary with lists of dates for each category
        """
        significant_days = {
            'crashes': [],
            'rallies': [],
            'volatility_spikes': [],
            'calm_days': []
        }
        
        # Merge data
        if len(vix_data) == 0:
            return significant_days
        
        # Ensure dates are in same format
        if 'date' not in daily_data.columns:
            daily_data['date'] = daily_data.index.date if hasattr(daily_data.index, 'date') else pd.to_datetime(daily_data.index).date
        
        if 'date' not in vix_data.columns:
            vix_data['date'] = vix_data.index.date if hasattr(vix_data.index, 'date') else pd.to_datetime(vix_data.index).date
        
        # Convert dates to same type for merging
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        vix_data['date'] = pd.to_datetime(vix_data['date'])
        
        merged = daily_data.merge(
            vix_data[['date', 'close']],
            on='date',
            how='inner',
            suffixes=('', '_vix')
        )
        
        if len(merged) == 0:
            return significant_days
        
        # Calculate daily returns
        merged['daily_return'] = merged['close'].pct_change()
        merged['vix_close'] = merged['close_vix']
        
        # Identify crashes (> -3% daily return)
        crashes = merged[merged['daily_return'] < -0.03]
        significant_days['crashes'] = crashes['date'].dt.strftime('%Y-%m-%d').tolist()
        
        # Identify rallies (> +3% daily return)
        rallies = merged[merged['daily_return'] > 0.03]
        significant_days['rallies'] = rallies['date'].dt.strftime('%Y-%m-%d').tolist()
        
        # Identify volatility spikes (VIX > 35)
        vol_spikes = merged[merged['vix_close'] > 35]
        significant_days['volatility_spikes'] = vol_spikes['date'].dt.strftime('%Y-%m-%d').tolist()
        
        # Identify calm days (VIX < 15)
        calm_days = merged[merged['vix_close'] < 15]
        significant_days['calm_days'] = calm_days['date'].dt.strftime('%Y-%m-%d').tolist()
        
        return significant_days
    
    def collect_full_dataset(self, save_intermediate: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Collect complete historical dataset for all symbols
        
        Args:
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Dictionary with DataFrames for each symbol
        """
        print("="*70)
        print("🚀 STARTING COMPREHENSIVE HISTORICAL DATA COLLECTION")
        print("="*70)
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print("="*70)
        
        dataset = {}
        vix_data = self.get_vix_data()
        
        # Save VIX data
        if save_intermediate and len(vix_data) > 0:
            vix_path = self.data_dir / "vix_daily.pkl"
            vix_data.to_pickle(vix_path)
            print(f"💾 Saved VIX data: {vix_path}")
        
        # Collect data for each symbol
        for symbol in self.symbols:
            print(f"\n{'='*70}")
            print(f"Processing: {symbol}")
            print(f"{'='*70}")
            
            daily_data = self.collect_daily_data(symbol)
            
            if len(daily_data) > 0:
                # Merge with VIX for regime classification
                if len(vix_data) > 0:
                    # Ensure date columns exist
                    if 'date' not in daily_data.columns:
                        daily_data['date'] = daily_data.index.date if hasattr(daily_data.index, 'date') else pd.to_datetime(daily_data.index).date
                    
                    if 'date' not in vix_data.columns:
                        vix_data['date'] = vix_data.index.date if hasattr(vix_data.index, 'date') else pd.to_datetime(vix_data.index).date
                    
                    # Convert dates to datetime for merging
                    daily_data['date'] = pd.to_datetime(daily_data['date'])
                    vix_data['date'] = pd.to_datetime(vix_data['date'])
                    
                    # Merge VIX data
                    daily_data = daily_data.merge(
                        vix_data[['date', 'close']],
                        on='date',
                        how='left',
                        suffixes=('', '_vix')
                    )
                    daily_data['vix'] = daily_data['close_vix'].fillna(20.0)
                    daily_data['regime'] = daily_data['vix'].apply(self.classify_market_regime)
                
                dataset[symbol] = daily_data
                
                # Save intermediate results
                if save_intermediate:
                    symbol_clean = symbol.replace('^', '')
                    daily_path = self.data_dir / f"{symbol_clean}_daily.pkl"
                    daily_data.to_pickle(daily_path)
                    print(f"💾 Saved daily data: {daily_path}")
        
        # Identify significant days
        if len(vix_data) > 0 and len(dataset) > 0:
            # Use SPY as reference for significant days
            if 'SPY' in dataset:
                significant_days = self.identify_significant_days(dataset['SPY'], vix_data)
                
                if save_intermediate:
                    sig_path = self.data_dir / "significant_days.json"
                    with open(sig_path, 'w') as f:
                        json.dump(significant_days, f, indent=2)
                    print(f"\n💾 Saved significant days: {sig_path}")
                    
                    # Print summary
                    print("\n📊 SIGNIFICANT DAYS SUMMARY:")
                    for category, dates in significant_days.items():
                        print(f"   {category}: {len(dates)} days")
        
        print("\n" + "="*70)
        print("✅ DATA COLLECTION COMPLETE")
        print("="*70)
        
        return dataset
    
    def create_training_dates(self, include_all: bool = True, sample_size: int = None) -> List[str]:
        """
        Create list of training dates, ensuring coverage of all regimes
        
        Args:
            include_all: Include all available dates
            sample_size: If specified, sample this many dates (balanced by regime)
            
        Returns:
            List of date strings (YYYY-MM-DD)
        """
        # Load VIX data to get regime information
        vix_path = self.data_dir / "vix_daily.pkl"
        if not vix_path.exists():
            print("⚠️ VIX data not found. Collecting...")
            vix_data = self.get_vix_data()
            if len(vix_data) > 0:
                vix_data.to_pickle(vix_path)
            else:
                return []
        else:
            vix_data = pd.read_pickle(vix_path)
        
        vix_data['regime'] = vix_data['close'].apply(self.classify_market_regime)
        vix_data['date'] = vix_data.index.date
        
        if include_all:
            dates = vix_data['date'].dt.strftime('%Y-%m-%d').tolist()
        else:
            # Balanced sampling by regime
            dates_by_regime = {}
            for regime in ['calm', 'normal', 'storm', 'crash']:
                regime_dates = vix_data[vix_data['regime'] == regime]['date'].dt.strftime('%Y-%m-%d').tolist()
                dates_by_regime[regime] = regime_dates
            
            # Sample evenly from each regime
            if sample_size:
                dates_per_regime = max(1, sample_size // 4)
                dates = []
                for regime_dates in dates_by_regime.values():
                    if len(regime_dates) > 0:
                        sampled = np.random.choice(regime_dates, min(dates_per_regime, len(regime_dates)), replace=False)
                        dates.extend(sampled.tolist())
            else:
                dates = []
                for regime_dates in dates_by_regime.values():
                    dates.extend(regime_dates)
        
        return sorted(list(set(dates)))
    
    def prepare_training_data_for_date(self, date: str, symbol: str = 'SPY') -> Optional[pd.DataFrame]:
        """
        Prepare complete training data for a specific date
        
        Args:
            date: Date string (YYYY-MM-DD)
            symbol: Trading symbol
            
        Returns:
            DataFrame with intraday data ready for training
        """
        # Collect intraday data
        intraday = self.collect_intraday_data(symbol, date, interval='1m')
        
        if len(intraday) == 0:
            return None
        
        # Load VIX data for regime
        vix_path = self.data_dir / "vix_daily.pkl"
        if vix_path.exists():
            vix_data = pd.read_pickle(vix_path)
            vix_data['date'] = vix_data.index.date
            date_obj = pd.Timestamp(date).date()
            vix_row = vix_data[vix_data['date'] == date_obj]
            
            if len(vix_row) > 0:
                vix_value = float(vix_row['close'].iloc[0])
                regime = self.classify_market_regime(vix_value)
                intraday['vix'] = vix_value
                intraday['regime'] = regime
            else:
                intraday['vix'] = 20.0
                intraday['regime'] = 'normal'
        else:
            intraday['vix'] = 20.0
            intraday['regime'] = 'normal'
        
        # Add date column
        intraday['date'] = date
        
        return intraday


def main():
    """Main function to collect historical data"""
    collector = HistoricalDataCollector(data_dir="training_data")
    
    print("\n🎯 Historical Data Collection for RL Training")
    print("="*70)
    print("This will collect:")
    print("  - Daily OHLCV data for SPY, QQQ, SPX (since 2002)")
    print("  - VIX data for regime classification")
    print("  - Identification of significant market days")
    print("="*70)
    
    response = input("\nStart collection? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        dataset = collector.collect_full_dataset(save_intermediate=True)
        
        print("\n✅ Collection complete!")
        print(f"\nData saved to: {collector.data_dir}")
        print("\nNext steps:")
        print("  1. Review collected data")
        print("  2. Run training pipeline")
        print("  3. Train RL model on historical data")
    else:
        print("Collection cancelled.")


if __name__ == "__main__":
    main()

