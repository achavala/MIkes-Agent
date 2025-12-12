#!/usr/bin/env python3
"""
Backtest the trained model on recent market data
Validates model performance before live deployment
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    print("Error: stable-baselines3 not installed")
    sys.exit(1)

# Import from live agent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from mike_agent_live_safe import prepare_observation, RiskManager, MODEL_PATH, LOOKBACK
except ImportError as e:
    print(f"Error importing from live agent: {e}")
    sys.exit(1)

def backtest_recent_data(days=30, symbol='SPY'):
    """Backtest model on recent N days of data"""
    print("=" * 80)
    print("📊 BACKTEST: RECENT DATA VALIDATION")
    print("=" * 80)
    print()
    
    # 1. Load model
    print("1️⃣ LOADING MODEL")
    print("-" * 80)
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return False
    
    try:
        model = PPO.load(MODEL_PATH)
        print(f"✅ Model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    print()
    
    # 2. Get recent data
    print(f"2️⃣ FETCHING RECENT DATA ({days} days)")
    print("-" * 80)
    try:
        ticker = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        hist = ticker.history(start=start_date, end=end_date, interval='1m')
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        hist = hist.dropna()
        
        print(f"✅ Data loaded: {len(hist)} bars")
        print(f"   Date range: {hist.index[0]} to {hist.index[-1]}")
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return False
    print()
    
    # 3. Run backtest
    print("3️⃣ RUNNING BACKTEST")
    print("-" * 80)
    risk_mgr = RiskManager()
    
    actions = []
    predictions = []
    timestamps = []
    
    # Process in windows of LOOKBACK
    for i in range(LOOKBACK, len(hist), 10):  # Step by 10 minutes
        window_data = hist.iloc[i-LOOKBACK:i]
        timestamp = hist.index[i]
        
        try:
            # Prepare observation
            obs = prepare_observation(window_data, risk_mgr, symbol=symbol)
            
            # Get prediction
            action_raw, _ = model.predict(obs, deterministic=True)
            action_value = float(action_raw[0] if isinstance(action_raw, (list, np.ndarray)) else action_raw)
            
            # Map to discrete action
            if abs(action_value) < 0.35:
                action = 0  # HOLD
            elif action_value > 0:
                action = 1  # BUY CALL
            else:
                action = 2  # BUY PUT
            
            actions.append(action)
            predictions.append(action_value)
            timestamps.append(timestamp)
            
        except Exception as e:
            print(f"   ⚠️  Error at {timestamp}: {e}")
            continue
    
    print(f"✅ Processed {len(actions)} predictions")
    print()
    
    # 4. Analyze results
    print("4️⃣ RESULTS ANALYSIS")
    print("-" * 80)
    actions_array = np.array(actions)
    predictions_array = np.array(predictions)
    
    action_counts = {
        'HOLD': np.sum(actions_array == 0),
        'BUY CALL': np.sum(actions_array == 1),
        'BUY PUT': np.sum(actions_array == 2)
    }
    
    print("Action Distribution:")
    total = len(actions)
    for action_name, count in action_counts.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"   {action_name}: {count} ({pct:.1f}%)")
    print()
    
    print("Prediction Statistics:")
    print(f"   Mean: {predictions_array.mean():.3f}")
    print(f"   Std: {predictions_array.std():.3f}")
    print(f"   Min: {predictions_array.min():.3f}")
    print(f"   Max: {predictions_array.max():.3f}")
    print()
    
    # 5. Summary
    print("5️⃣ SUMMARY")
    print("-" * 80)
    print("✅ BACKTEST COMPLETE")
    print(f"   Period: {days} days")
    print(f"   Symbol: {symbol}")
    print(f"   Predictions: {len(actions)}")
    print(f"   HOLD rate: {action_counts['HOLD']/total*100:.1f}%")
    print(f"   BUY CALL rate: {action_counts['BUY CALL']/total*100:.1f}%")
    print(f"   BUY PUT rate: {action_counts['BUY PUT']/total*100:.1f}%")
    print()
    
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    parser.add_argument('--symbol', type=str, default='SPY', help='Symbol to backtest')
    args = parser.parse_args()
    
    success = backtest_recent_data(days=args.days, symbol=args.symbol)
    sys.exit(0 if success else 1)

