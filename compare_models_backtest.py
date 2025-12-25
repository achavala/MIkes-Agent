#!/usr/bin/env python3
"""
Compare trained historical model vs current model performance
Runs backtest on recent data and compares metrics
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mike_agent_live_safe import (
    load_rl_model, prepare_observation_basic, RiskManager,
    get_market_data, init_alpaca, ACTION_MAP
)

def run_backtest(model, symbol='SPY', days=30, api=None):
    """Run simple backtest on recent data"""
    print(f"\n   Running backtest on {symbol} for last {days} days...")
    
    # Get data
    data = get_market_data(symbol, period=f'{days}d', interval='1m', api=api, risk_mgr=None)
    if data.empty:
        print(f"   ❌ No data for {symbol}")
        return None
    
    print(f"   ✅ Retrieved {len(data)} bars")
    
    # Create mock risk manager
    class MockRiskManager:
        def get_current_vix(self):
            return 20.0
    
    risk_mgr = MockRiskManager()
    
    # Simulate trading
    capital = 10000.0
    positions = []
    trades = []
    
    # Sample every 5 minutes (reduce computation)
    step_size = 5
    for i in range(20, len(data), step_size):
        try:
            # Get observation
            recent_data = data.iloc[:i+1]
            obs = prepare_observation_basic(recent_data, risk_mgr, symbol=symbol)
            
            # Get model prediction
            action, _ = model.predict(obs, deterministic=False)
            if isinstance(action, np.ndarray):
                action_value = int(action[0]) if len(action.shape) > 0 else int(action)
            else:
                action_value = int(action)
            
            # Simple trading logic (for comparison only)
            current_price = data['Close'].iloc[i]
            
            if action_value == 1:  # BUY CALL
                if len(positions) == 0:  # Only one position at a time
                    entry_price = current_price
                    positions.append({
                        'entry_price': entry_price,
                        'entry_time': data.index[i],
                        'type': 'call',
                        'action': action_value
                    })
            elif action_value == 2:  # BUY PUT
                if len(positions) == 0:
                    entry_price = current_price
                    positions.append({
                        'entry_price': entry_price,
                        'entry_time': data.index[i],
                        'type': 'put',
                        'action': action_value
                    })
            elif action_value == 5 and len(positions) > 0:  # FULL EXIT
                pos = positions.pop(0)
                exit_price = current_price
                pnl_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
                if pos['type'] == 'put':
                    pnl_pct = -pnl_pct
                
                trades.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': data.index[i],
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'type': pos['type']
                })
        
        except Exception as e:
            continue
    
    # Calculate metrics
    if len(trades) == 0:
        return {
            'num_trades': 0,
            'total_return': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        }
    
    trades_df = pd.DataFrame(trades)
    winning_trades = trades_df[trades_df['pnl_pct'] > 0]
    losing_trades = trades_df[trades_df['pnl_pct'] <= 0]
    
    total_return = trades_df['pnl_pct'].sum()
    win_rate = len(winning_trades) / len(trades) * 100 if len(trades) > 0 else 0
    avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
    
    return {
        'num_trades': len(trades),
        'total_return': total_return,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'trades': trades
    }

def compare_models():
    """Compare two models"""
    print("=" * 70)
    print("📊 MODEL COMPARISON: Trained vs Current")
    print("=" * 70)
    
    # Load both models
    print("\n1. Loading models...")
    
    # Trained model
    print("\n   Loading trained model (mike_historical_model.zip)...")
    try:
        # Temporarily change MODEL_PATH
        import mike_agent_live_safe
        original_path = mike_agent_live_safe.MODEL_PATH
        mike_agent_live_safe.MODEL_PATH = "models/mike_historical_model.zip"
        trained_model = load_rl_model()
        mike_agent_live_safe.MODEL_PATH = original_path
        print("   ✅ Trained model loaded")
    except Exception as e:
        print(f"   ❌ Failed to load trained model: {e}")
        return
    
    # Current model
    print("\n   Loading current model (mike_momentum_model_v3_lstm.zip)...")
    try:
        mike_agent_live_safe.MODEL_PATH = "models/mike_momentum_model_v3_lstm.zip"
        current_model = load_rl_model()
        mike_agent_live_safe.MODEL_PATH = original_path
        print("   ✅ Current model loaded")
    except Exception as e:
        print(f"   ⚠️  Failed to load current model: {e}")
        print("   Will only test trained model")
        current_model = None
    
    # Initialize API
    api = None
    try:
        api = init_alpaca()
        print("\n✅ Alpaca API initialized")
    except Exception as e:
        print(f"\n⚠️  Alpaca API not available: {e}")
        print("   Will use yfinance for data")
    
    # Run backtests
    print("\n2. Running backtests...")
    
    # Test trained model
    print("\n   Testing trained model...")
    trained_results = run_backtest(trained_model, symbol='SPY', days=30, api=api)
    
    # Test current model
    if current_model:
        print("\n   Testing current model...")
        current_results = run_backtest(current_model, symbol='SPY', days=30, api=api)
    else:
        current_results = None
    
    # Compare results
    print("\n" + "=" * 70)
    print("📊 COMPARISON RESULTS")
    print("=" * 70)
    
    print("\nTrained Model (mike_historical_model.zip):")
    if trained_results:
        print(f"   Trades: {trained_results['num_trades']}")
        print(f"   Total Return: {trained_results['total_return']:.2f}%")
        print(f"   Win Rate: {trained_results['win_rate']:.1f}%")
        print(f"   Avg Win: {trained_results['avg_win']:.2f}%")
        print(f"   Avg Loss: {trained_results['avg_loss']:.2f}%")
    else:
        print("   ❌ No results")
    
    if current_results:
        print("\nCurrent Model (mike_momentum_model_v3_lstm.zip):")
        print(f"   Trades: {current_results['num_trades']}")
        print(f"   Total Return: {current_results['total_return']:.2f}%")
        print(f"   Win Rate: {current_results['win_rate']:.1f}%")
        print(f"   Avg Win: {current_results['avg_win']:.2f}%")
        print(f"   Avg Loss: {current_results['avg_loss']:.2f}%")
        
        # Winner
        print("\n" + "=" * 70)
        print("🏆 WINNER")
        print("=" * 70)
        if trained_results and current_results:
            if trained_results['total_return'] > current_results['total_return']:
                print("✅ Trained model performs better")
            elif current_results['total_return'] > trained_results['total_return']:
                print("✅ Current model performs better")
            else:
                print("🤝 Models perform similarly")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    compare_models()





