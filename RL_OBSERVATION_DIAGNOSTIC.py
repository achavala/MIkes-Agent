#!/usr/bin/env python3
"""
Quick diagnostic to check what's going into the RL model
Run this to see what observations look like
"""
import numpy as np
import pandas as pd
from mike_agent_live_safe import prepare_observation, RiskManager, get_market_data, LOOKBACK

# Create test risk manager
class TestRiskManager:
    def __init__(self):
        self.open_positions = {}
        self.current_vix = 20.0
    
    def get_current_vix(self):
        return self.current_vix
    
    def log(self, msg, level="INFO"):
        print(f"[{level}] {msg}")

risk_mgr = TestRiskManager()

print("=" * 80)
print("RL OBSERVATION DIAGNOSTIC")
print("=" * 80)

for symbol in ['SPY', 'QQQ', 'SPX']:
    print(f"\n📊 Testing {symbol}...")
    
    try:
        # Get market data
        data = get_market_data(symbol, period="2d", interval="1m")
        print(f"  ✅ Market data: {len(data)} bars")
        print(f"  Columns: {list(data.columns)}")
        print(f"  Shape: {data.shape}")
        print(f"  Last close: ${data['Close'].iloc[-1] if 'Close' in data.columns else data['close'].iloc[-1]:.2f}")
        
        # Prepare observation
        obs = prepare_observation(data, risk_mgr, symbol=symbol)
        print(f"\n  📈 Observation shape: {obs.shape}")
        print(f"  Expected shape: ({LOOKBACK}, 10)")
        print(f"  Dtype: {obs.dtype}")
        
        # Check for issues
        if np.isnan(obs).any():
            print(f"  ❌ CONTAINS NaNs: {np.isnan(obs).sum()} NaN values")
        else:
            print(f"  ✅ No NaNs")
        
        if np.isinf(obs).any():
            print(f"  ❌ CONTAINS INFINITIES: {np.isinf(obs).sum()} inf values")
        else:
            print(f"  ✅ No infinities")
        
        if (obs == 0).all():
            print(f"  ❌ ALL ZEROS - THIS IS THE PROBLEM!")
        else:
            print(f"  ✅ Not all zeros")
        
        # Show stats
        print(f"\n  Statistics:")
        print(f"    Min: {obs.min():.6f}")
        print(f"    Max: {obs.max():.6f}")
        print(f"    Mean: {obs.mean():.6f}")
        print(f"    Std: {obs.std():.6f}")
        
        # Show first row (features)
        print(f"\n  First timestep (most recent):")
        print(f"    OHLC: {obs[0, 0:4]}")
        print(f"    Volume: {obs[0, 4]:.6f}")
        print(f"    VIX: {obs[0, 5]:.6f}")
        print(f"    Greeks: {obs[0, 6:10]}")
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)





