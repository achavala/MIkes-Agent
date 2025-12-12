#!/usr/bin/env python3
"""
Paper Trading Validation Script
Tests model integration with Alpaca paper trading account
"""

import os
import sys
import time
import warnings
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    print("Error: alpaca-trade-api not installed")
    sys.exit(1)

try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    print("Error: stable-baselines3 not installed")
    sys.exit(1)

import config

# Import from live agent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from mike_agent_live_safe import (
        MODEL_PATH, init_alpaca, load_rl_model, 
        prepare_observation_basic, get_market_data, RiskManager, LOOKBACK
    )
except ImportError as e:
    print(f"Error importing from live agent: {e}")
    sys.exit(1)

def validate_paper_trading():
    """Validate paper trading setup"""
    print("=" * 80)
    print("🧪 PAPER TRADING VALIDATION")
    print("=" * 80)
    print()
    
    # 1. Check API connection
    print("1️⃣ ALPACA API CONNECTION")
    print("-" * 80)
    try:
        api = init_alpaca()
        account = api.get_account()
        print("✅ Alpaca API connected")
        print(f"   Account status: {account.status}")
        print(f"   Equity: ${float(account.equity):,.2f}")
        print(f"   Buying power: ${float(account.buying_power):,.2f}")
    except Exception as e:
        print(f"❌ Alpaca API error: {e}")
        return False
    print()
    
    # 2. Check model loading
    print("2️⃣ MODEL LOADING")
    print("-" * 80)
    try:
        model = load_rl_model()
        print("✅ Model loaded successfully")
        print(f"   Path: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False
    print()
    
    # 3. Test observation preparation
    print("3️⃣ OBSERVATION PREPARATION")
    print("-" * 80)
    try:
        hist = get_market_data("SPY", period="2d", interval="1m")
        if len(hist) < LOOKBACK:
            print(f"⚠️  Insufficient data for observation (need {LOOKBACK} bars, got {len(hist)})")
            return False
        
        risk_mgr = RiskManager()
        obs = prepare_observation_basic(hist, risk_mgr, symbol='SPY')
        print(f"✅ Observation prepared")
        print(f"   Shape: {obs.shape}")
        print(f"   Expected: ({LOOKBACK}, 10)")
        
        if obs.shape != (LOOKBACK, 10):
            print(f"❌ Observation shape mismatch!")
            print(f"   Model expects ({LOOKBACK}, 10) - {LOOKBACK} timesteps, 10 features (OHLCV + VIX + Greeks)")
            return False
        else:
            print(f"   ✅ Shape matches model expectations")
            print(f"   Features: OHLCV (5) + VIX (1) + Greeks (4) = 10 features")
    except Exception as e:
        print(f"❌ Observation preparation error: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # 4. Test model inference
    print("4️⃣ MODEL INFERENCE")
    print("-" * 80)
    try:
        # Model outputs discrete actions (Discrete action space)
        action_raw, _ = model.predict(obs, deterministic=True)
        
        # Extract action value (discrete integer)
        # Handle different return types from stable_baselines3
        if isinstance(action_raw, np.ndarray):
            if action_raw.ndim == 0:
                action_value = int(action_raw.item())
            else:
                action_value = int(action_raw[0] if len(action_raw) > 0 else action_raw.item())
        elif isinstance(action_raw, (list, tuple)):
            action_value = int(action_raw[0] if len(action_raw) > 0 else 0)
        else:
            action_value = int(action_raw)
        
        # Map discrete action to trading action
        # Model was trained with discrete actions, so output is already an integer
        action_names = {
            0: "HOLD",
            1: "BUY CALL", 
            2: "BUY PUT",
            3: "TRIM 50%",
            4: "TRIM 70%",
            5: "FULL EXIT"
        }
        
        action_name = action_names.get(action_value, f"UNKNOWN ({action_value})")
        print(f"✅ Inference successful")
        print(f"   Raw output: {action_value}")
        print(f"   Action: {action_name}")
    except Exception as e:
        print(f"❌ Inference error: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # 5. Check positions
    print("5️⃣ CURRENT POSITIONS")
    print("-" * 80)
    try:
        positions = api.list_positions()
        option_positions = [p for p in positions if p.asset_class == 'option']
        print(f"Total positions: {len(positions)}")
        print(f"Option positions: {len(option_positions)}")
        
        if option_positions:
            print("   Option positions:")
            for pos in option_positions:
                print(f"      {pos.symbol}: {pos.qty} @ ${float(pos.avg_entry_price):.2f}")
    except Exception as e:
        print(f"⚠️  Error fetching positions: {e}")
    print()
    
    # 6. Summary
    print("6️⃣ VALIDATION SUMMARY")
    print("-" * 80)
    print("✅ PAPER TRADING VALIDATION: PASSED")
    print()
    print("System is ready for paper trading:")
    print("   ✅ Alpaca API connected")
    print("   ✅ Model loaded")
    print("   ✅ Observation preparation working")
    print("   ✅ Model inference working")
    print()
    print("Next steps:")
    print("   1. Run: python mike_agent_live_safe.py")
    print("   2. Monitor trades in Alpaca dashboard")
    print("   3. Check logs/ directory for detailed logs")
    print()
    
    return True

if __name__ == "__main__":
    success = validate_paper_trading()
    sys.exit(0 if success else 1)

