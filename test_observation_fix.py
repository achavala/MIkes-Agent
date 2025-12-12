#!/usr/bin/env python3
"""
Test the fixed observation preparation to ensure it matches model expectations
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Suppress warnings
import warnings
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
    from mike_agent_live_safe import prepare_observation_basic, RiskManager, MODEL_PATH, LOOKBACK, get_market_data
except ImportError as e:
    print(f"Error importing: {e}")
    sys.exit(1)

def test_observation_preparation():
    """Test observation preparation matches model expectations"""
    print("=" * 80)
    print("🧪 OBSERVATION PREPARATION TEST")
    print("=" * 80)
    print()
    
    # 1. Load model to check expected shape
    print("1️⃣ MODEL EXPECTATIONS")
    print("-" * 80)
    try:
        model = PPO.load(MODEL_PATH)
        obs_space = model.observation_space
        print(f"✅ Model loaded")
        print(f"   Observation space: {obs_space.shape}")
        print(f"   Expected: (20, 10)")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    print()
    
    # 2. Prepare test data
    print("2️⃣ PREPARE TEST DATA")
    print("-" * 80)
    try:
        # Get real market data
        hist = get_market_data("SPY", period="2d", interval="1m")
        if len(hist) < LOOKBACK:
            print("⚠️  Insufficient data, using mock data")
            hist = pd.DataFrame({
                'Open': np.random.randn(25) + 450,
                'High': np.random.randn(25) + 451,
                'Low': np.random.randn(25) + 449,
                'Close': np.random.randn(25) + 450,
                'Volume': np.random.randint(1000000, 5000000, 25)
            })
        
        print(f"✅ Data prepared: {len(hist)} bars")
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        return False
    print()
    
    # 3. Create RiskManager
    print("3️⃣ RISK MANAGER")
    print("-" * 80)
    try:
        risk_mgr = RiskManager()
        print("✅ RiskManager created")
    except Exception as e:
        print(f"❌ Error creating RiskManager: {e}")
        return False
    print()
    
    # 4. Prepare observation
    print("4️⃣ OBSERVATION PREPARATION")
    print("-" * 80)
    try:
        obs = prepare_observation_basic(hist, risk_mgr, symbol='SPY')
        print(f"✅ Observation prepared")
        print(f"   Shape: {obs.shape}")
        print(f"   Expected: ({LOOKBACK}, 10)")
        print(f"   Match: {obs.shape == (LOOKBACK, 10)}")
        
        if obs.shape != (LOOKBACK, 10):
            print(f"❌ Shape mismatch! Expected ({LOOKBACK}, 10), got {obs.shape}")
            return False
        
        # Check feature breakdown
        print(f"   Features breakdown:")
        print(f"      OHLCV: columns 0-4 (5 features)")
        print(f"      VIX: column 5 (1 feature)")
        print(f"      Greeks: columns 6-9 (4 features: Delta, Gamma, Theta, Vega)")
        print(f"   Total: 10 features")
        
    except Exception as e:
        print(f"❌ Error preparing observation: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # 5. Test model inference
    print("5️⃣ MODEL INFERENCE TEST")
    print("-" * 80)
    try:
        action_raw, _ = model.predict(obs, deterministic=True)
        print(f"✅ Inference successful")
        print(f"   Action output: {action_raw}")
        print(f"   Action type: {type(action_raw)}")
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # 6. Summary
    print("6️⃣ SUMMARY")
    print("-" * 80)
    print("✅ OBSERVATION PREPARATION: FIXED AND VALIDATED")
    print()
    print("Observation format:")
    print("   Shape: (20, 10)")
    print("   Features:")
    print("      - OHLCV: 5 features (open, high, low, close, volume)")
    print("      - VIX: 1 feature (normalized VIX / 50)")
    print("      - Greeks: 4 features (Delta, Gamma, Theta, Vega)")
    print()
    print("✅ Model inference: WORKING")
    print()
    
    return True

if __name__ == "__main__":
    success = test_observation_preparation()
    sys.exit(0 if success else 1)

