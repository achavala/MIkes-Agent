#!/usr/bin/env python3
"""
Test script to verify trained model integration
- Loads the trained historical model
- Verifies observation space matches
- Tests inference on sample data
- Validates action outputs
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

# Import model loading function
from mike_agent_live_safe import load_rl_model, prepare_observation_basic, prepare_observation, RiskManager, MODEL_PATH
from mike_agent_live_safe import get_market_data, init_alpaca

def test_model_loading():
    """Test 1: Load the trained model"""
    print("=" * 70)
    print("TEST 1: MODEL LOADING")
    print("=" * 70)
    
    try:
        model = load_rl_model()
        print(f"✅ Model loaded successfully")
        print(f"   Model type: {type(model).__name__}")
        
        # Check observation space
        if hasattr(model, 'observation_space'):
            obs_space = model.observation_space
            print(f"   Observation space: {obs_space}")
            if hasattr(obs_space, 'shape'):
                print(f"   Expected shape: {obs_space.shape}")
        
        # Check action space
        if hasattr(model, 'action_space'):
            action_space = model.action_space
            print(f"   Action space: {action_space}")
        
        return model
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_observation_space():
    """Test 2: Verify observation space matches"""
    print("\n" + "=" * 70)
    print("TEST 2: OBSERVATION SPACE VERIFICATION")
    print("=" * 70)
    
    # Create sample data
    dates = pd.date_range(end=datetime.now(), periods=50, freq='1min')
    sample_data = pd.DataFrame({
        'open': np.random.randn(50).cumsum() + 450,
        'high': np.random.randn(50).cumsum() + 451,
        'low': np.random.randn(50).cumsum() + 449,
        'close': np.random.randn(50).cumsum() + 450,
        'volume': np.random.randint(1000000, 5000000, 50)
    }, index=dates)
    
    # Rename columns to match expected format
    sample_data.columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Create mock risk manager
    class MockRiskManager:
        def __init__(self):
            self.open_positions = {}  # Add open_positions attribute
        
        def get_current_vix(self):
            return 20.0
    
    risk_mgr = MockRiskManager()
    
    try:
        # Use prepare_observation which routes to correct version based on MODEL_PATH
        obs = prepare_observation(sample_data, risk_mgr, symbol='SPY')
        print(f"✅ Observation created successfully")
        print(f"   Shape: {obs.shape}")
        # Check which model we're using
        if "mike_historical_model" in MODEL_PATH:
            expected_shape = (20, 10)
        else:
            expected_shape = (20, 23)
        print(f"   Expected: {expected_shape}")
        print(f"   Match: {'✅ YES' if obs.shape == expected_shape else '❌ NO'}")
        print(f"   Data type: {obs.dtype}")
        print(f"   Min value: {obs.min():.4f}")
        print(f"   Max value: {obs.max():.4f}")
        print(f"   Mean value: {obs.mean():.4f}")
        print(f"   Has NaN: {np.isnan(obs).any()}")
        print(f"   Has Inf: {np.isinf(obs).any()}")
        
        # Verify feature count
        feature_count = obs.shape[1]
        print(f"\n   Feature breakdown:")
        if "mike_historical_model" in MODEL_PATH:
            print(f"   - OHLCV: 5 features")
            print(f"   - VIX: 1 feature (vix_norm)")
            print(f"   - Greeks: 4 features (delta, gamma, theta, vega)")
            print(f"   Total: {feature_count} features (10-feature model)")
            expected_shape = (20, 10)
        else:
            print(f"   - OHLCV: 5 features")
            print(f"   - VIX: 2 features (vix_norm, vix_delta_norm)")
            print(f"   - Technical: 11 features (EMA, VWAP, RSI, MACD, ATR, etc.)")
            print(f"   - Greeks: 4 features (delta, gamma, theta, vega)")
            print(f"   - Other: 1 feature (trend_strength)")
            print(f"   Total: {feature_count} features (23-feature model)")
            expected_shape = (20, 23)
        
        return obs.shape == expected_shape
    except Exception as e:
        print(f"❌ Observation creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_inference(model):
    """Test 3: Test model inference"""
    print("\n" + "=" * 70)
    print("TEST 3: MODEL INFERENCE")
    print("=" * 70)
    
    if model is None:
        print("❌ Cannot test inference - model not loaded")
        return False
    
    # Create sample observation - check which model we're using
    if "mike_historical_model" in MODEL_PATH:
        obs_shape = (20, 10)  # Historical model uses 10 features
    else:
        obs_shape = (20, 23)  # Momentum model uses 23 features
    
    obs = np.random.randn(*obs_shape).astype(np.float32)
    obs = np.clip(obs, -10.0, 10.0)
    
    try:
        # Test prediction
        if hasattr(model, 'predict'):
            action, _ = model.predict(obs, deterministic=False)
            print(f"✅ Inference successful")
            print(f"   Action: {action}")
            print(f"   Action type: {type(action)}")
            
            # Handle different action formats
            if isinstance(action, np.ndarray):
                action_value = int(action[0]) if len(action.shape) > 0 else int(action)
            else:
                action_value = int(action)
            
            print(f"   Action value: {action_value}")
            
            # Map action to name
            action_map = {
                0: "HOLD",
                1: "BUY CALL",
                2: "BUY PUT",
                3: "TRIM 50%",
                4: "TRIM 70%",
                5: "FULL EXIT"
            }
            action_name = action_map.get(action_value, "UNKNOWN")
            print(f"   Action name: {action_name}")
            
            # Test multiple predictions
            print(f"\n   Testing 10 predictions:")
            actions = []
            for i in range(10):
                action, _ = model.predict(obs, deterministic=False)
                if isinstance(action, np.ndarray):
                    action_value = int(action[0]) if len(action.shape) > 0 else int(action)
                else:
                    action_value = int(action)
                actions.append(action_value)
                print(f"   Prediction {i+1}: {action_value} ({action_map.get(action_value, 'UNKNOWN')})")
            
            # Check action distribution
            unique, counts = np.unique(actions, return_counts=True)
            print(f"\n   Action distribution:")
            for action_val, count in zip(unique, counts):
                print(f"   {action_map.get(int(action_val), 'UNKNOWN')}: {count}/10 ({count*10}%)")
            
            return True
        else:
            print("❌ Model does not have predict method")
            return False
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_data_inference(model):
    """Test 4: Test inference on real market data"""
    print("\n" + "=" * 70)
    print("TEST 4: REAL DATA INFERENCE")
    print("=" * 70)
    
    if model is None:
        print("❌ Cannot test - model not loaded")
        return False
    
    try:
        # Initialize Alpaca (optional, will use yfinance if not available)
        api = None
        try:
            api = init_alpaca()
            print("✅ Alpaca API initialized")
        except Exception as e:
            print(f"⚠️  Alpaca API not available: {e}")
            print("   Will use yfinance for data")
        
        # Get real market data
        print("\n   Fetching SPY data...")
        data = get_market_data('SPY', period='2d', interval='1m', api=api, risk_mgr=None)
        
        if data.empty:
            print("❌ No data retrieved")
            return False
        
        print(f"✅ Retrieved {len(data)} bars")
        
        # Create mock risk manager
        class MockRiskManager:
            def __init__(self):
                self.open_positions = {}  # Add open_positions attribute
            
            def get_current_vix(self):
                return 20.0
        
        risk_mgr = MockRiskManager()
        
        # Prepare observation (routes to correct version based on MODEL_PATH)
        obs = prepare_observation(data, risk_mgr, symbol='SPY')
        print(f"✅ Observation prepared: {obs.shape}")
        
        # Run inference
        action, _ = model.predict(obs, deterministic=False)
        if isinstance(action, np.ndarray):
            action_value = int(action[0]) if len(action.shape) > 0 else int(action)
        else:
            action_value = int(action)
        
        action_map = {
            0: "HOLD",
            1: "BUY CALL",
            2: "BUY PUT",
            3: "TRIM 50%",
            4: "TRIM 70%",
            5: "FULL EXIT"
        }
        
        print(f"✅ Inference on real data successful")
        print(f"   Action: {action_value} ({action_map.get(action_value, 'UNKNOWN')})")
        print(f"   Current SPY price: ${data['Close'].iloc[-1]:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Real data inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("🧪 TRAINED MODEL INTEGRATION TEST")
    print("=" * 70)
    print(f"Model: models/mike_historical_model.zip")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Model loading
    model = test_model_loading()
    results['model_loading'] = model is not None
    
    # Test 2: Observation space
    results['observation_space'] = test_observation_space()
    
    # Test 3: Model inference
    if model:
        results['model_inference'] = test_model_inference(model)
        results['real_data_inference'] = test_real_data_inference(model)
    else:
        results['model_inference'] = False
        results['real_data_inference'] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED - Model ready for integration")
    else:
        print("❌ SOME TESTS FAILED - Review errors above")
    print("=" * 70)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

