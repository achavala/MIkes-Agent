# 🚀 QUICK INTEGRATION GUIDE - Institutional Features

## Step-by-Step Integration

### Step 1: Install Additional Dependencies (Optional but Recommended)

```bash
pip install ta  # Technical analysis library
# Note: ta-lib requires system installation first
```

### Step 2: Test Feature Engine (5 minutes)

Create a test script:

```python
# test_institutional_features.py
from institutional_features import InstitutionalFeatureEngine
import yfinance as yf
import pandas as pd

# Get test data
spy = yf.Ticker("SPY")
data = spy.history(period="1d", interval="1m")
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Create feature engine
engine = InstitutionalFeatureEngine(lookback_minutes=20)

# Extract features
features, groups = engine.extract_all_features(data, symbol='SPY')

print(f"✅ Extracted {features.shape[1]} features from {features.shape[0]} bars")
print(f"✅ Feature groups: {list(groups.keys())}")
print(f"✅ Feature counts per group:")
for name, group_features in groups.items():
    print(f"   {name}: {group_features.shape[1]} features")
```

Run it:
```bash
python test_institutional_features.py
```

### Step 3: Integrate into Live Trading (10 minutes)

**Modify `mike_agent_live_safe.py`:**

```python
# Add import at top
from institutional_features import InstitutionalFeatureEngine

# Initialize feature engine (add after imports)
feature_engine = InstitutionalFeatureEngine(lookback_minutes=20)

# Replace prepare_observation function (around line 1010)
def prepare_observation_enhanced(data: pd.DataFrame, risk_mgr: RiskManager) -> np.ndarray:
    """
    Prepare observation with institutional-grade features
    Returns shape: (1, n_bars, n_features)
    """
    # Extract all features
    all_features, feature_groups = feature_engine.extract_all_features(
        data, 
        symbol='SPY',  # Or get from context
        risk_mgr=risk_mgr,
        include_microstructure=True
    )
    
    # Take last LOOKBACK bars
    if len(all_features) >= LOOKBACK:
        recent_features = all_features[-LOOKBACK:]
    else:
        # Pad if needed
        padding = np.zeros((LOOKBACK - len(all_features), all_features.shape[1]))
        recent_features = np.vstack([padding, all_features])
    
    # Reshape for RL model: (1, LOOKBACK, n_features)
    return recent_features.reshape(1, LOOKBACK, -1)

# Update the call in main loop (around line 1238)
# Change from:
# obs = prepare_observation(hist, risk_mgr)
# To:
obs = prepare_observation_enhanced(hist, risk_mgr)
```

### Step 4: Update RL Model (After Testing)

**Important:** The current RL model expects 5 features, but we now have 500+.

**Options:**

1. **Quick Option:** Use feature selection (PCA or top features)
   ```python
   from sklearn.decomposition import PCA
   
   # Reduce to 50 most important features
   pca = PCA(n_components=50)
   features_reduced = pca.fit_transform(recent_features)
   ```

2. **Best Option:** Retrain model with new features
   - Update `mike_rl_agent.py` observation space
   - Retrain PPO model
   - Use new model in live trading

### Step 5: Test Integration

1. Run in paper trading mode
2. Monitor feature extraction times (should be < 1 second)
3. Check model predictions
4. Verify no errors

---

## 🎯 Quick Wins (Use Immediately)

Even without full integration, you can use features for:

1. **Better entry/exit signals**
   ```python
   features, groups = engine.extract_all_features(data, symbol='SPY')
   
   # Use volatility features for position sizing
   volatility_features = groups['volatility']
   current_vol = volatility_features[-1, 0]  # Latest RV
   
   # Adjust position size based on volatility
   if current_vol > threshold:
       reduce_position_size()
   ```

2. **Enhanced risk management**
   ```python
   # Use cross-asset features for regime detection
   cross_asset = groups['cross_asset']
   vix_level = cross_asset[-1, 0]
   
   if vix_level > 25:
       # High volatility regime - reduce risk
       adjust_risk_parameters()
   ```

3. **Better stop-loss placement**
   ```python
   # Use technical indicators
   technical = groups['technical']
   rsi = technical[-1, 0]  # Latest RSI
   
   if rsi > 70:  # Overbought
       tighten_stop_loss()
   ```

---

## ⚠️ Important Notes

1. **Feature extraction takes time** (~0.5-1 second per call)
   - Cache features when possible
   - Extract once per bar, not per model prediction

2. **Memory usage**
   - 500+ features × 20 bars = 10,000+ values per observation
   - Monitor memory if running for hours

3. **Model compatibility**
   - Current model trained on 5 features
   - Need to retrain or use feature selection
   - Recommendation: Retrain with all features

---

## ✅ Success Checklist

- [ ] Feature engine tested successfully
- [ ] Features extracted correctly (500+ features)
- [ ] Integration into `mike_agent_live_safe.py` complete
- [ ] Model updated or feature selection added
- [ ] Paper trading test successful
- [ ] Performance monitoring active

---

## 🚀 Next Steps After Integration

1. **Phase 1.2:** Implement LSTM backbone
2. **Phase 1.3:** Add advanced risk metrics
3. **Phase 2:** Build multi-agent system

---

**Ready to integrate? Start with Step 1 (test script) and work your way through! 🎯**

