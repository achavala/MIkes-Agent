# 🔄 UPDATE LIVE AGENT TO USE 23-FEATURE MODEL

**After training completes, follow these steps:**

---

## ✅ STEP 1: Verify Training Completed

```bash
# Check model file exists
ls -lh models/mike_23feature_model.zip

# Should see: ~0.5-1 MB file
```

---

## ✅ STEP 2: Update MODEL_PATH

**File:** `mike_agent_live_safe.py`  
**Line:** 395

**Change:**
```python
# FROM:
MODEL_PATH = "models/mike_historical_model.zip"

# TO:
MODEL_PATH = "models/mike_23feature_model.zip"
```

---

## ✅ STEP 3: Update start_cloud.sh

**File:** `start_cloud.sh`  
**Line:** 43

**Change:**
```bash
# FROM:
MODEL_PATH="models/mike_historical_model.zip"

# TO:
MODEL_PATH="models/mike_23feature_model.zip"
```

---

## ✅ STEP 4: Verify prepare_observation() Logic

**File:** `mike_agent_live_safe.py`  
**Line:** ~2425

**Should already be updated** (we just fixed it), but verify it looks like:

```python
if "mike_23feature_model" in MODEL_PATH or "mike_momentum_model" in MODEL_PATH:
    # Use 23-feature observation
    obs = prepare_observation_basic(data, risk_mgr, symbol)
    # Validate shape (20, 23)
elif "mike_historical_model" in MODEL_PATH:
    # Use 10-feature observation
    obs = prepare_observation_10_features_inline(data, risk_mgr, symbol)
```

---

## ✅ STEP 5: Deploy

```bash
fly deploy --app mike-agent-project
```

---

## ✅ STEP 6: Verify

```bash
# Check model loads
fly logs --app mike-agent-project | grep "Model loaded"

# Check observation shape
fly logs --app mike-agent-project | grep "Observation"

# Check trading activity
fly logs --app mike-agent-project | grep "RL Decision"
```

---

**That's it! The agent will now use all 23 features. 🎯**





