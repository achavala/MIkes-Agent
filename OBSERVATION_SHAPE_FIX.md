# ✅ Observation Shape Mismatch - FIXED

## Problem

**Error:** `ValueError: could not broadcast input array from shape (20,5) into shape (20,10)`

**Location:** Line 649 in training script during `env.reset()`

**Root Cause:**
- When `use_greeks=True`, observation space was defined as `(20, 10)` features
- But the environment returned `(20, 5)` when no position existed
- Stable-Baselines3 requires observation shape to be consistent

---

## Fix Applied

**File:** `historical_training_system.py` - `_get_obs()` method

**Changes:**
1. ✅ When `use_greeks=True`, **always** return `(20, 10)` features
2. ✅ When no position exists, use **zeros for Greeks** (maintains shape)
3. ✅ When position exists, calculate **actual Greeks**
4. ✅ Observation shape is now **100% consistent**

---

## Technical Details

### Before (Broken):
```python
if self.greeks_calc and self.position:
    # Return (20, 10) with actual Greeks
else:
    # Return (20, 5) - SHAPE MISMATCH!
```

### After (Fixed):
```python
if self.use_greeks and self.greeks_calc:
    # Always return (20, 10)
    if self.position:
        # Use actual Greeks
    else:
        # Use zeros for Greeks (maintains shape)
else:
    # Return (20, 5) - only when Greeks disabled
```

---

## Verification

✅ **Observation space:** `(20, 10)`  
✅ **Reset observation:** `(20, 10)`  
✅ **No position observation:** `(20, 10)`  
✅ **Shape consistency:** **100% consistent**

---

## Status

**FIXED ✅** - Training should now work correctly!

You can now run:
```bash
source venv/bin/activate
python train_historical_model.py \
    --symbols SPY,QQQ \
    --start-date 2002-01-01 \
    --timesteps 5000000 \
    --use-greeks \
    --regime-balanced
```

---

**The observation shape mismatch is completely resolved!** 🎯
