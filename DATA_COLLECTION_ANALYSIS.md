# 🔍 DATA COLLECTION ANALYSIS & FIXES

## 🚨 CRITICAL BUG FOUND

### Problem: Only 50 Bars Returned Instead of 2 Days

**Current Code (Line 1044 & 1062):**
```python
return hist.tail(50 if interval == '1m' else len(hist))
```

**What This Does:**
- Requests: 2 days of 1-minute data (~2,880 bars)
- Returns: Only last 50 bars (50 minutes of data!)
- **This is a BUG!**

**Impact:**
- RL model needs 20 bars minimum (LOOKBACK=20)
- But should have 2 days of context for better predictions
- Only getting 50 minutes instead of 2 days
- Missing important historical context

---

## ✅ VERIFICATION: What Data Is Actually Collected?

### Test Results

Run this to verify:
```bash
python3 verify_data_collection.py
```

**Expected Output:**
```
📊 DATA ANALYSIS:
   Expected bars (2 days × 1,440 min/day): 2,880
   Actual bars received: 50
   
⚠️  WARNING: Only 50 bars received, expected 2,880
   This is a BUG in the code!
```

---

## 🔧 FIXES NEEDED

### Fix 1: Remove 50-Bar Limit

**Current (WRONG):**
```python
return hist.tail(50 if interval == '1m' else len(hist))
```

**Fixed (CORRECT):**
```python
# Return all data, not just last 50 bars
# For 1-minute data, we want full 2 days for context
return hist
```

### Fix 2: Use Alpaca API for Data (You're Already Paying!)

**Why Use Alpaca:**
- ✅ You're already paying for Alpaca API
- ✅ Real-time data included
- ✅ No need for yfinance fallback
- ✅ More reliable than free APIs

**Alpaca Data Access:**
```python
# Alpaca provides get_bars() method
bars = api.get_bars(
    symbol,
    timeframe=TimeFrame.Minute,
    start=start_date,
    end=end_date,
    limit=5000  # Max 5000 bars per request
)
```

**For 2 days of 1-minute data:**
- Need: ~2,880 bars
- Alpaca limit: 5,000 bars per request
- ✅ Can get 2 days in one request

### Fix 3: Priority Order Should Be

1. **Alpaca API** (you're paying for it!)
2. **Massive API** (if available)
3. **yfinance** (last resort, free but delayed)

---

## 📊 CURRENT DATA SOURCE PRIORITY

**Current (WRONG):**
1. Massive API
2. yfinance (free, delayed)

**Should Be:**
1. **Alpaca API** ← You're paying for this!
2. Massive API
3. yfinance (last resort)

---

## 🔍 HOW TO VERIFY DATA IS REAL

### Method 1: Run Verification Script
```bash
python3 verify_data_collection.py
```

This will show:
- How many bars are actually received
- Data quality checks
- Price validation
- Volume validation
- Timestamp validation

### Method 2: Check Logs
```bash
fly logs --app mike-agent-project | grep "Observation\|get_market_data"
```

Look for:
- "Observation: shape=(20, 23)" - Should show actual shape
- Data collection errors
- Fallback messages

### Method 3: Add Data Logging

Add this to `get_market_data()`:
```python
risk_mgr.log(f"📊 {symbol} Data: {len(hist)} bars, period={period}, interval={interval}, date_range={hist.index.min()} to {hist.index.max()}", "INFO")
```

---

## 🛠️ IMPLEMENTATION PLAN

### Step 1: Fix 50-Bar Limit
- Remove `.tail(50)` restriction
- Return full dataset

### Step 2: Add Alpaca Data Source
- Use `api.get_bars()` for historical data
- Priority: Alpaca → Massive → yfinance

### Step 3: Add Data Validation
- Log actual data received
- Verify data quality
- Warn if insufficient data

### Step 4: Add Data Verification
- Show data sample in logs
- Verify prices are reasonable
- Check timestamps are sequential

---

## 📋 DATA REQUIREMENTS

**For RL Inference:**
- Minimum: 20 bars (LOOKBACK=20)
- Recommended: 2 days of 1-minute data (~2,880 bars)
- Why: More context = better predictions

**Current Reality:**
- Getting: 50 bars (50 minutes)
- Missing: ~2,830 bars of context
- Impact: Model has less historical context

---

## ✅ RECOMMENDED FIXES

### Fix 1: Remove Bar Limit
```python
# OLD (WRONG):
return hist.tail(50 if interval == '1m' else len(hist))

# NEW (CORRECT):
# Return all data for full context
if len(hist) >= LOOKBACK:  # At least 20 bars
    return hist
else:
    return pd.DataFrame()  # Not enough data
```

### Fix 2: Add Alpaca Data Source
```python
def get_market_data(symbol: str, period: str = "2d", interval: str = "1m", api=None) -> pd.DataFrame:
    """
    Get market data - tries Alpaca first, then Massive, then yfinance
    """
    # Priority 1: Alpaca API (you're paying for it!)
    if api:
        try:
            from alpaca_trade_api.rest import TimeFrame
            from datetime import datetime, timedelta
            
            # Calculate date range
            if period == "2d":
                days = 2
            elif period == "1d":
                days = 1
            else:
                days = 2
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Map interval
            if interval == "1m":
                timeframe = TimeFrame.Minute
            elif interval == "5m":
                timeframe = TimeFrame(5, TimeFrame.Minute)
            else:
                timeframe = TimeFrame.Minute
            
            # Get bars from Alpaca
            bars = api.get_bars(
                symbol,
                timeframe=timeframe,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                limit=5000
            ).df
            
            if len(bars) > 0:
                # Alpaca returns: open, high, low, close, volume
                # Rename to match expected format
                bars.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                return bars
            
        except Exception as e:
            # Fallback to Massive/yfinance
            pass
    
    # Priority 2: Massive API
    # ... existing code ...
    
    # Priority 3: yfinance
    # ... existing code ...
```

---

## 🔍 DATA VALIDATION CHECKS

### Check 1: Bar Count
```python
expected_bars = 2 * 1440  # 2 days × 1,440 min/day
actual_bars = len(hist)
if actual_bars < expected_bars * 0.8:  # At least 80% of expected
    risk_mgr.log(f"⚠️ {symbol}: Only {actual_bars} bars, expected ~{expected_bars}", "WARNING")
```

### Check 2: Price Reasonableness
```python
if 'Close' in hist.columns:
    close_prices = hist['Close']
    if symbol == 'SPY' and not (300 < close_prices.mean() < 800):
        risk_mgr.log(f"⚠️ {symbol}: Unusual prices: ${close_prices.mean():.2f}", "WARNING")
```

### Check 3: Data Freshness
```python
latest_timestamp = hist.index.max()
age_minutes = (datetime.now() - latest_timestamp).total_seconds() / 60
if age_minutes > 15:
    risk_mgr.log(f"⚠️ {symbol}: Data is {age_minutes:.0f} minutes old", "WARNING")
```

---

## 📊 EXPECTED VS ACTUAL

### Expected (2 days, 1-minute):
- Bars: ~2,880 (2 days × 1,440 minutes/day)
- Time span: 2 days
- Data source: Alpaca (real-time)

### Actual (Current Bug):
- Bars: 50 (limited by `.tail(50)`)
- Time span: ~50 minutes
- Data source: yfinance (delayed)

### After Fix:
- Bars: ~2,880 (full 2 days)
- Time span: 2 days
- Data source: Alpaca (real-time)

---

## 🚀 NEXT STEPS

1. **Fix the 50-bar limit** (critical bug)
2. **Add Alpaca as primary data source** (you're paying for it!)
3. **Add data validation logging** (verify data quality)
4. **Test with verification script** (confirm fixes work)

---

**Status:** Ready for implementation
**Priority:** HIGH (affects all trading decisions)





