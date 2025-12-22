# 🔍 Data Source Investigation Report

## 📊 Test Results

### **Direct Data Source Tests:**

1. **yfinance (Direct):**
   - ✅ Price: $680.84
   - ✅ Range: $677.00 - $680.99
   - **Status:** CORRECT - matches actual market

2. **Massive API (Direct):**
   - ✅ Price: $680.68
   - ✅ Range: $671.89 - $680.84
   - **Status:** CORRECT - close to actual market

3. **Alpaca API:**
   - ⚠️ No API keys found in environment
   - **Status:** NOT BEING USED

### **Agent Logs Show:**
- ❌ Prices: $683.55, $683.76, $684.00, etc.
- ❌ **$3-4 HIGHER** than actual market
- **Status:** WRONG DATA

---

## 🔍 Root Cause Analysis

### **Hypothesis 1: Stale/Cached Data**
- Agent might be using cached data from previous day
- Data might not be refreshing properly
- **Evidence:** Logs don't show data source logs (should show "📊 SPY Data: X bars from Y")

### **Hypothesis 2: Wrong Data Source**
- Agent might be using a different data source than expected
- Alpaca API might be returning wrong data (but API keys not found)
- **Evidence:** No data source logs found in mike.log

### **Hypothesis 3: Data Transformation Issue**
- Price might be getting transformed/adjusted somewhere
- Could be using adjusted prices instead of raw prices
- **Evidence:** Code shows `adjustment='raw'` for Alpaca, but Alpaca not being used

### **Hypothesis 4: Timezone/Date Issue**
- Agent might be getting data from wrong day
- Timezone mismatch causing wrong date selection
- **Evidence:** Date calculations in get_market_data() function

---

## 🔧 Investigation Steps

### **Step 1: Check Data Source Logging**
The `log_data_source()` function should log:
```
📊 SPY Data: X bars from [Alpaca API|Massive API|yfinance]
```

**Finding:** No such logs found in mike.log
**Conclusion:** Either:
- Logging is filtered out
- Data source logging isn't happening
- Different code path is being used

### **Step 2: Check Which Source Is Actually Used**
Priority order:
1. Alpaca API (if available)
2. Massive API (if available)
3. yfinance (fallback)

**Finding:** 
- Alpaca: No API keys → Not used
- Massive: Available → Should be used
- yfinance: Available → Fallback

**Conclusion:** Should be using Massive API or yfinance, both return correct prices

### **Step 3: Check for Data Caching**
**Finding:** No obvious caching in code
**Conclusion:** Need to check if pandas/yfinance is caching

### **Step 4: Check Date/Time Calculations**
**Finding:** Code uses:
```python
end_date = datetime.now()
start_date = end_date - timedelta(days=days)
start_str = start_date.strftime("%Y-%m-%d")
end_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
```

**Potential Issue:** 
- `datetime.now()` might be in wrong timezone
- Date strings might be selecting wrong day
- **Need to verify:** What dates are actually being requested?

---

## 🎯 Most Likely Cause

### **Stale Data from Previous Day**

**Theory:**
1. Agent started before market open
2. Got data from previous day (Dec 18)
3. Previous day's high was ~$684
4. Data never refreshed during trading day
5. Agent kept using stale data

**Evidence:**
- Agent logs show prices ~$684 (previous day's range)
- Actual market today: $676-680 (lower range)
- No data source refresh logs

**Fix:**
- Add data freshness check
- Force refresh every minute
- Validate last bar timestamp

---

## 🔧 Recommended Fixes

### **Fix 1: Add Data Freshness Validation**
```python
# After getting data, check if it's fresh
last_bar_time = hist.index[-1]
now = datetime.now()
time_diff = (now - last_bar_time).total_seconds() / 60

if time_diff > 5:  # More than 5 minutes old
    risk_mgr.log(f"⚠️ WARNING: Data is {time_diff:.1f} minutes old - may be stale", "WARNING")
    # Force refresh or use alternative source
```

### **Fix 2: Add Data Source Logging**
```python
# Log which source is actually being used
risk_mgr.log(f"📊 {symbol} Data: {len(hist)} bars from {source} | Last price: ${hist['Close'].iloc[-1]:.2f} at {hist.index[-1]}", "INFO")
```

### **Fix 3: Add Price Validation**
```python
# Cross-check price with multiple sources
yf_price = get_yfinance_price(symbol)
if abs(current_price - yf_price) > 0.50:
    risk_mgr.log(f"⚠️ PRICE MISMATCH: {symbol} - Agent: ${current_price:.2f}, yfinance: ${yf_price:.2f}", "WARNING")
```

### **Fix 4: Force Data Refresh**
```python
# Clear any caches before fetching
import yfinance as yf
yf.pdr_override = False  # Disable pandas_datareader cache
ticker = yf.Ticker(symbol)
ticker.history(period=period, interval=interval, clear_cache=True)
```

---

## 📝 Next Steps

1. ✅ **Test data sources** (DONE - both return correct prices)
2. ⏳ **Check agent's actual data fetch** (need to test get_market_data function)
3. ⏳ **Add data freshness validation** (implement fix)
4. ⏳ **Add price cross-checking** (implement fix)
5. ⏳ **Add data source logging** (implement fix)

---

## ⚠️ CRITICAL FINDING

**The data sources themselves are CORRECT** (yfinance and Massive both return $680.84).

**The agent is seeing WRONG prices** ($683-684).

**This means:**
- Data is being cached/stale
- OR wrong date is being requested
- OR data transformation is happening

**DO NOT TRADE until this is fixed!**

