# 🔧 Data Source Fix Plan

## 🎯 Root Cause Identified

**Problem:** Agent is seeing prices $3-4 higher than actual market
- Agent sees: $683-684
- Actual market: $676-680
- **Discrepancy:** ~$3.50

**Root Cause:** **STALE DATA FROM PREVIOUS DAY**

### **Evidence:**
1. ✅ Direct tests show data sources return CORRECT prices ($680.84)
2. ❌ Agent logs show WRONG prices ($683-684)
3. ❌ No data source refresh logs found
4. ⚠️ Previous day (Dec 18) SPY was around $684
5. ✅ Today (Dec 19) SPY is $676-680

**Conclusion:** Agent is using cached/stale data from Dec 18, not refreshing to Dec 19 data.

---

## 🔧 Fixes Required

### **Fix 1: Add Data Freshness Validation** ⚠️ **CRITICAL**

**Location:** `get_market_data()` function, after data fetch

**Code:**
```python
# After getting data, validate freshness
if len(bars) > 0:
    last_bar_time = bars.index[-1]
    now = datetime.now(pytz.timezone('US/Eastern'))
    
    # Convert last_bar_time to EST if needed
    if hasattr(last_bar_time, 'tzinfo') and last_bar_time.tzinfo:
        last_bar_est = last_bar_time.astimezone(pytz.timezone('US/Eastern'))
    else:
        # Assume UTC if no timezone
        last_bar_est = pytz.utc.localize(last_bar_time).astimezone(pytz.timezone('US/Eastern'))
    
    time_diff_minutes = (now - last_bar_est).total_seconds() / 60
    
    # Check if data is stale (more than 5 minutes old during market hours)
    if time_diff_minutes > 5:
        if risk_mgr and hasattr(risk_mgr, 'log'):
            risk_mgr.log(
                f"⚠️ WARNING: {symbol} data is {time_diff_minutes:.1f} minutes old "
                f"(last bar: {last_bar_est.strftime('%Y-%m-%d %H:%M:%S %Z')}, "
                f"now: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}) - may be stale!",
                "WARNING"
            )
        
        # If data is more than 1 hour old, it's definitely stale
        if time_diff_minutes > 60:
            if risk_mgr and hasattr(risk_mgr, 'log'):
                risk_mgr.log(
                    f"❌ CRITICAL: {symbol} data is {time_diff_minutes:.1f} minutes old - "
                    f"REJECTING STALE DATA. Last bar date: {last_bar_est.date()}, Today: {now.date()}",
                    "ERROR"
                )
            # Don't return stale data - let it fall through to next source
            bars = pd.DataFrame()
```

### **Fix 2: Force Data Refresh Every Iteration** ⚠️ **CRITICAL**

**Location:** Main trading loop, before getting market data

**Code:**
```python
# Clear any caches before fetching fresh data
import yfinance as yf
# Disable caching for yfinance
if hasattr(yf, 'pdr_override'):
    yf.pdr_override = False

# Force fresh fetch
hist = get_market_data("SPY", period="2d", interval="1m", api=api, risk_mgr=risk_mgr)

# Validate data is from today
if len(hist) > 0:
    last_bar_date = hist.index[-1].date()
    today = datetime.now(pytz.timezone('US/Eastern')).date()
    
    if last_bar_date != today:
        risk_mgr.log(
            f"❌ CRITICAL: Data is from {last_bar_date}, not today ({today})! "
            f"Rejecting stale data.",
            "ERROR"
        )
        # Skip this iteration, wait for fresh data
        time.sleep(30)
        continue
```

### **Fix 3: Add Data Source Logging** ✅ **IMPORTANT**

**Location:** `get_market_data()` function, `log_data_source()` calls

**Enhancement:**
```python
def log_data_source(source: str, bars: int, symbol: str):
    if risk_mgr and hasattr(risk_mgr, 'log'):
        if len(bars) > 0:
            last_price = bars['Close'].iloc[-1]
            last_time = bars.index[-1]
            risk_mgr.log(
                f"📊 {symbol} Data: {bars} bars from {source} | "
                f"Last: ${last_price:.2f} at {last_time} | "
                f"period={period}, interval={interval}",
                "INFO"
            )
        else:
            risk_mgr.log(
                f"📊 {symbol} Data: 0 bars from {source} | period={period}, interval={interval}",
                "WARNING"
            )
```

### **Fix 4: Add Price Cross-Validation** ✅ **IMPORTANT**

**Location:** Main trading loop, after getting current_price

**Code:**
```python
current_price = hist['Close'].iloc[-1]

# Cross-validate price with yfinance
try:
    yf_ticker = yf.Ticker('SPY')
    yf_hist = yf_ticker.history(period='1d', interval='1m')
    if len(yf_hist) > 0:
        yf_price = yf_hist['Close'].iloc[-1]
        price_diff = abs(current_price - yf_price)
        
        if price_diff > 0.50:  # More than $0.50 difference
            risk_mgr.log(
                f"⚠️ PRICE MISMATCH: SPY - Agent: ${current_price:.2f}, "
                f"yfinance: ${yf_price:.2f}, diff: ${price_diff:.2f}",
                "WARNING"
            )
            
            # If difference is more than $2, reject and use yfinance
            if price_diff > 2.00:
                risk_mgr.log(
                    f"❌ CRITICAL PRICE MISMATCH: ${price_diff:.2f} difference! "
                    f"Using yfinance price ${yf_price:.2f} instead.",
                    "ERROR"
                )
                current_price = yf_price
except Exception as e:
    risk_mgr.log(f"⚠️ Price validation failed: {e}", "WARNING")
```

### **Fix 5: Fix Timezone Handling** ✅ **IMPORTANT**

**Location:** `get_market_data()` function, date calculations

**Code:**
```python
# Use EST timezone for all date calculations
est = pytz.timezone('US/Eastern')
now_est = datetime.now(est)
end_date = now_est
start_date = end_date - timedelta(days=days)

# For 2 days, extend start date
if period == "2d":
    start_date = start_date - timedelta(hours=12)

# Format dates
start_str = start_date.strftime("%Y-%m-%d")
end_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")

# Log what we're requesting
if risk_mgr and hasattr(risk_mgr, 'log'):
    risk_mgr.log(
        f"🔍 Requesting {symbol} data: {start_str} to {end_str} "
        f"(EST: {now_est.strftime('%Y-%m-%d %H:%M:%S %Z')})",
        "DEBUG"
    )
```

---

## 📋 Implementation Priority

1. **Fix 1: Data Freshness Validation** - ⚠️ **CRITICAL** (prevents stale data)
2. **Fix 2: Force Data Refresh** - ⚠️ **CRITICAL** (ensures fresh data)
3. **Fix 5: Timezone Handling** - ✅ **IMPORTANT** (prevents date issues)
4. **Fix 3: Data Source Logging** - ✅ **IMPORTANT** (debugging)
5. **Fix 4: Price Cross-Validation** - ✅ **IMPORTANT** (safety check)

---

## 🧪 Testing Plan

1. **Test with stale data:**
   - Simulate agent starting before market open
   - Verify it rejects previous day's data
   - Verify it waits for fresh data

2. **Test with fresh data:**
   - Verify agent uses today's data
   - Verify prices match actual market
   - Verify logging shows correct data source

3. **Test price validation:**
   - Simulate price mismatch
   - Verify warning is logged
   - Verify agent uses correct price

---

## ⚠️ CRITICAL WARNING

**DO NOT TRADE until these fixes are implemented!**

Current issue:
- Agent using stale data from previous day
- Prices $3-4 higher than actual market
- Would cause wrong strike selection
- Would cause wrong position sizing
- Would cause higher risk than intended

**Fix data source issue before resuming trading.**

