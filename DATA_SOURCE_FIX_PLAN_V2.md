# 🔧 Data Source Fix Plan V2 - Prioritize Paid Sources

## 🎯 Updated Strategy

**Priority Order (CORRECT):**
1. **Alpaca API** - Real-time, options data, paid subscription ✅
2. **Massive API** - Real-time, 1-minute granular, paid subscription ✅
3. **yfinance** - **ONLY for validation/cross-check**, NOT primary source ⚠️

**Why:**
- 0DTE trading needs **real-time data** (yfinance is delayed)
- Options data requires **Alpaca/Massive** (yfinance doesn't have options)
- Paid subscriptions should be **utilized**, not ignored

---

## 🔍 Current Status

### **Alpaca API:**
- ✅ Keys configured in `config.py`
- ✅ API initialized successfully
- ⚠️ **Issue:** May not be getting TODAY's data (stale data from previous day)

### **Massive API:**
- ✅ Client initialized
- ✅ Available for use
- ⚠️ **Issue:** May not be getting TODAY's data (stale data from previous day)

### **yfinance:**
- ✅ Available as fallback
- ⚠️ **Should NOT be primary source** (delayed, no options)

---

## 🔧 Fixes Required

### **Fix 1: Ensure Alpaca Gets TODAY's Data** ⚠️ **CRITICAL**

**Problem:** Date calculation might be requesting wrong day

**Location:** `get_market_data()` function, Alpaca section

**Fix:**
```python
# Use EST timezone for all date calculations
est = pytz.timezone('US/Eastern')
now_est = datetime.now(est)
today_est = now_est.date()

# Calculate date range
if period == "2d":
    days = 2
elif period == "1d":
    days = 1
else:
    days = 2

# Start from (days) days ago, end at TODAY (inclusive)
start_date = now_est - timedelta(days=days)
if period == "2d":
    start_date = start_date - timedelta(hours=12)  # Buffer for extended hours

# CRITICAL: Ensure end date includes TODAY
# Use tomorrow as end (exclusive) to get today's data
end_date = now_est + timedelta(days=1)

start_str = start_date.strftime("%Y-%m-%d")
end_str = end_date.strftime("%Y-%m-%d")

# Log what we're requesting
if risk_mgr and hasattr(risk_mgr, 'log'):
    risk_mgr.log(
        f"🔍 Alpaca API: Requesting {symbol} from {start_str} to {end_str} "
        f"(EST: {now_est.strftime('%Y-%m-%d %H:%M:%S %Z')}, Today: {today_est})",
        "INFO"
    )

# Get bars from Alpaca
bars = api.get_bars(
    symbol,
    timeframe,
    start_str,
    end_str,
    limit=5000,
    adjustment='raw'  # Raw prices, not adjusted
).df

# CRITICAL: Validate data is from TODAY
if len(bars) > 0:
    last_bar_date = bars.index[-1].date()
    if hasattr(bars.index[-1], 'tzinfo') and bars.index[-1].tzinfo:
        # Convert to EST if timezone-aware
        last_bar_est = bars.index[-1].astimezone(est)
        last_bar_date = last_bar_est.date()
    
    if last_bar_date != today_est:
        if risk_mgr and hasattr(risk_mgr, 'log'):
            risk_mgr.log(
                f"❌ CRITICAL: Alpaca data is from {last_bar_date}, not today ({today_est})! "
                f"Rejecting stale data. Last bar: {bars.index[-1]}",
                "ERROR"
            )
        # Don't return stale data - fall through to Massive API
        bars = pd.DataFrame()
    else:
        # Data is fresh - validate timestamp
        last_bar_time = bars.index[-1]
        if hasattr(last_bar_time, 'tzinfo') and last_bar_time.tzinfo:
            last_bar_est = last_bar_time.astimezone(est)
        else:
            last_bar_est = est.localize(last_bar_time)
        
        time_diff_minutes = (now_est - last_bar_est).total_seconds() / 60
        
        if time_diff_minutes > 5:
            if risk_mgr and hasattr(risk_mgr, 'log'):
                risk_mgr.log(
                    f"⚠️ WARNING: Alpaca data is {time_diff_minutes:.1f} minutes old "
                    f"(last bar: {last_bar_est.strftime('%H:%M:%S %Z')})",
                    "WARNING"
                )
        
        # Log successful fetch
        log_data_source("Alpaca API", len(bars), symbol)
        if risk_mgr and hasattr(risk_mgr, 'log'):
            risk_mgr.log(
                f"✅ Alpaca API: {len(bars)} bars, last price: ${bars['Close'].iloc[-1]:.2f} "
                f"at {last_bar_est.strftime('%Y-%m-%d %H:%M:%S %Z')}",
                "INFO"
            )
        return bars
```

### **Fix 2: Ensure Massive Gets TODAY's Data** ⚠️ **CRITICAL**

**Location:** `get_market_data()` function, Massive section

**Fix:**
```python
# Use EST timezone
est = pytz.timezone('US/Eastern')
now_est = datetime.now(est)
today_est = now_est.date()

# Calculate date range
if period == "2d":
    days = 2
elif period == "1d":
    days = 1
else:
    days = 2

# Massive API needs date strings
end_date_str = (now_est + timedelta(days=1)).strftime("%Y-%m-%d")  # Tomorrow to include today
start_date_str = (now_est - timedelta(days=days)).strftime("%Y-%m-%d")

# Log request
if risk_mgr and hasattr(risk_mgr, 'log'):
    risk_mgr.log(
        f"🔍 Massive API: Requesting {symbol} from {start_date_str} to {end_date_str} "
        f"(EST: {now_est.strftime('%Y-%m-%d %H:%M:%S %Z')}, Today: {today_est})",
        "INFO"
    )

# Get data from Massive
hist = massive_client.get_historical_data(
    massive_symbol, 
    start_date_str, 
    end_date_str, 
    interval=interval
)

# CRITICAL: Validate data is from TODAY
if len(hist) > 0:
    last_bar_date = hist.index[-1].date()
    if hasattr(hist.index[-1], 'tzinfo') and hist.index[-1].tzinfo:
        last_bar_est = hist.index[-1].astimezone(est)
        last_bar_date = last_bar_est.date()
    
    if last_bar_date != today_est:
        if risk_mgr and hasattr(risk_mgr, 'log'):
            risk_mgr.log(
                f"❌ CRITICAL: Massive data is from {last_bar_date}, not today ({today_est})! "
                f"Rejecting stale data.",
                "ERROR"
            )
        # Don't return stale data - fall through to yfinance
        hist = pd.DataFrame()
    else:
        # Data is fresh - log success
        log_data_source("Massive API", len(hist), symbol)
        if risk_mgr and hasattr(risk_mgr, 'log'):
            last_price = hist['Close'].iloc[-1] if 'Close' in hist.columns else hist['close'].iloc[-1]
            risk_mgr.log(
                f"✅ Massive API: {len(hist)} bars, last price: ${last_price:.2f} "
                f"at {hist.index[-1]}",
                "INFO"
            )
        return hist
```

### **Fix 3: Use yfinance ONLY for Validation** ✅ **IMPORTANT**

**Location:** Main trading loop, after getting current_price from Alpaca/Massive

**Fix:**
```python
# Get price from primary source (Alpaca or Massive)
current_price = hist['Close'].iloc[-1]

# CRITICAL: Cross-validate with yfinance ONLY (not as primary source)
# This is a safety check, not the primary data source
try:
    yf_ticker = yf.Ticker('SPY')
    yf_hist = yf_ticker.history(period='1d', interval='1m')
    if len(yf_hist) > 0:
        yf_price = yf_hist['Close'].iloc[-1]
        price_diff = abs(current_price - yf_price)
        
        # Log validation
        if risk_mgr and hasattr(risk_mgr, 'log'):
            risk_mgr.log(
                f"🔍 Price Validation: Primary source: ${current_price:.2f}, "
                f"yfinance (delayed): ${yf_price:.2f}, diff: ${price_diff:.2f}",
                "DEBUG"
            )
        
        # If difference is significant, warn but use primary source
        if price_diff > 2.00:  # More than $2 difference
            if risk_mgr and hasattr(risk_mgr, 'log'):
                risk_mgr.log(
                    f"⚠️ PRICE MISMATCH: ${price_diff:.2f} difference between primary source "
                    f"(${current_price:.2f}) and yfinance (${yf_price:.2f}). "
                    f"Using primary source (Alpaca/Massive) - yfinance is delayed.",
                    "WARNING"
                )
        elif price_diff > 0.50:  # More than $0.50 difference
            if risk_mgr and hasattr(risk_mgr, 'log'):
                risk_mgr.log(
                    f"⚠️ Price difference: ${price_diff:.2f} (primary: ${current_price:.2f}, "
                    f"yfinance: ${yf_price:.2f}). Using primary source.",
                    "INFO"
                )
except Exception as e:
    # Validation failed - not critical, just log
    if risk_mgr and hasattr(risk_mgr, 'log'):
        risk_mgr.log(f"⚠️ Price validation failed: {e} (non-critical)", "WARNING")

# Continue with primary source price (Alpaca/Massive)
# DO NOT use yfinance price - it's delayed and doesn't have options data
```

### **Fix 4: Never Use yfinance as Primary Source** ✅ **CRITICAL**

**Location:** `get_market_data()` function, yfinance section

**Fix:**
```python
# ========== PRIORITY 3: YFINANCE (LAST RESORT - DELAYED DATA) ==========
# ⚠️ WARNING: yfinance is DELAYED (15-20 minutes) and has NO OPTIONS DATA
# Only use if Alpaca AND Massive both fail
# For 0DTE trading, this is NOT acceptable - should fail rather than use delayed data

try:
    if risk_mgr and hasattr(risk_mgr, 'log'):
        risk_mgr.log(
            f"⚠️ WARNING: Both Alpaca and Massive failed for {symbol}. "
            f"Falling back to yfinance (DELAYED - NOT SUITABLE FOR 0DTE TRADING)",
            "WARNING"
        )
    
    # Map symbol for yfinance
    yf_symbol = symbol
    if symbol == 'SPX':
        yf_symbol = '^GSPC'
    
    ticker = yf.Ticker(yf_symbol)
    hist = ticker.history(period=period, interval=interval)
    
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    hist = hist.dropna()
    
    if len(hist) > 0:
        # CRITICAL: Validate this is TODAY's data
        last_bar_date = hist.index[-1].date()
        today = datetime.now(pytz.timezone('US/Eastern')).date()
        
        if last_bar_date != today:
            if risk_mgr and hasattr(risk_mgr, 'log'):
                risk_mgr.log(
                    f"❌ CRITICAL: yfinance data is from {last_bar_date}, not today ({today})! "
                    f"Cannot use stale data for 0DTE trading.",
                    "ERROR"
                )
            return pd.DataFrame()  # Return empty - better than wrong data
        
        log_data_source("yfinance (DELAYED - LAST RESORT)", len(hist), symbol)
        if risk_mgr and hasattr(risk_mgr, 'log'):
            risk_mgr.log(
                f"⚠️ Using yfinance (DELAYED) for {symbol} - NOT SUITABLE FOR 0DTE! "
                f"Last price: ${hist['Close'].iloc[-1]:.2f} at {hist.index[-1]}",
                "WARNING"
            )
        return hist
except Exception as e:
    if risk_mgr and hasattr(risk_mgr, 'log'):
        risk_mgr.log(f"❌ All data sources failed for {symbol}: {e}", "ERROR")
    return pd.DataFrame()
```

---

## 📋 Implementation Priority

1. **Fix 1: Alpaca TODAY validation** - ⚠️ **CRITICAL** (ensure real-time data)
2. **Fix 2: Massive TODAY validation** - ⚠️ **CRITICAL** (ensure real-time data)
3. **Fix 4: Never use yfinance as primary** - ✅ **CRITICAL** (0DTE needs real-time)
4. **Fix 3: yfinance validation only** - ✅ **IMPORTANT** (safety check)

---

## 🎯 Key Principles

1. **Alpaca First** - Real-time, options data, paid subscription
2. **Massive Second** - Real-time, 1-minute granular, paid subscription
3. **yfinance Last** - ONLY if both fail, and ONLY for validation
4. **Never Trade on Delayed Data** - If only yfinance available, don't trade
5. **Always Validate TODAY** - Reject any data not from today

---

## ⚠️ CRITICAL WARNING

**For 0DTE trading:**
- ❌ **NEVER use yfinance as primary source** (delayed 15-20 minutes)
- ❌ **NEVER trade on stale data** (must be from TODAY)
- ✅ **ALWAYS use Alpaca or Massive** (real-time, paid subscriptions)
- ✅ **ALWAYS validate data freshness** (check date and timestamp)

**If Alpaca and Massive both fail:**
- **DO NOT TRADE** - Better to skip trades than use delayed data
- Log error and wait for data sources to recover

