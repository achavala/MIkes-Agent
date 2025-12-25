# 🔍 PRICE DATA SOURCE INVESTIGATION

**Issue:** Agent generating strike $688 when SPY is actually ~$680.59  
**Date:** December 19, 2025  
**Critical:** Using wrong/stale price data

---

## 🚨 PROBLEM

### User Observation:
- **SPY actual price:** ~$680.59 (from market data)
- **Agent generated strike:** $688
- **Option symbol:** SPY251219C00688000
- **Calculation:** Strike = Price + $2, so Price must be $686

**This means the agent is using $686 as the price, which is WRONG!**

---

## 🔍 DATA SOURCE FLOW

### Price Data Sources (Priority Order):

1. **`get_market_data()`** - Used for main SPY data
   - Alpaca API → Massive API → yfinance
   - Returns: `hist['Close'].iloc[-1]` as `current_price`

2. **`get_current_price()`** - Used for symbol-specific prices
   - Massive API → yfinance
   - Returns: Latest price from data source

### Where Price is Used:

**Line 3288:** `current_price = hist['Close'].iloc[-1]`
- From `get_market_data("SPY", ...)`
- This is the PRIMARY price used

**Line 3988:** `symbol_price = get_current_price(current_symbol)`
- Used for strike calculation
- Falls back to `current_price` if None

**Line 4011:** `strike = find_atm_strike(symbol_price, option_type='call')`
- Calculates: `strike = symbol_price + 2.0`
- If `symbol_price = $686`, then `strike = $688` ✅ This matches!

---

## 🔍 ROOT CAUSE ANALYSIS

### Hypothesis 1: Stale Data from `get_market_data()`
- `get_market_data()` might be returning stale data
- Even with freshness validation, might be using cached data
- Last bar might be from previous day

### Hypothesis 2: Wrong Data Source
- `get_current_price()` might be using wrong source
- Massive API might return stale data
- yfinance might return delayed data

### Hypothesis 3: Price Calculation Error
- Price might be calculated incorrectly
- Might be using High instead of Close
- Might be using wrong timestamp

---

## 🔧 INVESTIGATION NEEDED

### Check 1: What price is `get_market_data()` returning?
```python
hist = get_market_data("SPY", period="2d", interval="1m", api=api, risk_mgr=risk_mgr)
current_price = hist['Close'].iloc[-1]
# What is current_price? Should be ~$680.59
```

### Check 2: What price is `get_current_price()` returning?
```python
symbol_price = get_current_price("SPY")
# What is symbol_price? Should be ~$680.59
```

### Check 3: What is the last bar timestamp?
```python
last_bar_time = hist.index[-1]
# Is this from today or yesterday?
```

---

## ✅ FIXES NEEDED

### Fix #1: Add Price Validation
- Validate price is within reasonable range ($600-$700 for SPY)
- Reject if price is outside expected range
- Log which data source provided the price

### Fix #2: Add Price Source Logging
- Log which source provided the price (Alpaca/Massive/yfinance)
- Log the price value and timestamp
- Log data freshness

### Fix #3: Cross-Validate Prices
- Compare `get_market_data()` price vs `get_current_price()` price
- Reject if difference > $2
- Use most recent/reliable source

### Fix #4: Add Price Sanity Checks
- Check if price matches expected range
- Check if price changed dramatically (might be wrong)
- Validate against multiple sources

---

## 📊 EXPECTED BEHAVIOR

### For SPY on Dec 19:
- **Expected Price Range:** $680-$681
- **Expected Strike:** $682-$683 (price + $2)
- **Expected Option Symbol:** SPY251219C00682000 or SPY251219C00683000

### If Price Shows $686:
- **Strike would be:** $688
- **This is WRONG** - suggests stale data
- **Should be rejected** by validation

---

## 🎯 IMMEDIATE ACTIONS

1. **Add price validation** before strike calculation
2. **Add data source logging** to see where price comes from
3. **Add price sanity checks** to reject unreasonable prices
4. **Cross-validate** prices from multiple sources

---

**Status:** 🔴 CRITICAL - Price data validation needed


