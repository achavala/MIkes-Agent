# 🔍 DETAILED ANALYSIS: Option Expiration Date Issue

**Date:** December 19, 2025  
**Issue:** Agent trading expired options (Dec 5, Dec 10) instead of today's (Dec 19)

---

## 🚨 CRITICAL FINDING

### Option Symbols Found in Logs:
1. **SPY251210C00687000** = SPY, Dec 10, 2025 (251210), Call, Strike $687
2. **SPY251210C00688000** = SPY, Dec 10, 2025 (251210), Call, Strike $688
3. **SPY251205C00686000** = SPY, Dec 5, 2025 (251205), Call, Strike $686

**Expected:** SPY251219C... (Dec 19, 2025)

---

## 🔍 ROOT CAUSE ANALYSIS

### Issue #1: `get_option_symbol()` Function

**Location:** Line 1566-1572

**Original Code:**
```python
def get_option_symbol(underlying: str, strike: float, option_type: str) -> str:
    """Generate Alpaca option symbol"""
    expiration = datetime.now()  # ❌ Uses local timezone, not EST!
    date_str = expiration.strftime('%y%m%d')
    ...
```

**Problem:**
- Uses `datetime.now()` which uses **system local timezone**
- For 0DTE options, we **MUST use EST timezone**
- If system is in different timezone, wrong date is generated
- No validation that expiration is TODAY

**Fix Applied:**
```python
def get_option_symbol(underlying: str, strike: float, option_type: str) -> str:
    """Generate Alpaca option symbol - MUST use EST for 0DTE options"""
    est = pytz.timezone('US/Eastern')
    expiration = datetime.now(est)  # ✅ Use EST explicitly
    expiration_date = expiration.date()
    today_est = datetime.now(est).date()
    
    # Validate expiration is today
    if expiration_date != today_est:
        print(f"⚠️ WARNING: Option expiration date mismatch!")
    
    date_str = expiration.strftime('%y%m%d')
    ...
```

---

## 📊 INVESTIGATION: When Were These Trades Executed?

### Hypothesis 1: Trades Executed on Dec 10/5
- If trades were executed on Dec 10 or Dec 5, the option symbols would be correct for those dates
- Need to check log timestamps

### Hypothesis 2: Wrong Date Generation
- If `datetime.now()` was using wrong timezone or cached date
- System timezone issue
- Date calculation error

### Hypothesis 3: Stale Data
- If market data was stale (from Dec 10/5), it might have influenced date calculation
- But option symbol generation shouldn't depend on market data

---

## 🔧 FIXES IMPLEMENTED

### Fix #1: EST Timezone in `get_option_symbol()`
- ✅ Changed to use `datetime.now(est)` explicitly
- ✅ Added date validation
- ✅ Added logging for debugging

### Fix #2: Additional Validation Needed
- ⚠️ Need to add validation before order execution
- ⚠️ Need to check if option expiration is today
- ⚠️ Need to reject orders for expired options

---

## 📋 DETAILED DATA SOURCE ANALYSIS

### Option Symbol Generation Flow:

1. **Entry Point:** Line 4001
   ```python
   symbol = get_option_symbol(current_symbol, strike, 'call')
   ```

2. **Function Call:** Line 1566
   ```python
   def get_option_symbol(underlying: str, strike: float, option_type: str) -> str:
       expiration = datetime.now()  # ❌ OLD: Wrong timezone
       expiration = datetime.now(est)  # ✅ NEW: EST timezone
   ```

3. **Date String Generation:**
   ```python
   date_str = expiration.strftime('%y%m%d')  # YYMMDD format
   # Example: 251219 for Dec 19, 2025
   ```

4. **Symbol Construction:**
   ```python
   option_symbol = f"{underlying}{date_str}{type_str}{strike_str}"
   # Example: SPY251219C00688000
   ```

### Data Dependencies:
- **Input:** `current_symbol` (SPY, QQQ, etc.), `strike` (688.00), `option_type` ('call')
- **Date Source:** `datetime.now()` → **MUST be EST**
- **Output:** Option symbol string

---

## 🎯 VALIDATION CHECKS NEEDED

### Check 1: Date Validation
```python
# Before generating symbol
est = pytz.timezone('US/Eastern')
today_est = datetime.now(est).date()
# Ensure expiration date matches today
```

### Check 2: Option Expiration Validation
```python
# After generating symbol, validate it's not expired
expiration_date = parse_expiration_from_symbol(option_symbol)
if expiration_date < today_est:
    raise ValueError(f"Option {option_symbol} is expired!")
```

### Check 3: Logging
```python
# Log generated symbol with expiration date
risk_mgr.log(f"Generated option: {option_symbol} | Expiration: {expiration_date} (EST)", "INFO")
```

---

## 🔍 ADDITIONAL INVESTIGATION

### Questions to Answer:

1. **When were these trades actually executed?**
   - Check log timestamps for SPY251210C00687000
   - Were they on Dec 10 or Dec 19?

2. **Is Alpaca accepting expired option symbols?**
   - Do orders for expired options get rejected?
   - Or are they executing on wrong contracts?

3. **Is there date caching?**
   - Is `datetime.now()` being cached somewhere?
   - Are there any date calculations that might be wrong?

4. **System timezone:**
   - What timezone is the system running in?
   - Is it different from EST?

---

## ✅ FIXES COMPLETED

1. ✅ **Fixed `get_option_symbol()` to use EST timezone**
2. ✅ **Added date validation**
3. ✅ **Added logging for debugging**

## ⚠️ FIXES STILL NEEDED

1. ⚠️ **Add expiration date validation before order execution**
2. ⚠️ **Reject orders for expired options**
3. ⚠️ **Add option chain validation (verify option exists)**
4. ⚠️ **Check all datetime.now() calls for EST timezone**

---

## 📊 TESTING REQUIRED

1. **Test option symbol generation:**
   ```python
   symbol = get_option_symbol('SPY', 688.00, 'call')
   # Should be: SPY251219C00688000 (Dec 19)
   # NOT: SPY251210C00688000 (Dec 10)
   ```

2. **Test with different times:**
   - Test at market open (9:30 AM EST)
   - Test during market hours
   - Test after market close

3. **Test timezone edge cases:**
   - Test at midnight EST
   - Test during DST transitions

---

**Status:** 🔴 CRITICAL - Fix applied, validation needed


