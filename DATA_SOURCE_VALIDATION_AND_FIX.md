# 🔍 DATA SOURCE VALIDATION & FIX - COMPLETE ANALYSIS

**Date:** December 17, 2025  
**Issue:** Ensure training uses Alpaca/Massive (paid) NOT yfinance  
**Status:** ⚠️ **CRITICAL ISSUE FOUND - NEEDS FIX**

---

## ❌ CRITICAL FINDING

**Current State:**
- ✅ Training script uses `--data-source massive` (correct)
- ❌ **NO ALPACA INTEGRATION in training data collection**
- ❌ Training only uses Massive API OR yfinance (no Alpaca fallback)
- ⚠️ If Massive API fails, it falls back to yfinance (NOT Alpaca)

---

## 📊 CURRENT DATA FLOW (LINE BY LINE)

### **1. Training Script Entry Point**
**File:** `train_historical_model.py`  
**Line:** 339-344

```python
# If human_momentum is enabled, default to intraday data unless explicitly overridden.
resolved_source = data_source
if human_momentum and data_source == "enriched":
    resolved_source = "massive"
```

**✅ GOOD:** If `--human-momentum` is used, it defaults to "massive"

---

### **2. Data Source Selection**
**File:** `train_historical_model.py`  
**Line:** 344-381

```python
if resolved_source in ("massive", "polygon", "intraday"):
    print("📥 LOADING INTRADAY (1m) HISTORICAL DATA...")
    print("   Source: Massive/Polygon (cached)")
    
    for symbol in symbols:
        df = collector.get_historical_data_massive(
            symbol=symbol,
            start_date=intraday_start,
            end_date=intraday_end,
            interval="1m",
            use_cache=True,
        )
```

**✅ GOOD:** Uses `get_historical_data_massive()` when `--data-source massive`

**❌ PROBLEM:** No Alpaca fallback if Massive fails

---

### **3. Massive API Data Collection**
**File:** `historical_training_system.py`  
**Line:** 60-114

```python
def get_historical_data_massive(
    self,
    symbol: str,
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = "1m",
    use_cache: bool = True,
) -> pd.DataFrame:
    # ... cache check ...
    
    # Lazy import
    try:
        from massive_api_client import MassiveAPIClient
    except Exception as e:
        raise ImportError("massive_api_client.py not available...")
    
    massive_symbol = self.massive_symbol_map.get(symbol, symbol)
    client = MassiveAPIClient()
    df = client.get_historical_data(massive_symbol, start_date=start_date, end_date=end_date, interval=interval)
    
    return df
```

**✅ GOOD:** Uses Massive API directly  
**❌ PROBLEM:** No Alpaca fallback if Massive fails  
**❌ PROBLEM:** Raises exception if Massive fails (no graceful fallback)

---

### **4. Fallback to yfinance (PROBLEM)**
**File:** `historical_training_system.py`  
**Line:** 116-243

```python
def get_historical_data(
    self,
    symbol: str,
    start_date: str = "2002-01-01",
    end_date: Optional[str] = None,
    interval: str = '1m',
    use_cache: bool = True
) -> pd.DataFrame:
    # ... cache check ...
    
    print(f"📥 Downloading {symbol} data: {start_date} to {end_date} ({interval})...")
    
    try:
        ticker = yf.Ticker(yf_symbol)  # ❌ USES YFINANCE!
        # ... yfinance download logic ...
```

**❌ PROBLEM:** This function ONLY uses yfinance (no Alpaca/Massive)

**This function is NOT called when `--data-source massive` is used, but if Massive fails, there's no Alpaca fallback.**

---

## 🔧 REQUIRED FIXES

### **FIX #1: Add Alpaca as Priority Data Source**

**File:** `historical_training_system.py`  
**Add new method:** `get_historical_data_alpaca()`

```python
def get_historical_data_alpaca(
    self,
    symbol: str,
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = "1m",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Get historical OHLCV data from Alpaca API (PRIORITY 1 for paid members).
    
    Requires ALPACA_KEY and ALPACA_SECRET env vars.
    Caches results to disk.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Cache file path
    safe_symbol = symbol.replace(":", "_")
    cache_file = self.cache_dir / f"{safe_symbol}_{interval}_{start_date}_{end_date}_alpaca.pkl"
    
    if use_cache and cache_file.exists():
        print(f"📂 Loading cached Alpaca data: {cache_file.name}")
        try:
            with open(cache_file, "rb") as f:
                df = pickle.load(f)
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                return df
        except Exception as e:
            print(f"⚠️ Alpaca cache load failed: {e}, downloading fresh data")
    
    # Check for Alpaca credentials
    api_key = os.getenv('ALPACA_KEY') or os.getenv('ALPACA_API_KEY') or os.getenv('APCA_API_KEY_ID')
    api_secret = os.getenv('ALPACA_SECRET') or os.getenv('ALPACA_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY')
    
    if not api_key or not api_secret:
        print(f"⚠️ Alpaca credentials not found. Skipping Alpaca data source.")
        return pd.DataFrame()
    
    try:
        import alpaca_trade_api as tradeapi
        from alpaca_trade_api.rest import TimeFrame
        
        # Use paper trading URL (or live if configured)
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        
        # Convert interval to Alpaca TimeFrame
        if interval == '1m':
            tf = TimeFrame.Minute
        elif interval == '5m':
            tf = TimeFrame(5, TimeFrame.Minute)
        elif interval == '1h':
            tf = TimeFrame.Hour
        elif interval == '1d':
            tf = TimeFrame.Day
        else:
            tf = TimeFrame.Minute
        
        print(f"📥 Alpaca download: {symbol} {start_date}→{end_date} ({interval})")
        
        # Get bars from Alpaca
        bars = api.get_bars(
            symbol,
            tf,
            start=start_date,
            end=end_date
        ).df
        
        if len(bars) == 0:
            print(f"⚠️ Alpaca returned 0 bars for {symbol}")
            return pd.DataFrame()
        
        # Normalize column names
        bars.columns = [col.lower() for col in bars.columns]
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in bars.columns:
                if col == 'volume':
                    bars['volume'] = 0
                else:
                    bars[col] = bars.get('close', 0)
        
        # Filter to trading hours for minute data
        if interval == '1m' and len(bars) > 0:
            bars = bars.between_time('09:30', '16:00')
        
        # Remove duplicates and sort
        bars = bars.drop_duplicates().sort_index()
        
        # Cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(bars, f)
            print(f"💾 Cached {len(bars)} bars from Alpaca to {cache_file.name}")
        except Exception as e:
            print(f"⚠️ Could not write Alpaca cache: {e}")
        
        return bars
        
    except ImportError:
        print(f"⚠️ alpaca-trade-api not installed. Skipping Alpaca data source.")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Alpaca API error: {e}")
        return pd.DataFrame()
```

---

### **FIX #2: Update get_historical_data_massive() with Alpaca Fallback**

**File:** `historical_training_system.py`  
**Modify:** `get_historical_data_massive()`

```python
def get_historical_data_massive(
    self,
    symbol: str,
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = "1m",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Get historical OHLCV data with PRIORITY: Alpaca → Massive → yfinance
    """
    # PRIORITY 1: Try Alpaca first (you're paying for this!)
    print(f"🔑 Attempting Alpaca API for {symbol}...")
    alpaca_data = self.get_historical_data_alpaca(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        use_cache=use_cache
    )
    
    if alpaca_data is not None and len(alpaca_data) > 0:
        print(f"✅ Got {len(alpaca_data)} bars from Alpaca API (PAID SERVICE)")
        return alpaca_data
    
    # PRIORITY 2: Try Massive API
    print(f"🔑 Attempting Massive API for {symbol}...")
    try:
        massive_data = self._get_historical_data_massive_internal(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            use_cache=use_cache
        )
        
        if massive_data is not None and len(massive_data) > 0:
            print(f"✅ Got {len(massive_data)} bars from Massive API (PAID SERVICE)")
            return massive_data
    except Exception as e:
        print(f"⚠️ Massive API failed: {e}")
    
    # PRIORITY 3: Fallback to yfinance (FREE, but limited)
    print(f"⚠️ Paid services failed, falling back to yfinance (FREE, LIMITED)...")
    return self.get_historical_data(symbol, start_date, end_date, interval, use_cache)
```

---

### **FIX #3: Update Training Script to Log Data Source**

**File:** `train_historical_model.py`  
**Add logging to show which data source was used:**

```python
for symbol in symbols:
    print(f"\n📊 Loading {symbol} intraday data (1m)...")
    print(f"   🔑 Priority: Alpaca → Massive → yfinance")
    
    df = collector.get_historical_data_massive(
        symbol=symbol,
        start_date=intraday_start,
        end_date=intraday_end,
        interval="1m",
        use_cache=True,
    )
    
    # Log which source was actually used (check cache file name)
    cache_file = collector.cache_dir / f"{symbol}_1m_{intraday_start}_{intraday_end}_*.pkl"
    # ... check which cache file exists to determine source ...
```

---

## 🚀 FLY.IO TRAINING (CAN IT RUN?)

**Current State:**
- ❌ Training is designed for **local execution** (saves to `data/historical/`)
- ❌ Fly.io is configured for **live trading agent** (not training)
- ⚠️ Training requires:
  - Large disk space (2 years of 1-minute data)
  - Long runtime (4-6 hours)
  - GPU/CPU intensive

**Options:**
1. **Run training locally** (recommended) - Your laptop, can run overnight
2. **Run training on Fly.io** (possible but complex) - Would need:
   - Separate Fly.io app for training
   - Persistent volume for data cache
   - Larger VM (more CPU/memory)
   - Higher cost

**Recommendation:** Run training **locally** (laptop can be on), then deploy the trained model to Fly.io.

---

## ✅ VALIDATION CHECKLIST

After fixes, verify:

- [ ] Alpaca API credentials are checked
- [ ] Alpaca is tried FIRST (before Massive)
- [ ] Massive is tried SECOND (if Alpaca fails)
- [ ] yfinance is LAST resort (if both fail)
- [ ] Logging shows which source was used
- [ ] Cache files are named by source (`_alpaca.pkl`, `_massive.pkl`, `.pkl`)
- [ ] Training script logs data source for each symbol

---

## 📋 IMMEDIATE ACTION PLAN

1. **Add Alpaca data collection method** to `historical_training_system.py`
2. **Update `get_historical_data_massive()`** to try Alpaca first
3. **Add logging** to show which data source was used
4. **Test locally** with Alpaca credentials
5. **Verify data quality** (check bar counts, date ranges)

---

**This ensures you're using your PAID data sources (Alpaca/Massive) and NOT free yfinance! 🎯**





