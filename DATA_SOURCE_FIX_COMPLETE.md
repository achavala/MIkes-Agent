# Data Source Fix - Alpaca + Massive API Integration

## ✅ Fix Applied

### Problem Identified:
- Alpaca API was working but returning only ~1,800 bars (market hours only)
- Expected ~2,880 bars for full 2 days of 1-minute data
- Massive API was never being used because Alpaca was succeeding

### Solution Implemented:
**Smart Fallback Logic:**
1. Try Alpaca API first
2. **Check if Alpaca returned sufficient data** (< 2,000 bars for 2 days)
3. If insufficient, **automatically try Massive API** (you have 1-minute granular package!)
4. Use whichever returns more complete data

## 🔧 Code Changes

### Alpaca API Section:
- Added data sufficiency check
- If Alpaca returns < 2,000 bars for 2 days, it logs a warning and continues to Massive API
- If Alpaca returns sufficient data, it returns immediately

### Massive API Section:
- Enhanced logging to show when Massive API is being used
- Better comparison messages when Massive provides more complete data
- Validates data completeness

## 📊 Expected Behavior

### Scenario 1: Alpaca Returns Insufficient Data
```
🔍 Alpaca API available for SPY, attempting data fetch...
📊 SPY Data: 1823 bars from Alpaca API | period=2d, interval=1m
⚠️ Alpaca API returned only 1823 bars for 2 days (expected ~2,880). Trying Massive API for complete data...
🔍 Attempting Massive API fetch for SPY: 2025-12-15 to 2025-12-17, interval=1m
📊 SPY Data: 2880 bars from Massive API | period=2d, interval=1m
✅ Massive API returned 2880 bars - using complete dataset for SPY
```

### Scenario 2: Alpaca Returns Sufficient Data
```
🔍 Alpaca API available for SPY, attempting data fetch...
📊 SPY Data: 2880 bars from Alpaca API | period=2d, interval=1m
(Returns immediately - no need for Massive API)
```

## 🚀 Next Steps

1. **Deploy the fix:**
   ```bash
   fly deploy
   ```

2. **Monitor logs:**
   ```bash
   fly logs --app mike-agent-project | grep -E "Alpaca|Massive|Data:|bars|attempting"
   ```

3. **Expected Output:**
   - Should see Massive API being attempted when Alpaca returns < 2,000 bars
   - Should see ~2,880 bars from Massive API (with your 1-minute granular package)
   - Should see success message: "✅ Massive API returned X bars - using complete dataset"

## ✅ Benefits

1. **Automatic Optimization:** System automatically uses the best available data source
2. **Maximum Value:** Uses your paid Massive API subscription when needed
3. **Better Data:** Gets full 2 days of 1-minute data instead of just market hours
4. **Transparent:** Clear logging shows which API is being used and why

## 📝 Notes

- **Massive API Key is already set** in Fly.io secrets ✅
- **1-minute granular package** is active ✅
- **System will now automatically use Massive API** when Alpaca returns insufficient data ✅
- **Both APIs are paid subscriptions** - you're getting maximum value! ✅





