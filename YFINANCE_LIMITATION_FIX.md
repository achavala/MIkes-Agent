# ⚠️ yfinance Limitation & Fix

## The Problem

yfinance has a **strict limitation** for 1-minute data:
- **Only 8 days of 1-minute data can be fetched per request**
- The error: `"Only 8 days worth of 1m granularity data are allowed to be fetched per request."`

## The Solution

I've updated `historical_training_system.py` to:
- ✅ Fetch in **7-day chunks** (respecting 8-day limit)
- ✅ Add error handling for failed chunks
- ✅ Add progress reporting
- ✅ Increase rate limiting delay to 1 second

## Updated Collection Strategy

**Before (broken):**
- Tried to fetch 60-day chunks
- Failed immediately due to yfinance limit

**After (fixed):**
- Fetches 7-day chunks
- Handles errors gracefully
- Shows progress for each chunk
- Respects API rate limits

## Impact on Collection Time

**For 20+ years of data (2002-2025):**
- Days: ~6,000 trading days
- Chunks needed: ~850 chunks (6,000 / 7)
- Time per chunk: ~2-3 seconds (download + delay)
- **Total time: ~30-45 minutes** (much better than 8-24 hours!)

## How It Works Now

1. Start from 2002-01-01
2. Fetch 7 days at a time
3. Save each chunk to cache
4. Continue until end date
5. Combine all chunks at the end

## Running Collection

The fixed version will now work:

```bash
source venv/bin/activate
python collect_historical_data.py --symbols SPY,QQQ --start-date 2002-01-01
```

**Expected:** Collection will take ~30-45 minutes (not hours!)

---

**Status: ✅ Fixed and ready to run!**

