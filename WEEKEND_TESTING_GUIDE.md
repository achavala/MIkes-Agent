# 🧪 Weekend Testing Guide

## Overview

Test gap detection and trading logic during weekends when the market is closed. This allows you to validate everything before Monday's live trading.

## Quick Start

### Test Single Day
```bash
python weekend_backtest.py --symbol SPY --date 2025-12-05
```

### Test Date Range
```bash
python weekend_backtest.py --symbol SPY --start 2025-12-01 --end 2025-12-05
```

### Test Last 5 Trading Days (Default)
```bash
python weekend_backtest.py --symbol SPY
```

### Quick Gap Detection Test
```bash
python test_gap_detection.py 2025-12-05 SPY
```

## What Gets Tested

### 1. Gap Detection
- ✅ Detects overnight gaps
- ✅ Calculates gap percentage and points
- ✅ Determines fade vs follow bias
- ✅ Checks early strength

### 2. Gap-Based Actions
- ✅ Gap up + weak → BUY PUT
- ✅ Gap up + strong → BUY CALL
- ✅ Gap down + weak → BUY PUT
- ✅ Gap down + bounce → BUY CALL

### 3. Trading Logic
- ✅ Executes gap-based trades
- ✅ Uses correct strike selection
- ✅ Applies position sizing
- ✅ Respects time windows (9:30-10:35 AM)

## Test Scenarios

### Scenario 1: Test Recent Days
```bash
# Test last week
python weekend_backtest.py --symbol SPY --start 2025-11-25 --end 2025-12-01
```

### Scenario 2: Test Specific High-Volatility Days
```bash
# Test specific dates you know had gaps
python weekend_backtest.py --symbol SPY --date 2025-12-05
python weekend_backtest.py --symbol SPY --date 2025-12-04
```

### Scenario 3: Test Multiple Symbols
```bash
python weekend_backtest.py --symbol SPY --start 2025-12-01 --end 2025-12-05
python weekend_backtest.py --symbol QQQ --start 2025-12-01 --end 2025-12-05
python weekend_backtest.py --symbol SPX --start 2025-12-01 --end 2025-12-05
```

### Scenario 4: Quick Gap Check
```bash
# Check if gaps were detected on specific dates
python test_gap_detection.py 2025-12-05 SPY
python test_gap_detection.py 2025-12-04 SPY
python test_gap_detection.py 2025-12-03 SPY
```

## Expected Output

### Gap Detected:
```
📊 GAP DETECTED: UP $5.71 (+0.83%) | Prev Close: $686.50 | Open: $692.21 | Bias: FADE | Strength: WEAK
🎯 GAP-BASED ACTION: 2 (BUY PUT) | Overriding RL signal for first 60 min
[09:35] 🎯 GAP TRADE: BUY 2x 692 PUT @ $11.10
```

### No Gap:
```
ℹ️  No significant gap detected (< 0.35% and < $2.50)
```

## Validation Checklist

Before Monday, verify:

- [ ] Gap detection works on historical data
- [ ] Gap actions are correct (fade vs follow)
- [ ] Trades execute during first 60 minutes
- [ ] No trades after 10:35 AM (gap logic disabled)
- [ ] Multiple symbols work (SPY, QQQ, SPX)
- [ ] Different gap sizes trigger correctly
- [ ] Early strength detection works

## Testing Plan

### Saturday Testing:
1. Test last 10 trading days
2. Verify gap detection accuracy
3. Check action logic (fade vs follow)
4. Validate time windows

### Sunday Testing:
1. Test specific high-gap days
2. Test edge cases (small gaps, large gaps)
3. Test multiple symbols
4. Final validation before Monday

## Troubleshooting

### No Data Available
- Check if date is a trading day (not weekend/holiday)
- Verify symbol is correct (SPY, QQQ, SPX)
- Try a different date range

### Gap Not Detected
- Check if gap meets threshold (≥0.35% OR ≥$2.50)
- Verify historical data includes previous day's close
- Check time window (9:30-10:35 AM)

### Import Errors
- Ensure `gap_detection.py` is in same directory
- Check Python environment is activated
- Verify all dependencies installed

## Next Steps

1. **Saturday**: Run comprehensive tests on last 2 weeks
2. **Sunday**: Test edge cases and validate everything
3. **Monday**: Ready for live trading!

## Files

- `weekend_backtest.py` - Full backtesting environment
- `test_gap_detection.py` - Quick gap detection test
- `gap_detection.py` - Gap detection module (used by both)

Happy testing! 🧪

