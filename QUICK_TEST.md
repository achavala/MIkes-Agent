# 🧪 Quick Testing Commands

## Weekend Testing - Ready to Use!

### Test Single Day
```bash
python weekend_backtest.py --symbol SPY --date 2025-12-05
```

### Test Date Range
```bash
python weekend_backtest.py --symbol SPY --start 2025-12-01 --end 2025-12-05
```

### Quick Gap Check
```bash
python test_gap_detection.py 2025-12-05 SPY
```

### Run Full Test Suite
```bash
./run_weekend_tests.sh
```

## What You'll See

### When Gap is Detected:
```
📊 GAP DETECTED: UP $5.71 (+0.83%) | Prev Close: $686.50 | Open: $692.21 | Bias: FADE | Strength: WEAK
🎯 GAP-BASED ACTION: 2 (BUY PUT) | Overriding RL signal for first 60 min
[09:35] 🎯 GAP TRADE: BUY 2x 692 PUT @ $11.10
```

### When No Gap:
```
ℹ️  No significant gap detected (< 0.35% and < $2.50)
```

## Test Plan

### Saturday:
- Test last 10 trading days
- Verify gap detection works
- Check action logic

### Sunday:
- Test specific dates
- Final validation
- Ready for Monday!

