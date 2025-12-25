# 🎯 Setup Analytics Dashboard

Real-time analytics tool to monitor setup validation, picking, and rejection in the Mike Agent.

## Features

- ✅ **Current Status**: Shows what's currently being validated, picked, rejected, or executed
- 📊 **Statistics**: Total counts, average confidence, rejection reasons
- 📈 **Symbol Activity**: Track activity per symbol (SPY, QQQ, IWM, SPX)
- 🕐 **Recent Activity**: Last 10 events in each category
- 🔄 **Watch Mode**: Auto-refresh every 5 seconds

## Usage

### Basic Usage
```bash
python3 setup_analytics.py
```

### Watch Mode (Auto-refresh)
```bash
python3 setup_analytics.py --watch
# or
python3 setup_analytics.py -w
```

### Export to JSON
```bash
python3 setup_analytics.py --export
# or
python3 setup_analytics.py -e
```

### Use Specific Log File
```bash
python3 setup_analytics.py --file logs/mike_agent_safe_20251219.log
# or
python3 setup_analytics.py -f agent_output.log
```

## Output Example

```
================================================================================
🎯 SETUP ANALYTICS DASHBOARD
================================================================================
📅 Last Updated: 2025-12-19 13:07:36
📊 Log File: agent_output.log

🔍 CURRENT STATUS
--------------------------------------------------------------------------------
  ⏳ Validating: SPY | Action=1 | Strength=0.580 | Time: 14:15:35
  ✅ Picked: SPY | Strength=0.630 | Time: 14:15:36
  ❌ Rejected: None
  🚀 Executed: SPY251210C00688000 | Qty: 33x | Time: 14:15:38

📈 STATISTICS
--------------------------------------------------------------------------------
  Total Validated: 127
  Total Picked: 45
  Total Rejected: 82
  Total Executed: 18
  Avg Confidence: 0.542

❌ REJECTION REASONS
--------------------------------------------------------------------------------
  Low Confidence: 45
  Stale Data: 20
  No Eligible Symbols: 17

📊 SYMBOL ACTIVITY
--------------------------------------------------------------------------------
  SPY: 89 events
  QQQ: 23 events
  IWM: 8 events
  SPX: 7 events

🕐 RECENT ACTIVITY (Last 10)
--------------------------------------------------------------------------------

  Validating:
    [14:15:35] SPY | Action=1 | Strength=0.580
    [14:20:23] QQQ | Action=0 | Strength=0.501
    [14:25:12] SPY | Action=2 | Strength=0.520

  Picked:
    [14:15:36] SPY | Strength=0.630
    [14:30:45] QQQ | Strength=0.580

  Rejected:
    [14:20:25] QQQ | Reason: Low Confidence
    [14:25:15] SPY | Reason: Stale Data

  Executed:
    [14:15:38] SPY251210C00688000 | Qty: 33x
    [14:30:50] QQQ251210C00630000 | Qty: 25x
```

## What It Tracks

### Validating
- RL inference events per symbol
- Action (0=HOLD, 1=BUY CALL, 2=BUY PUT)
- Confidence strength (0.0 - 1.0)
- TA pattern detection

### Picked
- Symbols selected for trading
- Confidence strength
- Selection time

### Rejected
- Rejection reasons:
  - Low Confidence (< 0.52 threshold)
  - Stale Data (not from today or > 5 min old)
  - No Eligible Symbols (cooldown, max positions, etc.)
  - Price Mismatch (cross-validation failed)

### Executed
- Successfully executed trades
- Symbol and quantity
- Execution time

## Integration

The analytics tool automatically:
- Finds the most recent log file
- Parses log entries in real-time
- Tracks setup validation flow
- Calculates statistics

## Notes

- Logs are read from `logs/mike_agent_safe_YYYYMMDD.log` or `agent_output.log`
- Only analyzes last 500 lines by default (configurable)
- Keeps last 50 events in each category
- Watch mode refreshes every 5 seconds


