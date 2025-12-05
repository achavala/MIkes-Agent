# Mike Agent v3 - Backtest Guide

## 📊 Available Data Ranges

Based on Yahoo Finance historical data:

- **SPY**: 1993-01-29 to 2025-12-04 (8,270 trading days)
- **VIX**: 1990-01-02 to 2025-12-04 (9,049 trading days)
- **QQQ**: 1999-03-10 to 2025-12-04 (6,728 trading days)
- **SPX**: Available via SPY proxy

## 🚀 Quick Start

### Basic Backtest (Last 30 Days)
```bash
python backtest_mike_agent_v3.py --start 2024-11-01 --end 2024-12-04
```

### Custom Date Range
```bash
python backtest_mike_agent_v3.py --start 2024-01-01 --end 2024-12-31 --symbol SPY
```

### Different Intervals
```bash
# 1-minute bars (most detailed, slower)
python backtest_mike_agent_v3.py --start 2024-11-01 --end 2024-12-04 --interval 1m

# 5-minute bars (recommended)
python backtest_mike_agent_v3.py --start 2024-11-01 --end 2024-12-04 --interval 5m

# 15-minute bars (faster)
python backtest_mike_agent_v3.py --start 2024-11-01 --end 2024-12-04 --interval 15m

# Daily bars (fastest, less detail)
python backtest_mike_agent_v3.py --start 2024-01-01 --end 2024-12-31 --interval 1d
```

### Export Results
```bash
python backtest_mike_agent_v3.py --start 2024-11-01 --end 2024-12-04 --export equity_curve.csv
```

## 📋 Command Line Options

```
--start DATE          Start date (YYYY-MM-DD) [default: 2024-11-01]
--end DATE            End date (YYYY-MM-DD) [default: 2024-12-04]
--symbol SYMBOL        Symbol to backtest (SPY, QQQ, SPX) [default: SPY]
--interval INTERVAL    Data interval (1m, 5m, 15m, 1h, 1d) [default: 5m]
--capital AMOUNT       Initial capital [default: 10000.0]
--export FILE          Export equity curve to CSV
```

## ⚠️ Important: Yahoo Finance Data Limitations

**Intraday Data (1m, 5m, 15m, etc.)**: Only available for the **last 60 days**
- If your start date is more than 60 days ago, the script will automatically switch to daily (1d) data
- For historical backtests beyond 60 days, always use `--interval 1d`

**Daily Data (1d)**: Available for full historical range (1993+ for SPY)
- Use `--interval 1d` for any date range
- Recommended for backtests longer than 60 days

## 🎯 Recommended Backtest Periods

### Quick Tests (Fast)
- **Last 7 days**: `--start 2024-11-27 --end 2024-12-04`
- **Last 30 days**: `--start 2024-11-01 --end 2024-12-04`
- **Last 90 days**: `--start 2024-09-01 --end 2024-12-04`

### Comprehensive Tests (Slower)
- **Last 6 months**: `--start 2024-06-01 --end 2024-12-04 --interval 1d`
- **Last 1 year**: `--start 2024-01-01 --end 2024-12-31 --interval 1d`
- **Last 2 years**: `--start 2023-01-01 --end 2024-12-31 --interval 1d`

**Note**: For periods > 60 days, use `--interval 1d` (daily data)

### Stress Tests
- **2020 COVID Crash**: `--start 2020-02-01 --end 2020-04-30`
- **2022 Bear Market**: `--start 2022-01-01 --end 2022-12-31`
- **2023 Recovery**: `--start 2023-01-01 --end 2023-12-31`

## 📈 What the Backtest Includes

✅ **Full Agent Logic**:
- RL model predictions (if model available)
- 5-tier take-profit system (TP1-TP5)
- Fixed -15% stop-loss
- Trailing stop after TP4
- Volatility regime engine (4 regimes)
- Position sizing based on risk %

✅ **Risk Management**:
- Max 10 concurrent positions
- Daily loss limit (-15%)
- Max position size (25% of equity)
- VIX kill switch (>28)
- Time-of-day filters

✅ **Realistic Simulation**:
- Black-Scholes option pricing
- Transaction fees (0.1% per contract)
- Slippage modeling
- Market hours enforcement

## 📊 Output Metrics

The backtest reports:

- **Overall Performance**:
  - Total Return (%)
  - Max Drawdown (%)
  - Sharpe Ratio
  - Win Rate (%)
  - Profit Factor

- **Regime Performance**:
  - Performance by volatility regime (Calm, Normal, Storm, Crash)
  - Trades, win rate, and P&L per regime

- **Exit Reasons**:
  - Breakdown of why positions were closed (TP1, TP2, TP3, TP4, TP5, Stop-Loss, Trailing Stop)

## 🔧 Requirements

Make sure you have:
- `mike_agent_live_safe.py` (for importing agent logic)
- `mike_rl_agent.zip` (optional, for RL model predictions)
- Required packages: `yfinance`, `pandas`, `numpy`, `scipy`, `stable_baselines3`

## 💡 Tips

1. **Start Small**: Test with 7-30 days first to verify everything works
2. **Use 5m Interval**: Good balance between detail and speed
3. **Check Data Availability**: Some symbols may have limited historical data
4. **RL Model**: If `mike_rl_agent.zip` is missing, the backtest will use random actions
5. **Export Results**: Use `--export` to save equity curve for analysis

## 🐛 Troubleshooting

### "No data available"
- Check date range is valid
- Ensure symbol is correct (SPY, QQQ, SPX)
- Try a different date range

### "RL model not found"
- This is OK - backtest will use random actions
- To use RL model: ensure `mike_rl_agent.zip` exists in project directory

### "Import errors"
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check that `mike_agent_live_safe.py` is in the same directory

## 📝 Example Output

```
================================================================================
MIKE AGENT v3 - COMPREHENSIVE BACKTEST
================================================================================
Symbol: SPY
Period: 2024-11-01 to 2024-12-04
Initial Capital: $10,000.00
Interval: 5m
================================================================================
Downloading data...
✓ Downloaded 1,234 bars
  Date range: 2024-11-01 09:30:00 to 2024-12-04 16:00:00
================================================================================
Running backtest...
  Progress: 10.0% | Capital: $10,150.00 | Positions: 2
  Progress: 20.0% | Capital: $10,320.00 | Positions: 3
  ...

================================================================================
BACKTEST RESULTS
================================================================================

OVERALL PERFORMANCE:
  Initial Capital: $10,000.00
  Final Capital: $12,450.00
  Total Return: +24.50%
  Max Drawdown: -8.20%
  Sharpe Ratio: 1.85
  Win Rate: 72.5% (58/80)
  Profit Factor: 2.15

REGIME PERFORMANCE:
Regime      Trades   Win Rate   Total P&L
--------------------------------------------------
CALM        25       80.0%      $1,250.00
NORMAL      40       70.0%      $1,100.00
STORM       12       66.7%      $100.00
CRASH       3        33.3%      -$50.00

EXIT REASONS:
  TP1: 35
  TP2: 15
  STOP_LOSS: 20
  TP3: 8
  TRAILING_STOP: 2
================================================================================
```

## 🎓 Next Steps

1. Run a quick 7-day test to verify setup
2. Run a 30-day backtest to see performance
3. Export equity curve and analyze in Excel/Python
4. Compare different date ranges and market conditions
5. Adjust parameters in `mike_agent_live_safe.py` and re-run

---

**Happy Backtesting! 📈**

