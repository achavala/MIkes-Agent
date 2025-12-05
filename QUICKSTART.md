# Quick Start Guide

## Setup

1. **Activate virtual environment:**
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Verify installation:**
```bash
python test_imports.py
```

## Running Backtests

Test the strategy on historical data:

```bash
# Basic backtest (default dates: Nov 3 - Dec 1, 2025)
python main.py --mode backtest --symbols SPY,QQQ

# Custom date range
python main.py --mode backtest --symbols SPY --start_date 2025-11-01 --end_date 2025-11-30

# Custom capital
python main.py --mode backtest --symbols SPY,QQQ --capital 5000
```

## Paper Trading

Run simulated trading (no real money):

```bash
# Paper trade SPY and QQQ with $10,000
python main.py --mode paper --symbols SPY,QQQ --capital 10000
```

Press `Ctrl+C` to stop paper trading.

## Example Output

### Backtest Output:
```
============================================================
Backtesting MikeAgent
Period: 2025-11-03 to 2025-12-01
Symbols: SPY, QQQ
Initial Capital: $10,000.00
============================================================

Backtesting SPY...
  [2025-11-03] ENTRY: PUT strike=450.00 size=2 premium=$2.50
  [2025-11-03] EXIT: trim_30 PnL=$125.00 (+25.0%) Capital=$10,125.00

============================================================
BACKTEST RESULTS
============================================================
Total Trades: 15
Winning Trades: 12
Losing Trades: 3
Win Rate: 80.0%
Average Win: $210.50
Average Loss: $-15.20
Total PnL: $2,450.00
Final Capital: $12,450.00
Total Return: 24.50%
============================================================
```

## Integration with FutBot-Pro

To integrate MikeAgent into FutBot-Pro:

1. **Copy the agent:**
```bash
cp core/agents/mike_agent.py /path/to/FutBot-Pro/core/agents/
```

2. **Update FutBot-Pro's strategy_manager.py:**
```python
from core.agents.mike_agent import MikeAgent

# In StrategyManager.__init__:
self.agents.append(
    MikeAgent(
        self.options_feed,
        self.microstructure,
        self.regime_engine,
        self.risk_manager
    )
)
```

## Troubleshooting

### Import Errors
If you get import errors, make sure:
- Virtual environment is activated
- All dependencies are installed: `pip install -r requirements.txt`

### No Trades Generated
The agent only trades when:
- Market is in neutral regime
- Significant gap detected (>0.5%)
- Options chain data is available

### Data Issues
If backtest fails with data errors:
- Check internet connection (needs Yahoo Finance API)
- Verify date range has trading days
- Try a different symbol (SPY, QQQ work best)

## Next Steps

1. **Run backtests** on different date ranges to validate strategy
2. **Paper trade** for at least 1 week before going live
3. **Monitor performance** and adjust parameters if needed
4. **Go live** only after thorough testing (start with small capital)

## Support

For issues or questions:
- Check README.md for detailed documentation
- Review code comments in `core/agents/mike_agent.py`
- Test with `test_imports.py` to verify setup

