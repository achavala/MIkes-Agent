# 📊 TRADING PLATFORM INFORMATION

## 🏦 Trading Platform

**Platform:** **Alpaca Markets**

### Configuration:
- **Mode:** Paper Trading (default)
- **Paper Trading URL:** `https://paper-api.alpaca.markets`
- **Live Trading URL:** `https://api.alpaca.markets`
- **Current Setting:** Paper Trading (controlled by `USE_PAPER` environment variable)

### How to Check:
```python
USE_PAPER = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'
BASE_URL = PAPER_URL if USE_PAPER else LIVE_URL
```

### To Switch to Live Trading:
1. Set environment variable: `export ALPACA_PAPER=false`
2. Or modify code: `USE_PAPER = False`

---

## 📈 Trade Database

**Database:** SQLite (`trades_database.db`)

All trades are saved to a persistent SQLite database for tracking and analysis.

---

## 🔍 Recent Trades

See `query_recent_trades.py` to query recent trades from the database.


