# 📊 RECENT TRADES SUMMARY

## 🏦 Trading Platform

**Platform:** **Alpaca Markets**
- **Mode:** Paper Trading (default)
- **Paper Trading URL:** `https://paper-api.alpaca.markets`
- **Live Trading URL:** `https://api.alpaca.markets`
- **Current Setting:** Paper Trading

**Note:** The agent is currently configured for **paper trading** (simulated trading with fake money). To switch to live trading, set the environment variable `ALPACA_PAPER=false`.

---

## 📈 Recent Trades

Based on the trade database, here are the most recent trades:

### Trade Activity (Last 20 Trades - December 12, 2025)

**Most Recent Trade:**
- **Time:** 2025-12-12 19:08:52 UTC
- **Symbol:** QQQ251212C00615000 (QQQ Call, Strike $615, Exp Dec 12)
- **Action:** SELL
- **Quantity:** 40 contracts
- **Fill Price:** $0.85

**Previous Trade:**
- **Time:** 2025-12-12 19:08:41 UTC
- **Symbol:** QQQ251212C00615000
- **Action:** BUY
- **Quantity:** 40 contracts
- **Fill Price:** $0.91

### Trading Pattern Observed:

1. **Symbols Traded:**
   - SPY (SPDR S&P 500 ETF) - Multiple strikes ($682, $683)
   - QQQ (Invesco QQQ Trust) - Multiple strikes ($615, $616)

2. **Option Types:**
   - Mostly **CALL** options
   - Some **PUT** options (e.g., SPY251212P00682000)

3. **Strikes:**
   - SPY: $682, $683
   - QQQ: $615, $616

4. **Quantities:**
   - Range: 34-42 contracts per trade

5. **Trading Activity:**
   - Multiple round-trip trades (BUY then SELL)
   - All trades on December 12, 2025 (0DTE - same day expiration)

### Important Notes:

⚠️ **P&L Data:** The database shows `pnl = 0.0` and `entry_premium = 0.0` for these trades. This suggests:
- These trades were synced from Alpaca (`source: alpaca_sync`)
- Premium data may not have been captured at the time of trade
- P&L calculation may need to be recalculated from fill prices

### To View All Recent Trades:

Run the query script:
```bash
python3 query_recent_trades.py
```

This will show the last 50 trades with full details.

---

## 🔍 Trade Database

**Location:** `trades_database.db` (SQLite)

**Schema:**
- All trades are permanently stored
- Includes: timestamp, symbol, action, quantity, strike, premiums, P&L, etc.
- Trades are synced from Alpaca API

---

## 📊 Summary Statistics

From the last 20 trades:
- **Total Trades:** 20 (10 BUY, 10 SELL)
- **Symbols:** SPY, QQQ
- **Option Types:** Mostly CALL, some PUT
- **Expiration:** All 0DTE (same day expiration)
- **Date:** December 12, 2025

---

**Last Updated:** Based on database query on December 20, 2025


