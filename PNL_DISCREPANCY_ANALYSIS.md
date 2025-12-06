# 📊 P&L Discrepancy Analysis - GUI vs Report

## The Issue

**GUI Shows:**
- Trades Today: **6**
- Today's P&L: **+$762.30**

**Report Shows:**
- Total Trades: **45**
- Total Realized P&L: **+$574.00**

## Root Cause Analysis

### 1. Trade Count Discrepancy ✅ **BOTH ARE CORRECT - JUST DIFFERENT DEFINITIONS**

**GUI Count (6 trades):**
- Counts **BUY orders only** (entry orders)
- Today there were **6 BUY orders** placed:
  1. 37 contracts @ $0.31
  2. 35 contracts @ $0.50
  3. 35 contracts @ $0.50
  4. 37 contracts @ $0.38
  5. 36 contracts @ $0.50
  6. 37 contracts @ $0.45

**Report Count (45 trades):**
- Counts **completed round-trip trades** (buy + sell pairs)
- Uses FIFO matching to pair each BUY with corresponding SELL orders
- Each partial exit creates a separate completed trade

**Why the difference:**
- 6 entry orders were partially exited multiple times
- Each partial exit (e.g., selling 5 contracts) creates a separate completed trade
- Example: 37 contracts bought → sold in 7 partial exits = 7 completed trades

---

### 2. P&L Discrepancy ⚠️ **DIFFERENT CALCULATION METHODS**

**GUI P&L (+$762.30):**
- Uses `daily_pnl_dollar = current_equity - last_equity`
- Includes:
  - ✅ Realized P&L from closed trades
  - ✅ Unrealized P&L from open positions
  - ✅ Fees (commissions)
  - ✅ Other account changes (dividends, interest, etc.)
- **This is account-level change, not just trade P&L**

**Report P&L (+$574.00):**
- Uses **realized P&L only** from matched buy/sell pairs
- Calculates: `(sell_price - buy_price) * quantity * 100` for each completed trade
- Does NOT include:
  - ❌ Unrealized P&L
  - ❌ Fees
  - ❌ Other account changes

---

## Expected Calculations

### Account Equity Method (GUI):
```
Daily P&L = Current Equity - Last Equity
          = Account balance change from midnight to now
          = Realized P&L + Unrealized P&L + Fees + Other
```

### Trade Matching Method (Report):
```
Realized P&L = Sum of all (Sell Price - Buy Price) * Quantity * 100
              = Only completed round-trip trades
              = Does not include open positions
```

---

## The Math Check

Based on the report:
- **Realized P&L:** +$574.00
- **Open Position Unrealized P&L:** -$60.00 (2 contracts down -96.77%)
- **Net Trade P&L:** +$514.00

**GUI shows:** +$762.30

**Difference:** +$762.30 - $514.00 = **+$248.30**

This difference could be:
1. **Fees deducted from trades** (should reduce P&L, not increase)
2. **Last equity is from earlier** (not midnight, so includes previous day's gains)
3. **Other account credits** (interest, dividends, etc.)
4. **Calculation timing** (GUI and report run at different times)

---

## The Problem: `last_equity` May Be Incorrect

The GUI uses:
```python
daily_pnl_dollar = portfolio_value - last_equity
```

Where `last_equity` comes from `account.last_equity`, which is:
- Alpaca's internal tracking of equity at some point (not necessarily midnight)
- May include previous trading activity
- May not reset at market open

---

## Recommendations

### 1. Fix Trade Count Labeling
The GUI should clarify what it's counting:

**Option A:** Change label to be more specific
```
"Trade Entries Today: 6"
"Completed Trades: 45"
```

**Option B:** Count completed trades (match report)
- Count matched buy/sell pairs instead of just BUY orders

### 2. Fix P&L Calculation
The GUI should use the same method as the report for consistency:

**Option A:** Show both metrics
```
Today's Realized P&L: $+574.00
Today's Total P&L (with unrealized): $+514.00
Account Change: $+762.30
```

**Option B:** Calculate realized P&L the same way as report
- Match buy/sell pairs using FIFO
- Sum only completed trades
- Add unrealized P&L separately

### 3. Fix `last_equity` Reference Point
Store equity at market open (9:30 AM ET) and use that as baseline:
```python
# Store at market open
opening_equity = current_equity  # At 9:30 AM

# Calculate during day
daily_pnl = current_equity - opening_equity
```

---

## Immediate Fix: Make GUI Match Report

To make the GUI show the same numbers as the report:

1. **Trade Count:** Count completed trades (matched pairs) instead of BUY orders
2. **P&L:** Calculate realized P&L the same way (FIFO matching of buy/sell pairs)

This would show:
- Trades Today: **45** (completed trades)
- Today's P&L: **+$574.00** (realized only) or **+$514.00** (with unrealized)

---

## Conclusion

Both are technically correct, but they're measuring different things:

- **GUI:** Account-level daily change (6 entries, +$762.30 account change)
- **Report:** Trade-level analysis (45 completed trades, +$574.00 realized)

**For consistency, we should:**
1. Make GUI count completed trades (45) instead of entries (6)
2. Make GUI calculate P&L the same way as report (+$574.00 realized)
3. Show unrealized P&L separately (-$60.00)

This will make both show the same numbers and be easier to understand.

