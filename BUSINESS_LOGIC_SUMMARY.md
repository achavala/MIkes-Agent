# Mike Agent v3 — Business Logic Summary

## 📊 Core Trading Strategy

### Trading Symbols
- **SPY** (S&P 500 ETF)
- **QQQ** (Nasdaq 100 ETF)
- **SPX** (S&P 500 Index)

### Position Limits
- **Max Concurrent Positions**: 10 positions at any time
- **Max Position Size**: 25% of equity per position (volatility-adjusted)
- **Max Notional per Order**: $50,000

---

## 🎯 Entry Logic

### Signal Generation
- **RL Model**: Uses PPO (Proximal Policy Optimization) trained on historical data
- **Action Space**: 
  - 0 = HOLD/FLAT
  - 1 = BUY CALL
  - 2 = BUY PUT
  - 3 = TRIM (partial exit)
  - 4 = EXIT (full close)
  - 5 = REJECT (no entry)

### Entry Conditions
- Market must be open (9:30 AM - 4:00 PM EST)
- No trades after 2:30 PM EST
- VIX < 28 (kill switch)
- IV Rank ≥ 30%
- Max 10 positions already open
- Duplicate order protection (5-minute window)

### Position Sizing
- **Base Risk**: 7% per trade (volatility-adjusted)
- **IV-Adjusted Risk**:
  - Low IV (<20%): 10% risk (cheaper premiums, higher conviction)
  - Normal IV (20-50%): 7% risk (standard)
  - High IV (>50%): 4% risk (expensive, volatile)
- **Calculation**: Based on premium cost, not notional value
- **Strike Selection**: At-the-money (ATM) for 0DTE options

---

## 🛡️ Risk Management Safeguards (13 Layers)

1. **Daily Loss Limit**: -15% daily loss → Full shutdown
2. **Max Position Size**: 25% of equity per position
3. **Max Concurrent Positions**: 10 positions maximum
4. **VIX Kill Switch**: No trades if VIX > 28
5. **IV Rank Minimum**: 30% minimum IV Rank
6. **Time-of-Day Filter**: No new entries after 2:30 PM EST
7. **Max Drawdown**: -30% from peak → Full shutdown
8. **Max Notional**: $50,000 per order limit
9. **Duplicate Order Protection**: 5-minute cooldown between orders
10. **Manual Kill Switch**: Ctrl+C to flatten all positions
11. **Fixed Stop-Loss**: -15% always (regardless of volatility)
12. **5-Tier Take-Profit System**: Partial exits at multiple levels
13. **Trailing Stop**: Activates after TP4, locks in +100% minimum

---

## 📈 Volatility Regime Engine

The agent adapts risk parameters based on VIX levels:

### CALM (VIX < 18)
- **Risk**: 10% per trade
- **Max Position Size**: 30% of equity
- **Stop-Loss**: -15% (fixed, overrides regime)
- **Take-Profits**: Same 5-tier system

### NORMAL (VIX 18-25)
- **Risk**: 7% per trade
- **Max Position Size**: 25% of equity
- **Stop-Loss**: -15% (fixed, overrides regime)
- **Take-Profits**: Same 5-tier system

### STORM (VIX 25-35)
- **Risk**: 5% per trade
- **Max Position Size**: 20% of equity
- **Stop-Loss**: -15% (fixed, overrides regime)
- **Take-Profits**: Same 5-tier system

### CRASH (VIX > 35)
- **Risk**: 3% per trade
- **Max Position Size**: 15% of equity
- **Stop-Loss**: -15% (fixed, overrides regime)
- **Take-Profits**: Same 5-tier system

**Note**: Stop-loss is ALWAYS -15% regardless of regime (fixed rule).

---

## 💰 5-Tier Take-Profit System

### TP1: +40% → Sell 50% of Position
- Triggers at +40% profit
- Sells 50% of current position
- Remaining 50% continues for higher targets

### TP2: +60% → Sell 20% of Remaining
- Triggers at +60% profit
- Sells 20% of remaining position
- Remaining 80% continues

### TP3: +100% → Sell 10% of Remaining
- Triggers at +100% profit
- Sells 10% of remaining position
- Remaining 90% continues

### TP4: +150% → Sell 10% of Remaining
- Triggers at +150% profit
- Sells 10% of remaining position
- **Activates Trailing Stop** after this level
- Remaining 90% continues

### TP5: +200% → Full Exit
- Triggers at +200% profit
- Closes entire remaining position
- Maximum profit target

### Trailing Stop (After TP4)
- **Activates**: After TP4 is hit
- **Lock Level**: +100% minimum profit
- **Behavior**: Trails price up, locks in +100% minimum
- **Exit**: If price drops below trailing stop level

---

## 🚨 Stop-Loss Logic

### Fixed Stop-Loss: -15% ALWAYS
- **Priority**: Checked FIRST (before any take-profits)
- **Override**: Overrides all volatility-adjusted stops
- **Action**: Immediate full exit
- **Applies to**: All positions, all volatility regimes

### Calculation
- **Entry Premium**: From Alpaca's `avg_entry_price` (actual fill price)
- **Current Premium**: `market_value / (qty * 100)`
- **P&L**: `(current_premium - entry_premium) / entry_premium`
- **Trigger**: If P&L ≤ -15%, close position immediately

---

## 📊 Position Tracking & P&L Calculation

### Entry Premium
- **Source**: Alpaca's `avg_entry_price` (actual fill price)
- **Fallback**: `cost_basis / (qty * 100)` if `avg_entry_price` unavailable
- **Update**: Always synced from Alpaca for accuracy

### Current Premium
- **Calculation**: `market_value / (qty * 100)`
- **Source**: Direct from Alpaca position data
- **Fallback**: Estimate if market_value unavailable

### P&L Calculation
- **Formula**: `(current_premium - entry_premium) / entry_premium`
- **Updates**: Every cycle (every ~55 seconds)
- **Accuracy**: Uses actual Alpaca fill prices, not estimates

---

## 🔄 Position Management

### Position Syncing
- **On Startup**: Syncs all existing Alpaca positions
- **During Runtime**: Continuously syncs with Alpaca
- **Entry Premium**: Always uses Alpaca's `avg_entry_price`
- **Quantity Tracking**: Updates `qty_remaining` after partial sells

### Partial Sells
- **TP1**: Sells 50% using `submit_order()` (not `close_position()`)
- **TP2**: Sells 20% of remaining
- **TP3**: Sells 10% of remaining
- **TP4**: Sells 10% of remaining
- **TP5**: Full exit using `close_position()`

### Position Cleanup
- Removes positions from tracking when closed
- Handles positions closed externally (Alpaca UI, etc.)

---

## 💵 Capital Management

### $10K Capital Mode
- **Forced Capital**: $10,000 (ignores Alpaca balance)
- **P&L Tracking**: Tracks P&L separately from Alpaca
- **Equity Calculation**: `FORCE_CAPITAL + tracked_pnl`
- **Use Case**: Paper trading with fixed capital for testing

### Position Sizing Formula
```
risk_amount = equity * risk_pct
estimated_premium = Black-Scholes estimate
contracts = risk_amount / (estimated_premium * 100)
```

### Order Safety Checks
- **Notional Check**: `premium * 100 * qty` (not `strike * 100`)
- **Position Size Check**: Total exposure ≤ max position size
- **Concurrent Check**: Total positions ≤ 10
- **Duplicate Check**: No same-symbol orders within 5 minutes

---

## 📝 Trade Logging

### Database Storage
- **File**: `mike_agent_trades.csv`
- **Fields**: timestamp, symbol, action, qty, price, entry_premium, exit_premium, pnl, pnl_pct, capital_before, capital_after, reason, regime, vix
- **All Trades**: BUY, SELL, TP1, TP2, TP3, TP4, TP5, Stop-Loss, Trailing Stop

### Logging Triggers
- **BUY**: On entry execution
- **SELL**: On TP1, TP2, TP3, TP4, TP5, Stop-Loss, Trailing Stop
- **P&L**: Calculated and logged for all exits

---

## ⏰ Market Hours & Timing

### Trading Hours
- **Market Open**: 9:30 AM EST
- **Market Close**: 4:00 PM EST
- **No New Entries After**: 2:30 PM EST
- **Weekend Handling**: Skips weekends, waits for next trading day

### Agent Cycle
- **Check Interval**: ~55 seconds
- **Actions Per Cycle**:
  1. Check all positions for stop-loss/take-profit
  2. Check for new entry signals
  3. Execute trades if conditions met
  4. Log all activity

---

## 🎲 Entry Signal Generation (RL Model)

### Observation Space
- **Shape**: (20, 5) - 20 bars of OHLCV data
- **Features**: Open, High, Low, Close, Volume
- **Normalization**: Standardized for model input

### Action Mapping
- **Continuous → Discrete**: RL model outputs continuous value, mapped to discrete actions (0-5)
- **Deterministic**: Uses deterministic predictions for live trading

### Entry Execution
- **Strike**: At-the-money (ATM) - rounded to nearest dollar
- **Expiry**: 0DTE (same-day expiry)
- **Order Type**: Market order
- **Time in Force**: Day order

---

## 🔍 Key Business Rules Summary

1. **Max 10 positions** at any time
2. **Fixed -15% stop-loss** always (highest priority)
3. **5-tier take-profit** with partial sells
4. **Trailing stop** activates after TP4, locks +100%
5. **Volatility-adjusted** risk and position sizing
6. **VIX kill switch** at VIX > 28
7. **Time filter** - no entries after 2:30 PM EST
8. **Duplicate protection** - 5-minute cooldown
9. **Daily loss limit** - -15% daily → shutdown
10. **Max drawdown** - -30% from peak → shutdown
11. **$10K capital mode** - fixed capital for testing
12. **Multi-symbol** - SPY, QQQ, SPX rotation
13. **RL-driven** - AI model generates entry signals

---

## 📊 Example Trade Flow

1. **Entry**: RL model signals BUY CALL on SPY
   - Check: Market open? VIX < 28? < 10 positions? Before 2:30 PM?
   - Calculate: Position size based on 7% risk
   - Execute: Market buy order for ATM 0DTE call
   - Log: Entry to database with `entry_premium` from Alpaca

2. **Monitoring**: Every 55 seconds
   - Calculate: Current premium = `market_value / (qty * 100)`
   - Calculate: P&L = `(current - entry) / entry`
   - Check: Stop-loss? Take-profits? Trailing stop?

3. **TP1 Trigger** (+40%):
   - Action: Sell 50% of position
   - Update: `qty_remaining` reduced by 50%
   - Log: SELL trade to database
   - Continue: Remaining 50% for TP2/TP3/TP4/TP5

4. **TP2 Trigger** (+60%):
   - Action: Sell 20% of remaining
   - Update: `qty_remaining` reduced by 20%
   - Log: SELL trade to database
   - Continue: Remaining 80% for TP3/TP4/TP5

5. **TP4 Trigger** (+150%):
   - Action: Sell 10% of remaining
   - Activate: Trailing stop at +100% minimum
   - Log: SELL trade to database
   - Continue: Remaining 90% with trailing stop active

6. **Stop-Loss Trigger** (-15%):
   - Action: Immediate full exit
   - Priority: Highest (checked first)
   - Log: SELL trade to database
   - Remove: Position from tracking

---

## 🎯 Key Differentiators

1. **Fixed -15% Stop-Loss**: Always enforced, regardless of volatility
2. **5-Tier Partial Exits**: Scales out profits gradually
3. **Trailing Stop After TP4**: Locks in +100% minimum after big wins
4. **Volatility Regime Engine**: Adapts to market conditions
5. **RL-Driven Entries**: AI model generates signals
6. **Multi-Symbol Rotation**: SPY, QQQ, SPX
7. **Accurate P&L**: Uses Alpaca's actual fill prices
8. **13-Layer Safeguards**: Institutional-grade risk management

---

**Last Updated**: December 4, 2025
**Version**: Mike Agent v3 — Final Fixed Edition

