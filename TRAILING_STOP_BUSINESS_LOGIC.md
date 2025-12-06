# Trailing Stop Business Logic - Detailed Specification

## Overview

After each Take-Profit tier executes, implement a **trailing stop system** that:
1. **Locks in 80% of remaining position** at TP level - 20%
2. **Runs 20% of remaining position** until EOD or -15% stop loss

This creates a **"runner" strategy** - lock most profits, let a small portion run for maximum gains.

---

## Current System (Before Update)

### Take-Profit Tiers
- **TP1 (+40%)**: Sells 50% of position
- **TP2 (+80%)**: Sells 60% of remaining
- **TP3 (+150%)**: Full exit (sells 100% of remaining)

### Example: 10 Calls Entry
1. **Entry**: 10 calls @ $2.00 premium
2. **TP1 (+40%)**: Price hits $2.80 → Sells 5 calls → **5 calls remaining**
3. **TP2 (+80%)**: Price hits $3.60 → Sells 3 calls (60% of 5) → **2 calls remaining**
4. **TP3 (+150%)**: Price hits $5.00 → Sells 2 calls → **0 calls remaining**

---

## New System (After Update)

### Trailing Stop Logic After Each TP

#### After TP1 (+40%)
**Scenario**: 10 calls entry, TP1 sells 5 calls → 5 calls remaining

**Trailing Stop Action**:
1. **Calculate TP1 - 20%**: +40% - 20% = **+20%** (price = $2.40)
2. **When price drops to +20%**: Sell 80% of remaining = **4 calls** (80% of 5)
3. **Keep 20% of remaining = 1 call** running until:
   - **EOD** (End of Day - market close at 4:00 PM EST), OR
   - **-15% stop loss** from entry premium ($2.00 → $1.70)

**Result**:
- 5 calls sold at TP1 (+40%)
- 4 calls sold at trailing stop (+20%)
- 1 call runs until EOD or -15% stop

---

#### After TP2 (+80%)
**Scenario**: After TP1, we have 5 calls remaining. TP2 sells 3 calls → 2 calls remaining

**Trailing Stop Action**:
1. **Calculate TP2 - 20%**: +80% - 20% = **+60%** (price = $3.20)
2. **When price drops to +60%**: Sell 80% of remaining = **1.6 calls → 1 call** (80% of 2, rounded)
3. **Keep 20% of remaining = 0.4 calls → 1 call** running until:
   - **EOD**, OR
   - **-15% stop loss** from entry premium ($2.00 → $1.70)

**Result**:
- 3 calls sold at TP2 (+80%)
- 1 call sold at trailing stop (+60%)
- 1 call runs until EOD or -15% stop

---

#### After TP3 (+150%)
**Scenario**: After TP2, we have 2 calls remaining. TP3 is full exit.

**Option A**: TP3 remains full exit (sells all remaining)
**Option B**: Apply trailing stop logic to TP3 as well

**If Option B (Trailing Stop on TP3)**:
1. **Calculate TP3 - 20%**: +150% - 20% = **+130%** (price = $4.60)
2. **When price drops to +130%**: Sell 80% of remaining = **1.6 calls → 1 call** (80% of 2)
3. **Keep 20% of remaining = 0.4 calls → 1 call** running until:
   - **EOD**, OR
   - **-15% stop loss** from entry premium ($2.00 → $1.70)

**Result**:
- 1 call sold at trailing stop (+130%)
- 1 call runs until EOD or -15% stop

---

## Complete Example Flow

### Entry: 10 Calls @ $2.00 Premium

#### Step 1: TP1 Hits (+40% = $2.80)
- **Action**: Sell 5 calls (50%)
- **Remaining**: 5 calls
- **Trailing Stop Activated**: Monitor for +20% ($2.40)

#### Step 2: Price Drops to +20% ($2.40)
- **Action**: Sell 4 calls (80% of remaining 5)
- **Remaining**: 1 call
- **Runner Activated**: 1 call runs until EOD or -15% stop

#### Step 3: Runner Management
- **If price continues up**: 1 call can hit TP2 (+80% = $3.60) or TP3 (+150% = $5.00)
- **If price drops to -15%** ($1.70): Exit 1 call
- **If EOD (4:00 PM)**: Exit 1 call at market close

---

## Business Logic Rules

### Rule 1: Trailing Stop Calculation
```
Trailing Stop Price = TP Level - 20%
Example: TP1 at +40% → Trailing Stop at +20%
```

### Rule 2: Position Sizing After TP
```
After TP executes:
  - Remaining Position = Original Position - TP Sell Amount
  - Trailing Stop Sell = 80% of Remaining Position
  - Runner Position = 20% of Remaining Position
```

### Rule 3: Runner Exit Conditions
```
Runner exits when ANY of these conditions are met:
  1. Price drops to -15% from entry premium
  2. Market closes (4:00 PM EST)
  3. (Optional) TP2 or TP3 hits (if runner continues)
```

### Rule 4: Trailing Stop Priority
```
Trailing Stop is checked AFTER TP execution
- Only activates after TP1, TP2, or TP3 executes
- Monitors price continuously
- Triggers when price drops to TP - 20%
```

---

## Edge Cases

### Case 1: Price Never Drops to Trailing Stop
- **Scenario**: TP1 hits, price continues up to TP2
- **Action**: Trailing stop never triggers, position continues to TP2
- **Result**: Apply trailing stop logic after TP2 instead

### Case 2: Price Drops Immediately After TP
- **Scenario**: TP1 hits, price immediately drops to +20%
- **Action**: Trailing stop triggers immediately
- **Result**: 80% sold, 20% runner activated

### Case 3: Multiple TPs Hit Before Trailing Stop
- **Scenario**: TP1 hits, price continues to TP2 before trailing stop triggers
- **Action**: Skip TP1 trailing stop, apply trailing stop after TP2
- **Result**: Trailing stop based on TP2 level

### Case 4: Runner Hits Another TP
- **Scenario**: 1 call runner continues and hits TP2 (+80%)
- **Action**: Apply TP2 logic to runner (sell 60% of 1 = 0.6 → 1 call, but it's already the runner)
- **Result**: Could exit runner at TP2, or continue as runner

---

## Implementation Considerations

### State Tracking
Need to track:
- `tp1_done`: Boolean (TP1 executed)
- `tp2_done`: Boolean (TP2 executed)
- `tp3_done`: Boolean (TP3 executed)
- `trail_active`: Boolean (Trailing stop monitoring)
- `trail_triggered`: Boolean (Trailing stop executed)
- `runner_active`: Boolean (Runner position active)
- `trail_price`: Float (Trailing stop price level)
- `tp1_level`: Float (TP1 price level for trailing stop calc)
- `tp2_level`: Float (TP2 price level for trailing stop calc)
- `tp3_level`: Float (TP3 price level for trailing stop calc)

### Price Monitoring
- Check trailing stop price on every price update
- Compare current premium to trailing stop price
- Trigger when: `current_premium <= trail_price`

### Position Management
- Track `qty_remaining` after each TP
- Calculate trailing stop sell: `int(qty_remaining * 0.8)`
- Calculate runner: `qty_remaining - trailing_stop_sell`

### EOD Check
- Monitor time: Check if current time >= 4:00 PM EST
- Exit runner at market close if still active

---

## Profit Protection Strategy

### Why This Works
1. **Locks Most Profits**: 80% of remaining sold at TP - 20% protects gains
2. **Lets Winners Run**: 20% runner can capture additional gains
3. **Protects Against Reversals**: Trailing stop prevents giving back all profits
4. **EOD Safety**: Runner exits at close to avoid overnight risk

### Example Profit Scenarios

#### Scenario A: Price Reverses After TP1
- Entry: 10 calls @ $2.00
- TP1: 5 calls @ $2.80 (+40%) = **+$400 profit**
- Trailing Stop: 4 calls @ $2.40 (+20%) = **+$160 profit**
- Runner: 1 call @ $2.20 (+10%) = **+$20 profit**
- **Total: +$580 profit** (vs. giving back all if no trailing stop)

#### Scenario B: Price Continues Up
- Entry: 10 calls @ $2.00
- TP1: 5 calls @ $2.80 (+40%) = **+$400 profit**
- Price continues to $3.60 (+80%)
- Trailing Stop: 4 calls @ $3.20 (+60%) = **+$480 profit**
- Runner: 1 call @ $3.60 (+80%) = **+$80 profit**
- **Total: +$960 profit** (runner captured additional gains)

---

## Summary

### Key Principles
1. **After each TP**: Sell 80% of remaining at TP - 20%
2. **Keep 20% runner**: Until EOD or -15% stop loss
3. **Protect profits**: Trailing stop prevents giving back gains
4. **Let winners run**: Small position can capture additional upside

### Implementation Flow
```
TP1 Executes → Calculate Trail Price (TP1 - 20%) → Monitor Price
  ↓
Price Drops to Trail → Sell 80% Remaining → Activate Runner
  ↓
Runner: Monitor until EOD or -15% Stop
```

This creates a **"lock and run"** strategy that maximizes profit protection while allowing for additional upside capture.


