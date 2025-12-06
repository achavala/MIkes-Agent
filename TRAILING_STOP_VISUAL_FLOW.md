# Trailing Stop Visual Flow - Step by Step

## Complete Example: 10 Calls Entry

### Initial State
```
Entry: 10 calls @ $2.00 premium
Entry Price: $450.00 (SPY)
Total Cost: $2,000 (10 × $2.00 × 100)
```

---

## Step 1: TP1 Hits (+40% = $2.80)

### Action
```
Sell: 5 calls (50% of 10)
Price: $2.80
Profit: 5 × ($2.80 - $2.00) × 100 = +$400
```

### State After TP1
```
Remaining: 5 calls
Trailing Stop Activated: Monitor for +20% ($2.40)
Runner: Not yet activated
```

### Trailing Stop Setup
```
TP1 Level: +40% ($2.80)
Trailing Stop Level: TP1 - 20% = +20% ($2.40)
Trigger Condition: current_premium <= $2.40
```

---

## Step 2A: Price Drops to Trailing Stop (+20% = $2.40)

### Action
```
Sell: 4 calls (80% of remaining 5)
Price: $2.40
Profit: 4 × ($2.40 - $2.00) × 100 = +$160
```

### State After Trailing Stop
```
Remaining: 1 call (20% of original 5)
Runner Activated: YES
Runner Exit Conditions:
  - EOD (4:00 PM EST), OR
  - -15% stop loss ($1.70)
```

### Runner Management
```
Runner: 1 call
Entry Premium: $2.00
Stop Loss: $1.70 (-15%)
EOD Exit: 4:00 PM EST
```

---

## Step 2B: Price Continues Up (Alternative Path)

### If Price Never Drops to +20%
```
Price continues: $2.80 → $3.20 → $3.60
Trailing Stop: Never triggers
Position: Continues to TP2
```

### When TP2 Hits (+80% = $3.60)
```
Sell: 3 calls (60% of remaining 5)
Price: $3.60
Profit: 3 × ($3.60 - $2.00) × 100 = +$480
Remaining: 2 calls
```

### New Trailing Stop After TP2
```
TP2 Level: +80% ($3.60)
Trailing Stop Level: TP2 - 20% = +60% ($3.20)
Trigger Condition: current_premium <= $3.20
```

---

## Step 3: Trailing Stop After TP2 (+60% = $3.20)

### Action
```
Sell: 1 call (80% of remaining 2, rounded)
Price: $3.20
Profit: 1 × ($3.20 - $2.00) × 100 = +$120
```

### State After Trailing Stop
```
Remaining: 1 call (20% of remaining 2)
Runner Activated: YES
Runner Exit Conditions:
  - EOD (4:00 PM EST), OR
  - -15% stop loss ($1.70)
```

---

## Step 4: Runner Management

### Runner: 1 Call

#### Scenario A: Runner Hits -15% Stop Loss
```
Price drops to: $1.70 (-15% from $2.00)
Action: Exit 1 call
Loss: 1 × ($1.70 - $2.00) × 100 = -$30
Total Trade P&L: +$400 + $160 + $480 + $120 - $30 = +$1,130
```

#### Scenario B: Runner Runs Until EOD
```
Time: 4:00 PM EST
Price: $3.00 (+50%)
Action: Exit 1 call at market close
Profit: 1 × ($3.00 - $2.00) × 100 = +$100
Total Trade P&L: +$400 + $160 + $480 + $120 + $100 = +$1,260
```

#### Scenario C: Runner Hits TP3 (+150% = $5.00)
```
Price continues to: $5.00 (+150%)
Action: Exit 1 call at TP3
Profit: 1 × ($5.00 - $2.00) × 100 = +$300
Total Trade P&L: +$400 + $160 + $480 + $120 + $300 = +$1,460
```

---

## Complete Flow Diagram

```
ENTRY: 10 calls @ $2.00
│
├─→ TP1 (+40% = $2.80)
│   ├─→ Sell 5 calls (50%)
│   └─→ Remaining: 5 calls
│       │
│       ├─→ Price Drops to +20% ($2.40)
│       │   ├─→ Sell 4 calls (80% of 5)
│       │   └─→ Runner: 1 call
│       │       ├─→ EOD → Exit
│       │       ├─→ -15% Stop → Exit
│       │       └─→ TP2/TP3 → Exit
│       │
│       └─→ Price Continues Up
│           └─→ TP2 (+80% = $3.60)
│               ├─→ Sell 3 calls (60% of 5)
│               └─→ Remaining: 2 calls
│                   │
│                   ├─→ Price Drops to +60% ($3.20)
│                   │   ├─→ Sell 1 call (80% of 2)
│                   │   └─→ Runner: 1 call
│                   │       ├─→ EOD → Exit
│                   │       ├─→ -15% Stop → Exit
│                   │       └─→ TP3 → Exit
│                   │
│                   └─→ Price Continues Up
│                       └─→ TP3 (+150% = $5.00)
│                           └─→ Sell 2 calls (full exit)
```

---

## State Tracking Requirements

### Position Data Structure
```python
pos_data = {
    'qty_remaining': 5,              # After TP1
    'tp1_done': True,
    'tp1_level': 0.40,                # +40% for trailing calc
    'tp2_done': False,
    'tp2_level': 0.80,                # +80% for trailing calc
    'tp3_done': False,
    'tp3_level': 1.50,                # +150% for trailing calc
    'trail_active': True,              # After TP1
    'trail_price': 2.40,               # TP1 - 20% = $2.40
    'trail_triggered': False,
    'runner_active': False,            # Set to True after trail triggers
    'runner_qty': 0,                   # 20% of remaining after trail
    'entry_premium': 2.00,
    'entry_price': 450.00
}
```

---

## Calculation Formulas

### After TP1 Executes
```python
# TP1 sells 50%
qty_remaining = original_qty * 0.5

# Trailing stop calculation
tp1_level = entry_premium * (1 + tp1_pct)  # $2.00 * 1.40 = $2.80
trail_price = entry_premium * (1 + tp1_pct - 0.20)  # $2.00 * 1.20 = $2.40

# When trail triggers
trail_sell_qty = int(qty_remaining * 0.8)  # 80% of remaining
runner_qty = qty_remaining - trail_sell_qty  # 20% of remaining
```

### After TP2 Executes
```python
# TP2 sells 60% of remaining
qty_remaining = qty_after_tp1 * 0.4

# Trailing stop calculation
tp2_level = entry_premium * (1 + tp2_pct)  # $2.00 * 1.80 = $3.60
trail_price = entry_premium * (1 + tp2_pct - 0.20)  # $2.00 * 1.60 = $3.20

# When trail triggers
trail_sell_qty = int(qty_remaining * 0.8)  # 80% of remaining
runner_qty = qty_remaining - trail_sell_qty  # 20% of remaining
```

### Runner Exit Conditions
```python
# Check every price update
if runner_active:
    # Condition 1: -15% stop loss
    if current_premium <= entry_premium * 0.85:
        exit_runner()
    
    # Condition 2: EOD (4:00 PM EST)
    current_time = datetime.now(est_timezone)
    if current_time.hour == 16 and current_time.minute >= 0:
        exit_runner()
    
    # Condition 3: TP2 or TP3 (if runner continues)
    if pnl_pct >= tp2_pct:
        # Apply TP2 logic to runner
    elif pnl_pct >= tp3_pct:
        # Apply TP3 logic to runner
```

---

## Edge Cases & Special Scenarios

### Case 1: Trailing Stop Triggers Immediately
```
TP1 executes → Price immediately drops to +20%
Action: Trailing stop triggers in same tick
Result: 80% sold, 20% runner activated immediately
```

### Case 2: Price Never Drops to Trailing Stop
```
TP1 executes → Price continues up to TP2
Action: Skip TP1 trailing stop, apply trailing stop after TP2
Result: Trailing stop based on TP2 level (+60%)
```

### Case 3: Runner Hits Another TP
```
Runner: 1 call continues
Price: Hits TP2 (+80%)
Action: Could exit runner, or continue as runner
Decision: Exit runner at TP2 (capture the gain)
```

### Case 4: Multiple TPs Before Trailing Stop
```
TP1 → Price continues → TP2 → Price continues → TP3
Action: Apply trailing stop after most recent TP
Result: Trailing stop based on TP3 level (+130%)
```

### Case 5: Fractional Contracts
```
Remaining: 2 calls
Trailing Stop Sell: 80% of 2 = 1.6 → Round to 2 (sell all)
Runner: 0 calls (edge case - all sold)
```

---

## Implementation Priority

### Phase 1: After TP1
1. Calculate trailing stop price (TP1 - 20%)
2. Monitor price for trailing stop trigger
3. When triggered: Sell 80% of remaining
4. Activate runner: 20% of remaining

### Phase 2: After TP2
1. Calculate trailing stop price (TP2 - 20%)
2. Monitor price for trailing stop trigger
3. When triggered: Sell 80% of remaining
4. Activate runner: 20% of remaining

### Phase 3: Runner Management
1. Monitor runner position
2. Check -15% stop loss condition
3. Check EOD condition (4:00 PM EST)
4. Exit runner when condition met

---

## Profit Protection Summary

### Why This Works
1. **Locks 80% of remaining** at TP - 20% → Protects most profits
2. **Runs 20% until EOD** → Captures additional upside
3. **-15% stop on runner** → Limits downside risk
4. **EOD exit** → Avoids overnight risk on 0DTE

### Expected Outcomes
- **Best Case**: Runner hits TP2/TP3 → Maximum profit
- **Normal Case**: Runner exits at EOD → Good profit locked
- **Worst Case**: Runner hits -15% stop → Still profitable overall

This creates a **"lock and run"** strategy that maximizes profit protection while allowing for additional upside capture.


