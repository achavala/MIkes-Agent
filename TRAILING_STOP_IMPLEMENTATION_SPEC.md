# Trailing Stop Implementation Specification

## Business Logic Summary

### Core Rule
**After each TP tier executes:**
1. Calculate trailing stop price = **TP Level - 20%**
2. Monitor price continuously
3. When price drops to trailing stop: **Sell 80% of remaining position**
4. **Keep 20% as runner** until EOD or -15% stop loss

---

## Implementation Logic

### Step 1: After TP1 Executes

```python
# TP1 sells 50% of position
if TP1 triggers:
    sell_qty = int(qty_remaining * 0.5)  # 50%
    # Execute sell...
    qty_remaining = qty_remaining - sell_qty
    
    # Setup trailing stop
    tp1_price = entry_premium * (1 + tp1_pct)  # e.g., $2.00 * 1.40 = $2.80
    trail_price = entry_premium * (1 + tp1_pct - 0.20)  # $2.00 * 1.20 = $2.40
    
    pos_data['tp1_done'] = True
    pos_data['tp1_level'] = tp1_pct  # Store for trailing calc
    pos_data['trail_active'] = True
    pos_data['trail_price'] = trail_price
    pos_data['trail_tp_level'] = 1  # Track which TP this trail is for
```

### Step 2: Trailing Stop Check (After TP1)

```python
# Check every price update
if pos_data.get('trail_active') and pos_data.get('trail_tp_level') == 1:
    if current_premium <= pos_data['trail_price']:
        # Trigger trailing stop
        trail_sell_qty = int(qty_remaining * 0.8)  # 80% of remaining
        runner_qty = qty_remaining - trail_sell_qty  # 20% of remaining
        
        if trail_sell_qty > 0:
            # Sell 80%
            api.submit_order(symbol=symbol, qty=trail_sell_qty, side='sell', ...)
            qty_remaining = qty_remaining - trail_sell_qty
            
            # Activate runner
            pos_data['runner_active'] = True
            pos_data['runner_qty'] = runner_qty
            pos_data['trail_triggered'] = True
            pos_data['trail_active'] = False  # Trail done, now manage runner
```

### Step 3: After TP2 Executes

```python
# TP2 sells 60% of remaining
if TP2 triggers:
    sell_qty = int(qty_remaining * 0.6)  # 60% of remaining
    # Execute sell...
    qty_remaining = qty_remaining - sell_qty
    
    # Setup NEW trailing stop (overwrites TP1 trail if not triggered)
    tp2_price = entry_premium * (1 + tp2_pct)  # e.g., $2.00 * 1.80 = $3.60
    trail_price = entry_premium * (1 + tp2_pct - 0.20)  # $2.00 * 1.60 = $3.20
    
    pos_data['tp2_done'] = True
    pos_data['tp2_level'] = tp2_pct
    pos_data['trail_active'] = True
    pos_data['trail_price'] = trail_price
    pos_data['trail_tp_level'] = 2  # Track which TP this trail is for
```

### Step 4: Trailing Stop Check (After TP2)

```python
# Same logic as TP1 trailing stop
if pos_data.get('trail_active') and pos_data.get('trail_tp_level') == 2:
    if current_premium <= pos_data['trail_price']:
        trail_sell_qty = int(qty_remaining * 0.8)  # 80% of remaining
        runner_qty = qty_remaining - trail_sell_qty  # 20% of remaining
        
        if trail_sell_qty > 0:
            # Sell 80%
            api.submit_order(...)
            qty_remaining = qty_remaining - trail_sell_qty
            
            # Activate runner (or add to existing runner)
            if pos_data.get('runner_active'):
                pos_data['runner_qty'] += runner_qty
            else:
                pos_data['runner_active'] = True
                pos_data['runner_qty'] = runner_qty
            
            pos_data['trail_triggered'] = True
            pos_data['trail_active'] = False
```

### Step 5: Runner Management

```python
# Check runner every price update
if pos_data.get('runner_active') and pos_data.get('runner_qty', 0) > 0:
    runner_qty = pos_data['runner_qty']
    
    # Condition 1: -15% Stop Loss
    stop_loss_price = pos_data['entry_premium'] * 0.85  # -15%
    if current_premium <= stop_loss_price:
        # Exit runner
        api.submit_order(symbol=symbol, qty=runner_qty, side='sell', ...)
        pos_data['runner_active'] = False
        pos_data['runner_qty'] = 0
        risk_mgr.log(f"🛑 RUNNER STOP-LOSS: {symbol} @ -15%", "TRADE")
        continue
    
    # Condition 2: EOD (4:00 PM EST)
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    if now.hour == 16 and now.minute >= 0:  # 4:00 PM or later
        # Exit runner at market close
        api.submit_order(symbol=symbol, qty=runner_qty, side='sell', ...)
        pos_data['runner_active'] = False
        pos_data['runner_qty'] = 0
        risk_mgr.log(f"🕐 RUNNER EOD EXIT: {symbol} at market close", "TRADE")
        continue
    
    # Condition 3: Runner can hit TP2 or TP3 (optional)
    # If runner continues and hits another TP, could exit there
    # Or continue as runner (current implementation)
```

---

## State Variables Required

### New Position Data Fields
```python
pos_data = {
    # Existing fields...
    'qty_remaining': int,
    'tp1_done': bool,
    'tp2_done': bool,
    'tp3_done': bool,
    
    # NEW: Trailing stop tracking
    'tp1_level': float,        # TP1 percentage (e.g., 0.40 for +40%)
    'tp2_level': float,        # TP2 percentage (e.g., 0.80 for +80%)
    'tp3_level': float,        # TP3 percentage (e.g., 1.50 for +150%)
    'trail_active': bool,       # Is trailing stop monitoring active?
    'trail_price': float,      # Trailing stop price level
    'trail_tp_level': int,     # Which TP this trail is for (1, 2, or 3)
    'trail_triggered': bool,   # Has trailing stop been triggered?
    
    # NEW: Runner tracking
    'runner_active': bool,      # Is runner position active?
    'runner_qty': int,         # Quantity in runner position
}
```

---

## Execution Order (Priority)

### Check Order in `check_stop_losses()`:
1. **Hard Stop-Loss** (-35% or regime hard_sl) - Highest priority
2. **Take-Profit Tiers** (TP1 → TP2 → TP3) - Sequential
3. **Trailing Stop** (after TP1 or TP2) - Check if trail_active
4. **Runner Management** (EOD or -15% stop) - Check if runner_active
5. **Normal Stop-Loss** (-20% or regime sl) - Lower priority
6. **Rejection Detection** - Lowest priority

---

## Example Calculation

### Entry: 10 Calls @ $2.00

#### After TP1 (+40% = $2.80)
```
Sell: 5 calls (50%)
Remaining: 5 calls
Trailing Stop: $2.40 (+20%)
```

#### When Price Drops to $2.40
```
Sell: 4 calls (80% of 5 = 4)
Remaining: 1 call (20% of 5 = 1)
Runner: 1 call active
```

#### Runner Management
```
Runner: 1 call
Stop Loss: $1.70 (-15% from $2.00)
EOD: 4:00 PM EST
```

---

## Edge Cases

### Case 1: Trailing Stop Never Triggers
```
TP1 executes → Price continues up → TP2 executes
Action: Skip TP1 trailing stop, use TP2 trailing stop
Result: Trailing stop based on TP2 level
```

### Case 2: Multiple Trailing Stops
```
TP1 → Trail triggers → Runner active
TP2 → New trail setup → If triggers, adds to runner
Result: Runner can accumulate from multiple TPs
```

### Case 3: Fractional Contracts
```
Remaining: 2 calls
Trail Sell: 80% of 2 = 1.6 → Round to 2 (sell all)
Runner: 0 calls
Action: If trail_sell_qty >= qty_remaining, sell all (no runner)
```

### Case 4: Runner Hits TP
```
Runner: 1 call
Price: Hits TP2 (+80%)
Action: Exit runner at TP2 (capture the gain)
Result: Runner exits, no need to continue
```

---

## Implementation Checklist

### Phase 1: State Tracking
- [ ] Add `tp1_level`, `tp2_level`, `tp3_level` to position data
- [ ] Add `trail_tp_level` to track which TP trail is active
- [ ] Add `runner_active` and `runner_qty` to position data

### Phase 2: After TP1
- [ ] Calculate trailing stop price (TP1 - 20%)
- [ ] Set `trail_active = True`
- [ ] Store `trail_price` and `trail_tp_level = 1`

### Phase 3: After TP2
- [ ] Calculate trailing stop price (TP2 - 20%)
- [ ] Set `trail_active = True`
- [ ] Store `trail_price` and `trail_tp_level = 2`

### Phase 4: Trailing Stop Check
- [ ] Check `trail_active` and `trail_price`
- [ ] When triggered: Sell 80% of remaining
- [ ] Activate runner: 20% of remaining
- [ ] Set `trail_triggered = True`, `trail_active = False`

### Phase 5: Runner Management
- [ ] Check `runner_active` and `runner_qty`
- [ ] Check -15% stop loss condition
- [ ] Check EOD condition (4:00 PM EST)
- [ ] Exit runner when condition met

---

## Testing Scenarios

### Test 1: TP1 → Trail Triggers → Runner
```
Entry: 10 calls @ $2.00
TP1: 5 calls @ $2.80
Trail: 4 calls @ $2.40
Runner: 1 call until EOD
Expected: 9 calls sold, 1 call at EOD
```

### Test 2: TP1 → TP2 → Trail Triggers → Runner
```
Entry: 10 calls @ $2.00
TP1: 5 calls @ $2.80
TP2: 3 calls @ $3.60
Trail: 1 call @ $3.20
Runner: 1 call until EOD
Expected: 9 calls sold, 1 call at EOD
```

### Test 3: Runner Hits -15% Stop
```
Entry: 10 calls @ $2.00
TP1: 5 calls @ $2.80
Trail: 4 calls @ $2.40
Runner: 1 call → Drops to $1.70 → Exit
Expected: 10 calls sold, runner stopped at -15%
```

---

## Summary

**This implementation creates a "lock and run" strategy:**
- **Locks 80% of remaining** at TP - 20% → Protects profits
- **Runs 20% until EOD** → Captures additional upside
- **-15% stop on runner** → Limits downside
- **EOD exit** → Avoids overnight risk

**Ready for implementation!** 🚀


