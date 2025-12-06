# 🔍 STOP LOSS TROUBLESHOOTING ANALYSIS

## Problem Statement
Position `SPY251205C00686000` is down **-51.61%** but the **-15% stop loss is NOT triggering**.

## Root Cause Analysis

### Issue #1: Position Tracking Gap ⚠️ CRITICAL BUG
**Location:** `mike_agent_live_safe.py`, line 524

**The Problem:**
```python
for symbol, pos_data in list(risk_mgr.open_positions.items()):
```

`check_stop_losses()` **ONLY** checks positions that are already in `risk_mgr.open_positions`. 

**If a position exists in Alpaca but is NOT tracked:**
- ❌ It will NEVER be checked for stop loss
- ❌ Stop loss will NEVER trigger
- ❌ Position will keep losing money

**When this happens:**
1. Position opened before agent started
2. Position opened externally (manual trade)
3. Agent crashed/restarted and didn't sync properly
4. Position sync on startup failed

### Issue #2: Entry Premium Sync Problem
**Location:** `mike_agent_live_safe.py`, lines 1028-1037

**The Problem:**
When syncing positions on startup, the code uses:
```python
entry_premium = 0.5  # Default
if snapshot.bid_price:
    bid_float = float(snapshot.bid_price)
    if bid_float > 0:
        entry_premium = bid_float  # ❌ WRONG! This is CURRENT price, not ENTRY price!
```

**This is WRONG because:**
- `snapshot.bid_price` is the **CURRENT** bid price
- `entry_premium` should be the **ENTRY** price (what we paid)
- Alpaca provides `avg_entry_price` but we're not using it!

### Issue #3: Missing Position Sync in Stop Loss Check
**Location:** `mike_agent_live_safe.py`, line 494-522

**The Problem:**
`check_stop_losses()` gets Alpaca positions but:
1. Only checks positions already in tracking
2. Doesn't sync new positions from Alpaca
3. Doesn't handle untracked positions

## The Fix

### Fix #1: Check ALL Alpaca Positions (Not Just Tracked)
```python
def check_stop_losses(api, risk_mgr, current_price, trade_db=None):
    # Get ALL positions from Alpaca
    alpaca_positions = api.list_positions()
    alpaca_option_positions = {pos.symbol: pos for pos in alpaca_positions 
                               if is_option_position(pos)}
    
    # CRITICAL: Sync any positions from Alpaca that aren't tracked
    for symbol, alpaca_pos in alpaca_option_positions.items():
        if symbol not in risk_mgr.open_positions:
            # Sync this position!
            sync_position_from_alpaca(api, risk_mgr, symbol, alpaca_pos, current_price)
    
    # NOW check all positions (both tracked and newly synced)
    for symbol, pos_data in list(risk_mgr.open_positions.items()):
        # ... existing stop loss logic ...
```

### Fix #2: Use Alpaca's avg_entry_price
```python
def sync_position_from_alpaca(api, risk_mgr, symbol, alpaca_pos, current_price):
    # Get entry premium from Alpaca (CORRECT!)
    if hasattr(alpaca_pos, 'avg_entry_price') and alpaca_pos.avg_entry_price:
        entry_premium = float(alpaca_pos.avg_entry_price)
    else:
        # Fallback: use cost basis / (qty * 100)
        if hasattr(alpaca_pos, 'cost_basis') and alpaca_pos.cost_basis:
            qty = float(alpaca_pos.qty)
            entry_premium = float(alpaca_pos.cost_basis) / (qty * 100) if qty > 0 else 0.5
        else:
            entry_premium = 0.5  # Last resort default
```

### Fix #3: Add Continuous Sync
Every time `check_stop_losses()` runs:
1. Get ALL positions from Alpaca
2. Sync any missing positions
3. Remove positions that no longer exist
4. Check stop losses on ALL positions

## Implementation Plan

### Step 1: Create Position Sync Function
- Extract strike from symbol
- Get entry premium from Alpaca (avg_entry_price or cost_basis)
- Add to risk_mgr.open_positions with correct data

### Step 2: Modify check_stop_losses()
- Sync positions BEFORE checking stop losses
- Handle both tracked and untracked positions
- Add detailed logging

### Step 3: Enhanced Debug Logging
- Log when positions are synced
- Log entry_premium vs current_premium
- Log P&L calculation
- Log stop loss checks

### Step 4: Test & Validate
- Test with existing losing position
- Test with manually opened position
- Test with position opened before agent start

## Expected Behavior After Fix

1. **Position exists in Alpaca but not tracked:**
   - ✅ Position gets synced automatically
   - ✅ Entry premium uses Alpaca's avg_entry_price
   - ✅ Stop loss check happens immediately
   - ✅ Stop loss triggers if P&L <= -15%

2. **Position tracking stays in sync:**
   - ✅ Positions added/removed as needed
   - ✅ Entry premium stays accurate
   - ✅ Stop loss works consistently

## Priority: 🔴 CRITICAL

This bug can cause unlimited losses. Must fix immediately.

