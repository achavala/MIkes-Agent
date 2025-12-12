# 📋 Paper Trading Validation Guide

**Purpose**: Step-by-step guide to validate the QQQ/SPX symbol fixes in live paper trading

---

## 🚀 Quick Start

### 1. Start the Agent
```bash
cd /Users/chavala/Mike-agent-project
python3 mike_agent_live_safe.py
```

### 2. Monitor Logs
Watch for the enhanced logging patterns:

```bash
# In another terminal
tail -f logs/mike_agent_safe_$(date +%Y%m%d).log | grep -E "TRADE_OPENED|STOP_LOSS_CHECK|PNL_UPDATE|ROTATION"
```

---

## 📊 What to Look For

### ✅ Trade Opening Validation

**Expected Pattern**:
```
TRADE_OPENED | symbol=SPY | option=SPY251205C00690000 | symbol_price=$690.50 | entry_price=$690.50 | premium=$2.30 | qty=5 | strike=$691.00 | regime=CALM
```

**Validation Points**:
- [ ] `symbol_price` matches the underlying (SPY ~$690, QQQ ~$420, SPX ~$6,800)
- [ ] `entry_price` equals `symbol_price` (not SPY price for QQQ/SPX!)
- [ ] Premium is reasonable (SPY: $1-5, QQQ: $1-4, SPX: $10-50)

---

### ✅ Stop-Loss Check Validation

**Expected Pattern**:
```
STOP_LOSS_CHECK | symbol=QQQ | option=QQQ251205C00420000 | symbol_price=$415.00 | entry_price=$420.75 | pnl=-1.37% | threshold=-15.00% | entry_premium=$1.85 | current_premium=$1.82 | source=snapshot_bid
```

**Validation Points**:
- [ ] `symbol` matches the underlying (SPY, QQQ, or SPX)
- [ ] `symbol_price` is the correct underlying price (not SPY for QQQ/SPX!)
- [ ] `entry_price` matches the trade's entry price
- [ ] P&L calculation is based on correct prices

---

### ✅ P&L Update Validation

**Expected Pattern**:
```
PNL_UPDATE | symbol=SPX | option=SPX251205C00680000 | symbol_price=$6820.00 | entry_price=$6800.25 | entry_premium=$15.50 | current_premium=$16.20 | pnl=+4.52% | qty_remaining=3
```

**Validation Points**:
- [ ] `symbol_price` reflects the correct underlying price
- [ ] P&L moves independently for each symbol
- [ ] When SPY moves up but QQQ moves down, their P&L should reflect this independently

---

### ✅ Symbol Rotation Validation

**Expected Pattern**:
```
ROTATION | next_symbol=QQQ | open_positions=1 | all_symbols_have_positions=False
```

**Validation Sequence**:
1. **First Trade**: Should be SPY
   ```
   TRADE_OPENED | symbol=SPY | ...
   ```

2. **Second Trade**: Should be QQQ (when SPY has position)
   ```
   ROTATION | next_symbol=QQQ | open_positions=1
   TRADE_OPENED | symbol=QQQ | ...
   ```

3. **Third Trade**: Should be SPX (when SPY and QQQ have positions)
   ```
   ROTATION | next_symbol=SPX | open_positions=2
   TRADE_OPENED | symbol=SPX | ...
   ```

---

## 🔍 Detailed Validation Steps

### Step 1: Verify SPY Trade
1. Wait for first trade to open
2. Check log: `TRADE_OPENED | symbol=SPY`
3. Verify: `symbol_price` is around $690 (SPY price)
4. Verify: `entry_price` equals `symbol_price`

### Step 2: Verify QQQ Trade
1. Wait for second trade to open (after SPY has position)
2. Check log: `TRADE_OPENED | symbol=QQQ`
3. Verify: `symbol_price` is around $420 (QQQ price, NOT SPY!)
4. Verify: `entry_price` equals QQQ price (NOT SPY price!)

### Step 3: Verify SPX Trade
1. Wait for third trade to open (after SPY and QQQ have positions)
2. Check log: `TRADE_OPENED | symbol=SPX`
3. Verify: `symbol_price` is around $6,800 (SPX price, NOT SPY!)
4. Verify: `entry_price` equals SPX price (NOT SPY price!)

### Step 4: Verify Independent P&L
1. Monitor `PNL_UPDATE` logs for all three symbols
2. Verify: SPY P&L reflects SPY price movements
3. Verify: QQQ P&L reflects QQQ price movements (independent of SPY)
4. Verify: SPX P&L reflects SPX price movements (independent of SPY)

### Step 5: Verify Stop-Loss Per Symbol
1. Monitor `STOP_LOSS_CHECK` logs
2. Verify: QQQ stop-loss uses QQQ price (not SPY)
3. Verify: SPX stop-loss uses SPX price (not SPY)
4. Verify: Stop-loss triggers are based on correct symbol prices

---

## ⚠️ Red Flags to Watch For

### ❌ Wrong Price in Logs
```
TRADE_OPENED | symbol=QQQ | symbol_price=$690.50  # WRONG! Should be ~$420
```
**Action**: This indicates the fix didn't work - stop trading immediately.

### ❌ Cross-Contaminated P&L
```
PNL_UPDATE | symbol=QQQ | pnl=+5.00%  # But QQQ price didn't move, SPY did
```
**Action**: This indicates entry_price is wrong - stop trading immediately.

### ❌ Wrong Stop-Loss Trigger
```
STOP_LOSS_CHECK | symbol=QQQ | symbol_price=$690.50  # WRONG! Should be QQQ price
```
**Action**: This indicates stop-loss fix didn't work - stop trading immediately.

---

## ✅ Success Criteria

After 1-2 hours of paper trading, you should have:

1. ✅ At least one SPY trade with correct SPY price
2. ✅ At least one QQQ trade with correct QQQ price (not SPY)
3. ✅ At least one SPX trade with correct SPX price (not SPY)
4. ✅ Independent P&L behavior for each symbol
5. ✅ Stop-loss checks using correct symbol prices
6. ✅ Symbol rotation working as expected

**If all criteria are met**: System is **100% validated and ready for automatic trading**.

---

## 📝 Log Analysis Commands

### Extract All Trade Openings
```bash
grep "TRADE_OPENED" logs/mike_agent_safe_*.log | tail -20
```

### Extract All Stop-Loss Checks
```bash
grep "STOP_LOSS_CHECK" logs/mike_agent_safe_*.log | tail -20
```

### Extract All P&L Updates
```bash
grep "PNL_UPDATE" logs/mike_agent_safe_*.log | tail -20
```

### Extract Symbol Rotation
```bash
grep "ROTATION" logs/mike_agent_safe_*.log | tail -20
```

### Verify Price Consistency
```bash
# Check QQQ trades use QQQ prices
grep "TRADE_OPENED.*symbol=QQQ" logs/mike_agent_safe_*.log | grep -v "symbol_price=\$420\|symbol_price=\$41[0-9]\|symbol_price=\$43[0-9]"

# Check SPX trades use SPX prices
grep "TRADE_OPENED.*symbol=SPX" logs/mike_agent_safe_*.log | grep -v "symbol_price=\$6[0-9][0-9][0-9]"
```

---

## 🎯 Next Steps After Validation

Once validation is complete:

1. ✅ Document any issues found
2. ✅ Fix any remaining bugs
3. ✅ Re-run validation if fixes were needed
4. ✅ Proceed to automatic trading once all criteria are met

---

**Good luck with your validation! 🚀**

