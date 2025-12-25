# 🔍 Real-Time Agent Monitoring Guide

## Quick Start

### Option 1: Detailed Monitor (Recommended)
```bash
./monitor_agent.sh
```

Shows:
- ✅ RL inference signals with confidence levels
- ✅ Symbol selection decisions
- ✅ Ensemble signals
- ✅ Trade executions (with details)
- ✅ Blocked trades (with reasons)
- ✅ Safeguard checks
- ✅ Position exits (TP/SL)
- ✅ Color-coded output

### Option 2: Simple Monitor (Less Verbose)
```bash
./monitor_agent_simple.sh
```

Shows only:
- RL signals
- Symbol selections
- Trades
- Blocks
- Exits

### Option 3: Direct Fly Logs (Raw)
```bash
fly logs --app mike-agent-project
```

Shows all logs (very verbose)

---

## What You'll See

### RL Signal Example
```
[RL] SPY → BUY CALL (72.5%) [10:35:22]
```
- Shows which symbol
- Action type (BUY CALL/PUT, HOLD)
- Confidence percentage
- Green if >= 65% (will execute)
- Yellow if < 65% (will be blocked)

### Symbol Selection Example
```
[SELECT] SPY selected for BUY CALL [10:35:23]
```
- Shows which symbol was chosen
- Why it was selected

### Trade Execution Example
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ TRADE EXECUTED SPY251216C00678000
   Qty: 35 contracts
   Premium: $2.50
   Strike: $678.00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Blocked Trade Example
```
⛔ BLOCKED SPY: Confidence too low (0.58 < 0.65) [10:35:24]
⛔ BLOCKED QQQ: Stop-loss cooldown active (1.5 min remaining) [10:35:25]
⛔ BLOCKED SPY: Position would exceed 25% limit [10:35:26]
```

### Ensemble Signal Example
```
[ENSEMBLE] SPY confidence: 78.5% [10:35:21]
```

### Exit Example
```
[EXIT] SPY251216C00678000 STOP-LOSS EXIT @ -21.2%
[EXIT] SPY251216C00678000 TAKE-PROFIT TP1 @ +42.5%
```

---

## Understanding the Output

### Color Coding
- **Green**: Successful trades, high confidence signals
- **Yellow**: Warnings, low confidence, blocks
- **Red**: Errors, blocked trades, safeguards triggered
- **Blue**: Ensemble signals, market data
- **Cyan**: Symbol names
- **Magenta**: Selection decisions
- **White**: General info

### Key Patterns

**1. RL Signal Flow**
```
[RL] SPY → BUY CALL (72.5%)     ← Model says buy
[RL] QQQ → HOLD (50.0%)         ← Model says hold
[ENSEMBLE] SPY confidence: 78.5% ← Ensemble confirms
[SELECT] SPY selected for BUY CALL ← Symbol chosen
✅ TRADE EXECUTED                ← Trade placed
```

**2. Blocked Trade Flow**
```
[RL] SPY → BUY CALL (58.0%)     ← Model says buy
⛔ BLOCKED SPY: Confidence too low (0.58 < 0.65) ← Blocked
```

**3. Cooldown Example**
```
[RL] SPY → BUY CALL (75.0%)     ← Model says buy
⛔ BLOCKED SPY: Stop-loss cooldown active (2 min remaining) ← Blocked
```

**4. Position Limit Example**
```
[RL] SPY → BUY CALL (80.0%)     ← Model says buy
[SELECT] SPY selected for BUY CALL
⛔ BLOCKED SPY: Position would exceed 25% limit ← Blocked
```

---

## Monitoring Tips

### 1. Watch for Patterns
- **Many blocks with "Confidence too low"**: Market conditions not favorable
- **Many blocks with "cooldown"**: Agent is being conservative (good!)
- **Many blocks with "position limit"**: Already at max exposure

### 2. Check RL Confidence
- **> 70%**: Strong signal, likely to execute
- **65-70%**: Good signal, will execute
- **< 65%**: Weak signal, will be blocked

### 3. Monitor Symbol Selection
- See which symbols are being considered
- Understand why one symbol is chosen over another
- Check if rotation is working (fair distribution)

### 4. Track Trade Execution
- See exact details of each trade
- Monitor position sizing
- Check strike prices

### 5. Watch for Safeguards
- If you see "SAFEGUARD TRIGGERED", agent will shutdown
- Daily loss limit, drawdown, VIX kill switch

---

## Advanced Monitoring

### Filter for Specific Events
```bash
fly logs --app mike-agent-project | grep "TRADE_OPENED"
fly logs --app mike-agent-project | grep "BLOCKED"
fly logs --app mike-agent-project | grep "RL Action"
```

### Save Logs to File
```bash
./monitor_agent.sh | tee agent_activity_$(date +%Y%m%d).log
```

### Monitor Specific Symbol
```bash
fly logs --app mike-agent-project | grep "SPY"
```

### Count Events
```bash
fly logs --app mike-agent-project | grep -c "TRADE_OPENED"
fly logs --app mike-agent-project | grep -c "BLOCKED"
```

---

## Troubleshooting

### No Output
- Check if agent is running: `fly status --app mike-agent-project`
- Check if logs are available: `fly logs --app mike-agent-project | head -20`

### Too Much Output
- Use `monitor_agent_simple.sh` instead
- Or filter: `fly logs --app mike-agent-project | grep -E "(TRADE|BLOCKED|RL Action)"`

### Missing Events
- Some events may be logged at different levels
- Check raw logs: `fly logs --app mike-agent-project`

---

## Example Full Session

```
╔══════════════════════════════════════════════════════════════╗
║     🔍 MIKE AGENT REAL-TIME ACTIVITY MONITOR                 ║
╚══════════════════════════════════════════════════════════════╝

[10:35:20] [RL] SPY → BUY CALL (72.5%)
[10:35:20] [RL] QQQ → HOLD (50.0%)
[10:35:21] [ENSEMBLE] SPY confidence: 78.5%
[10:35:22] [SELECT] SPY selected for BUY CALL
[10:35:23] [DATA] Preparing observation for SPY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ TRADE EXECUTED SPY251216C00678000
   Qty: 35 contracts
   Premium: $2.50
   Strike: $678.00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[10:36:15] [RL] SPY → BUY PUT (58.0%)
⛔ BLOCKED SPY: Confidence too low (0.58 < 0.65)

[10:37:20] [RL] QQQ → BUY CALL (75.0%)
[10:37:21] [SELECT] QQQ selected for BUY CALL
⛔ BLOCKED QQQ: Stop-loss cooldown active (1.5 min remaining)

[10:40:30] [EXIT] SPY251216C00678000 TAKE-PROFIT TP1 @ +42.5%
```

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `./monitor_agent.sh` | Full detailed monitor |
| `./monitor_agent_simple.sh` | Simple monitor (key events only) |
| `fly logs --app mike-agent-project` | Raw logs (all output) |
| `fly status --app mike-agent-project` | Check if agent is running |

---

## Pro Tips

1. **Run in separate terminal**: Keep monitor running while you work
2. **Save important sessions**: Use `tee` to save logs
3. **Watch for patterns**: Learn what conditions trigger trades
4. **Check confidence levels**: Understand why trades are/aren't executing
5. **Monitor cooldowns**: See when agent is being conservative

---

Enjoy monitoring your agent! 🚀





