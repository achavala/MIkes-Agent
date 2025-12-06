# 📋 Paper Trading Monitoring Checklist

**Use this checklist daily during paper trading validation**

---

## 🌅 **MORNING CHECKLIST** (Before Market Open)

### Agent Status
- [ ] Agent process is running (`ps aux | grep mike_agent`)
- [ ] No crashes in overnight logs
- [ ] Last log entry is recent (within last hour)

### Previous Day Review
- [ ] Check final positions in Alpaca dashboard
- [ ] Review yesterday's trade history CSV
- [ ] Note any issues from logs
- [ ] Check daily P&L

### System Health
- [ ] Log files are updating
- [ ] Dashboard accessible
- [ ] Alpaca dashboard accessible
- [ ] No error messages in logs

---

## 📊 **DURING MARKET HOURS** (9:30 AM - 4:00 PM EST)

### Hourly Checks (Recommended Every 1-2 Hours)

**Dashboard Monitoring:**
- [ ] Market status shows "MARKET OPEN"
- [ ] Positions table updating
- [ ] Trading history showing new trades
- [ ] Portfolio summary accurate

**Alpaca Dashboard:**
- [ ] Positions tab shows correct positions
- [ ] Orders tab shows executed orders
- [ ] Account value updating
- [ ] No unexpected positions

**Log Files:**
- [ ] No error messages
- [ ] Order submissions successful
- [ ] Position updates occurring
- [ ] Risk warnings (if any) are expected

### Key Times to Watch

**9:30 AM EST - Market Open:**
- [ ] Agent detects market open
- [ ] First trades (if signals present)
- [ ] Position sizes are reasonable

**2:30 PM EST - Trading Window Close:**
- [ ] No new entries after 2:30 PM
- [ ] Existing positions continue to be monitored
- [ ] Stop-losses and take-profits still active

**4:00 PM EST - Market Close:**
- [ ] Agent detects market close
- [ ] Final positions recorded
- [ ] Daily summary logged

---

## 📈 **DAILY METRICS TO TRACK**

### Performance Metrics

**Starting Values:**
- Starting Equity: $________
- Starting Cash: $________
- Starting Positions: ________

**Ending Values:**
- Ending Equity: $________
- Ending Cash: $________
- Ending Positions: ________

**Daily Totals:**
- Daily P&L: $________ (%________)
- Number of Trades: ________
- Win Rate: ________%
- Average Win: $________
- Average Loss: $________
- Largest Win: $________
- Largest Loss: $________

### Risk Metrics

**Position Limits:**
- Max Position Size Used: $________ (% of equity)
- Max Concurrent Positions: ________
- Daily Loss Limit: -15% (check if hit)

**Risk Events:**
- [ ] Daily loss limit NOT hit
- [ ] Position size limits respected
- [ ] Stop-losses triggered correctly
- [ ] Take-profits triggered correctly
- [ ] No position exceeded 25% equity

### Trading Activity

**Trades Today:**
- Total Trades: ________
- Winning Trades: ________
- Losing Trades: ________
- Breakeven Trades: ________

**Entry/Exit Analysis:**
- Calls Opened: ________
- Puts Opened: ________
- Calls Closed: ________
- Puts Closed: ________

---

## ✅ **END OF DAY CHECKLIST**

### Position Review
- [ ] All positions from Alpaca match agent tracking
- [ ] No unexpected open positions
- [ ] Position sizes are correct
- [ ] P&L calculations match Alpaca

### Trade Review
- [ ] All orders executed correctly
- [ ] Fill prices reasonable
- [ ] No failed orders
- [ ] Stop-losses executed when triggered
- [ ] Take-profits executed when triggered

### System Review
- [ ] No errors in logs
- [ ] All safeguards functioned correctly
- [ ] Risk limits respected
- [ ] Time-of-day filter worked

### Documentation
- [ ] Daily P&L recorded
- [ ] Trade count recorded
- [ ] Issues documented
- [ ] Observations noted

---

## 📝 **WEEKLY SUMMARY**

### Performance Summary
- Total Return (Week): ________%
- Total P&L (Week): $________
- Win Rate (Week): ________%
- Average Daily Return: ________%

### Trade Statistics
- Total Trades: ________
- Average Trade Size: $________
- Best Day: $________ (Date: ________)
- Worst Day: $________ (Date: ________)

### Risk Analysis
- Max Drawdown: ________%
- Largest Position: $________ (% of equity)
- Risk Limit Triggers: ________
- Stop-Loss Triggers: ________

### Issues & Observations
- Issues Encountered: ________
- Solutions Applied: ________
- Patterns Noted: ________
- Improvements Needed: ________

---

## 🎯 **VALIDATION CHECKLIST**

### Week 1 Validation

**Technical:**
- [ ] Agent runs 24/7 without crashes
- [ ] All 13 safeguards functioning
- [ ] Orders execute correctly
- [ ] No critical errors
- [ ] Dashboard works properly
- [ ] Logging works correctly

**Performance:**
- [ ] Daily metrics tracked
- [ ] Trade patterns documented
- [ ] Win rate acceptable
- [ ] Risk limits respected
- [ ] Position sizing correct

### Week 2-4 Validation

**Consistency:**
- [ ] Performance metrics consistent
- [ ] No unexpected behaviors
- [ ] All edge cases handled
- [ ] System stability confirmed

**Decision Ready:**
- [ ] Minimum 2-4 weeks paper trading completed
- [ ] All issues resolved
- [ ] Performance acceptable
- [ ] Confidence in system built

---

## ⚠️ **RED FLAGS TO WATCH FOR**

### Immediate Action Required

- [ ] Daily loss limit hit (-15%)
- [ ] Agent crashed/stopped running
- [ ] Multiple failed orders
- [ ] Position sizes exceed limits
- [ ] Unexpected large losses
- [ ] Error messages in logs

### Monitor Closely

- [ ] Consistent losing days
- [ ] Stop-losses not triggering
- [ ] Take-profits not executing
- [ ] Position tracking errors
- [ ] Unusual market behavior

---

## 📞 **QUICK REFERENCE**

### Daily Monitoring Commands

```bash
# Check agent status
ps aux | grep mike_agent_live_safe

# View today's log
tail -f logs/mike_agent_safe_$(date +%Y%m%d).log

# View recent trades
tail -20 mike_agent_trades.csv

# Start dashboard
streamlit run app.py
```

### Key URLs

- **Alpaca Dashboard:** https://app.alpaca.markets/paper/dashboard
- **Streamlit Dashboard:** http://localhost:8501
- **Alpaca Status:** https://status.alpaca.markets

---

## 📅 **DAILY LOG TEMPLATE**

### Date: ________

**Market Conditions:**
- VIX: ________
- SPY Price: $________
- Market Regime: ________

**Trades:**
- Trades Today: ________
- Win/Loss: ________/________
- P&L: $________

**Issues:**
- ________

**Observations:**
- ________

**Next Steps:**
- ________

---

*Use this checklist daily during paper trading validation period*


