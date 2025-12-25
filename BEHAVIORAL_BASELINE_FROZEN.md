# ✅ BEHAVIORAL CONFIGURATION - FROZEN BASELINE

**Date:** December 13, 2025  
**Status:** ✅ **FROZEN - DO NOT MODIFY**

---

## ✅ BEHAVIORAL PHASE COMPLETE

**Final Verdict:** REVISE  
**Trade Activity:** 5 trades, 1.00 trades/day  
**Scorecards:** Behavior: 0.30, Risk: 0.70, Execution: 0.50, Learning: 0.50

**This configuration is now the baseline for behavioral testing.**

---

## 🔒 FROZEN CONFIGURATION

### Risk Manager Settings
- **Gamma limit multiplier:** 1.5x (50% more lenient)
- **Delta limit multiplier:** 1.5x (50% more lenient)
- **VIX kill switch:** OFF (disabled for behavioral)
- **Daily loss limit multiplier:** 2.0x (more lenient)

### Ensemble Settings
- **Min agent agreement:** 1 (allow single agent to propose)
- **Confidence threshold:** 0.25 (lowered from 0.3)

### Signal Floor Settings
- **Enabled:** True
- **RL confidence minimum:** 0.52
- **Ensemble confidence minimum:** 0.50
- **Min size multiplier:** 0.5 (smaller size for exploratory trades)

### Action Nudge Settings
- **Enabled:** True
- **RL action raw threshold:** 0.15
- **Force probe trade:** True (will be disabled for PAPER mode)
- **Probe trade size:** 0.1 (10% of normal)

### Execution Settings
- **Apply IV crush:** False (still log, but don't penalize)
- **Apply theta penalty:** False (still log, but don't penalize)
- **Slippage multiplier:** 0.5 (reduce slippage impact)

### Trading Settings
- **Min trades per day:** 1
- **Max trades per day:** 20

---

## 🚫 DO NOT MODIFY

**This configuration is frozen. Do NOT change:**
- ❌ Thresholds
- ❌ Verdict logic
- ❌ Risk limits
- ❌ Ensemble agreement
- ❌ Signal floor settings

**This is the validated baseline for behavioral testing.**

---

## 📋 NEXT: PAPER MODE CONFIGURATION

See `PAPER_MODE_CONFIGURATION.md` for PAPER mode settings.





