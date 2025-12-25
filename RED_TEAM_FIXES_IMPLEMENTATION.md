# 🔴 RED-TEAM FIXES - IMPLEMENTATION PLAN

**Date:** December 22, 2025  
**Priority:** CRITICAL - Stop the bleeding before any more trading

---

## 📋 IMPLEMENTATION STATUS

### **Phase 0 - STOP THE BLEEDING** (CRITICAL - Do First)

- [x] **0.1: Remove resampling logic entirely**
- [ ] **0.2: Add trade gating (spread, quote age, expected move)**
- [ ] **0.3: Disable SPX, restrict IWM**
- [ ] **0.4: Raise confidence threshold (0.52 → 0.60)**

### **Phase 1 - STRUCTURAL EDGE** (HIGH PRIORITY)

- [ ] **1.1: Add VIX1D, IV rank/skew, Expected move calculation**
- [ ] **1.2: Add Gamma wall proxy calculation**
- [ ] **1.3: Convert ensemble from averaging → gating network**
- [ ] **1.4: Make liquidity & vol agents hard vetoes**
- [ ] **1.5: Restrict RL to entry timing, sizing, exit (not trade selection)**

### **Phase 2 - MODEL RE-ARCHITECTURE** (MEDIUM PRIORITY)

- [ ] **2.1: Add supervised ML regime classifier**
- [ ] **2.2: Condition RL on regime**
- [ ] **2.3: Add realistic slippage/spreads/failed fills to environment**

### **Phase 3 - INSTITUTIONAL LEVEL** (LONG TERM)

- [ ] **3.1: Quote-level data**
- [ ] **3.2: Real option chain snapshots**
- [ ] **3.3: GEX & vanna ingestion**
- [ ] **3.4: Offline RL (CQL/IQL)**
- [ ] **3.5: Walk-forward stress tests**

---

## 🎯 KEY PRINCIPLES

1. **RL is for tactics, not trade selection**
2. **Gate trades by volatility & liquidity first**
3. **Accept lower frequency, higher quality**
4. **Kill the idea that "more trades = learning"**


