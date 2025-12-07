# 📊 Comprehensive Quant Features Collection Plan

**Date:** December 7, 2025  
**Status:** ✅ **SYSTEM CREATED - READY TO COLLECT**

---

## 🎯 Objective

Collect **all institutional-grade quant features** for historical data before training, including:

1. ✅ **IV (Implied Volatility)** from VIX
2. ✅ **Greeks** (Delta, Gamma, Vega, Theta)
3. ✅ **Theta decay model**
4. ✅ **Market microstructure** (order flow imbalance)
5. ✅ **Cross-asset correlations** (SPY-QQQ-VIX-SPX)
6. ✅ **Volatility regime classification**
7. ✅ **Market Profile/TPO signals**

---

## ✅ Implementation Status

### Files Created

1. **`quant_features_collector.py`** ✅
   - Comprehensive quant features collection module
   - All 7 requested feature categories implemented
   - Integrates with existing Greeks calculator
   - Supports both ATM and custom strike prices

2. **`collect_quant_features.py`** ✅
   - Script to collect all quant features for historical data
   - Integrates with existing data collection pipeline
   - Handles cross-asset correlations
   - Saves enriched data to separate directory

---

## 📋 Feature Details

### 1. IV (Implied Volatility) ✅

**Source:** VIX data (already collected)

**Features:**
- `vix`: Raw VIX level
- `iv_from_vix`: IV derived from VIX (decimal form)
- `iv_0dte`: IV scaled for 0DTE options

**Calculation:**
```python
iv_0dte = VIX / 100 * sqrt(1 / 30)  # Scale for 0DTE time horizon
```

---

### 2. Greeks (Delta, Gamma, Vega, Theta) ✅

**Calculation:** Black-Scholes model (via `GreeksCalculator`)

**Features:**
- `call_delta`, `put_delta`: Price sensitivity
- `call_gamma`, `put_gamma`: Convexity/acceleration
- `call_vega`, `put_vega`: Volatility sensitivity
- `call_theta`, `put_theta`: Time decay

**For Custom Strikes:**
- `call_k{strike}_delta`, `put_k{strike}_delta`, etc.

**Default:** ATM strikes (current price rounded to nearest dollar)

---

### 3. Theta Decay Model ✅

**Features:**
- `time_to_exp`: Time to expiration (days)
- `theta_decay_rate_call`: Theta decay rate for calls
- `theta_decay_rate_put`: Theta decay rate for puts
- `theta_decay_1h`: Expected decay over next hour

**Model:**
- Exponential decay based on Greeks theta
- Scaled for 0DTE (1 day = 1/252 years)
- Hourly decay approximation

---

### 4. Market Microstructure ✅

**Features:**
- `ofi`: Order Flow Imbalance (buy pressure - sell pressure)
- `buy_pressure`: Estimated buy volume
- `sell_pressure`: Estimated sell volume
- `vwap`: Volume-Weighted Average Price
- `vwap_distance`: Distance from VWAP
- `price_impact`: Returns per unit volume
- `spread_proxy`: High-low range as % of close

**Calculation:**
```python
ofi = (buy_pressure - sell_pressure) / total_volume
buy_pressure = volume when returns > 0
sell_pressure = volume when returns < 0
```

---

### 5. Cross-Asset Correlations ✅

**Features:**
- `corr_spy`: Correlation with SPY (if not SPY)
- `corr_qqq`: Correlation with QQQ (if not QQQ)
- `corr_spx`: Correlation with SPX (if not SPX)
- `corr_vix`: Correlation with VIX

**Method:**
- Rolling 30-day correlation windows
- Calculated on returns
- Updates dynamically

---

### 6. Volatility Regime Classification ✅

**Regimes:**
- `calm`: VIX < 18
- `normal`: VIX 18-25
- `storm`: VIX 25-35
- `crash`: VIX > 35

**Features:**
- `vix_level`: Current VIX level
- `vol_regime`: Regime name (string)
- `vol_regime_encoded`: Regime number (0-3)

---

### 7. Market Profile/TPO Signals ✅

**Features:**
- `value_area_high`: Upper bound of value area
- `value_area_low`: Lower bound of value area
- `poc`: Point of Control (VWAP)
- `volume_density`: Volume per price range
- `distance_from_value_area`: Distance from value area (normalized)

**Method:**
- Rolling 20-day windows
- Volume-weighted price distribution
- Simplified TPO approximation

---

## 🚀 Usage

### Collect Quant Features

```bash
source venv/bin/activate

# Collect all quant features for all symbols
python collect_quant_features.py \
    --symbols SPY,QQQ,SPX \
    --start-date 2002-01-01 \
    --interval 1d
```

### Options

- `--symbols`: Comma-separated symbols (default: SPY,QQQ,SPX)
- `--start-date`: Start date (default: 2002-01-01)
- `--end-date`: End date (default: today)
- `--interval`: Data interval (default: 1d)
- `--output-dir`: Output directory (default: data/historical/enriched)
- `--skip-existing`: Skip symbols that already have enriched data

---

## 📁 Output Structure

```
data/historical/
├── SPY_1d_2002-01-01_2025-12-07.pkl        # Base OHLCV data
├── QQQ_1d_2002-01-01_2025-12-07.pkl
├── SPX_1d_2002-01-01_2025-12-07.pkl
├── VIX_daily_2002-01-01_2025-12-07.pkl
└── enriched/                                # NEW: Enriched data
    ├── SPY_enriched_2002-01-01_latest.pkl  # Base + Quant features
    ├── QQQ_enriched_2002-01-01_latest.pkl
    └── SPX_enriched_2002-01-01_latest.pkl
```

---

## 📊 Expected Feature Count

**Base OHLCV:** 7-8 columns
- open, high, low, close, volume, dividends, stock splits

**Added Quant Features:** ~50-100 columns
- IV features: 3
- Greeks (ATM): 8 (4 Greeks × 2 types)
- Theta decay: 4
- Microstructure: 7
- Regime: 3
- Correlations: 4 (if all symbols)
- Market Profile: 5
- **Total:** ~34 base features

**With Custom Strikes:** +8 per strike/type combination

---

## ⏱️ Collection Time Estimate

**Per Symbol:**
- Base data: Already collected ✅
- Feature calculation: ~5-15 minutes per symbol
- Cross-asset correlations: ~2-5 minutes
- **Total:** ~20-30 minutes for all symbols

**Total Time:** ~30-45 minutes for complete collection

---

## ✅ Validation Checklist

After collection, validate:

- [ ] IV features present (`vix`, `iv_from_vix`, `iv_0dte`)
- [ ] Greeks present (delta, gamma, vega, theta for calls/puts)
- [ ] Theta decay features present
- [ ] Microstructure features present (OFI, pressure, impact)
- [ ] Regime classification present
- [ ] Correlations present (if multiple symbols)
- [ ] Market Profile features present
- [ ] No missing values in key features
- [ ] Feature values in expected ranges

---

## 🔄 Integration with Training

After collection, enriched data can be used in:

1. **Historical Training Environment:**
   - Load enriched data instead of base OHLCV
   - Use quant features as observation space
   - Train with full feature set

2. **Feature Engineering:**
   - Combine with existing institutional features
   - Create derived features from quant features
   - Build feature pipelines

3. **Regime-Aware Training:**
   - Filter by volatility regime
   - Balance training across regimes
   - Test regime-specific performance

---

## 📝 Next Steps

1. ✅ **System Created** - Quant features collector module ready
2. ⏳ **Collect Features** - Run collection script for all symbols
3. ⏳ **Validate** - Verify all features are present and correct
4. ⏳ **Integrate** - Update training pipeline to use enriched data
5. ⏳ **Train** - Start training with full quant features

---

**Status:** ✅ **READY TO COLLECT**  
**Created:** December 7, 2025

