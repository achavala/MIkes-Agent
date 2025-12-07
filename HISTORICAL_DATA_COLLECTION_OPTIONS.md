# 📊 Historical Data Collection - Options & Recommendations

## ⚠️ Important Discovery

**yfinance limitation:** Only allows **8 days of 1-minute data per request**

This means:
- For 20+ years (2002-2025): **~850 API calls needed**
- At 2-3 seconds per call: **~30-45 minutes minimum**
- Realistically: **2-4 hours** (with rate limiting and errors)

---

## ✅ Recommended Approach

### Option 1: Use Daily Data (Fastest - Recommended)

**Best for training on long historical periods:**

```bash
# Collect daily data (much faster - single API call)
python collect_historical_data.py \
    --symbols SPY,QQQ \
    --start-date 2002-01-01 \
    --interval 1d
```

**Benefits:**
- ✅ Single API call per symbol (seconds, not hours)
- ✅ 20+ years of data available
- ✅ Sufficient for RL training (daily bars work great)
- ✅ All market regimes included

**For training:** Daily data is actually **better** for learning long-term patterns and market regimes!

---

### Option 2: Recent Minute Data Only

**For more granular training on recent data:**

```bash
# Collect minute data for recent period (last 2-3 years)
python collect_historical_data.py \
    --symbols SPY,QQQ \
    --start-date 2022-01-01 \
    --interval 1m
```

**Benefits:**
- ✅ Recent minute data available
- ✅ More granular for intraday patterns
- ✅ Takes ~30-60 minutes (not hours)

---

### Option 3: Hybrid Approach (Best of Both)

**Use daily data for long history, minute data for recent:**

1. **Daily data (2002-present):**
   ```bash
   python collect_historical_data.py --symbols SPY,QQQ --start-date 2002-01-01 --interval 1d
   ```

2. **Minute data (last 2 years):**
   ```bash
   python collect_historical_data.py --symbols SPY,QQQ --start-date 2023-01-01 --interval 1m
   ```

**Training strategy:**
- Train on daily data for regime understanding (2002-2025)
- Fine-tune on minute data for intraday patterns (2023-2025)

---

## 📊 Data Availability

### What's Available:

| Interval | Years Available | API Calls Needed | Time |
|----------|----------------|------------------|------|
| **Daily (1d)** | 2002-present | 1 per symbol | ~30 seconds |
| **Hourly (1h)** | ~2 years | ~730 per symbol | ~30-60 min |
| **Minute (1m)** | ~8 days per request | ~850+ per symbol | 2-4 hours |

### Recommendation:

**For your goal (train on 2002-present with all regimes):**

✅ **Use daily data** - it's perfect for:
- Learning market regimes (calm, normal, storm, crash)
- Understanding long-term patterns
- Training on all worst days (2008, 2020, 2022)
- Much faster collection (seconds, not hours)

Then optionally add minute data for recent years if needed.

---

## 🚀 Quick Start

### Recommended: Daily Data Collection

```bash
source venv/bin/activate

# Collect daily data (2002-present) - FAST!
python collect_historical_data.py \
    --symbols SPY,QQQ \
    --start-date 2002-01-01 \
    --interval 1d
```

**Time:** ~30 seconds per symbol ✅

### Then Train:

```bash
python train_historical_model.py \
    --symbols SPY,QQQ \
    --start-date 2002-01-01 \
    --timesteps 5000000 \
    --use-greeks \
    --regime-balanced
```

---

## 💡 Why Daily Data Works Great

For RL training on market regimes:
- ✅ Daily bars capture regime changes perfectly
- ✅ VIX changes are daily anyway
- ✅ All worst days (2008, 2020, 2022) are included
- ✅ Much faster to collect and train
- ✅ Standard practice in quantitative finance

**Minute data** is mainly useful for:
- Intraday patterns (gaps, reversals)
- Execution timing
- High-frequency strategies

For your goal (learn from all regimes since 2002), **daily data is perfect!**

---

## ⏭️ Next Steps

1. **Collect daily data** (recommended):
   ```bash
   python collect_historical_data.py --symbols SPY,QQQ --start-date 2002-01-01 --interval 1d
   ```

2. **Verify data:**
   ```python
   from historical_training_system import HistoricalDataCollector
   collector = HistoricalDataCollector()
   data = collector.get_historical_data('SPY', '2002-01-01', None, '1d', use_cache=True)
   print(f"SPY daily bars: {len(data):,}")
   ```

3. **Start training:**
   ```bash
   python train_historical_model.py --symbols SPY,QQQ --start-date 2002-01-01 --timesteps 1000000
   ```

---

**Recommendation: Start with daily data - it's perfect for your goal!** 🎯

