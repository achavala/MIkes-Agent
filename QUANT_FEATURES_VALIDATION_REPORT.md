# Quant Features Validation Report

**Date:** 2025-12-07 09:01:34

## Validation Results

### IV

- **Status:** ✅ PASS
- **Features:** 8
- **Feature List:**
  - dividends
  - vix
  - iv_0dte
  - iv_from_vix
  - vix_level
  - rv_iv_spread
  - rv_iv_ratio
  - corr_vix

### GREEKS

- **Status:** ✅ PASS
- **Features:** 11
- **Feature List:**
  - call_delta
  - put_delta
  - call_gamma
  - put_gamma
  - call_vega
  - put_vega
  - call_theta
  - put_theta
  - theta_decay_rate_call
  - theta_decay_rate_put
  - theta_decay_1h

### THETA_DECAY

- **Status:** ✅ PASS
- **Features:** 3
- **Feature List:**
  - theta_decay_rate_call
  - theta_decay_rate_put
  - theta_decay_1h

### MICROSTRUCTURE

- **Status:** ✅ PASS
- **Features:** 8
- **Feature List:**
  - buy_pressure
  - sell_pressure
  - ofi
  - vwap
  - vwap_distance
  - price_impact
  - spread_proxy
  - rv_iv_spread

### CORRELATIONS

- **Status:** ✅ PASS
- **Features:** 3
- **Feature List:**
  - corr_qqq
  - corr_spx
  - corr_vix

### REGIME

- **Status:** ✅ PASS
- **Features:** 11
- **Feature List:**
  - vol_regime
  - vol_regime_encoded
  - regime_change
  - time_in_regime
  - regime_to_calm
  - regime_to_normal
  - regime_to_storm
  - regime_to_crash
  - regime_transition_direction
  - regime_stability
  - regime_change_probability

### MARKET_PROFILE

- **Status:** ✅ PASS
- **Features:** 5
- **Feature List:**
  - volume_density
  - value_area_high
  - value_area_low
  - poc
  - distance_from_value_area

