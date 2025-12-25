"""
Observation preparation for 10-feature model (OHLCV + VIX + 4 Greeks)
Matches the trained historical model observation space
"""

import numpy as np
import pandas as pd
from typing import Optional

def prepare_observation_10_features(data: pd.DataFrame, risk_mgr, symbol: str = 'SPY', greeks_calc=None) -> np.ndarray:
    """
    Prepare observation for 10-feature model (matches historical training)
    
    Features:
    1. Open (normalized %)
    2. High (normalized %)
    3. Low (normalized %)
    4. Close (normalized %)
    5. Volume (normalized)
    6. VIX (normalized)
    7. Delta (Greeks)
    8. Gamma (Greeks)
    9. Theta (Greeks)
    10. Vega (Greeks)
    
    Returns: (20, 10) numpy array
    """
    LOOKBACK = 20
    
    # Pad if needed
    if len(data) < LOOKBACK:
        padding = pd.concat([data.iloc[[-1]]] * (LOOKBACK - len(data)))
        data = pd.concat([padding, data])
    
    recent = data.tail(LOOKBACK).copy()
    
    # Handle column name variations
    if 'Close' in recent.columns:
        recent = recent.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    elif 'close' not in recent.columns:
        # Try to find close column
        for col in recent.columns:
            if col.lower() in ['close', 'c']:
                recent = recent.rename(columns={col: 'close'})
    
    # Extract OHLCV
    closes = recent['close'].astype(float).values
    highs  = recent['high'].astype(float).values
    lows   = recent['low'].astype(float).values
    opens  = recent['open'].astype(float).values
    vols   = recent['volume'].astype(float).values
    
    # Base price for normalization
    base = float(closes[0]) if float(closes[0]) != 0 else 1.0
    
    # Normalize OHLC (% change from base)
    o = (opens  - base) / base * 100.0
    h = (highs  - base) / base * 100.0
    l = (lows   - base) / base * 100.0
    c = (closes - base) / base * 100.0
    
    # Normalized volume
    maxv = vols.max() if vols.max() > 0 else 1.0
    v = vols / maxv
    
    # VIX (constant across window)
    vix_value = risk_mgr.get_current_vix() if risk_mgr else 20.0
    vix_norm = np.full(LOOKBACK, (vix_value / 50.0) if vix_value else 0.4, dtype=np.float32)
    
    # Greeks (delta/gamma/theta/vega) - constant across window if no position
    greeks = np.zeros((LOOKBACK, 4), dtype=np.float32)
    
    # Try to get Greeks if we have a position and calculator
    position = None
    if risk_mgr and hasattr(risk_mgr, 'open_positions') and risk_mgr.open_positions:
        first_pos = list(risk_mgr.open_positions.values())[0]
        position = {
            "strike": first_pos.get('strike', closes[-1]),
            "option_type": first_pos.get('type', 'call')
        }
    
    if position and greeks_calc:
        try:
            g = greeks_calc.calculate_greeks(
                S=closes[-1],
                K=position["strike"],
                T=(1.0 / (252 * 6.5)),  # 0DTE approximation
                sigma=(vix_value / 100.0) * 1.3 if vix_value else 0.20,
                option_type=position["option_type"]
            )
            # Fill all bars with same Greeks (constant for window)
            greeks[:] = [
                float(np.clip(g.get("delta", 0), -1, 1)),
                float(np.tanh(g.get("gamma", 0) * 100)),
                float(np.tanh(g.get("theta", 0) / 10)),
                float(np.tanh(g.get("vega", 0) / 10)),
            ]
        except Exception:
            pass  # Keep zeros
    
    # FINAL OBSERVATION (20 × 10)
    obs = np.column_stack([
        o, h, l, c, v,                    # 5 features: OHLCV
        vix_norm,                         # 1 feature: VIX
        greeks[:,0],                      # 1 feature: Delta
        greeks[:,1],                      # 1 feature: Gamma
        greeks[:,2],                      # 1 feature: Theta
        greeks[:,3],                      # 1 feature: Vega
    ]).astype(np.float32)
    
    # Ensure shape is exactly (20, 10)
    if obs.shape != (20, 10):
        if obs.shape[1] > 10:
            obs = obs[:, :10]
        elif obs.shape[1] < 10:
            # Pad with zeros if somehow less
            padding = np.zeros((20, 10 - obs.shape[1]), dtype=np.float32)
            obs = np.column_stack([obs, padding])
    
    return np.clip(obs, -10.0, 10.0)





