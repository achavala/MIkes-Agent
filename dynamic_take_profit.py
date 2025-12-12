"""
Dynamic Take-Profit System
Adapts TP1, TP2, TP3 based on:
- ATR (Average True Range)
- TrendStrength
- VIX regime
- Ticker personality profiles
- Meta-policy confidence
- Option IV Rank
- Directional momentum
"""

from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np


# ==================== TICKER PERSONALITY PROFILES ====================
TICKER_PERSONALITY = {
    # High volatility, strong trenders
    "NVDA": 1.4,   # Strong Trender
    "TSLA": 1.3,   # Volatile Explosive
    "MSTR": 1.5,   # BTC correlation
    "PLTR": 1.15,  # Breakout Driven
    
    # Trend followers
    "META": 1.2,   # Trend Follower
    "AMZN": 1.2,   # Smooth Trender
    "AMD": 1.1,    # Sympathy mover
    
    # Slow movers
    "AAPL": 0.8,   # Slow mover
    "GOOG": 0.8,   # Slow mover
    "MSFT": 0.85,  # Controlled
    "AVGO": 0.8,   # Low liquidity
    "INTC": 0.75,  # Very slow
    
    # Default for unknown tickers
    "DEFAULT": 1.0
}


def get_ticker_personality_factor(ticker: str) -> float:
    """Get personality factor for a ticker"""
    # Extract underlying from option symbol if needed
    underlying = ticker[:3] if len(ticker) > 3 and ticker[:3] in TICKER_PERSONALITY else ticker
    return TICKER_PERSONALITY.get(underlying, TICKER_PERSONALITY["DEFAULT"])


# ==================== ATR CALCULATION ====================
def calculate_atr(hist_data: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate Average True Range (ATR) from historical data
    
    Args:
        hist_data: DataFrame with OHLC columns
        period: Period for ATR calculation (default 14)
    
    Returns:
        ATR value as float
    """
    if hist_data.empty or len(hist_data) < period + 1:
        return 0.0
    
    try:
        high = hist_data['High'].values
        low = hist_data['Low'].values
        close = hist_data['Close'].values
        
        # Calculate True Range
        tr_list = []
        for i in range(1, len(hist_data)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr_list.append(max(tr1, tr2, tr3))
        
        if len(tr_list) < period:
            return np.mean(tr_list) if tr_list else 0.0
        
        # Calculate ATR as Simple Moving Average of TR
        atr = np.mean(tr_list[-period:])
        return float(atr)
    except Exception:
        return 0.0


def get_atr_factor(hist_data: pd.DataFrame, ticker: str = "SPY") -> float:
    """
    Get ATR-based adjustment factor
    
    Args:
        hist_data: Historical price data
        ticker: Ticker symbol (for reference)
    
    Returns:
        ATR factor (0.85 to 1.35)
    """
    atr = calculate_atr(hist_data)
    if atr == 0.0:
        return 1.0  # Default neutral factor
    
    # Calculate ATR percentiles (simplified - using recent data)
    recent_atrs = []
    for i in range(min(100, len(hist_data) - 14), len(hist_data)):
        if i >= 14:
            window = hist_data.iloc[i-14:i+1]
            window_atr = calculate_atr(window)
            if window_atr > 0:
                recent_atrs.append(window_atr)
    
    if not recent_atrs:
        return 1.0
    
    atr_50th = np.percentile(recent_atrs, 50)
    atr_70th = np.percentile(recent_atrs, 70)
    atr_90th = np.percentile(recent_atrs, 90)
    
    # Assign factor based on current ATR position
    if atr > atr_90th:
        return 1.35  # High ATR → wider TPs
    elif atr > atr_70th:
        return 1.20
    elif atr > atr_50th:
        return 1.00  # Neutral
    else:
        return 0.85  # Low ATR → tighter TPs


# ==================== TREND STRENGTH EXTRACTION ====================
def extract_trend_strength(hist_data: pd.DataFrame, period: int = 20) -> float:
    """
    Extract trend strength from historical data
    Uses RSI, ADX, and momentum indicators
    
    Args:
        hist_data: DataFrame with OHLC columns
        period: Period for calculations
    
    Returns:
        TrendStrength value (0.0 to 1.0)
    """
    if hist_data.empty or len(hist_data) < period:
        return 0.5  # Default neutral
    
    try:
        close = hist_data['Close'].values
        high = hist_data['High'].values
        low = hist_data['Low'].values
        
        # Simple momentum-based trend strength
        if len(close) < period:
            return 0.5
        
        # Calculate rate of change
        roc = (close[-1] - close[-period]) / close[-period]
        
        # Calculate RSI (simplified)
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else np.mean(gains) if len(gains) > 0 else 0.01
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else np.mean(losses) if len(losses) > 0 else 0.01
        
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            rsi = 100 - (100 / (1 + rs))
        
        # Calculate ADX-like indicator (simplified)
        tr_sum = 0
        plus_dm_sum = 0
        minus_dm_sum = 0
        
        for i in range(1, min(period + 1, len(hist_data))):
            idx = len(hist_data) - i
            if idx > 0:
                tr = max(high[idx] - low[idx], 
                        abs(high[idx] - close[idx-1]), 
                        abs(low[idx] - close[idx-1]))
                tr_sum += tr
                
                plus_dm = max(0, high[idx] - high[idx-1]) if high[idx] > high[idx-1] else 0
                minus_dm = max(0, low[idx-1] - low[idx]) if low[idx-1] > low[idx] else 0
                
                plus_dm_sum += plus_dm
                minus_dm_sum += minus_dm
        
        if tr_sum > 0:
            plus_di = 100 * (plus_dm_sum / tr_sum)
            minus_di = 100 * (minus_dm_sum / tr_sum)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
        else:
            dx = 0
        
        # Combine indicators into trend strength (0.0 to 1.0)
        # RSI normalized (0-100 → 0-1)
        rsi_norm = rsi / 100.0
        
        # DX normalized (0-100 → 0-1)
        dx_norm = dx / 100.0
        
        # Momentum strength (absolute ROC normalized)
        momentum_strength = min(1.0, abs(roc) * 10)  # Scale ROC
        
        # Weighted combination
        trend_strength = (rsi_norm * 0.3 + dx_norm * 0.4 + momentum_strength * 0.3)
        
        # Ensure range [0.0, 1.0]
        return max(0.0, min(1.0, trend_strength))
    
    except Exception:
        return 0.5  # Default neutral


def get_trend_strength_factor(trend_strength: float) -> float:
    """
    Get trend strength adjustment factor
    
    Args:
        trend_strength: TrendStrength value (0.0 to 1.0)
    
    Returns:
        Trend factor (0.90 to 1.40)
    """
    if trend_strength > 0.7:
        return 1.40  # Strong trend → wider TPs
    elif trend_strength > 0.5:
        return 1.20
    else:
        return 0.90  # Weak trend → tighter TPs


# ==================== VIX FACTOR ====================
def get_vix_factor(vix: float) -> float:
    """
    Get VIX-based adjustment factor
    
    Args:
        vix: Current VIX level
    
    Returns:
        VIX factor (0.90 to 1.30)
    """
    if vix >= 25:
        return 1.30  # High VIX → wider TPs
    elif vix >= 18:
        return 1.10
    else:
        return 0.90  # Low VIX → tighter TPs


# ==================== CONFIDENCE FACTOR ====================
def get_confidence_factor(confidence: Optional[float] = None, rl_action_raw: Optional[float] = None) -> float:
    """
    Get confidence-based adjustment factor
    
    Args:
        confidence: Explicit confidence value (0.0 to 1.0)
        rl_action_raw: RL model raw output (absolute value used as confidence proxy)
    
    Returns:
        Confidence factor (0.90 to 1.30)
    """
    if confidence is not None:
        conf_value = float(confidence)
    elif rl_action_raw is not None:
        # Use absolute value of RL action as confidence proxy
        conf_value = min(1.0, abs(float(rl_action_raw)) * 2.0)  # Scale to 0-1
    else:
        return 1.0  # Default neutral
    
    if conf_value > 0.60:
        return 1.30  # High confidence → wider TPs
    elif conf_value > 0.40:
        return 1.10
    else:
        return 0.90  # Low confidence → tighter TPs


# ==================== DYNAMIC TP CALCULATION ====================
def compute_dynamic_tp_factors(
    hist_data: pd.DataFrame,
    ticker: str,
    vix: float,
    confidence: Optional[float] = None,
    rl_action_raw: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute all adjustment factors for dynamic take-profits
    
    Args:
        hist_data: Historical price data
        ticker: Ticker symbol
        vix: Current VIX level
        confidence: Explicit confidence value (optional)
        rl_action_raw: RL model raw output (optional, used as confidence proxy)
    
    Returns:
        Dictionary of factors: {'atr', 'trend', 'vix', 'personality', 'confidence', 'total'}
    """
    # Extract ticker from option symbol if needed
    underlying = ticker[:3] if len(ticker) > 10 and ticker[:3] in TICKER_PERSONALITY else ticker
    
    # Calculate ATR factor
    atr_factor = get_atr_factor(hist_data, underlying)
    
    # Extract and calculate trend strength factor
    trend_strength = extract_trend_strength(hist_data)
    trend_factor = get_trend_strength_factor(trend_strength)
    
    # VIX factor
    vix_factor = get_vix_factor(vix)
    
    # Personality factor
    personality_factor = get_ticker_personality_factor(underlying)
    
    # Confidence factor
    confidence_factor = get_confidence_factor(confidence, rl_action_raw)
    
    # Total adjustment (multiplicative)
    total_factor = atr_factor * trend_factor * vix_factor * personality_factor * confidence_factor
    
    return {
        'atr': atr_factor,
        'trend': trend_factor,
        'vix': vix_factor,
        'personality': personality_factor,
        'confidence': confidence_factor,
        'total': total_factor,
        'trend_strength': trend_strength
    }


def compute_dynamic_takeprofits(
    base_tp1: float,
    base_tp2: float,
    base_tp3: float,
    adjustment_factors: Dict[str, float]
) -> Tuple[float, float, float]:
    """
    Compute dynamic take-profit levels
    
    Args:
        base_tp1: Base TP1 level (e.g., 0.40 = 40%)
        base_tp2: Base TP2 level (e.g., 0.80 = 80%)
        base_tp3: Base TP3 level (e.g., 1.50 = 150%)
        adjustment_factors: Dictionary from compute_dynamic_tp_factors()
    
    Returns:
        Tuple of (dynamic_tp1, dynamic_tp2, dynamic_tp3) with caps applied
    """
    total_factor = adjustment_factors['total']
    
    # Calculate dynamic TPs
    dynamic_tp1 = base_tp1 * total_factor
    dynamic_tp2 = base_tp2 * total_factor
    dynamic_tp3 = base_tp3 * total_factor
    
    # Apply caps with overflow protection
    # TP1: 20% to 80%
    dynamic_tp1 = max(0.20, min(dynamic_tp1, 0.80))
    
    # TP2: 40% to 120%
    dynamic_tp2 = max(0.40, min(dynamic_tp2, 1.20))
    
    # TP3: 80% to 200% (with overflow protection for volatility explosions)
    if dynamic_tp3 > 2.0:
        # Log warning if TP3 would exceed 200% (volatility explosion protection)
        dynamic_tp3 = 2.0
        # Note: Warning will be logged by caller if needed
    
    dynamic_tp3 = max(0.80, min(dynamic_tp3, 2.00))
    
    return (dynamic_tp1, dynamic_tp2, dynamic_tp3)

