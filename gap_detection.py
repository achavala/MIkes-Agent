"""
Gap Detection Module - Mike's Strategy Foundation
Detects overnight gaps and provides trade bias for first 45-60 minutes
"""
from datetime import datetime
import pytz
import pandas as pd
import yfinance as yf
from typing import Optional, Dict, Tuple


def detect_overnight_gap(symbol: str, current_price: float, hist: pd.DataFrame, 
                         risk_mgr=None) -> Optional[Dict]:
    """
    Detect overnight gap at market open (9:35 AM ET)
    
    Logic:
    - prev_close = yesterday's close
    - open_price = today's 9:30-9:40 fair price (first 10-min VWAP or 9:35 price)
    - gap_pct = (open_price - prev_close) / prev_close
    - gap_points = open_price - prev_close
    
    Threshold: abs(gap_pct) > 0.35% OR abs(gap_points) > 2.50
    
    Returns:
    {
        'detected': bool,
        'gap_pct': float,
        'gap_points': float,
        'direction': 'up' or 'down',
        'prev_close': float,
        'open_price': float,
        'bias': 'fade' or 'follow',
        'strength': 'weak' or 'strong'
    }
    """
    try:
        est = pytz.timezone('US/Eastern')
        now = datetime.now(est)
        current_time = now.hour * 100 + now.minute  # HHMM format
        
        # Only detect gap during market open (9:30 AM - 10:35 AM ET)
        if current_time < 930 or current_time > 1035:
            return None
        
        # Get yesterday's close - always use daily data for accuracy
        ticker = yf.Ticker(symbol)
        hist_daily = ticker.history(period="5d", interval="1d")
        if isinstance(hist_daily.columns, pd.MultiIndex):
            hist_daily.columns = hist_daily.columns.get_level_values(0)
        
        if len(hist_daily) < 2:
            return None
        
        # Get yesterday's close (second to last day in daily data)
        # Daily data is sorted by date, so -2 is previous trading day
        prev_close = float(hist_daily['Close'].iloc[-2])
        
        # Get today's open price (9:30-9:40 fair price)
        # Use first 10 minutes VWAP or 9:35 price
        today_bars = hist[hist.index.date == now.date()]
        
        if len(today_bars) == 0:
            return None
        
        # Calculate VWAP for first 10 minutes (9:30-9:40)
        early_bars = today_bars[today_bars.index.hour == 9]
        early_bars = early_bars[early_bars.index.minute <= 40]
        
        if len(early_bars) > 0:
            # Use VWAP of first 10 minutes
            typical_price = (early_bars['High'] + early_bars['Low'] + early_bars['Close']) / 3
            vwap = (typical_price * early_bars['Volume']).sum() / early_bars['Volume'].sum()
            open_price = float(vwap)
        else:
            # Fallback: use current price if we're early in the day
            open_price = current_price
        
        # Calculate gap
        gap_points = open_price - prev_close
        gap_pct = (gap_points / prev_close) * 100 if prev_close > 0 else 0
        
        # Debug logging
        if risk_mgr:
            risk_mgr.log(f"Gap Calculation: Prev Close=${prev_close:.2f}, Open=${open_price:.2f}, Gap=${gap_points:.2f} ({gap_pct:+.2f}%)", "DEBUG")
        
        # Check threshold: abs(gap_pct) > 0.35% OR abs(gap_points) > 2.50
        if abs(gap_pct) < 0.35 and abs(gap_points) < 2.50:
            if risk_mgr:
                risk_mgr.log(f"Gap below threshold: {abs(gap_pct):.2f}% < 0.35% AND ${abs(gap_points):.2f} < $2.50", "DEBUG")
            return None
        
        # Determine direction
        direction = 'up' if gap_points > 0 else 'down'
        
        # Determine bias based on gap size (Mike's logic)
        # Fade gaps > 0.6%, follow gaps < 0.4% in first 30 min
        if abs(gap_pct) > 0.6:
            bias = 'fade'  # Large gap → fade
        elif abs(gap_pct) < 0.4:
            bias = 'follow'  # Small gap → follow
        else:
            # Medium gap (0.4% - 0.6%) → determine by early strength
            # Check if price is holding (strong) or fading (weak)
            if len(today_bars) >= 5:
                recent_close = float(today_bars['Close'].iloc[-1])
                if direction == 'up':
                    strength = 'strong' if recent_close >= open_price * 0.998 else 'weak'
                else:
                    strength = 'strong' if recent_close <= open_price * 1.002 else 'weak'
                bias = 'fade' if strength == 'weak' else 'follow'
            else:
                bias = 'fade'  # Default to fade if uncertain
        
        # Check early strength
        if len(today_bars) >= 5:
            recent_close = float(today_bars['Close'].iloc[-1])
            if direction == 'up':
                strength = 'strong' if recent_close >= open_price * 0.998 else 'weak'
            else:
                strength = 'strong' if recent_close <= open_price * 1.002 else 'weak'
        else:
            strength = 'unknown'
        
        result = {
            'detected': True,
            'gap_pct': gap_pct,
            'gap_points': gap_points,
            'direction': direction,
            'prev_close': prev_close,
            'open_price': open_price,
            'bias': bias,
            'strength': strength
        }
        
        if risk_mgr:
            risk_mgr.log(f"📊 GAP DETECTED: {direction.upper()} ${gap_points:.2f} ({gap_pct:+.2f}%) | Prev Close: ${prev_close:.2f} | Open: ${open_price:.2f} | Bias: {bias.upper()} | Strength: {strength.upper()}", "INFO")
        
        return result
        
    except Exception as e:
        if risk_mgr:
            risk_mgr.log(f"Error detecting gap: {e}", "WARNING")
        return None


def get_gap_based_action(gap_data: Dict, current_price: float, current_time: int) -> Optional[int]:
    """
    Get action based on gap detection (Mike's logic)
    
    Overrides RL signal for first 45-60 minutes (until 10:35 AM ET)
    
    Logic:
    - Gap up + weak open → BUY PUT (fade)
    - Gap up + strong → BUY CALL (follow)
    - Gap down + weak → BUY PUT (follow)
    - Gap down + bouncing → BUY CALL (fade)
    
    Returns:
    - Action 1 (BUY CALL) or Action 2 (BUY PUT) or None (no override)
    """
    if not gap_data or not gap_data.get('detected'):
        return None
    
    # Only apply gap logic during first 60 minutes (9:30 - 10:35 AM ET)
    if current_time < 930 or current_time > 1035:
        return None
    
    direction = gap_data['direction']
    bias = gap_data['bias']
    strength = gap_data.get('strength', 'unknown')
    
    # Mike's gap-based logic
    if direction == 'up':  # Gap up
        if bias == 'fade' or strength == 'weak':
            # Gap up + weak → fade (buy puts)
            return 2  # BUY PUT
        elif bias == 'follow' or strength == 'strong':
            # Gap up + strong → follow (buy calls)
            return 1  # BUY CALL
    else:  # Gap down
        if bias == 'follow' or strength == 'weak':
            # Gap down + weak → follow (buy puts)
            return 2  # BUY PUT
        elif bias == 'fade' or strength == 'strong':
            # Gap down + bouncing → fade (buy calls)
            return 1  # BUY CALL
    
    return None

