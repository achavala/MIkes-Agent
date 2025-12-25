#!/usr/bin/env python3
"""
Diagnose why patterns aren't being detected on Dec 17, 2025
Inspect actual market data to understand what Mike saw
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

try:
    from massive_api_client import MassiveAPIClient
    MASSIVE_AVAILABLE = True
except ImportError:
    MASSIVE_AVAILABLE = False

from mike_agent_live_safe import get_market_data

def get_data_for_time(symbol: str, date: str, end_time: str):
    """Get data up to a specific time"""
    trade_date = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M:%S")
    trade_date = pytz.timezone('America/New_York').localize(trade_date)
    
    # Get 2 days of data for context
    start_date = trade_date - pd.Timedelta(days=2)
    
    # Try to get data
    try:
        if ALPACA_AVAILABLE:
            api = tradeapi.REST(
                os.getenv('ALPACA_KEY') or os.getenv('ALPACA_API_KEY'),
                os.getenv('ALPACA_SECRET') or os.getenv('ALPACA_SECRET_KEY'),
                base_url='https://paper-api.alpaca.markets',
                api_version='v2'
            )
            
            start_str = start_date.strftime('%Y-%m-%dT%H:%M:%S-05:00')
            end_str = trade_date.strftime('%Y-%m-%dT%H:%M:%S-05:00')
            
            bars = api.get_bars(
                symbol,
                tradeapi.TimeFrame.Minute,
                start=start_str,
                end=end_str
            ).df
            
            if len(bars) > 0:
                bars.columns = [col.lower() for col in bars.columns]
                # Filter to trade date only
                bars = bars[bars.index.date == trade_date.date()]
                return bars
    except Exception as e:
        print(f"⚠️ Error: {e}")
    
    return None

def analyze_price_action(data: pd.DataFrame, trade_time: str):
    """Analyze price action to understand what patterns might be present"""
    
    trade_datetime = datetime.strptime(f"2025-12-17 {trade_time}", "%Y-%m-%d %H:%M:%S")
    trade_datetime = pytz.timezone('America/New_York').localize(trade_datetime)
    
    # Get data up to trade time
    data_before = data[data.index <= trade_datetime]
    
    if len(data_before) < 20:
        return None
    
    # Get recent price action (last 30 bars)
    recent = data_before.tail(30)
    
    # Calculate key levels
    highs = recent['high'].values
    lows = recent['low'].values
    closes = recent['close'].values
    opens = recent['open'].values
    
    current_price = closes[-1]
    current_high = highs[-1]
    current_low = lows[-1]
    current_open = opens[-1]
    
    # Calculate resistance/support
    resistance = np.max(highs[-10:])
    support = np.min(lows[-10:])
    
    # Calculate previous resistance (bars 15-20)
    prev_resistance = np.max(highs[-10:-5]) if len(highs) >= 10 else resistance
    prev_support = np.min(lows[-10:-5]) if len(lows) >= 10 else support
    
    # Price change
    price_change = (closes[-1] - closes[-20]) / closes[-20] if len(closes) >= 20 else 0
    
    # Check for gaps
    if len(recent) >= 2:
        gap_up = opens[-1] > closes[-2] * 1.005
        gap_down = opens[-1] < closes[-2] * 0.995
    else:
        gap_up = gap_down = False
    
    # Check for rejections
    rejections_at_resistance = 0
    rejections_at_support = 0
    for i in range(max(0, len(highs)-5), len(highs)):
        if highs[i] >= resistance * 0.998 and closes[i] < resistance * 0.995:
            rejections_at_resistance += 1
        if lows[i] <= support * 1.002 and closes[i] > support * 1.005:
            rejections_at_support += 1
    
    return {
        'time': trade_time,
        'current_price': current_price,
        'current_high': current_high,
        'current_low': current_low,
        'resistance': resistance,
        'support': support,
        'prev_resistance': prev_resistance,
        'prev_support': prev_support,
        'price_change_pct': price_change * 100,
        'gap_up': gap_up,
        'gap_down': gap_down,
        'rejections_at_resistance': rejections_at_resistance,
        'rejections_at_support': rejections_at_support,
        'price_above_resistance': current_high > resistance,
        'price_below_support': current_low < support,
        'recent_highs': highs[-10:].tolist(),
        'recent_lows': lows[-10:].tolist(),
        'recent_closes': closes[-10:].tolist(),
    }

def main():
    """Diagnose patterns"""
    
    print("=" * 80)
    print("🔍 DIAGNOSING DEC 17, 2025 PATTERNS")
    print("=" * 80)
    print()
    
    trades = [
        ('08:57:00', 'SPY', 682, 'call', 'PT - $680/$682'),
        ('09:08:00', 'SPY', 675, 'put', '$675 range PT, breakdown to gamma zone'),
        ('09:36:00', 'SPY', 672, 'put', 'Breakdown after 10:30, $674/$675 PT'),
        ('09:52:00', 'SPY', 670, 'put', '$670 PT'),
        ('10:40:00', 'SPY', 669, 'put', '$670 PT, $673.5 breakdown'),
    ]
    
    for trade_time, symbol, strike, option_type, reason in trades:
        print(f"\n{'='*80}")
        print(f"📊 {symbol} ${strike:.0f} {option_type.upper()} @ {trade_time}")
        print(f"{'='*80}")
        print(f"Mike's Reason: {reason}")
        print()
        
        # Get data
        data = get_data_for_time(symbol, "2025-12-17", trade_time)
        
        if data is None or len(data) < 20:
            print(f"❌ No data available")
            continue
        
        # Analyze
        analysis = analyze_price_action(data, trade_time)
        
        if analysis:
            print(f"Price Action Analysis:")
            print(f"  Current Price: ${analysis['current_price']:.2f}")
            print(f"  Current High: ${analysis['current_high']:.2f}")
            print(f"  Current Low: ${analysis['current_low']:.2f}")
            print(f"  Resistance: ${analysis['resistance']:.2f}")
            print(f"  Support: ${analysis['support']:.2f}")
            print(f"  Previous Resistance: ${analysis['prev_resistance']:.2f}")
            print(f"  Previous Support: ${analysis['prev_support']:.2f}")
            print(f"  Price Change (20 bars): {analysis['price_change_pct']:.2f}%")
            print(f"  Gap Up: {analysis['gap_up']}")
            print(f"  Gap Down: {analysis['gap_down']}")
            print(f"  Rejections at Resistance: {analysis['rejections_at_resistance']}")
            print(f"  Rejections at Support: {analysis['rejections_at_support']}")
            print(f"  Price Above Resistance: {analysis['price_above_resistance']}")
            print(f"  Price Below Support: {analysis['price_below_support']}")
            print()
            
            # Check what patterns should be detected
            print(f"Pattern Checks:")
            
            # False breakout check
            if analysis['price_above_resistance'] and analysis['current_price'] < analysis['resistance']:
                print(f"  ✅ FALSE BREAKOUT: Price broke above ${analysis['resistance']:.2f} but closed below")
            elif analysis['price_below_support'] and analysis['current_price'] > analysis['support']:
                print(f"  ✅ FALSE BREAKOUT: Price broke below ${analysis['support']:.2f} but closed above")
            else:
                print(f"  ❌ No false breakout")
            
            # Gap fill check
            if analysis['gap_up']:
                print(f"  ✅ GAP UP DETECTED: Potential gap fill down")
            elif analysis['gap_down']:
                print(f"  ✅ GAP DOWN DETECTED: Potential gap fill up")
            else:
                print(f"  ❌ No gap detected")
            
            # Rejection check
            if analysis['rejections_at_resistance'] >= 2:
                print(f"  ✅ REJECTION PATTERN: {analysis['rejections_at_resistance']} rejections at resistance ${analysis['resistance']:.2f}")
            elif analysis['rejections_at_support'] >= 2:
                print(f"  ✅ REJECTION PATTERN: {analysis['rejections_at_support']} rejections at support ${analysis['support']:.2f}")
            else:
                print(f"  ❌ No rejection pattern")
            
            # Trendline break check
            if analysis['current_high'] > analysis['prev_resistance'] * 1.003:
                print(f"  ✅ TRENDLINE BREAK: Price broke above previous resistance ${analysis['prev_resistance']:.2f}")
            elif analysis['current_low'] < analysis['prev_support'] * 0.997:
                print(f"  ✅ TRENDLINE BREAK: Price broke below previous support ${analysis['prev_support']:.2f}")
            else:
                print(f"  ❌ No trendline break")
            
            print()
            print(f"Recent Price Action (last 10 bars):")
            print(f"  Highs: {[f'${h:.2f}' for h in analysis['recent_highs']]}")
            print(f"  Lows: {[f'${l:.2f}' for l in analysis['recent_lows']]}")
            print(f"  Closes: {[f'${c:.2f}' for c in analysis['recent_closes']]}")

if __name__ == "__main__":
    main()





