#!/usr/bin/env python3
"""
Quick Gap Detection Test
Tests gap detection logic on specific dates
"""
from gap_detection import detect_overnight_gap, get_gap_based_action
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz

def test_gap_on_date(symbol: str, date: str):
    """Test gap detection on a specific date"""
    print(f"\n{'='*70}")
    print(f"Testing Gap Detection: {symbol} on {date}")
    print(f"{'='*70}\n")
    
    # Get historical data
    target_date = pd.to_datetime(date)
    start_date = target_date - timedelta(days=5)
    end_date = target_date + timedelta(days=1)
    
    symbol_map = {'SPY': 'SPY', 'QQQ': 'QQQ', 'SPX': '^SPX'}
    yf_symbol = symbol_map.get(symbol, symbol)
    
    ticker = yf.Ticker(yf_symbol)
    hist = ticker.history(start=start_date.strftime('%Y-%m-%d'),
                         end=end_date.strftime('%Y-%m-%d'),
                         interval='1m')
    
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    
    # Filter to target date
    hist = hist[hist.index.date == target_date.date()]
    
    if len(hist) == 0:
        print(f"❌ No data for {symbol} on {date}")
        return
    
    # Get first price
    current_price = float(hist['Close'].iloc[0])
    
    # Get extended history for gap detection
    hist_extended = ticker.history(start=start_date.strftime('%Y-%m-%d'),
                                   end=end_date.strftime('%Y-%m-%d'),
                                   interval='1d')
    if isinstance(hist_extended.columns, pd.MultiIndex):
        hist_extended.columns = hist_extended.columns.get_level_values(0)
    
    # Test gap detection
    class MockRM:
        def log(self, msg, level="INFO"):
            print(f"  [{level}] {msg}")
    
    gap_data = detect_overnight_gap(symbol, current_price, hist_extended, MockRM())
    
    if gap_data and gap_data.get('detected'):
        print(f"✅ GAP DETECTED!")
        print(f"   Direction: {gap_data['direction'].upper()}")
        print(f"   Gap: ${gap_data['gap_points']:.2f} ({gap_data['gap_pct']:+.2f}%)")
        print(f"   Prev Close: ${gap_data['prev_close']:.2f}")
        print(f"   Open: ${gap_data['open_price']:.2f}")
        print(f"   Bias: {gap_data['bias'].upper()}")
        print(f"   Strength: {gap_data['strength'].upper()}")
        
        # Get action
        gap_action = get_gap_based_action(gap_data, current_price, 935)
        if gap_action:
            action_desc = 'BUY CALL' if gap_action == 1 else 'BUY PUT'
            print(f"\n🎯 ACTION: {action_desc}")
    else:
        print("ℹ️  No significant gap detected")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        date = sys.argv[1]
        symbol = sys.argv[2] if len(sys.argv) > 2 else 'SPY'
    else:
        # Test recent dates
        dates = [
            '2025-12-05',  # Today (if market was open)
            '2025-12-04',
            '2025-12-03',
            '2025-12-02',
            '2025-12-01',
        ]
        
        for date in dates:
            test_gap_on_date('SPY', date)

