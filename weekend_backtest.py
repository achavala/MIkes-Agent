#!/usr/bin/env python3
"""
Weekend Backtesting Environment
Tests gap detection and trading logic on historical data
Works during weekends when market is closed
"""
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
import pytz
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional, Dict, Any

# Set environment variables BEFORE importing torch/gym
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['OMP_NUM_THREADS'] = '1'

warnings.filterwarnings("ignore")

# Import gap detection
try:
    from gap_detection import detect_overnight_gap, get_gap_based_action
    GAP_DETECTION_AVAILABLE = True
except ImportError:
    GAP_DETECTION_AVAILABLE = False
    print("Warning: gap_detection module not found")

# Import helper functions from main agent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class MockRiskManager:
    """Mock risk manager for backtesting"""
    def __init__(self):
        self.open_positions = {}
        self.logs = []
        self.daily_pnl = 0.0
        self.start_of_day_equity = 100000.0
        self.peak_equity = 100000.0
        
    def log(self, msg: str, level: str = "INFO"):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"{timestamp} | [{level}] {msg}"
        self.logs.append(log_msg)
        print(log_msg)
    
    def get_equity(self, api=None):
        return self.start_of_day_equity
    
    def get_current_vix(self):
        # Mock VIX for backtesting
        return 16.0
    
    def get_vol_regime(self, vix):
        if vix < 18:
            return 'calm'
        elif vix < 25:
            return 'normal'
        elif vix < 35:
            return 'storm'
        else:
            return 'crash'
    
    def get_vol_params(self, regime):
        regimes = {
            'calm': {'risk': 0.10, 'max_pct': 0.30},
            'normal': {'risk': 0.07, 'max_pct': 0.25},
            'storm': {'risk': 0.05, 'max_pct': 0.20},
            'crash': {'risk': 0.03, 'max_pct': 0.15}
        }
        return regimes.get(regime, regimes['normal'])


def get_historical_data(symbol: str, date: str, days_before: int = 5) -> pd.DataFrame:
    """
    Get historical minute data for a specific date
    
    Args:
        symbol: Trading symbol (SPY, QQQ, SPX)
        date: Date string in format 'YYYY-MM-DD'
        days_before: How many days before to fetch
    
    Returns:
        DataFrame with minute bars
    """
    try:
        # Convert date to datetime
        target_date = pd.to_datetime(date)
        
        # Get data for the period
        start_date = target_date - timedelta(days=days_before)
        end_date = target_date + timedelta(days=1)
        
        # Map symbol for yfinance
        symbol_map = {
            'SPY': 'SPY',
            'QQQ': 'QQQ',
            'SPX': '^SPX'
        }
        yf_symbol = symbol_map.get(symbol, symbol)
        
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                             end=end_date.strftime('%Y-%m-%d'), 
                             interval='1m')
        
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        
        # Filter to target date only
        hist = hist[hist.index.date == target_date.date()]
        
        return hist.dropna()
        
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()


def simulate_trading_day(symbol: str, date: str, verbose: bool = True) -> Dict:
    """
    Simulate a full trading day with gap detection and trading logic
    
    Args:
        symbol: Trading symbol to test
        date: Date string in format 'YYYY-MM-DD'
        verbose: Print detailed logs
    
    Returns:
        Dictionary with results
    """
    print("=" * 70)
    print(f"BACKTESTING: {symbol} on {date}")
    print("=" * 70)
    print()
    
    # Get historical data
    hist = get_historical_data(symbol, date, days_before=2)
    
    if len(hist) == 0:
        print(f"❌ No data available for {symbol} on {date}")
        return {'success': False, 'reason': 'No data'}
    
    # Create mock risk manager
    risk_mgr = MockRiskManager()
    
    # Filter to market hours (9:30 AM - 4:00 PM ET)
    est = pytz.timezone('US/Eastern')
    # Handle timezone - yfinance may already be timezone-aware
    if hist.index.tz is None:
        hist_est = hist.tz_localize('UTC').tz_convert(est)
    else:
        hist_est = hist.tz_convert(est)
    market_hours = hist_est[(hist_est.index.hour >= 9) & (hist_est.index.hour < 16)]
    
    if len(market_hours) == 0:
        print(f"❌ No market hours data for {symbol} on {date}")
        return {'success': False, 'reason': 'No market hours'}
    
    # Get first 10 minutes for gap detection
    first_10min = market_hours[market_hours.index.hour == 9]
    first_10min = first_10min[first_10min.index.minute <= 40]
    
    if len(first_10min) == 0:
        print(f"❌ No early morning data for gap detection")
        return {'success': False, 'reason': 'No early data'}
    
    # Get current price (first bar after 9:30)
    current_price = float(first_10min['Close'].iloc[0])
    
    # Get extended history for gap detection (need previous day's close)
    hist_extended = get_historical_data(symbol, date, days_before=5)
    
    print(f"📊 Market Data:")
    print(f"   First price: ${current_price:.2f}")
    print(f"   Total bars: {len(market_hours)}")
    print()
    
    # ========== GAP DETECTION ==========
    gap_data = None
    gap_action = None
    
    if GAP_DETECTION_AVAILABLE:
        print("🔍 Testing Gap Detection...")
        print("-" * 70)
        
        # Simulate 9:35 AM ET
        test_time = datetime.strptime(f"{date} 09:35:00", "%Y-%m-%d %H:%M:%S")
        test_time_est = est.localize(test_time)
        
        # Use extended history for gap detection
        gap_data = detect_overnight_gap(symbol, current_price, hist_extended, risk_mgr)
        
        if gap_data and gap_data.get('detected'):
            print(f"✅ GAP DETECTED!")
            print(f"   Direction: {gap_data['direction'].upper()}")
            print(f"   Gap Points: ${gap_data['gap_points']:.2f}")
            print(f"   Gap %: {gap_data['gap_pct']:+.2f}%")
            print(f"   Prev Close: ${gap_data['prev_close']:.2f}")
            print(f"   Open Price: ${gap_data['open_price']:.2f}")
            print(f"   Bias: {gap_data['bias'].upper()}")
            print(f"   Strength: {gap_data['strength'].upper()}")
            print()
            
            # Get gap-based action
            current_time_int = 935  # 9:35 AM
            gap_action = get_gap_based_action(gap_data, current_price, current_time_int)
            
            if gap_action:
                action_desc = 'BUY CALL' if gap_action == 1 else 'BUY PUT'
                print(f"🎯 GAP-BASED ACTION: {gap_action} ({action_desc})")
                print()
        else:
            print("ℹ️  No significant gap detected (< 0.35% and < $2.50)")
            print()
    else:
        print("⚠️  Gap detection module not available")
        print()
    
    # ========== SIMULATE TRADING THROUGH THE DAY ==========
    trades = []
    positions = {}
    
    # Simulate minute-by-minute
    for idx, (timestamp, bar) in enumerate(market_hours.iterrows()):
        price = float(bar['Close'])
        hour = timestamp.hour
        minute = timestamp.minute
        current_time_int = hour * 100 + minute
        
        # Only trade during first 60 minutes if gap detected
        if gap_action and 930 <= current_time_int <= 1035 and len(positions) == 0:
            # Execute gap-based trade
            if gap_action == 1:  # BUY CALL
                strike = round(price)
                premium = max(0.10, abs(price - strike) * 0.5)  # Estimate premium
                qty = 2  # Example size
                
                positions['call'] = {
                    'strike': strike,
                    'premium': premium,
                    'qty': qty,
                    'entry_time': timestamp,
                    'entry_price': price
                }
                
                trades.append({
                    'time': timestamp,
                    'action': 'BUY CALL',
                    'strike': strike,
                    'premium': premium,
                    'qty': qty,
                    'price': price,
                    'source': 'GAP'
                })
                
                if verbose:
                    print(f"[{timestamp.strftime('%H:%M')}] 🎯 GAP TRADE: BUY {qty}x {strike} CALL @ ${premium:.2f}")
                break  # Only one entry per simulation
                
            elif gap_action == 2:  # BUY PUT
                strike = round(price)
                premium = max(0.10, abs(price - strike) * 0.5)  # Estimate premium
                qty = 2  # Example size
                
                positions['put'] = {
                    'strike': strike,
                    'premium': premium,
                    'qty': qty,
                    'entry_time': timestamp,
                    'entry_price': price
                }
                
                trades.append({
                    'time': timestamp,
                    'action': 'BUY PUT',
                    'strike': strike,
                    'premium': premium,
                    'qty': qty,
                    'price': price,
                    'source': 'GAP'
                })
                
                if verbose:
                    print(f"[{timestamp.strftime('%H:%M')}] 🎯 GAP TRADE: BUY {qty}x {strike} PUT @ ${premium:.2f}")
                break  # Only one entry per simulation
    
    # ========== RESULTS ==========
    result = {
        'success': True,
        'date': date,
        'symbol': symbol,
        'gap_detected': gap_data is not None and gap_data.get('detected', False),
        'gap_data': gap_data,
        'gap_action': gap_action,
        'trades': trades,
        'positions': positions,
        'first_price': current_price,
        'last_price': float(market_hours['Close'].iloc[-1]),
        'price_change': float(market_hours['Close'].iloc[-1]) - current_price,
        'price_change_pct': ((float(market_hours['Close'].iloc[-1]) - current_price) / current_price) * 100
    }
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Date: {date}")
    print(f"Symbol: {symbol}")
    print(f"Gap Detected: {'✅ YES' if result['gap_detected'] else '❌ NO'}")
    if result['gap_detected']:
        print(f"Gap Action: {gap_action} ({'BUY CALL' if gap_action == 1 else 'BUY PUT' if gap_action == 2 else 'NONE'})")
    print(f"Trades Executed: {len(trades)}")
    print(f"Price Range: ${current_price:.2f} → ${result['last_price']:.2f}")
    print(f"Price Change: ${result['price_change']:.2f} ({result['price_change_pct']:+.2f}%)")
    print()
    
    return result


def test_multiple_days(symbol: str, start_date: str, end_date: str, verbose: bool = True):
    """
    Test multiple days in a row
    
    Args:
        symbol: Trading symbol
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        verbose: Print detailed logs
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    results = []
    current = start
    
    print("=" * 70)
    print(f"WEEKEND BACKTESTING: {symbol} from {start_date} to {end_date}")
    print("=" * 70)
    print()
    
    while current <= end:
        # Skip weekends
        if current.weekday() < 5:  # Monday = 0, Friday = 4
            date_str = current.strftime('%Y-%m-%d')
            result = simulate_trading_day(symbol, date_str, verbose=verbose)
            if result['success']:
                results.append(result)
            print()
        
        current += timedelta(days=1)
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Days Tested: {len(results)}")
    print(f"Days with Gaps: {sum(1 for r in results if r['gap_detected'])}")
    print(f"Days with Trades: {sum(1 for r in results if len(r['trades']) > 0)}")
    print()
    
    if results:
        gap_days = [r for r in results if r['gap_detected']]
        if gap_days:
            print("Gap Days Details:")
            for r in gap_days:
                gap = r['gap_data']
                print(f"  {r['date']}: {gap['direction'].upper()} ${gap['gap_points']:.2f} ({gap['gap_pct']:+.2f}%) → Action: {r['gap_action']}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Weekend Backtesting Environment')
    parser.add_argument('--symbol', type=str, default='SPY', help='Symbol to test (SPY, QQQ, SPX)')
    parser.add_argument('--date', type=str, help='Single date to test (YYYY-MM-DD)')
    parser.add_argument('--start', type=str, help='Start date for range (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date for range (YYYY-MM-DD)')
    parser.add_argument('--quiet', action='store_true', help='Less verbose output')
    
    args = parser.parse_args()
    
    if args.date:
        # Test single day
        simulate_trading_day(args.symbol, args.date, verbose=not args.quiet)
    elif args.start and args.end:
        # Test date range
        test_multiple_days(args.symbol, args.start, args.end, verbose=not args.quiet)
    else:
        # Default: test last 5 trading days
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=7)
        test_multiple_days(args.symbol, start_date.strftime('%Y-%m-%d'), 
                          end_date.strftime('%Y-%m-%d'), verbose=not args.quiet)

