#!/usr/bin/env python3
"""
Backtest Bot vs Mike's Actual Trades - Dec 18, 2025
Uses real Alpaca/Massive data to validate what the bot would have done
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional, Tuple
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import Alpaca and Massive APIs
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠️ alpaca-trade-api not available")

try:
    from massive_api_client import MassiveAPIClient
    MASSIVE_AVAILABLE = True
except ImportError:
    MASSIVE_AVAILABLE = False
    print("⚠️ massive_api_client not available")

# Import bot logic
from mike_agent_live_safe import (
    get_market_data,
    find_atm_strike,
    get_current_price,
    TRADING_SYMBOLS
)

# Mike's actual trades from Dec 18, 2025
MIKE_TRADES = [
    {
        'time': '08:38:00',
        'symbol': 'SPY',
        'option_type': 'put',
        'strike': 672,
        'entry_price': 0.50,
        'entry_underlying': 675.0,  # Approximate
        'reason': 'False trend-line breakout, invalidated if $678.6 reclaims',
        'target': 675.0,
        'exit_price': 0.90,
        'exit_time': '08:49:00',
        'profit_pct': 80.0,
        'notes': 'Took majority at $0.75 (50% profit)'
    },
    {
        'time': '09:11:00',
        'symbol': 'QQQ',
        'option_type': 'put',
        'strike': 603,
        'entry_price': 0.60,
        'entry_underlying': 609.0,  # Approximate
        'reason': '$605 targets, size accordingly',
        'target': 605.0,
        'exit_price': 0.84,
        'exit_time': '09:14:00',
        'profit_pct': 40.0,
        'notes': 'Sold majority at 40%'
    },
    {
        'time': '09:29:00',
        'symbol': 'SPY',
        'option_type': 'call',
        'strike': 681,
        'entry_price': 0.50,
        'entry_underlying': 680.0,  # Approximate
        'reason': 'Upside structure confirmed, trendline break on 15M',
        'target': 682.0,
        'exit_price': 1.10,
        'exit_time': '09:39:00',
        'profit_pct': 130.0,
        'notes': 'Took majority at $0.65 (30-40% profit)'
    },
    {
        'time': '11:11:00',
        'symbol': 'SPY',
        'option_type': 'put',
        'strike': 672,
        'entry_price': 0.40,
        'entry_underlying': 675.0,  # Approximate
        'reason': 'Looking for gap fill',
        'target': 672.0,
        'exit_price': 0.73,
        'exit_time': '11:18:00',
        'profit_pct': 110.0,
        'notes': 'Sold majority at 50-60% profit'
    },
    {
        'time': '13:34:00',
        'symbol': 'SPY',
        'option_type': 'put',
        'strike': 670,
        'entry_price': 1.05,
        'entry_underlying': 675.0,  # Approximate
        'reason': 'Double rejection on SPY, IV high',
        'target': 672.8,
        'exit_price': 1.45,
        'exit_time': '13:57:00',
        'profit_pct': 55.0,
        'notes': '1DTE, sold majority at 45% profit'
    },
]

def get_historical_data_alpaca(symbol: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Get historical data from Alpaca API"""
    if not ALPACA_AVAILABLE:
        return None
    
    try:
        api = tradeapi.REST(
            os.getenv('ALPACA_KEY') or os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET') or os.getenv('ALPACA_SECRET_KEY'),
            base_url='https://paper-api.alpaca.markets',
            api_version='v2'
        )
        
        # Convert to timezone-aware
        start_str = start.strftime('%Y-%m-%dT%H:%M:%S-05:00')  # EST
        end_str = end.strftime('%Y-%m-%dT%H:%M:%S-05:00')
        
        bars = api.get_bars(
            symbol,
            tradeapi.TimeFrame.Minute,
            start=start_str,
            end=end_str
        ).df
        
        if len(bars) > 0:
            bars.columns = [col.lower() for col in bars.columns]
            return bars
    except Exception as e:
        print(f"⚠️ Alpaca API error for {symbol}: {e}")
    
    return None

def get_historical_data_massive(symbol: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Get historical data from Massive API"""
    if not MASSIVE_AVAILABLE:
        return None
    
    try:
        api_key = os.getenv('MASSIVE_API_KEY') or os.getenv('POLYGON_API_KEY')
        if not api_key:
            return None
        
        client = MassiveAPIClient(api_key)
        
        # Massive symbol mapping
        massive_map = {
            'SPY': 'SPY',
            'QQQ': 'QQQ',
            'IWM': 'IWM'
        }
        
        massive_symbol = massive_map.get(symbol)
        if not massive_symbol:
            return None
        
        # Get 1-minute bars
        data = client.get_bars(
            massive_symbol,
            start=start,
            end=end,
            timeframe='1min'
        )
        
        if data and len(data) > 0:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
    except Exception as e:
        print(f"⚠️ Massive API error for {symbol}: {e}")
    
    return None

def simulate_bot_decision(symbol: str, timestamp: datetime, price: float, option_type: str) -> Dict:
    """Simulate what the bot would decide at this moment"""
    
    # Get historical data for context
    start = timestamp - timedelta(days=2)
    end = timestamp
    
    # Try Alpaca first, then Massive
    hist_data = get_historical_data_alpaca(symbol, start, end)
    if hist_data is None or len(hist_data) < 20:
        hist_data = get_historical_data_massive(symbol, start, end)
    
    if hist_data is None or len(hist_data) < 20:
        return {
            'would_trade': False,
            'reason': 'Insufficient data',
            'strike': None,
            'confidence': 0.0
        }
    
    # Simulate bot's strike selection
    if option_type == 'call':
        bot_strike = find_atm_strike(price, option_type='call')
    else:
        bot_strike = find_atm_strike(price, option_type='put')
    
    # Simulate bot's confidence (simplified - would need actual RL model)
    # For now, assume moderate confidence if price is moving
    price_change = (hist_data['close'].iloc[-1] - hist_data['close'].iloc[-20]) / hist_data['close'].iloc[-20]
    confidence = 0.50 + abs(price_change) * 2  # Higher confidence if price moving
    confidence = min(0.90, max(0.40, confidence))
    
    # Bot would trade if confidence > 0.52
    would_trade = confidence >= 0.52
    
    return {
        'would_trade': would_trade,
        'reason': f'Confidence: {confidence:.3f}',
        'strike': bot_strike,
        'confidence': confidence,
        'price_change': price_change
    }

def analyze_mike_vs_bot():
    """Compare Mike's trades vs what bot would do"""
    
    print("=" * 80)
    print("🧠 MIKE vs BOT ANALYSIS - Dec 18, 2025")
    print("=" * 80)
    print()
    
    results = []
    
    for trade in MIKE_TRADES:
        print(f"\n{'='*80}")
        print(f"📊 TRADE: {trade['symbol']} ${trade['strike']:.0f} {trade['option_type'].upper()}")
        print(f"{'='*80}")
        print(f"Time: {trade['time']}")
        print(f"Mike's Entry: ${trade['entry_price']:.2f} @ ${trade['entry_underlying']:.2f}")
        print(f"Mike's Strike: ${trade['strike']:.0f}")
        print(f"Mike's Reason: {trade['reason']}")
        print(f"Mike's Target: ${trade['target']:.2f}")
        print(f"Mike's Exit: ${trade['exit_price']:.2f} ({trade['profit_pct']:.0f}% profit)")
        print()
        
        # Parse time
        trade_time = datetime.strptime(f"2025-12-18 {trade['time']}", "%Y-%m-%d %H:%M:%S")
        trade_time = pytz.timezone('America/New_York').localize(trade_time)
        
        # Simulate bot decision
        bot_decision = simulate_bot_decision(
            trade['symbol'],
            trade_time,
            trade['entry_underlying'],
            trade['option_type']
        )
        
        print(f"🤖 BOT DECISION:")
        print(f"  Would Trade: {'✅ YES' if bot_decision['would_trade'] else '❌ NO'}")
        print(f"  Strike: ${bot_decision['strike']:.2f}" if bot_decision['strike'] else "  Strike: N/A")
        print(f"  Confidence: {bot_decision['confidence']:.3f}")
        print(f"  Reason: {bot_decision['reason']}")
        print()
        
        # Compare
        print(f"📊 COMPARISON:")
        
        # Strike comparison
        if bot_decision['strike']:
            strike_diff = abs(bot_decision['strike'] - trade['strike'])
            strike_match = strike_diff <= 2.0  # Within $2
            print(f"  Strike Match: {'✅' if strike_match else '❌'} (Bot: ${bot_decision['strike']:.0f}, Mike: ${trade['strike']:.0f}, Diff: ${strike_diff:.2f})")
        else:
            strike_match = False
            print(f"  Strike Match: ❌ (Bot: N/A, Mike: ${trade['strike']:.0f})")
        
        # Trade decision
        decision_match = bot_decision['would_trade']
        print(f"  Would Trade: {'✅' if decision_match else '❌'} (Bot: {'YES' if bot_decision['would_trade'] else 'NO'}, Mike: YES)")
        
        # Overall match
        overall_match = strike_match and decision_match
        print(f"  Overall Match: {'✅ YES' if overall_match else '❌ NO'}")
        
        results.append({
            'trade': trade,
            'bot_decision': bot_decision,
            'strike_match': strike_match,
            'decision_match': decision_match,
            'overall_match': overall_match
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}")
    
    total_trades = len(results)
    strike_matches = sum(1 for r in results if r['strike_match'])
    decision_matches = sum(1 for r in results if r['decision_match'])
    overall_matches = sum(1 for r in results if r['overall_match'])
    
    print(f"Total Trades: {total_trades}")
    print(f"Strike Matches: {strike_matches}/{total_trades} ({strike_matches/total_trades*100:.0f}%)")
    print(f"Decision Matches: {decision_matches}/{total_trades} ({decision_matches/total_trades*100:.0f}%)")
    print(f"Overall Matches: {overall_matches}/{total_trades} ({overall_matches/total_trades*100:.0f}%)")
    print()
    
    # Save results
    with open('mike_vs_bot_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("✅ Results saved to mike_vs_bot_analysis.json")
    
    return results

if __name__ == "__main__":
    results = analyze_mike_vs_bot()





