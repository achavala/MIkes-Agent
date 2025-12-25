#!/usr/bin/env python3
"""
Validate Technical Analysis Engine against Mike's Dec 17, 2025 trades
Uses real Alpaca/Massive data to test if bot can detect Mike's patterns
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional
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

# Import TA engine
from technical_analysis_engine import TechnicalAnalysisEngine

# Import bot's data fetching
from mike_agent_live_safe import get_market_data, get_current_price

# Mike's actual trades from Dec 17, 2025
MIKE_TRADES_DEC17 = [
    {
        'time': '08:57:00',
        'symbol': 'SPY',
        'option_type': 'call',
        'strike': 682,
        'entry_price': 0.50,
        'entry_underlying': 680.0,  # Approximate
        'reason': 'PT - $680/$682, expect volatility, avg down',
        'target': 682.0,
        'exit_price': 0.60,
        'exit_time': '09:08:00',
        'profit_pct': 15.0,
        'notes': 'Exited near breakeven, failure to reclaim upside'
    },
    {
        'time': '09:08:00',
        'symbol': 'SPY',
        'option_type': 'put',
        'strike': 675,
        'entry_price': 0.50,
        'entry_underlying': 678.0,  # Approximate
        'reason': '$675 range PT, looking for breakdown to gamma zone',
        'target': 675.0,
        'exit_price': 0.65,
        'exit_time': '09:19:00',
        'profit_pct': 30.0,
        'notes': 'Took profits at $0.65, runners left with SL'
    },
    {
        'time': '09:36:00',
        'symbol': 'SPY',
        'option_type': 'put',
        'strike': 672,
        'entry_price': 0.35,
        'entry_underlying': 675.0,  # Approximate
        'reason': 'Breakdown after 10:30, $674/$675 PT',
        'target': 675.0,
        'exit_price': 0.88,
        'exit_time': '09:53:00',
        'profit_pct': 160.0,
        'notes': 'Took majority profits near $0.60, runners to $0.88'
    },
    {
        'time': '09:52:00',
        'symbol': 'SPY',
        'option_type': 'put',
        'strike': 670,
        'entry_price': 0.40,
        'entry_underlying': 673.0,  # Approximate
        'reason': '$670 PT, high risk play, avg down',
        'target': 670.0,
        'exit_price': 0.60,
        'exit_time': '10:15:00',
        'profit_pct': 90.0,
        'notes': 'Took majority profits near $0.60'
    },
    {
        'time': '10:40:00',
        'symbol': 'SPY',
        'option_type': 'put',
        'strike': 669,
        'entry_price': 0.24,
        'entry_underlying': 671.0,  # Approximate
        'reason': '$670 PT, $673.5 breakdown essential',
        'target': 670.0,
        'exit_price': 0.37,
        'exit_time': '10:52:00',
        'profit_pct': 80.0,
        'notes': 'Took majority profits, 5-5 on the day'
    },
]

def get_historical_data_for_validation(symbol: str, date: str, start_time: str, end_time: str) -> Optional[pd.DataFrame]:
    """Get historical data from Alpaca or Massive for validation"""
    
    # Parse date and times
    trade_date = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M:%S")
    trade_date = pytz.timezone('America/New_York').localize(trade_date)
    
    end_datetime = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M:%S")
    end_datetime = pytz.timezone('America/New_York').localize(end_datetime)
    
    # Get data starting 2 days before to have context
    start_datetime = trade_date - timedelta(days=2)
    
    # Try Alpaca first
    if ALPACA_AVAILABLE:
        try:
            api = tradeapi.REST(
                os.getenv('ALPACA_KEY') or os.getenv('ALPACA_API_KEY'),
                os.getenv('ALPACA_SECRET') or os.getenv('ALPACA_SECRET_KEY'),
                base_url='https://paper-api.alpaca.markets',
                api_version='v2'
            )
            
            start_str = start_datetime.strftime('%Y-%m-%dT%H:%M:%S-05:00')
            end_str = end_datetime.strftime('%Y-%m-%dT%H:%M:%S-05:00')
            
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
            print(f"⚠️ Alpaca API error for {symbol}: {e}")
    
    # Try Massive API
    if MASSIVE_AVAILABLE:
        try:
            api_key = os.getenv('MASSIVE_API_KEY') or os.getenv('POLYGON_API_KEY')
            if api_key:
                client = MassiveAPIClient(api_key)
                data = client.get_bars(
                    symbol,
                    start=start_datetime,
                    end=end_datetime,
                    timeframe='1min'
                )
                
                if data and len(data) > 0:
                    df = pd.DataFrame(data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    # Filter to trade date only
                    df = df[df.index.date == trade_date.date()]
                    return df
        except Exception as e:
            print(f"⚠️ Massive API error for {symbol}: {e}")
    
    return None

def validate_trade(trade: Dict, ta_engine: TechnicalAnalysisEngine) -> Dict:
    """Validate if bot would detect this trade"""
    
    print(f"\n{'='*80}")
    print(f"📊 VALIDATING: {trade['symbol']} ${trade['strike']:.0f} {trade['option_type'].upper()}")
    print(f"{'='*80}")
    print(f"Time: {trade['time']}")
    print(f"Mike's Entry: ${trade['entry_price']:.2f} @ ${trade['entry_underlying']:.2f}")
    print(f"Mike's Strike: ${trade['strike']:.0f}")
    print(f"Mike's Reason: {trade['reason']}")
    print(f"Mike's Target: ${trade['target']:.2f}")
    print()
    
    # Get historical data
    trade_date = "2025-12-17"
    trade_time = trade['time']
    
    # Get data up to trade time (need context before trade)
    data = get_historical_data_for_validation(
        trade['symbol'],
        trade_date,
        "08:00:00",  # Start from 8 AM
        trade_time
    )
    
    if data is None or len(data) < 20:
        print(f"❌ Insufficient data for {trade['symbol']} at {trade_time}")
        return {
            'trade': trade,
            'data_available': False,
            'pattern_detected': False,
            'reason': 'Insufficient data'
        }
    
    # Get price at trade time
    trade_datetime = datetime.strptime(f"{trade_date} {trade_time}", "%Y-%m-%d %H:%M:%S")
    trade_datetime = pytz.timezone('America/New_York').localize(trade_datetime)
    
    # Find closest bar to trade time
    data_before_trade = data[data.index <= trade_datetime]
    if len(data_before_trade) < 20:
        print(f"❌ Not enough data before trade time")
        return {
            'trade': trade,
            'data_available': False,
            'pattern_detected': False,
            'reason': 'Not enough data before trade time'
        }
    
    current_price = data_before_trade['close'].iloc[-1]
    
    # Run technical analysis
    ta_result = ta_engine.analyze_symbol(
        data=data_before_trade,
        symbol=trade['symbol'],
        current_price=current_price
    )
    
    print(f"🤖 BOT ANALYSIS:")
    print(f"  Current Price: ${current_price:.2f}")
    
    if ta_result and ta_result.get('patterns'):
        print(f"  Patterns Detected: {len(ta_result['patterns'])}")
        for pattern in ta_result['patterns']:
            print(f"    • {pattern['pattern_type']}: {pattern.get('reason', 'N/A')} (confidence: {pattern.get('confidence', 0):.2f})")
        
        if ta_result.get('best_pattern'):
            best = ta_result['best_pattern']
            print(f"  Best Pattern: {best['pattern_type']} ({best['direction']})")
            print(f"  Confidence: {best.get('confidence', 0):.2f}")
            print(f"  Confidence Boost: +{ta_result.get('confidence_boost', 0):.3f}")
            
            if ta_result.get('strike_suggestion'):
                print(f"  Strike Suggestion: ${ta_result['strike_suggestion']:.2f}")
            
            if ta_result.get('targets'):
                targets = ta_result['targets']
                print(f"  Targets: ${targets.get('target1', 0):.2f} / ${targets.get('target2', 0):.2f}")
    else:
        print(f"  No patterns detected")
    
    print()
    
    # Compare to Mike's trade
    print(f"📊 COMPARISON:")
    
    # Check if pattern matches Mike's direction
    pattern_match = False
    direction_match = False
    strike_match = False
    
    if ta_result and ta_result.get('best_pattern'):
        best_pattern = ta_result['best_pattern']
        pattern_direction = best_pattern.get('direction', '')
        
        # Check direction match
        if trade['option_type'] == 'call' and pattern_direction == 'bullish':
            direction_match = True
        elif trade['option_type'] == 'put' and pattern_direction == 'bearish':
            direction_match = True
        
        pattern_match = True
        
        # Check strike match
        if ta_result.get('strike_suggestion'):
            strike_diff = abs(ta_result['strike_suggestion'] - trade['strike'])
            strike_match = strike_diff <= 3.0  # Within $3
            print(f"  Strike Match: {'✅' if strike_match else '❌'} (Bot: ${ta_result['strike_suggestion']:.2f}, Mike: ${trade['strike']:.0f}, Diff: ${strike_diff:.2f})")
        else:
            print(f"  Strike Match: ❌ (Bot: N/A, Mike: ${trade['strike']:.0f})")
        
        print(f"  Direction Match: {'✅' if direction_match else '❌'} (Bot: {pattern_direction}, Mike: {trade['option_type']})")
        print(f"  Pattern Detected: ✅ ({best_pattern['pattern_type']})")
    else:
        print(f"  Pattern Detected: ❌ (No pattern found)")
        print(f"  Direction Match: ❌")
        print(f"  Strike Match: ❌")
    
    # Overall match
    overall_match = pattern_match and direction_match
    print(f"  Overall Match: {'✅ YES' if overall_match else '❌ NO'}")
    
    return {
        'trade': trade,
        'data_available': True,
        'pattern_detected': pattern_match,
        'direction_match': direction_match,
        'strike_match': strike_match,
        'overall_match': overall_match,
        'ta_result': ta_result,
        'current_price': current_price
    }

def main():
    """Main validation function"""
    
    print("=" * 80)
    print("🧠 VALIDATING TECHNICAL ANALYSIS ENGINE - DEC 17, 2025")
    print("=" * 80)
    print()
    print("Testing if bot can detect Mike's patterns using real market data")
    print()
    
    # Initialize TA engine
    ta_engine = TechnicalAnalysisEngine(lookback_bars=50)
    
    # Validate each trade
    results = []
    for trade in MIKE_TRADES_DEC17:
        result = validate_trade(trade, ta_engine)
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    total_trades = len(results)
    data_available = sum(1 for r in results if r.get('data_available', False))
    patterns_detected = sum(1 for r in results if r.get('pattern_detected', False))
    direction_matches = sum(1 for r in results if r.get('direction_match', False))
    strike_matches = sum(1 for r in results if r.get('strike_match', False))
    overall_matches = sum(1 for r in results if r.get('overall_match', False))
    
    print(f"Total Trades: {total_trades}")
    print(f"Data Available: {data_available}/{total_trades} ({data_available/total_trades*100:.0f}%)")
    print(f"Patterns Detected: {patterns_detected}/{data_available} ({patterns_detected/data_available*100:.0f}% of available)" if data_available > 0 else "Patterns Detected: N/A")
    print(f"Direction Matches: {direction_matches}/{data_available} ({direction_matches/data_available*100:.0f}% of available)" if data_available > 0 else "Direction Matches: N/A")
    print(f"Strike Matches: {strike_matches}/{data_available} ({strike_matches/data_available*100:.0f}% of available)" if data_available > 0 else "Strike Matches: N/A")
    print(f"Overall Matches: {overall_matches}/{data_available} ({overall_matches/data_available*100:.0f}% of available)" if data_available > 0 else "Overall Matches: N/A")
    print()
    
    # Save results
    with open('dec17_validation_results.json', 'w') as f:
        # Convert to JSON-serializable format
        json_results = []
        for r in results:
            json_r = {
                'trade': r['trade'],
                'data_available': r.get('data_available', False),
                'pattern_detected': r.get('pattern_detected', False),
                'direction_match': r.get('direction_match', False),
                'strike_match': r.get('strike_match', False),
                'overall_match': r.get('overall_match', False),
                'current_price': r.get('current_price', 0)
            }
            if r.get('ta_result'):
                ta_r = r['ta_result']
                json_r['ta_result'] = {
                    'best_pattern': ta_r.get('best_pattern', {}),
                    'strike_suggestion': ta_r.get('strike_suggestion'),
                    'confidence_boost': ta_r.get('confidence_boost', 0),
                    'recommended_action': ta_r.get('recommended_action', 'HOLD')
                }
            json_results.append(json_r)
        
        json.dump(json_results, f, indent=2, default=str)
    
    print("✅ Results saved to dec17_validation_results.json")
    print()
    
    return results

if __name__ == "__main__":
    results = main()





