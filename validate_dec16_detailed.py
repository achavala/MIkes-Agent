#!/usr/bin/env python3
"""
Detailed Validation: Dec 16, 2025 - Compare Mike's Trades vs Bot's Decisions
Uses real Alpaca/Massive data for comprehensive analysis
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional
import json

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

from technical_analysis_engine import TechnicalAnalysisEngine
from mike_agent_live_safe import get_market_data, get_current_price

# Mike's actual trades from Dec 16, 2025
MIKE_TRADES_DEC16 = [
    {
        'time': '08:34:00',
        'symbol': 'SPY',
        'option_type': 'put',
        'strike': 674,
        'entry_price': 0.43,
        'entry_avg': 0.40,
        'entry_underlying': 678.0,  # Approximate
        'reason': '$676.6-$675 PT range, looking for bounces to avg down',
        'target': 675.0,
        'exit_price': 0.84,
        'exit_time': '08:46:00',
        'profit_pct': 110.0,
        'notes': 'Sold majority at 90% profit, runners to 110%'
    },
    {
        'time': '09:20:00',
        'symbol': 'QQQ',
        'option_type': 'put',
        'strike': 604,
        'entry_price': 0.50,
        'entry_underlying': 610.0,  # Approximate
        'reason': 'PTs = $605 #1, $602 #2, $605/$602 range expected',
        'target': 605.0,
        'exit_price': 1.03,
        'exit_time': '09:25:00',
        'profit_pct': 107.0,
        'notes': 'Sold majority, 2-2 on the day with both alerts running 100%+'
    },
    {
        'time': '12:12:00',
        'symbol': 'SPY',
        'option_type': 'put',
        'strike': 673,
        'entry_price': 0.40,
        'entry_avg': 0.32,
        'entry_underlying': 675.0,  # Approximate
        'reason': 'High risk setup, LOD sweep into $674/$673, size for $0 or skip',
        'target': 673.0,
        'exit_price': 0.38,
        'exit_time': '12:19:00',
        'profit_pct': 20.0,
        'notes': 'Sold majority at 20% profit due to risk, hands off'
    },
    {
        'time': '12:47:00',
        'symbol': 'SPY',
        'option_type': 'call',
        'strike': 679,
        'entry_price': 0.45,
        'entry_underlying': 675.0,  # Approximate
        'reason': 'EOD V shape recovery back to $678/680 range, main move near 2:30',
        'target': 680.0,
        'exit_price': 0.67,
        'exit_time': '12:51:00',
        'profit_pct': 50.0,
        'notes': 'Sold majority around $0.65, took 50% profit'
    },
]

def get_historical_data(symbol: str, date: str, start_time: str, end_time: str) -> Optional[pd.DataFrame]:
    """Get historical data from Alpaca or Massive"""
    trade_date = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M:%S")
    trade_date = pytz.timezone('America/New_York').localize(trade_date)
    
    end_datetime = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M:%S")
    end_datetime = pytz.timezone('America/New_York').localize(end_datetime)
    
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
                bars = bars[bars.index.date == trade_date.date()]
                return bars
        except Exception as e:
            print(f"⚠️ Alpaca error: {e}")
    
    # Try Massive
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
                    df = df[df.index.date == trade_date.date()]
                    return df
        except Exception as e:
            print(f"⚠️ Massive error: {e}")
    
    return None

def analyze_price_action_detailed(data: pd.DataFrame, trade_time: str, symbol: str):
    """Detailed price action analysis"""
    trade_datetime = datetime.strptime(f"2025-12-16 {trade_time}", "%Y-%m-%d %H:%M:%S")
    trade_datetime = pytz.timezone('America/New_York').localize(trade_datetime)
    
    data_before = data[data.index <= trade_datetime]
    
    if len(data_before) < 20:
        return None
    
    recent = data_before.tail(50)
    
    highs = recent['high'].values
    lows = recent['low'].values
    closes = recent['close'].values
    opens = recent['open'].values
    volumes = recent['volume'].values if 'volume' in recent.columns else np.ones(len(recent))
    
    current_price = closes[-1]
    current_high = highs[-1]
    current_low = lows[-1]
    current_open = opens[-1]
    
    # Calculate key levels
    resistance = np.max(highs[-20:])
    support = np.min(lows[-20:])
    prev_resistance = np.max(highs[-20:-10]) if len(highs) >= 20 else resistance
    prev_support = np.min(lows[-20:-10]) if len(lows) >= 20 else support
    
    # Price changes
    change_5 = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
    change_10 = (closes[-1] - closes[-10]) / closes[-10] if len(closes) >= 10 else 0
    change_20 = (closes[-1] - closes[-20]) / closes[-20] if len(closes) >= 20 else 0
    
    # Trend analysis
    lower_lows = np.sum(np.diff(lows[-10:]) < 0) >= 3
    lower_highs = np.sum(np.diff(highs[-10:]) < 0) >= 3
    higher_lows = np.sum(np.diff(lows[-10:]) > 0) >= 3
    higher_highs = np.sum(np.diff(highs[-10:]) > 0) >= 3
    
    # Gap analysis
    gap_up = False
    gap_down = False
    if len(recent) >= 2:
        gap_up = opens[-1] > closes[-2] * 1.005
        gap_down = opens[-1] < closes[-2] * 0.995
    
    # Volume analysis
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 1
    current_volume = volumes[-1] if len(volumes) > 0 else 1
    volume_spike = current_volume > avg_volume * 1.5
    
    return {
        'time': trade_time,
        'current_price': current_price,
        'current_high': current_high,
        'current_low': current_low,
        'resistance': resistance,
        'support': support,
        'prev_resistance': prev_resistance,
        'prev_support': prev_support,
        'change_5_pct': change_5 * 100,
        'change_10_pct': change_10 * 100,
        'change_20_pct': change_20 * 100,
        'lower_lows': lower_lows,
        'lower_highs': lower_highs,
        'higher_lows': higher_lows,
        'higher_highs': higher_highs,
        'gap_up': gap_up,
        'gap_down': gap_down,
        'volume_spike': volume_spike,
        'distance_to_resistance': ((resistance - current_price) / current_price) * 100,
        'distance_to_support': ((current_price - support) / current_price) * 100,
    }

def detailed_validation(trade: Dict, ta_engine: TechnicalAnalysisEngine) -> Dict:
    """Detailed validation with comprehensive analysis"""
    
    print(f"\n{'='*100}")
    print(f"📊 DETAILED ANALYSIS: {trade['symbol']} ${trade['strike']:.0f} {trade['option_type'].upper()} @ {trade['time']}")
    print(f"{'='*100}")
    
    # Mike's reasoning
    print(f"\n🎯 MIKE'S REASONING:")
    print(f"  Entry: ${trade['entry_price']:.2f} @ ${trade['entry_underlying']:.2f}")
    print(f"  Strike: ${trade['strike']:.0f}")
    print(f"  Target: ${trade['target']:.2f}")
    print(f"  Reason: {trade['reason']}")
    print(f"  Result: ${trade['exit_price']:.2f} ({trade['profit_pct']:.0f}% profit)")
    print()
    
    # Get data
    data = get_historical_data(trade['symbol'], "2025-12-16", "08:00:00", trade['time'])
    
    if data is None or len(data) < 20:
        print(f"❌ Insufficient data")
        return {'data_available': False}
    
    # Price action analysis
    price_analysis = analyze_price_action_detailed(data, trade['time'], trade['symbol'])
    
    if price_analysis:
        print(f"📈 PRICE ACTION ANALYSIS:")
        print(f"  Current Price: ${price_analysis['current_price']:.2f}")
        print(f"  Resistance: ${price_analysis['resistance']:.2f} ({price_analysis['distance_to_resistance']:.2f}% away)")
        print(f"  Support: ${price_analysis['support']:.2f} ({price_analysis['distance_to_support']:.2f}% away)")
        print(f"  Price Change: {price_analysis['change_5_pct']:.2f}% (5 bars), {price_analysis['change_10_pct']:.2f}% (10 bars)")
        print(f"  Trend: Lower Lows={price_analysis['lower_lows']}, Lower Highs={price_analysis['lower_highs']}")
        print(f"  Gap: Up={price_analysis['gap_up']}, Down={price_analysis['gap_down']}")
        print(f"  Volume: Spike={price_analysis['volume_spike']}")
        print()
    
    # Technical analysis
    trade_datetime = datetime.strptime(f"2025-12-16 {trade['time']}", "%Y-%m-%d %H:%M:%S")
    trade_datetime = pytz.timezone('America/New_York').localize(trade_datetime)
    data_before = data[data.index <= trade_datetime]
    
    if len(data_before) < 20:
        print(f"❌ Not enough data before trade time")
        return {'data_available': False}
    
    current_price = data_before['close'].iloc[-1]
    
    # Extract target from trade if available
    target_levels = [trade.get('target')] if trade.get('target') else None
    
    ta_result = ta_engine.analyze_symbol(
        data=data_before,
        symbol=trade['symbol'],
        current_price=current_price,
        target_levels=target_levels
    )
    
    print(f"🤖 BOT'S ANALYSIS:")
    print(f"  Current Price: ${current_price:.2f}")
    
    if ta_result and ta_result.get('patterns'):
        print(f"  Patterns Detected: {len(ta_result['patterns'])}")
        for i, pattern in enumerate(ta_result['patterns'], 1):
            print(f"    {i}. {pattern['pattern_type']}: {pattern.get('reason', 'N/A')}")
            print(f"       Direction: {pattern.get('direction', 'N/A')}, Confidence: {pattern.get('confidence', 0):.2f}")
        
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
    
    # Comparison
    print(f"📊 COMPARISON:")
    
    pattern_detected = ta_result and ta_result.get('best_pattern') is not None
    direction_match = False
    strike_match = False
    
    if pattern_detected:
        best_pattern = ta_result['best_pattern']
        pattern_direction = best_pattern.get('direction', '')
        
        # Direction match
        if trade['option_type'] == 'call' and pattern_direction == 'bullish':
            direction_match = True
        elif trade['option_type'] == 'put' and pattern_direction == 'bearish':
            direction_match = True
        
        # Strike match
        if ta_result.get('strike_suggestion'):
            strike_diff = abs(ta_result['strike_suggestion'] - trade['strike'])
            strike_match = strike_diff <= 3.0
            print(f"  Strike Match: {'✅' if strike_match else '❌'} (Bot: ${ta_result['strike_suggestion']:.2f}, Mike: ${trade['strike']:.0f}, Diff: ${strike_diff:.2f})")
        else:
            print(f"  Strike Match: ❌ (Bot: N/A, Mike: ${trade['strike']:.0f})")
        
        print(f"  Direction Match: {'✅' if direction_match else '❌'} (Bot: {pattern_direction}, Mike: {trade['option_type']})")
        print(f"  Pattern Detected: ✅ ({best_pattern['pattern_type']})")
    else:
        print(f"  Pattern Detected: ❌ (No pattern found)")
        print(f"  Direction Match: ❌")
        print(f"  Strike Match: ❌")
    
    overall_match = pattern_detected and direction_match
    print(f"  Overall Match: {'✅ YES' if overall_match else '❌ NO'}")
    print()
    
    # Why analysis
    print(f"🔍 WHY ANALYSIS:")
    print(f"  Why Mike Picked:")
    print(f"    • {trade['reason']}")
    if price_analysis:
        if trade['option_type'] == 'put':
            if price_analysis['lower_lows']:
                print(f"    • Price making lower lows (bearish structure)")
            if price_analysis['change_5_pct'] < -0.1:
                print(f"    • Downward momentum ({price_analysis['change_5_pct']:.2f}% in 5 bars)")
            if price_analysis['distance_to_support'] < 1.0:
                print(f"    • Price near support (${price_analysis['support']:.2f})")
        else:  # call
            if price_analysis['higher_highs']:
                print(f"    • Price making higher highs (bullish structure)")
            if price_analysis['change_5_pct'] > 0.1:
                print(f"    • Upward momentum ({price_analysis['change_5_pct']:.2f}% in 5 bars)")
    
    print(f"  Why Bot {'Picked' if pattern_detected else 'Missed'}:")
    if pattern_detected:
        best = ta_result['best_pattern']
        print(f"    • Detected: {best['pattern_type']}")
        print(f"    • Reason: {best.get('reason', 'N/A')}")
        print(f"    • Confidence: {best.get('confidence', 0):.2f}")
    else:
        print(f"    • No pattern detected by TA engine")
        if price_analysis:
            print(f"    • Price action: {price_analysis['change_5_pct']:.2f}% change (5 bars)")
            print(f"    • Structure: Lower lows={price_analysis['lower_lows']}, Lower highs={price_analysis['lower_highs']}")
    
    print()
    
    return {
        'trade': trade,
        'data_available': True,
        'pattern_detected': pattern_detected,
        'direction_match': direction_match,
        'strike_match': strike_match,
        'overall_match': overall_match,
        'ta_result': ta_result,
        'price_analysis': price_analysis
    }

def main():
    """Main validation"""
    
    print("=" * 100)
    print("🧠 DETAILED VALIDATION: DEC 16, 2025 - MIKE vs BOT")
    print("=" * 100)
    print()
    print("Using REAL data from Alpaca/Massive API")
    print()
    
    ta_engine = TechnicalAnalysisEngine(lookback_bars=50)
    
    results = []
    for trade in MIKE_TRADES_DEC16:
        result = detailed_validation(trade, ta_engine)
        results.append(result)
    
    # Summary
    print(f"\n{'='*100}")
    print("📊 VALIDATION SUMMARY")
    print(f"{'='*100}")
    
    total = len(results)
    data_available = sum(1 for r in results if r.get('data_available', False))
    patterns_detected = sum(1 for r in results if r.get('pattern_detected', False))
    direction_matches = sum(1 for r in results if r.get('direction_match', False))
    strike_matches = sum(1 for r in results if r.get('strike_match', False))
    overall_matches = sum(1 for r in results if r.get('overall_match', False))
    
    print(f"\nTotal Trades: {total}")
    print(f"Data Available: {data_available}/{total} ({data_available/total*100:.0f}%)")
    if data_available > 0:
        print(f"Patterns Detected: {patterns_detected}/{data_available} ({patterns_detected/data_available*100:.0f}%)")
        print(f"Direction Matches: {direction_matches}/{data_available} ({direction_matches/data_available*100:.0f}%)")
        print(f"Strike Matches: {strike_matches}/{data_available} ({strike_matches/data_available*100:.0f}%)")
        print(f"Overall Matches: {overall_matches}/{data_available} ({overall_matches/data_available*100:.0f}%)")
    
    print()
    
    # Detailed breakdown
    print(f"📋 TRADE-BY-TRADE BREAKDOWN:")
    for i, (trade, result) in enumerate(zip(MIKE_TRADES_DEC16, results), 1):
        if result.get('data_available'):
            match_status = "✅ MATCH" if result.get('overall_match') else "❌ NO MATCH"
            print(f"  {i}. {trade['symbol']} ${trade['strike']:.0f} {trade['option_type'].upper()} @ {trade['time']}: {match_status}")
            if result.get('pattern_detected'):
                pattern = result['ta_result']['best_pattern']
                print(f"     Pattern: {pattern['pattern_type']} ({pattern['direction']})")
            else:
                print(f"     Pattern: None detected")
    
    print()
    
    # Save results
    with open('dec16_detailed_validation.json', 'w') as f:
        json_results = []
        for r in results:
            json_r = {
                'trade': r.get('trade', {}),
                'data_available': r.get('data_available', False),
                'pattern_detected': r.get('pattern_detected', False),
                'direction_match': r.get('direction_match', False),
                'strike_match': r.get('strike_match', False),
                'overall_match': r.get('overall_match', False),
            }
            if r.get('ta_result'):
                json_r['ta_result'] = {
                    'best_pattern': r['ta_result'].get('best_pattern', {}),
                    'strike_suggestion': r['ta_result'].get('strike_suggestion'),
                    'confidence_boost': r['ta_result'].get('confidence_boost', 0)
                }
            if r.get('price_analysis'):
                json_r['price_analysis'] = r['price_analysis']
            json_results.append(json_r)
        
        json.dump(json_results, f, indent=2, default=str)
    
    print("✅ Detailed results saved to dec16_detailed_validation.json")
    
    return results

if __name__ == "__main__":
    results = main()

