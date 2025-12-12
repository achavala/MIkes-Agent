#!/usr/bin/env python3
"""
COMPREHENSIVE TRADE ANALYSIS
Analyzes all trades to understand:
1. Entry decision triggers (RL, Gap Detection, Market Conditions)
2. Exit decision triggers (TP, SL, Trailing Stop, EOD, etc.)
3. Why trades were closed early
4. Performance patterns and loss reasons
"""

import os
import sys
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import pytz

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Import config for API keys
try:
    import config
    API_KEY = getattr(config, 'ALPACA_KEY', os.getenv('ALPACA_KEY', ''))
    API_SECRET = getattr(config, 'ALPACA_SECRET', os.getenv('ALPACA_SECRET', ''))
    BASE_URL = getattr(config, 'ALPACA_BASE_URL', os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'))
except ImportError:
    API_KEY = os.getenv('ALPACA_KEY', '')
    API_SECRET = os.getenv('ALPACA_SECRET', '')
    BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

def init_alpaca():
    """Initialize Alpaca API"""
    if not ALPACA_AVAILABLE:
        raise ImportError("alpaca-trade-api not installed")
    return tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def get_all_trades(api: tradeapi.REST, days_back: int = 1) -> List[Dict]:
    """Get all trades from Alpaca"""
    est = pytz.timezone('US/Eastern')
    end_date = datetime.now(est)
    start_date = end_date - timedelta(days=days_back)
    
    orders = api.list_orders(
        status='filled',
        after=start_date.isoformat(),
        limit=500,
        nested=True
    )
    
    trades = []
    for order in orders:
        if hasattr(order, 'asset_class') and order.asset_class in ['option', 'us_option']:
            if order.filled_at:
                try:
                    if isinstance(order.filled_at, str):
                        filled_at = datetime.fromisoformat(order.filled_at.replace('Z', '+00:00')).astimezone(est)
                    else:
                        if hasattr(order.filled_at, 'astimezone'):
                            filled_at = order.filled_at.astimezone(est)
                        else:
                            filled_at = datetime.fromtimestamp(order.filled_at).astimezone(est)
                    
                    trades.append({
                        'id': order.id,
                        'symbol': order.symbol,
                        'side': order.side.lower(),
                        'qty': float(order.filled_qty),
                        'price': float(order.filled_avg_price),
                        'filled_at': filled_at,
                        'order_type': order.order_type,
                        'status': order.status
                    })
                except Exception as e:
                    continue
    
    return sorted(trades, key=lambda x: x['filled_at'])

def match_buy_sell_pairs(trades: List[Dict]) -> List[Dict]:
    """Match buy/sell pairs using FIFO"""
    open_positions = {}  # symbol -> list of buys
    matched_trades = []
    
    for trade in trades:
        symbol = trade['symbol']
        side = trade['side']
        
        if side == 'buy':
            if symbol not in open_positions:
                open_positions[symbol] = []
            open_positions[symbol].append(trade)
        
        elif side == 'sell':
            if symbol in open_positions and len(open_positions[symbol]) > 0:
                # Match with earliest buy (FIFO)
                buy = open_positions[symbol].pop(0)
                
                duration = (trade['filled_at'] - buy['filled_at']).total_seconds() / 60
                
                matched_trades.append({
                    'symbol': symbol,
                    'buy_time': buy['filled_at'],
                    'sell_time': trade['filled_at'],
                    'buy_price': buy['price'],
                    'sell_price': trade['price'],
                    'qty': buy['qty'],
                    'entry_cost': buy['price'] * buy['qty'] * 100,
                    'exit_value': trade['price'] * trade['qty'] * 100,
                    'pnl_dollar': (trade['price'] - buy['price']) * buy['qty'] * 100,
                    'pnl_pct': ((trade['price'] - buy['price']) / buy['price']) * 100 if buy['price'] > 0 else 0,
                    'duration_minutes': duration,
                    'buy_order_id': buy['id'],
                    'sell_order_id': trade['id']
                })
    
    # Unmatched buys (still open)
    for symbol, buys in open_positions.items():
        for buy in buys:
            matched_trades.append({
                'symbol': symbol,
                'buy_time': buy['filled_at'],
                'sell_time': None,
                'buy_price': buy['price'],
                'sell_price': None,
                'qty': buy['qty'],
                'entry_cost': buy['price'] * buy['qty'] * 100,
                'exit_value': None,
                'pnl_dollar': None,
                'pnl_pct': None,
                'duration_minutes': None,
                'buy_order_id': buy['id'],
                'sell_order_id': None,
                'status': 'OPEN'
            })
    
    return matched_trades

def parse_log_file(log_file: str) -> Dict[str, List[Dict]]:
    """Parse log file to extract decision triggers"""
    decision_logs = {
        'entries': [],
        'exits': [],
        'rl_decisions': [],
        'gap_detections': [],
        'stop_losses': [],
        'take_profits': [],
        'trailing_stops': []
    }
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            # Extract timestamp
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if not timestamp_match:
                continue
            timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
            
            # Entry decisions
            if 'EXECUTED: BUY' in line or 'GAP-BASED ACTION' in line:
                symbol_match = re.search(r'([A-Z]\d{8}[CP]\d{8})', line)
                if symbol_match:
                    decision_logs['entries'].append({
                        'timestamp': timestamp,
                        'symbol': symbol_match.group(1),
                        'line': line.strip(),
                        'source': 'GAP' if 'GAP-BASED' in line else 'RL'
                    })
            
            # Exit decisions
            if 'TP1' in line or 'TP2' in line or 'TP3' in line:
                symbol_match = re.search(r'([A-Z]\d{8}[CP]\d{8})', line)
                if symbol_match:
                    tp_match = re.search(r'TP(\d)', line)
                    decision_logs['take_profits'].append({
                        'timestamp': timestamp,
                        'symbol': symbol_match.group(1),
                        'tp_level': int(tp_match.group(1)) if tp_match else None,
                        'line': line.strip()
                    })
            
            if 'STOP-LOSS' in line or 'STOP LOSS' in line:
                symbol_match = re.search(r'([A-Z]\d{8}[CP]\d{8})', line)
                if symbol_match:
                    decision_logs['stop_losses'].append({
                        'timestamp': timestamp,
                        'symbol': symbol_match.group(1),
                        'line': line.strip()
                    })
            
            if 'TRAILING STOP' in line:
                symbol_match = re.search(r'([A-Z]\d{8}[CP]\d{8})', line)
                if symbol_match:
                    decision_logs['trailing_stops'].append({
                        'timestamp': timestamp,
                        'symbol': symbol_match.group(1),
                        'line': line.strip()
                    })
            
            if 'RL Debug' in line:
                decision_logs['rl_decisions'].append({
                    'timestamp': timestamp,
                    'line': line.strip()
                })
        
    except FileNotFoundError:
        pass
    
    return decision_logs

def analyze_trade_decision(trade: Dict, log_data: Dict) -> Dict:
    """Analyze what triggered a trade decision"""
    analysis = {
        'entry_trigger': 'UNKNOWN',
        'entry_source': 'UNKNOWN',  # RL, GAP, or UNKNOWN
        'exit_trigger': 'UNKNOWN',
        'exit_reason': 'UNKNOWN',
        'was_stop_loss': False,
        'was_take_profit': False,
        'was_trailing_stop': False,
        'was_early_exit': False,
        'was_eod_exit': False
    }
    
    # Determine entry trigger from logs
    entry_time = trade['buy_time']
    symbol = trade['symbol']
    
    # Find closest entry log within 5 minutes
    for entry_log in log_data['entries']:
        time_diff = abs((entry_log['timestamp'] - entry_time).total_seconds())
        if entry_log['symbol'] == symbol and time_diff < 300:  # 5 minutes
            analysis['entry_source'] = entry_log.get('source', 'UNKNOWN')
            if entry_log['source'] == 'GAP':
                analysis['entry_trigger'] = 'GAP_DETECTION'
            else:
                analysis['entry_trigger'] = 'RL_MODEL'
            break
    
    # Determine exit trigger
    if trade.get('status') == 'OPEN':
        analysis['exit_trigger'] = 'STILL_OPEN'
        analysis['exit_reason'] = 'Position still open'
    elif trade.get('pnl_pct') is not None:
        pnl_pct = trade['pnl_pct']
        sell_time = trade.get('sell_time')
        
        # Check logs for exit reasons
        exit_time = sell_time if sell_time else None
        
        if exit_time:
            # Check for stop loss
            for sl_log in log_data['stop_losses']:
                if sl_log['symbol'] == symbol:
                    time_diff = abs((sl_log['timestamp'] - exit_time).total_seconds())
                    if time_diff < 300:  # 5 minutes
                        analysis['exit_trigger'] = 'STOP_LOSS'
                        analysis['was_stop_loss'] = True
                        analysis['exit_reason'] = f'Stop loss at {pnl_pct:.1f}%'
                        break
            
            # Check for take profit
            for tp_log in log_data['take_profits']:
                if tp_log['symbol'] == symbol:
                    time_diff = abs((tp_log['timestamp'] - exit_time).total_seconds())
                    if time_diff < 300:  # 5 minutes
                        analysis['exit_trigger'] = f'TP{tp_log.get("tp_level", "?")}'
                        analysis['was_take_profit'] = True
                        analysis['exit_reason'] = f'Take profit TP{tp_log.get("tp_level", "?")} at {pnl_pct:.1f}%'
                        break
            
            # Check for trailing stop
            for trail_log in log_data['trailing_stops']:
                if trail_log['symbol'] == symbol:
                    time_diff = abs((trail_log['timestamp'] - exit_time).total_seconds())
                    if time_diff < 300:  # 5 minutes
                        analysis['exit_trigger'] = 'TRAILING_STOP'
                        analysis['was_trailing_stop'] = True
                        analysis['exit_reason'] = f'Trailing stop at {pnl_pct:.1f}%'
                        break
        
        # Infer from P&L if no log match
        if analysis['exit_trigger'] == 'UNKNOWN':
            if pnl_pct <= -15:
                analysis['exit_trigger'] = 'STOP_LOSS'
                analysis['was_stop_loss'] = True
                analysis['exit_reason'] = f'Stop loss at {pnl_pct:.1f}%'
            elif pnl_pct >= 40:
                analysis['exit_trigger'] = 'TAKE_PROFIT'
                analysis['was_take_profit'] = True
                analysis['exit_reason'] = f'Take profit at {pnl_pct:.1f}%'
            
            # Check if early exit
            duration = trade.get('duration_minutes', 0)
            if duration < 60:
                analysis['was_early_exit'] = True
                if analysis['exit_reason'] == 'UNKNOWN':
                    if pnl_pct > 0:
                        analysis['exit_reason'] = f'Early profit-taking at {pnl_pct:.1f}% after {duration:.0f} min'
                    else:
                        analysis['exit_reason'] = f'Early exit at {pnl_pct:.1f}% after {duration:.0f} min'
        
        # Check for EOD exit
        if exit_time:
            hour = exit_time.hour
            if hour == 15 and exit_time.minute >= 50:  # After 3:50 PM
                analysis['was_eod_exit'] = True
                if analysis['exit_reason'] == 'UNKNOWN':
                    analysis['exit_reason'] = f'EOD exit at {pnl_pct:.1f}%'
    
    return analysis

def generate_comprehensive_report(trades: List[Dict], log_data: Dict) -> str:
    """Generate comprehensive analysis report"""
    
    report = []
    report.append("=" * 100)
    report.append("📊 COMPREHENSIVE TRADE ANALYSIS REPORT")
    report.append("=" * 100)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Trades Analyzed: {len(trades)}")
    report.append("")
    
    # Calculate statistics
    completed = [t for t in trades if t.get('status') != 'OPEN' and t.get('pnl_pct') is not None]
    open_trades = [t for t in trades if t.get('status') == 'OPEN']
    
    if completed:
        total_pnl = sum(t['pnl_dollar'] for t in completed)
        avg_pnl = total_pnl / len(completed)
        win_rate = len([t for t in completed if t['pnl_pct'] > 0]) / len(completed) * 100
        winning_trades = [t for t in completed if t['pnl_pct'] > 0]
        losing_trades = [t for t in completed if t['pnl_pct'] < 0]
        avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0
        
        report.append("=" * 100)
        report.append("📈 SUMMARY STATISTICS")
        report.append("=" * 100)
        report.append(f"Completed Trades: {len(completed)}")
        report.append(f"Open Trades: {len(open_trades)}")
        report.append(f"Total Realized P&L: ${total_pnl:,.2f}")
        report.append(f"Average P&L per Trade: ${avg_pnl:,.2f}")
        report.append(f"Win Rate: {win_rate:.1f}%")
        report.append(f"Winning Trades: {len(winning_trades)}")
        report.append(f"Losing Trades: {len(losing_trades)}")
        if avg_win > 0:
            report.append(f"Average Win: +{avg_win:.2f}%")
        if avg_loss < 0:
            report.append(f"Average Loss: {avg_loss:.2f}%")
        report.append("")
    
    # Entry source analysis
    if completed:
        entry_sources = {}
        for trade in completed:
            analysis = analyze_trade_decision(trade, log_data)
            source = analysis['entry_source']
            entry_sources[source] = entry_sources.get(source, 0) + 1
        
        report.append("=" * 100)
        report.append("🎯 ENTRY DECISION ANALYSIS")
        report.append("=" * 100)
        for source, count in entry_sources.items():
            pct = (count / len(completed)) * 100
            report.append(f"{source}: {count} trades ({pct:.1f}%)")
        report.append("")
    
    # Exit trigger analysis
    if completed:
        exit_triggers = {}
        for trade in completed:
            analysis = analyze_trade_decision(trade, log_data)
            trigger = analysis['exit_trigger']
            exit_triggers[trigger] = exit_triggers.get(trigger, 0) + 1
        
        report.append("=" * 100)
        report.append("🚪 EXIT DECISION ANALYSIS")
        report.append("=" * 100)
        for trigger, count in sorted(exit_triggers.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(completed)) * 100
            report.append(f"{trigger}: {count} trades ({pct:.1f}%)")
        report.append("")
    
    # Trade-by-trade analysis
    report.append("=" * 100)
    report.append("📋 DETAILED TRADE-BY-TRADE ANALYSIS")
    report.append("=" * 100)
    report.append("")
    
    for i, trade in enumerate(completed, 1):
        analysis = analyze_trade_decision(trade, log_data)
        
        report.append(f"Trade #{i}: {trade['symbol']}")
        report.append("-" * 100)
        report.append(f"Entry Time: {trade['buy_time'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
        report.append(f"Entry Price: ${trade['buy_price']:.4f}")
        report.append(f"Quantity: {trade['qty']} contracts")
        report.append(f"Entry Cost: ${trade['entry_cost']:,.2f}")
        report.append("")
        report.append(f"Exit Time: {trade['sell_time'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
        report.append(f"Exit Price: ${trade['sell_price']:.4f}")
        report.append(f"Duration: {trade['duration_minutes']:.1f} minutes")
        report.append(f"P&L: ${trade['pnl_dollar']:,.2f} ({trade['pnl_pct']:+.2f}%)")
        report.append("")
        report.append(f"ENTRY DECISION:")
        report.append(f"  Source: {analysis['entry_source']}")
        report.append(f"  Trigger: {analysis['entry_trigger']}")
        report.append("")
        report.append(f"EXIT DECISION:")
        report.append(f"  Trigger: {analysis['exit_trigger']}")
        report.append(f"  Reason: {analysis['exit_reason']}")
        if analysis['was_early_exit']:
            report.append(f"  ⚠️  EARLY EXIT (before 60 min)")
        if analysis['was_eod_exit']:
            report.append(f"  🕐  EOD EXIT")
        report.append("")
        report.append("")
    
    # Loss analysis
    if losing_trades:
        report.append("=" * 100)
        report.append("🔴 LOSS ANALYSIS - WHY TRADES LOST MONEY")
        report.append("=" * 100)
        report.append(f"Total Losing Trades: {len(losing_trades)}")
        report.append(f"Total Loss: ${sum(t['pnl_dollar'] for t in losing_trades):,.2f}")
        report.append(f"Average Loss: ${sum(t['pnl_dollar'] for t in losing_trades) / len(losing_trades):,.2f}")
        report.append("")
        
        loss_triggers = {}
        for trade in losing_trades:
            analysis = analyze_trade_decision(trade, log_data)
            trigger = analysis['exit_trigger']
            loss_triggers[trigger] = loss_triggers.get(trigger, {'count': 0, 'total_loss': 0})
            loss_triggers[trigger]['count'] += 1
            loss_triggers[trigger]['total_loss'] += trade['pnl_dollar']
        
        report.append("Loss Breakdown by Exit Trigger:")
        for trigger, data in sorted(loss_triggers.items(), key=lambda x: x[1]['count'], reverse=True):
            avg_loss = data['total_loss'] / data['count']
            report.append(f"  {trigger}: {data['count']} trades | Total: ${data['total_loss']:,.2f} | Avg: ${avg_loss:,.2f}")
        report.append("")
        
        report.append("Losing Trades Detailed:")
        for trade in losing_trades:
            analysis = analyze_trade_decision(trade, log_data)
            report.append(f"  • {trade['symbol']}: {trade['pnl_pct']:.2f}% | {analysis['exit_reason']} | Duration: {trade['duration_minutes']:.1f} min")
        report.append("")
    
    # Pattern analysis
    if completed:
        report.append("=" * 100)
        report.append("🔍 PATTERN ANALYSIS")
        report.append("=" * 100)
        
        # Duration analysis
        durations = [t['duration_minutes'] for t in completed if t.get('duration_minutes')]
        if durations:
            report.append(f"Average Trade Duration: {np.mean(durations):.1f} minutes")
            report.append(f"Median Trade Duration: {np.median(durations):.1f} minutes")
            report.append(f"Shortest Trade: {min(durations):.1f} minutes")
            report.append(f"Longest Trade: {max(durations):.1f} minutes")
            report.append("")
        
        # Early exit analysis
        early_exits = [t for t in completed if t.get('duration_minutes') and t['duration_minutes'] < 60]
        if early_exits:
            report.append(f"Early Exits (< 60 min): {len(early_exits)} trades ({len(early_exits)/len(completed)*100:.1f}%)")
            early_exit_pnl = sum(t['pnl_pct'] for t in early_exits)
            early_exit_dollar = sum(t['pnl_dollar'] for t in early_exits)
            report.append(f"Early Exit Total P&L: {early_exit_pnl:+.2f}% (${early_exit_dollar:,.2f})")
            report.append("")
        
        # Profit target analysis
        tp_trades = [t for t in completed if analyze_trade_decision(t, log_data)['was_take_profit']]
        if tp_trades:
            report.append(f"Take Profit Trades: {len(tp_trades)} trades")
            tp_pnl = sum(t['pnl_pct'] for t in tp_trades)
            report.append(f"Total TP P&L: {tp_pnl:+.2f}%")
            report.append("")
        
        # Stop loss analysis
        sl_trades = [t for t in completed if analyze_trade_decision(t, log_data)['was_stop_loss']]
        if sl_trades:
            report.append(f"Stop Loss Trades: {len(sl_trades)} trades")
            sl_pnl = sum(t['pnl_pct'] for t in sl_trades)
            report.append(f"Total SL P&L: {sl_pnl:.2f}%")
            report.append("")
    
    return "\n".join(report)

def main():
    print("=" * 100)
    print("🔍 COMPREHENSIVE TRADE ANALYSIS")
    print("=" * 100)
    print()
    
    if not ALPACA_AVAILABLE:
        print("❌ Alpaca API not available. Cannot fetch trades.")
        return
    
    try:
        api = init_alpaca()
        print("✅ Connected to Alpaca API")
    except Exception as e:
        print(f"❌ Failed to connect to Alpaca: {e}")
        return
    
    print("📥 Fetching all trades...")
    all_trades = get_all_trades(api, days_back=1)
    print(f"   Found {len(all_trades)} filled orders")
    
    print("🔄 Matching buy/sell pairs...")
    matched_trades = match_buy_sell_pairs(all_trades)
    print(f"   Matched {len(matched_trades)} trades")
    
    print("📋 Parsing log files...")
    log_dir = "logs"
    log_files = []
    if os.path.exists(log_dir):
        log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.log')]
        log_files.sort(reverse=True)  # Most recent first
    
    log_data = {'entries': [], 'exits': [], 'rl_decisions': [], 'gap_detections': [], 'stop_losses': [], 'take_profits': [], 'trailing_stops': []}
    if log_files:
        print(f"   Found {len(log_files)} log files")
        # Parse most recent log file
        log_data = parse_log_file(log_files[0])
        print(f"   Found {len(log_data['entries'])} entries, {len(log_data['take_profits'])} TPs, {len(log_data['stop_losses'])} SLs")
    else:
        print("   ⚠️  No log files found")
    
    print("📊 Generating comprehensive analysis...")
    report = generate_comprehensive_report(matched_trades, log_data)
    
    # Save report
    report_file = f"comprehensive_trade_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print()
    print("=" * 100)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 100)
    print(f"Report saved to: {report_file}")
    print()
    print(report[:5000])  # Print first 5000 chars
    
    return matched_trades

if __name__ == "__main__":
    main()

