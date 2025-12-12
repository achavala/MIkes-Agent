#!/usr/bin/env python3
"""
Detailed Trade Analysis
Analyzes all trades to understand:
1. What triggered each entry decision
2. Why trades were closed early
3. Performance patterns
4. Loss reasons
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠️ Alpaca API not available")

try:
    from trade_database import TradeDatabase
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("⚠️ Trade database not available")

# Alpaca config
API_KEY = os.getenv('ALPACA_KEY', '')
API_SECRET = os.getenv('ALPACA_SECRET', '')
BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

def init_alpaca():
    """Initialize Alpaca API"""
    if not ALPACA_AVAILABLE:
        raise ImportError("alpaca-trade-api not installed")
    return tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def get_all_trades(api: tradeapi.REST, days_back: int = 30) -> List[Dict]:
    """Get all trades from Alpaca"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    orders = api.list_orders(
        status='filled',
        after=start_date.isoformat(),
        limit=500
    )
    
    trades = []
    for order in orders:
        if order.asset_class == 'option' and order.filled_at:
            trades.append({
                'id': order.id,
                'symbol': order.symbol,
                'side': order.side,
                'qty': float(order.filled_qty),
                'price': float(order.filled_avg_price),
                'filled_at': pd.to_datetime(order.filled_at),
                'order_type': order.order_type,
                'status': order.status
            })
    
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
                    'pnl_pct': ((trade['price'] - buy['price']) / buy['price']) * 100,
                    'duration_minutes': (trade['filled_at'] - buy['filled_at']).total_seconds() / 60,
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

def analyze_trade_decision(trade: Dict, risk_mgr_data: Optional[Dict] = None) -> Dict:
    """Analyze what triggered a trade decision"""
    analysis = {
        'entry_trigger': 'UNKNOWN',
        'exit_trigger': 'UNKNOWN',
        'exit_reason': 'UNKNOWN',
        'was_stop_loss': False,
        'was_take_profit': False,
        'was_early_exit': False
    }
    
    # Determine entry trigger
    # This would need access to log files or decision history
    # For now, we'll infer from trade characteristics
    
    # Determine exit trigger
    if trade.get('status') == 'OPEN':
        analysis['exit_trigger'] = 'STILL_OPEN'
    elif trade.get('pnl_pct') is not None:
        pnl_pct = trade['pnl_pct']
        
        # Check if it was a stop loss
        if pnl_pct <= -15:
            analysis['exit_trigger'] = 'STOP_LOSS'
            analysis['was_stop_loss'] = True
            analysis['exit_reason'] = f'Stop loss triggered at -{abs(pnl_pct):.1f}%'
        elif pnl_pct <= -20:
            analysis['exit_trigger'] = 'HARD_STOP'
            analysis['was_stop_loss'] = True
            analysis['exit_reason'] = f'Hard stop at -{abs(pnl_pct):.1f}%'
        
        # Check if it was a take profit
        elif pnl_pct >= 40:
            analysis['exit_trigger'] = 'TAKE_PROFIT'
            analysis['was_take_profit'] = True
            analysis['exit_reason'] = f'Take profit at +{pnl_pct:.1f}%'
        
        # Check if early exit
        elif trade.get('duration_minutes'):
            duration = trade['duration_minutes']
            if duration < 60 and pnl_pct > 0:
                analysis['exit_trigger'] = 'EARLY_EXIT'
                analysis['was_early_exit'] = True
                analysis['exit_reason'] = f'Early exit after {duration:.0f} min at +{pnl_pct:.1f}%'
            elif duration < 60 and pnl_pct < 0:
                analysis['exit_trigger'] = 'EARLY_STOP'
                analysis['exit_reason'] = f'Early stop after {duration:.0f} min at {pnl_pct:.1f}%'
    
    return analysis

def generate_detailed_report(trades: List[Dict]) -> str:
    """Generate detailed analysis report"""
    
    report = []
    report.append("=" * 80)
    report.append("📊 DETAILED TRADE ANALYSIS REPORT")
    report.append("=" * 80)
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
        avg_win = np.mean([t['pnl_pct'] for t in completed if t['pnl_pct'] > 0]) if [t for t in completed if t['pnl_pct'] > 0] else 0
        avg_loss = np.mean([t['pnl_pct'] for t in completed if t['pnl_pct'] < 0]) if [t for t in completed if t['pnl_pct'] < 0] else 0
        
        report.append("=" * 80)
        report.append("📈 SUMMARY STATISTICS")
        report.append("=" * 80)
        report.append(f"Completed Trades: {len(completed)}")
        report.append(f"Open Trades: {len(open_trades)}")
        report.append(f"Total Realized P&L: ${total_pnl:,.2f}")
        report.append(f"Average P&L per Trade: ${avg_pnl:,.2f}")
        report.append(f"Win Rate: {win_rate:.1f}%")
        if avg_win > 0:
            report.append(f"Average Win: +{avg_win:.2f}%")
        if avg_loss < 0:
            report.append(f"Average Loss: {avg_loss:.2f}%")
        report.append("")
    
    # Trade-by-trade analysis
    report.append("=" * 80)
    report.append("📋 TRADE-BY-TRADE DETAILED ANALYSIS")
    report.append("=" * 80)
    report.append("")
    
    for i, trade in enumerate(trades, 1):
        analysis = analyze_trade_decision(trade)
        
        report.append(f"Trade #{i}: {trade['symbol']}")
        report.append("-" * 80)
        report.append(f"Entry Time: {trade['buy_time']}")
        report.append(f"Entry Price: ${trade['buy_price']:.4f}")
        report.append(f"Quantity: {trade['qty']} contracts")
        report.append(f"Entry Cost: ${trade['entry_cost']:,.2f}")
        
        if trade.get('sell_time'):
            report.append(f"Exit Time: {trade['sell_time']}")
            report.append(f"Exit Price: ${trade['sell_price']:.4f}")
            report.append(f"Duration: {trade['duration_minutes']:.1f} minutes")
            report.append(f"P&L: ${trade['pnl_dollar']:,.2f} ({trade['pnl_pct']:+.2f}%)")
        else:
            report.append(f"Status: OPEN (not yet closed)")
        
        report.append(f"Entry Trigger: {analysis['entry_trigger']}")
        report.append(f"Exit Trigger: {analysis['exit_trigger']}")
        report.append(f"Exit Reason: {analysis['exit_reason']}")
        report.append("")
    
    # Loss analysis
    if completed:
        losing_trades = [t for t in completed if t['pnl_pct'] < 0]
        if losing_trades:
            report.append("=" * 80)
            report.append("🔴 LOSS ANALYSIS")
            report.append("=" * 80)
            report.append(f"Total Losing Trades: {len(losing_trades)}")
            report.append(f"Total Loss: ${sum(t['pnl_dollar'] for t in losing_trades):,.2f}")
            report.append(f"Average Loss: ${sum(t['pnl_dollar'] for t in losing_trades) / len(losing_trades):,.2f}")
            report.append("")
            
            report.append("Losing Trades Breakdown:")
            for trade in losing_trades:
                analysis = analyze_trade_decision(trade)
                report.append(f"  • {trade['symbol']}: {trade['pnl_pct']:.2f}% | {analysis['exit_reason']}")
            report.append("")
    
    # Pattern analysis
    if completed:
        report.append("=" * 80)
        report.append("🔍 PATTERN ANALYSIS")
        report.append("=" * 80)
        
        # Duration analysis
        durations = [t['duration_minutes'] for t in completed if t.get('duration_minutes')]
        if durations:
            report.append(f"Average Trade Duration: {np.mean(durations):.1f} minutes")
            report.append(f"Shortest Trade: {min(durations):.1f} minutes")
            report.append(f"Longest Trade: {max(durations):.1f} minutes")
            report.append("")
        
        # Early exit analysis
        early_exits = [t for t in completed if t.get('duration_minutes') and t['duration_minutes'] < 60]
        if early_exits:
            report.append(f"Early Exits (< 60 min): {len(early_exits)} trades")
            early_exit_pnl = sum(t['pnl_pct'] for t in early_exits)
            report.append(f"Early Exit Total P&L: {early_exit_pnl:+.2f}%")
            report.append("")
    
    return "\n".join(report)

def main():
    print("=" * 80)
    print("🔍 DETAILED TRADE ANALYSIS")
    print("=" * 80)
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
    all_trades = get_all_trades(api, days_back=7)
    print(f"   Found {len(all_trades)} filled orders")
    
    print("🔄 Matching buy/sell pairs...")
    matched_trades = match_buy_sell_pairs(all_trades)
    print(f"   Matched {len(matched_trades)} trades")
    
    print("📊 Generating detailed analysis...")
    report = generate_detailed_report(matched_trades)
    
    # Save report
    report_file = f"trade_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print()
    print("=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Report saved to: {report_file}")
    print()
    print(report)
    
    return matched_trades

if __name__ == "__main__":
    main()

