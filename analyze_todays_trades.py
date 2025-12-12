#!/usr/bin/env python3
"""
Detailed Analysis of Today's Trades
Analyzes why losses occurred and provides insights for improvement
"""

import os
import sys
from datetime import datetime, date
import pytz
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Error: alpaca-trade-api not installed")

import config

# Import trade database
try:
    from trade_database import TradeDatabase
    TRADE_DB_AVAILABLE = True
except ImportError:
    TRADE_DB_AVAILABLE = False
    print("Warning: trade_database module not available")

def analyze_todays_trades():
    """Comprehensive analysis of today's trades"""
    print("=" * 100)
    print("📊 DETAILED ANALYSIS: TODAY'S TRADES")
    print("=" * 100)
    print()
    
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).date()
    
    # 1. Get trades from Alpaca
    print("1️⃣ COLLECTING TRADE DATA")
    print("-" * 100)
    
    try:
        api = tradeapi.REST(
            config.ALPACA_KEY,
            config.ALPACA_SECRET,
            config.ALPACA_BASE_URL,
            api_version='v2'
        )
        print("✅ Connected to Alpaca API")
    except Exception as e:
        print(f"❌ Alpaca connection failed: {e}")
        return
    
    # Get all filled orders from today
    try:
        all_orders = api.list_orders(status='filled', limit=500, nested=True)
        today_orders = []
        
        for order in all_orders:
            try:
                # Get order date
                if hasattr(order, 'filled_at') and order.filled_at:
                    if isinstance(order.filled_at, str):
                        filled_dt = datetime.fromisoformat(order.filled_at.replace('Z', '+00:00')).astimezone(est)
                    else:
                        filled_dt = order.filled_at.astimezone(est)
                    order_date = filled_dt.date()
                else:
                    continue
                
                if order_date == today:
                    today_orders.append(order)
            except Exception:
                continue
        
        print(f"✅ Found {len(today_orders)} filled orders today")
        
    except Exception as e:
        print(f"❌ Error fetching orders: {e}")
        return
    
    print()
    
    # 2. Analyze orders by symbol
    print("2️⃣ TRADE BREAKDOWN BY SYMBOL")
    print("-" * 100)
    
    if not today_orders:
        print("⚠️  No trades found for today")
        return
    
    # Group by symbol
    trades_by_symbol = {}
    for order in today_orders:
        symbol = order.symbol
        if symbol not in trades_by_symbol:
            trades_by_symbol[symbol] = {'buys': [], 'sells': []}
        
        side = order.side.lower()
        if side == 'buy':
            trades_by_symbol[symbol]['buys'].append({
                'qty': float(order.filled_qty),
                'price': float(order.filled_avg_price),
                'time': filled_dt if 'filled_dt' in locals() else None,
                'order_id': order.id
            })
        else:
            trades_by_symbol[symbol]['sells'].append({
                'qty': float(order.filled_qty),
                'price': float(order.filled_avg_price),
                'time': filled_dt if 'filled_dt' in locals() else None,
                'order_id': order.id
            })
    
    # 3. Calculate P&L for each symbol
    print("3️⃣ P&L ANALYSIS BY SYMBOL")
    print("-" * 100)
    
    total_pnl = 0.0
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    
    detailed_trades = []
    
    for symbol, trades in trades_by_symbol.items():
        buys = sorted(trades['buys'], key=lambda x: x['time'] if x['time'] else datetime.min)
        sells = sorted(trades['sells'], key=lambda x: x['time'] if x['time'] else datetime.min)
        
        # Match buys and sells using FIFO
        buy_index = 0
        sell_index = 0
        
        while buy_index < len(buys) and sell_index < len(sells):
            buy = buys[buy_index]
            sell = sells[sell_index]
            
            # Determine quantity matched
            matched_qty = min(buy['qty'], sell['qty'])
            
            # Calculate P&L
            entry_cost = buy['price'] * matched_qty * 100  # Options: qty * price * 100
            exit_proceeds = sell['price'] * matched_qty * 100
            trade_pnl = exit_proceeds - entry_cost
            trade_pnl_pct = ((sell['price'] - buy['price']) / buy['price']) * 100 if buy['price'] > 0 else 0
            
            # Track trade
            total_trades += 1
            if trade_pnl > 0:
                winning_trades += 1
            else:
                losing_trades += 1
            
            total_pnl += trade_pnl
            
            detailed_trades.append({
                'symbol': symbol,
                'entry_price': buy['price'],
                'exit_price': sell['price'],
                'qty': matched_qty,
                'pnl_dollar': trade_pnl,
                'pnl_pct': trade_pnl_pct,
                'entry_time': buy['time'],
                'exit_time': sell['time'],
                'duration': (sell['time'] - buy['time']).total_seconds() / 60 if buy['time'] and sell['time'] else None
            })
            
            # Update quantities
            buy['qty'] -= matched_qty
            sell['qty'] -= matched_qty
            
            if buy['qty'] <= 0:
                buy_index += 1
            if sell['qty'] <= 0:
                sell_index += 1
    
    # 4. Display detailed analysis
    print(f"Total Completed Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades} ({winning_trades/total_trades*100:.1f}%)" if total_trades > 0 else "N/A")
    print(f"Losing Trades: {losing_trades} ({losing_trades/total_trades*100:.1f}%)" if total_trades > 0 else "N/A")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print()
    
    # Sort by P&L
    detailed_trades.sort(key=lambda x: x['pnl_dollar'])
    
    print("4️⃣ TRADE-BY-TRADE ANALYSIS (Sorted by P&L)")
    print("-" * 100)
    
    for i, trade in enumerate(detailed_trades, 1):
        pnl_sign = "✅" if trade['pnl_dollar'] > 0 else "❌"
        duration_str = f"{trade['duration']:.1f} min" if trade['duration'] else "N/A"
        
        print(f"\nTrade #{i}: {trade['symbol']} {pnl_sign}")
        print(f"   Entry: ${trade['entry_price']:.2f} @ {trade['entry_time'].strftime('%H:%M:%S') if trade['entry_time'] else 'N/A'}")
        print(f"   Exit:  ${trade['exit_price']:.2f} @ {trade['exit_time'].strftime('%H:%M:%S') if trade['exit_time'] else 'N/A'}")
        print(f"   Quantity: {trade['qty']} contracts")
        print(f"   Duration: {duration_str}")
        print(f"   P&L: ${trade['pnl_dollar']:,.2f} ({trade['pnl_pct']:+.2f}%)")
        
        # Analyze why loss occurred
        if trade['pnl_dollar'] < 0:
            print(f"   🔍 Loss Analysis:")
            price_change_pct = trade['pnl_pct']
            if price_change_pct < -15:
                print(f"      ⚠️  Large loss ({price_change_pct:.1f}%) - Stop loss should have triggered at -15%")
            elif price_change_pct < -10:
                print(f"      ⚠️  Moderate loss ({price_change_pct:.1f}%) - Risk management may need adjustment")
            else:
                print(f"      ℹ️  Small loss ({price_change_pct:.1f}%) - Within normal trading range")
    
    print()
    print("5️⃣ ROOT CAUSE ANALYSIS")
    print("-" * 100)
    
    # Analyze patterns
    losses = [t for t in detailed_trades if t['pnl_dollar'] < 0]
    wins = [t for t in detailed_trades if t['pnl_dollar'] > 0]
    
    if losses:
        avg_loss = sum(t['pnl_dollar'] for t in losses) / len(losses)
        max_loss = min(t['pnl_dollar'] for t in losses)
        avg_loss_pct = sum(t['pnl_pct'] for t in losses) / len(losses)
        max_loss_pct = min(t['pnl_pct'] for t in losses)
        
        print(f"Loss Statistics:")
        print(f"   Average Loss: ${avg_loss:,.2f} ({avg_loss_pct:.2f}%)")
        print(f"   Maximum Loss: ${max_loss:,.2f} ({max_loss_pct:.2f}%)")
        print(f"   Number of Losses: {len(losses)}")
        print()
        
        # Check stop loss execution
        stops_missed = [t for t in losses if t['pnl_pct'] < -15]
        if stops_missed:
            print(f"🚨 CRITICAL: {len(stops_missed)} trades exceeded -15% stop loss:")
            for trade in stops_missed:
                print(f"   - {trade['symbol']}: {trade['pnl_pct']:.2f}% loss (should have stopped at -15%)")
            print()
            print("   🔧 ROOT CAUSE: Stop loss not executing properly")
            print("      - Check stop loss logic in check_stop_losses()")
            print("      - Verify position tracking is synced with Alpaca")
            print("      - Ensure stop loss checks run frequently enough")
            print()
        
        # Check if losses are from early exits
        early_exits = [t for t in losses if t['duration'] and t['duration'] < 30]
        if early_exits:
            print(f"⚠️  {len(early_exits)} losses from positions held < 30 minutes:")
            print("   🔧 ROOT CAUSE: Exiting positions too early")
            print("      - May be cutting winners short")
            print("      - Consider holding longer for 0DTE moves")
            print()
    
    if wins:
        avg_win = sum(t['pnl_dollar'] for t in wins) / len(wins)
        max_win = max(t['pnl_dollar'] for t in wins)
        avg_win_pct = sum(t['pnl_pct'] for t in wins) / len(wins)
        max_win_pct = max(t['pnl_pct'] for t in wins)
        
        print(f"Win Statistics:")
        print(f"   Average Win: ${avg_win:,.2f} ({avg_win_pct:.2f}%)")
        print(f"   Maximum Win: ${max_win:,.2f} ({max_win_pct:.2f}%)")
        print(f"   Number of Wins: {len(wins)}")
        print()
        
        # Check if wins are too small
        small_wins = [t for t in wins if t['pnl_pct'] < 10 and t['pnl_dollar'] < 50]
        if small_wins:
            print(f"⚠️  {len(small_wins)} small wins (< 10% or < $50):")
            print("   🔧 ROOT CAUSE: Taking profits too early")
            print("      - Consider letting winners run longer")
            print("      - Adjust take-profit levels for 0DTE")
            print()
    
    # Overall assessment
    print("6️⃣ OVERALL ASSESSMENT")
    print("-" * 100)
    
    if total_pnl < 0:
        print(f"❌ NET LOSS TODAY: ${abs(total_pnl):,.2f}")
        print()
        print("Primary Issues Identified:")
        print()
        
        # Check win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        if win_rate < 50:
            print(f"1. ⚠️  LOW WIN RATE: {win_rate:.1f}%")
            print("   - Model may not be selecting optimal entries")
            print("   - Consider: Better entry timing, gap detection, market regime filters")
            print()
        
        # Check risk/reward
        if wins and losses:
            avg_win = sum(t['pnl_dollar'] for t in wins) / len(wins)
            avg_loss = abs(sum(t['pnl_dollar'] for t in losses) / len(losses))
            risk_reward = avg_win / avg_loss if avg_loss > 0 else 0
            
            if risk_reward < 1.0:
                print(f"2. ⚠️  POOR RISK/REWARD: {risk_reward:.2f} (should be > 1.0)")
                print("   - Average losses exceed average wins")
                print("   - Consider: Wider stop losses, better exit timing, trailing stops")
                print()
        
        # Check stop loss execution
        if stops_missed:
            print(f"3. 🚨 STOP LOSS FAILURES: {len(stops_missed)} trades")
            print("   - Stop losses not executing at -15%")
            print("   - This is the #1 priority fix")
            print()
    else:
        print(f"✅ NET PROFIT TODAY: ${total_pnl:,.2f}")
        print("   System is profitable, but can be optimized further")
    
    print()
    print("7️⃣ RECOMMENDATIONS")
    print("-" * 100)
    print()
    print("Priority 1: Fix Stop Loss Execution")
    print("   - Verify stop loss checks run every price update")
    print("   - Ensure position tracking is accurate")
    print("   - Test stop loss logic with paper trading")
    print()
    print("Priority 2: Improve Entry Selection")
    print("   - Review RL model predictions - are they accurate?")
    print("   - Verify gap detection is working")
    print("   - Check if model is getting correct observations")
    print()
    print("Priority 3: Optimize Exit Timing")
    print("   - Review take-profit levels - too aggressive?")
    print("   - Consider letting winners run longer")
    print("   - Implement better trailing stop logic")
    print()
    print("Priority 4: Model Performance")
    print("   - Review model training data quality")
    print("   - Consider retraining with more recent data")
    print("   - Validate model predictions match actual outcomes")
    print()
    
    print("=" * 100)

if __name__ == "__main__":
    analyze_todays_trades()

