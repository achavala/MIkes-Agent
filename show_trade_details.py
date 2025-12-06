#!/usr/bin/env python3
"""
Trade Details Viewer - Shows complete trade-by-trade breakdown with P&L
"""
import sys
import os
import config
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import pytz
from collections import defaultdict

def get_trade_details():
    """Get complete trade-by-trade details with P&L"""
    
    print("=" * 100)
    print("COMPLETE TRADE DETAILS - TRADE BY TRADE BREAKDOWN")
    print("=" * 100)
    print()
    
    # Initialize Alpaca API
    try:
        api = tradeapi.REST(
            config.ALPACA_KEY,
            config.ALPACA_SECRET,
            config.ALPACA_BASE_URL,
            api_version='v2'
        )
        print("✅ Connected to Alpaca API")
    except Exception as e:
        print(f"❌ Failed to connect to Alpaca: {e}")
        return
    
    print()
    
    # Get today's date in EST
    est = pytz.timezone('US/Eastern')
    now_est = datetime.now(est)
    today_date = now_est.date()
    
    print(f"📅 Analysis Date: {today_date.strftime('%Y-%m-%d')} ({now_est.strftime('%I:%M %p %Z')})")
    print()
    
    # Get all filled orders for today
    try:
        print("📊 Fetching all filled orders from Alpaca...")
        all_orders = api.list_orders(
            status='filled',
            limit=500,
            nested=True
        )
        
        # Filter for today's orders
        today_orders = []
        for order in all_orders:
            if order.filled_at:
                # Handle both string and datetime objects
                if isinstance(order.filled_at, str):
                    order_date = datetime.fromisoformat(order.filled_at.replace('Z', '+00:00')).astimezone(est).date()
                else:
                    # Already a datetime object
                    if hasattr(order.filled_at, 'date'):
                        order_date = order.filled_at.date()
                    else:
                        order_date = datetime.fromtimestamp(order.filled_at).astimezone(est).date()
                
                if order_date == today_date:
                    today_orders.append(order)
        
        print(f"✅ Found {len(today_orders)} filled orders today")
        print()
        
    except Exception as e:
        print(f"❌ Error fetching orders: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not today_orders:
        print("⚠️  No trades found for today")
        return
    
    # Separate buy and sell orders
    buy_orders = {}
    sell_orders = {}
    
    for order in today_orders:
        symbol = order.symbol
        side = order.side.lower()
        filled_qty = float(order.filled_qty)
        filled_avg_price = float(order.filled_avg_price)
        
        # Handle both string and datetime objects for filled_at
        if isinstance(order.filled_at, str):
            filled_at = datetime.fromisoformat(order.filled_at.replace('Z', '+00:00')).astimezone(est)
        else:
            if hasattr(order.filled_at, 'astimezone'):
                filled_at = order.filled_at.astimezone(est)
            else:
                filled_at = datetime.fromtimestamp(order.filled_at).astimezone(est)
        
        if side == 'buy':
            if symbol not in buy_orders:
                buy_orders[symbol] = []
            buy_orders[symbol].append({
                'order_id': order.id,
                'qty': filled_qty,
                'price': filled_avg_price,
                'time': filled_at,
                'notional': filled_qty * filled_avg_price * 100,
                'order': order
            })
        else:  # sell
            if symbol not in sell_orders:
                sell_orders[symbol] = []
            sell_orders[symbol].append({
                'order_id': order.id,
                'qty': filled_qty,
                'price': filled_avg_price,
                'time': filled_at,
                'notional': filled_qty * filled_avg_price * 100,
                'order': order
            })
    
    # Match buy/sell pairs using FIFO (First In, First Out)
    print("=" * 100)
    print("TRADE-BY-TRADE BREAKDOWN")
    print("=" * 100)
    print()
    
    all_symbols = set(list(buy_orders.keys()) + list(sell_orders.keys()))
    all_trades = []
    
    for symbol in sorted(all_symbols):
        print(f"📈 Symbol: {symbol}")
        print("-" * 100)
        
        buys = sorted(buy_orders.get(symbol, []), key=lambda x: x['time'])
        sells = sorted(sell_orders.get(symbol, []), key=lambda x: x['time'])
        
        buy_index = 0
        sell_index = 0
        trade_num = 1
        
        while buy_index < len(buys) or sell_index < len(sells):
            if buy_index < len(buys) and (sell_index >= len(sells) or buys[buy_index]['time'] <= sells[sell_index]['time']):
                # Process buy
                buy = buys[buy_index]
                buy_index += 1
                
                # Check if this is a full or partial position
                remaining_qty = buy['qty']
                
                # Match with sells (FIFO)
                while remaining_qty > 0.01 and sell_index < len(sells):
                    sell = sells[sell_index]
                    sell_qty = min(remaining_qty, sell['qty'])
                    
                    # Calculate P&L for this trade
                    entry_cost = buy['price'] * sell_qty * 100
                    exit_proceeds = sell['price'] * sell_qty * 100
                    pnl_dollar = exit_proceeds - entry_cost
                    pnl_pct = (sell['price'] - buy['price']) / buy['price'] if buy['price'] > 0 else 0
                    
                    trade_info = {
                        'symbol': symbol,
                        'trade_num': trade_num,
                        'buy_time': buy['time'],
                        'sell_time': sell['time'],
                        'qty': sell_qty,
                        'entry_price': buy['price'],
                        'exit_price': sell['price'],
                        'entry_cost': entry_cost,
                        'exit_proceeds': exit_proceeds,
                        'pnl_dollar': pnl_dollar,
                        'pnl_pct': pnl_pct,
                        'duration': sell['time'] - buy['time']
                    }
                    
                    all_trades.append(trade_info)
                    
                    print(f"  Trade #{trade_num}:")
                    print(f"    BUY:  {sell_qty:.0f} contracts @ ${buy['price']:.4f}  |  {buy['time'].strftime('%I:%M:%S %p')}")
                    print(f"    SELL: {sell_qty:.0f} contracts @ ${sell['price']:.4f}  |  {sell['time'].strftime('%I:%M:%S %p')}")
                    print(f"    Entry Cost:    ${entry_cost:,.2f}")
                    print(f"    Exit Proceeds: ${exit_proceeds:,.2f}")
                    print(f"    P&L:           ${pnl_dollar:,.2f} ({pnl_pct:+.2%})")
                    print(f"    Duration:      {trade_info['duration']}")
                    print()
                    
                    trade_num += 1
                    
                    # Update remaining quantities
                    remaining_qty -= sell_qty
                    if sell['qty'] - sell_qty < 0.01:
                        sell_index += 1
                    else:
                        sells[sell_index]['qty'] -= sell_qty
                
                # If there's remaining qty, it's still open
                if remaining_qty > 0.01:
                    print(f"  ⚠️  Partial position still open:")
                    print(f"    BUY:  {remaining_qty:.0f} contracts @ ${buy['price']:.4f}  |  {buy['time'].strftime('%I:%M:%S %p')}")
                    print(f"    Status: OPEN POSITION")
                    print()
            
            else:
                # Process sell (orphaned sell - shouldn't happen but handle it)
                sell = sells[sell_index]
                sell_index += 1
                print(f"  ⚠️  Orphaned SELL (no matching BUY):")
                print(f"    SELL: {sell['qty']:.0f} contracts @ ${sell['price']:.4f}  |  {sell['time'].strftime('%I:%M:%S %p')}")
                print()
    
    # Summary statistics
    print("=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    print()
    
    if all_trades:
        total_trades = len(all_trades)
        total_pnl = sum(t['pnl_dollar'] for t in all_trades)
        winning_trades = [t for t in all_trades if t['pnl_dollar'] > 0]
        losing_trades = [t for t in all_trades if t['pnl_dollar'] < 0]
        
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        avg_win = sum(t['pnl_dollar'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl_dollar'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        largest_win = max((t['pnl_dollar'] for t in all_trades), default=0)
        largest_loss = min((t['pnl_dollar'] for t in all_trades), default=0)
        
        print(f"Total Trades:        {total_trades}")
        print(f"Winning Trades:      {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"Losing Trades:       {len(losing_trades)} ({100-win_rate:.1f}%)")
        print(f"Total P&L:           ${total_pnl:,.2f}")
        print(f"Average Win:         ${avg_win:,.2f}")
        print(f"Average Loss:        ${avg_loss:,.2f}")
        print(f"Largest Win:         ${largest_win:,.2f}")
        print(f"Largest Loss:        ${largest_loss:,.2f}")
        if avg_loss != 0:
            profit_factor = abs(sum(t['pnl_dollar'] for t in winning_trades) / sum(t['pnl_dollar'] for t in losing_trades)) if losing_trades else 0
            print(f"Profit Factor:       {profit_factor:.2f}")
        print()
    else:
        print("No completed trades found (all positions are still open)")
        print()
    
    # Show open positions
    print("=" * 100)
    print("OPEN POSITIONS")
    print("=" * 100)
    print()
    
    try:
        positions = api.list_positions()
        option_positions = [pos for pos in positions 
                           if (hasattr(pos, 'asset_class') and pos.asset_class in ['option', 'us_option']) 
                           or (len(pos.symbol) >= 15 and ('C' in pos.symbol[-9:] or 'P' in pos.symbol[-9:]))]
        
        if option_positions:
            for pos in option_positions:
                symbol = pos.symbol
                qty = float(pos.qty)
                avg_entry = float(pos.avg_entry_price) if hasattr(pos, 'avg_entry_price') and pos.avg_entry_price else 0
                market_value = float(pos.market_value) if hasattr(pos, 'market_value') and pos.market_value else 0
                cost_basis = float(pos.cost_basis) if hasattr(pos, 'cost_basis') and pos.cost_basis else 0
                unrealized_pl = float(pos.unrealized_pl) if hasattr(pos, 'unrealized_pl') and pos.unrealized_pl else 0
                
                current_premium = market_value / (qty * 100) if qty > 0 else 0
                entry_premium = avg_entry if avg_entry > 0 else (cost_basis / (qty * 100) if qty > 0 and cost_basis > 0 else 0)
                
                if entry_premium > 0:
                    pnl_pct = (current_premium - entry_premium) / entry_premium
                else:
                    pnl_pct = 0
                
                print(f"Symbol: {symbol}")
                print(f"  Quantity:          {qty:.0f} contracts")
                print(f"  Entry Premium:     ${entry_premium:.4f}")
                print(f"  Current Premium:   ${current_premium:.4f}")
                print(f"  Entry Cost:        ${cost_basis:,.2f}")
                print(f"  Current Value:     ${market_value:,.2f}")
                print(f"  Unrealized P&L:    ${unrealized_pl:,.2f} ({pnl_pct:+.2%})")
                print()
        else:
            print("No open positions")
            print()
    except Exception as e:
        print(f"Error fetching open positions: {e}")
        print()
    
    print("=" * 100)
    print("END OF REPORT")
    print("=" * 100)

if __name__ == "__main__":
    get_trade_details()

