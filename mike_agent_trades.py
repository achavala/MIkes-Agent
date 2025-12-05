#!/usr/bin/env python3
"""
Mike Agent Trade Tracking System
Stores and tracks all trades made by Mike Agent (separate from Alpaca)
"""

import csv
import os
import pytz
from datetime import datetime
from typing import Dict, List, Optional

TRADE_DB_FILE = "mike_agent_trades.csv"

class MikeAgentTradeDB:
    """Database for tracking Mike Agent trades"""
    
    def __init__(self, db_file: str = TRADE_DB_FILE):
        self.db_file = db_file
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Create database file with headers if it doesn't exist"""
        if not os.path.exists(self.db_file):
            with open(self.db_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'symbol', 'action', 'qty', 'price', 
                    'strike', 'option_type', 'entry_premium', 'exit_premium',
                    'pnl', 'pnl_pct', 'capital_before', 'capital_after',
                    'reason', 'regime', 'vix'
                ])
                writer.writeheader()
    
    def log_trade(self, symbol: str, action: str, qty: int, price: float,
                  strike: Optional[float] = None, option_type: Optional[str] = None,
                  entry_premium: Optional[float] = None, exit_premium: Optional[float] = None,
                  pnl: Optional[float] = None, pnl_pct: Optional[float] = None,
                  capital_before: Optional[float] = None, capital_after: Optional[float] = None,
                  reason: Optional[str] = None, regime: Optional[str] = None,
                  vix: Optional[float] = None):
        """Log a trade to the database"""
        est = pytz.timezone('US/Eastern')
        timestamp = datetime.now(est)
        
        row = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'action': action.upper(),  # BUY, SELL
            'qty': qty,
            'price': price,
            'strike': strike or '',
            'option_type': option_type or '',
            'entry_premium': entry_premium or '',
            'exit_premium': exit_premium or '',
            'pnl': pnl or 0.0,
            'pnl_pct': pnl_pct or 0.0,
            'capital_before': capital_before or '',
            'capital_after': capital_after or '',
            'reason': reason or '',
            'regime': regime or '',
            'vix': vix or ''
        }
        
        with open(self.db_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
    
    def get_all_trades(self) -> List[Dict]:
        """Get all trades from database"""
        if not os.path.exists(self.db_file):
            return []
        
        trades = []
        with open(self.db_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                for key in ['qty', 'price', 'strike', 'entry_premium', 'exit_premium', 
                           'pnl', 'pnl_pct', 'capital_before', 'capital_after', 'vix']:
                    if row.get(key) and row[key] != '':
                        try:
                            row[key] = float(row[key])
                        except:
                            row[key] = 0.0
                    else:
                        row[key] = 0.0
                trades.append(row)
        
        return trades
    
    def get_open_positions(self) -> List[Dict]:
        """Get currently open positions (BUY without matching SELL)"""
        trades = self.get_all_trades()
        open_positions = {}
        
        for trade in trades:
            symbol = trade['symbol']
            action = trade['action']
            qty = trade['qty']
            
            if action == 'BUY':
                if symbol not in open_positions:
                    open_positions[symbol] = {
                        'symbol': symbol,
                        'qty': 0,
                        'entry_price': trade['price'],
                        'entry_premium': trade.get('entry_premium', 0),
                        'strike': trade.get('strike', 0),
                        'option_type': trade.get('option_type', ''),
                        'entry_timestamp': trade['timestamp'],
                        'regime': trade.get('regime', ''),
                        'vix': trade.get('vix', 0),
                        'total_cost': 0
                    }
                open_positions[symbol]['qty'] += qty
                open_positions[symbol]['total_cost'] += qty * trade.get('entry_premium', trade['price']) * 100
            elif action == 'SELL':
                if symbol in open_positions:
                    open_positions[symbol]['qty'] -= qty
                    if open_positions[symbol]['qty'] <= 0:
                        del open_positions[symbol]
        
        return list(open_positions.values())
    
    def get_total_pnl(self) -> Dict:
        """Calculate total P&L from all closed trades"""
        trades = self.get_all_trades()
        total_pnl = 0.0
        winning_trades = 0
        losing_trades = 0
        
        for trade in trades:
            if trade['action'] == 'SELL' and trade['pnl']:
                total_pnl += trade['pnl']
                if trade['pnl'] > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
        
        win_rate = (winning_trades / (winning_trades + losing_trades) * 100) if (winning_trades + losing_trades) > 0 else 0
        
        return {
            'total_pnl': total_pnl,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_trades': winning_trades + losing_trades
        }



