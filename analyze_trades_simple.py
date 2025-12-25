#!/usr/bin/env python3
"""
Simple Trade Analysis - Analyzes recent trades
"""

import sys
import os
from datetime import datetime, date
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from trade_database import TradeDatabase
except ImportError:
    print("❌ Trade database not available")
    sys.exit(1)

def analyze():
    trade_db = TradeDatabase()
    all_trades = trade_db.get_all_trades()
    
    if not all_trades:
        print("❌ No trades found")
        return
    
    df = pd.DataFrame(all_trades)
    
    # Parse timestamps more carefully
    df['ts'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['date'] = df['ts'].dt.date
    
    # Get most recent date
    valid_dates = df['date'].dropna()
    if len(valid_dates) == 0:
        print("❌ No valid dates found")
        return
    
    recent_date = valid_dates.max()
    print(f"📅 Analyzing: {recent_date}")
    print("=" * 80)
    
    day_trades = df[df['date'] == recent_date].copy()
    
    buys = day_trades[day_trades['action'] == 'BUY']
    sells = day_trades[day_trades['action'] == 'SELL']
    
    print(f"BUY Orders: {len(buys)}")
    print(f"SELL Orders: {len(sells)}")
    print()
    
    if len(sells) > 0:
        total_pnl = sells['pnl'].sum()
        print(f"💰 Total P&L: ${total_pnl:,.2f}")
        print()
        
        wins = sells[sells['pnl'] > 0]
        losses = sells[sells['pnl'] < 0]
        
        print(f"✅ Wins: {len(wins)} | ❌ Losses: {len(losses)}")
        if len(wins) > 0:
            print(f"   Avg Win: ${wins['pnl'].mean():.2f}")
        if len(losses) > 0:
            print(f"   Avg Loss: ${losses['pnl'].mean():.2f}")
        print()
        
        print("=" * 80)
        print("TRADE DETAILS:")
        print("=" * 80)
        
        for idx, sell in sells.iterrows():
            symbol = sell.get('symbol', 'UNK')
            pnl = sell.get('pnl', 0)
            pnl_pct = sell.get('pnl_pct', 0)
            reason = sell.get('reason', 'Unknown')
            entry_prem = sell.get('entry_premium', 0)
            exit_prem = sell.get('exit_premium', 0)
            ts = sell.get('ts', pd.NaT)
            
            print(f"\n{symbol}")
            print(f"  P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
            print(f"  Entry: ${entry_prem:.4f} → Exit: ${exit_prem:.4f}")
            print(f"  Reason: {reason}")
            if pd.notna(ts):
                print(f"  Time: {ts.strftime('%H:%M:%S')}")

if __name__ == "__main__":
    analyze()





