#!/usr/bin/env python3
"""Query recent trades from database"""
import sqlite3
import pandas as pd
from datetime import datetime
import os

db_path = "trades_database.db"
if not os.path.exists(db_path):
    print(f"❌ Trade database not found: {db_path}")
    exit(1)

try:
    conn = sqlite3.connect(db_path)
    
    # Query recent trades
    query = """
    SELECT 
        timestamp,
        symbol,
        underlying,
        expiration_date,
        strike_price,
        option_type,
        action,
        qty,
        entry_premium,
        exit_premium,
        fill_price,
        pnl,
        pnl_pct,
        reason,
        order_id
    FROM trades
    ORDER BY timestamp DESC
    LIMIT 50
    """
    
    df = pd.read_sql_query(query, conn)
    
    if len(df) == 0:
        print("❌ No trades found in database")
    else:
        print(f"✅ Found {len(df)} recent trades (showing last 50)")
        print("="*100)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        print(df.to_string(index=False))
        print("="*100)
        print()
        
        # Summary statistics
        print("📈 SUMMARY STATISTICS:")
        print(f"   Total trades in database: {len(df)}")
        if 'pnl' in df.columns and df['pnl'].notna().any():
            total_pnl = df['pnl'].sum()
            print(f"   Total P&L: ${total_pnl:.2f}")
        if 'action' in df.columns:
            buy_trades = len(df[df['action'].str.contains('BUY', case=False, na=False)])
            sell_trades = len(df[df['action'].str.contains('SELL', case=False, na=False)])
            print(f"   Buy trades: {buy_trades}")
            print(f"   Sell trades: {sell_trades}")
        
        # Most recent trade
        if len(df) > 0:
            latest = df.iloc[0]
            print()
            print("🕐 MOST RECENT TRADE:")
            for col in df.columns:
                val = latest[col]
                if pd.notna(val):
                    print(f"   {col}: {val}")
    
    conn.close()
    
except Exception as e:
    print(f"❌ Error querying database: {e}")
    import traceback
    traceback.print_exc()

