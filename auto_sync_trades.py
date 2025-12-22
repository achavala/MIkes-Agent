#!/usr/bin/env python3
"""
Automatic Trade Sync from Alpaca
Runs periodically to ensure database always has latest trades
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time
from datetime import datetime
import pytz

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

try:
    from trade_database import TradeDatabase
    TRADE_DB_AVAILABLE = True
except ImportError:
    TRADE_DB_AVAILABLE = False

try:
    import config
except ImportError:
    class Config:
        ALPACA_KEY = os.getenv('APCA_API_KEY_ID') or os.getenv('ALPACA_KEY', '')
        ALPACA_SECRET = os.getenv('APCA_API_SECRET_KEY') or os.getenv('ALPACA_SECRET', '')
        ALPACA_BASE_URL = os.getenv('APCA_API_BASE_URL') or os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    config = Config()

# Load credentials
API_KEY = os.getenv('ALPACA_KEY', config.ALPACA_KEY if hasattr(config, 'ALPACA_KEY') else 'YOUR_PAPER_KEY')
API_SECRET = os.getenv('ALPACA_SECRET', config.ALPACA_SECRET if hasattr(config, 'ALPACA_SECRET') else 'YOUR_PAPER_SECRET')

if not API_KEY or API_KEY == 'YOUR_PAPER_KEY':
    API_KEY = os.getenv('APCA_API_KEY_ID', '')
if not API_SECRET or API_SECRET == 'YOUR_PAPER_SECRET':
    API_SECRET = os.getenv('APCA_API_SECRET_KEY', '')

BASE_URL = os.getenv('ALPACA_BASE_URL', '')
if not BASE_URL:
    BASE_URL = os.getenv('APCA_API_BASE_URL', '')
if not BASE_URL:
    BASE_URL = config.ALPACA_BASE_URL if hasattr(config, 'ALPACA_BASE_URL') else 'https://paper-api.alpaca.markets'

USE_PAPER = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'
if USE_PAPER and 'paper-api' not in BASE_URL:
    BASE_URL = 'https://paper-api.alpaca.markets'
elif not USE_PAPER and 'api.alpaca.markets' not in BASE_URL:
    BASE_URL = 'https://api.alpaca.markets'

def convert_utc_to_est(utc_timestamp):
    """Convert UTC timestamp to EST"""
    est = pytz.timezone('US/Eastern')
    if not utc_timestamp:
        return ''
    try:
        ts_str = str(utc_timestamp)
        if isinstance(utc_timestamp, datetime):
            if utc_timestamp.tzinfo is None:
                dt_utc = pytz.utc.localize(utc_timestamp)
            else:
                dt_utc = utc_timestamp.astimezone(pytz.utc)
            dt_est = dt_utc.astimezone(est)
            return dt_est.strftime('%Y-%m-%d %H:%M:%S %Z')
        if 'T' in ts_str:
            ts_str = ts_str.replace('Z', '+00:00')
            dt_utc = datetime.fromisoformat(ts_str)
            if dt_utc.tzinfo is None:
                dt_utc = pytz.utc.localize(dt_utc)
            dt_est = dt_utc.astimezone(est)
            return dt_est.strftime('%Y-%m-%d %H:%M:%S %Z')
        return ts_str
    except:
        return str(utc_timestamp)

def sync_recent_trades(api, trade_db, hours_back=24, limit=500):
    """Sync recent trades from Alpaca (only new ones)"""
    try:
        # Get most recent trade timestamp from database
        import sqlite3
        conn = sqlite3.connect(trade_db.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(timestamp) as last_timestamp FROM trades")
        result = cursor.fetchone()
        last_timestamp_str = result[0] if result and result[0] else None
        conn.close()
        
        # Get filled orders from Alpaca
        orders = api.list_orders(status='filled', limit=limit)
        
        if not orders:
            return 0
        
        est = pytz.timezone('US/Eastern')
        count = 0
        new_count = 0
        
        for order in orders:
            try:
                filled_at_est = convert_utc_to_est(order.filled_at if hasattr(order, 'filled_at') else None)
                submitted_at_est = convert_utc_to_est(order.submitted_at if hasattr(order, 'submitted_at') else None)
                primary_timestamp = filled_at_est or submitted_at_est
                
                if not primary_timestamp:
                    continue
                
                # Skip if we already have this trade (check by order_id first, then timestamp)
                if hasattr(order, 'id') and order.id:
                    import sqlite3
                    check_conn = sqlite3.connect(trade_db.db_path)
                    check_cursor = check_conn.cursor()
                    check_cursor.execute("SELECT COUNT(*) FROM trades WHERE order_id = ?", (str(order.id),))
                    exists = check_cursor.fetchone()[0] > 0
                    check_conn.close()
                    if exists:
                        continue
                
                # Parse option symbol
                symbol = order.symbol
                is_0dte = 0
                if len(symbol) >= 15:
                    try:
                        date_str = symbol[3:9]
                        exp_year = 2000 + int(date_str[:2])
                        exp_month = int(date_str[2:4])
                        exp_day = int(date_str[4:6])
                        exp_date = datetime(exp_year, exp_month, exp_day).date()
                        today_est = datetime.now(est).date()
                        is_0dte = 1 if exp_date == today_est else 0
                    except:
                        pass
                
                trade_data = {
                    'timestamp': primary_timestamp,
                    'symbol': symbol,
                    'action': order.side.upper(),
                    'qty': float(order.filled_qty) if hasattr(order, 'filled_qty') else float(order.qty),
                    'fill_price': float(order.filled_avg_price) if hasattr(order, 'filled_avg_price') else 0.0,
                    'order_id': str(order.id),
                    'source': 'alpaca_auto_sync',
                    'pnl': 0,
                    'is_0dte': is_0dte,
                    'submitted_at': submitted_at_est,
                    'filled_at': filled_at_est
                }
                
                trade_db.save_trade(trade_data)
                count += 1
                new_count += 1
                
            except Exception as e:
                pass
        
        return new_count
        
    except Exception as e:
        print(f"⚠️ Sync error: {e}")
        return 0

def auto_sync_loop(interval_minutes=5):
    """Run automatic sync in a loop"""
    if not ALPACA_AVAILABLE or not TRADE_DB_AVAILABLE:
        print("❌ Alpaca API or Trade Database not available")
        return
    
    if not API_KEY or not API_SECRET or 'YOUR_PAPER_KEY' in API_KEY or 'PKXX' in API_KEY:
        print("❌ Invalid Alpaca credentials")
        return
    
    try:
        api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
        trade_db = TradeDatabase()
        
        print("="*80)
        print("AUTOMATIC TRADE SYNC - Running every {} minutes".format(interval_minutes))
        print("="*80)
        print("Press Ctrl+C to stop")
        print()
        
        while True:
            try:
                est = pytz.timezone('US/Eastern')
                now_est = datetime.now(est)
                print(f"[{now_est.strftime('%Y-%m-%d %H:%M:%S %Z')}] Syncing trades from Alpaca...")
                
                synced = sync_recent_trades(api, trade_db, hours_back=24, limit=500)
                
                if synced > 0:
                    print(f"✅ Synced {synced} new trades")
                else:
                    print("✅ No new trades to sync")
                
                print(f"⏳ Waiting {interval_minutes} minutes until next sync...\n")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                print("\n🛑 Auto-sync stopped by user")
                break
            except Exception as e:
                print(f"⚠️ Error in sync loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
                
    except Exception as e:
        print(f"❌ Failed to initialize auto-sync: {e}")

if __name__ == '__main__':
    # Run sync every 5 minutes by default
    auto_sync_loop(interval_minutes=5)

