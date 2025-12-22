#!/usr/bin/env python3
"""
Sync trades from Alpaca to Database
Converts UTC timestamps to EST
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime
import pytz

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("❌ alpaca-trade-api not installed")

try:
    from trade_database import TradeDatabase
    TRADE_DB_AVAILABLE = True
except ImportError:
    TRADE_DB_AVAILABLE = False
    print("❌ trade_database module not found")

try:
    import config
except ImportError:
    # Fallback if config.py doesn't exist
    class Config:
        ALPACA_KEY = os.getenv('APCA_API_KEY_ID') or os.getenv('ALPACA_KEY', '')
        ALPACA_SECRET = os.getenv('APCA_API_SECRET_KEY') or os.getenv('ALPACA_SECRET', '')
        ALPACA_BASE_URL = os.getenv('APCA_API_BASE_URL') or os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    config = Config()

# Load credentials with EXACT same priority as main agent (mike_agent_live_safe.py:390-391)
# Priority: 1. Environment variable ALPACA_KEY, 2. config.ALPACA_KEY, 3. Fallback
API_KEY = os.getenv('ALPACA_KEY', config.ALPACA_KEY if hasattr(config, 'ALPACA_KEY') else 'YOUR_PAPER_KEY')
API_SECRET = os.getenv('ALPACA_SECRET', config.ALPACA_SECRET if hasattr(config, 'ALPACA_SECRET') else 'YOUR_PAPER_SECRET')

# Also support Alpaca standard environment variable names (APCA_API_KEY_ID, etc.)
if not API_KEY or API_KEY == 'YOUR_PAPER_KEY':
    API_KEY = os.getenv('APCA_API_KEY_ID', '')
if not API_SECRET or API_SECRET == 'YOUR_PAPER_SECRET':
    API_SECRET = os.getenv('APCA_API_SECRET_KEY', '')

# Base URL - check environment first, then config
BASE_URL = os.getenv('ALPACA_BASE_URL', '')
if not BASE_URL:
    BASE_URL = os.getenv('APCA_API_BASE_URL', '')
if not BASE_URL:
    BASE_URL = config.ALPACA_BASE_URL if hasattr(config, 'ALPACA_BASE_URL') else 'https://paper-api.alpaca.markets'

# Determine paper vs live (same logic as main agent)
USE_PAPER = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'
if USE_PAPER and 'paper-api' not in BASE_URL:
    BASE_URL = 'https://paper-api.alpaca.markets'
elif not USE_PAPER and 'api.alpaca.markets' not in BASE_URL:
    BASE_URL = 'https://api.alpaca.markets'

def convert_utc_to_est(utc_timestamp):
    """Convert UTC timestamp from Alpaca to EST"""
    est = pytz.timezone('US/Eastern')
    
    if not utc_timestamp:
        return ''
    
    try:
        # Handle different timestamp formats from Alpaca
        ts_str = str(utc_timestamp)
        
        # If it's already a datetime object
        if isinstance(utc_timestamp, datetime):
            if utc_timestamp.tzinfo is None:
                dt_utc = pytz.utc.localize(utc_timestamp)
            else:
                dt_utc = utc_timestamp.astimezone(pytz.utc)
            dt_est = dt_utc.astimezone(est)
            return dt_est.strftime('%Y-%m-%d %H:%M:%S %Z')
        
        # If it's a string
        if 'T' in ts_str:
            # ISO format: 2025-12-18T10:15:16Z or 2025-12-18T10:15:16+00:00
            ts_str = ts_str.replace('Z', '+00:00')
            dt_utc = datetime.fromisoformat(ts_str)
            if dt_utc.tzinfo is None:
                dt_utc = pytz.utc.localize(dt_utc)
            dt_est = dt_utc.astimezone(est)
            return dt_est.strftime('%Y-%m-%d %H:%M:%S %Z')
        else:
            # Try parsing as datetime string
            try:
                dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                dt_utc = pytz.utc.localize(dt)
                dt_est = dt_utc.astimezone(est)
                return dt_est.strftime('%Y-%m-%d %H:%M:%S %Z')
            except:
                return ts_str  # Return as-is if parsing fails
    except Exception as e:
        print(f"⚠️ Error converting timestamp {utc_timestamp}: {e}")
        return str(utc_timestamp)

def sync_alpaca_trades(days_back=7, limit=500):
    """Sync trades from Alpaca to Database"""
    if not ALPACA_AVAILABLE:
        print("❌ Alpaca API not available")
        return -1
    
    if not TRADE_DB_AVAILABLE:
        print("❌ Trade database not available")
        return -1
    
    # Validate credentials
    if not API_KEY or not API_SECRET:
        print("❌ Alpaca API credentials not found!")
        print("   Please set ALPACA_KEY and ALPACA_SECRET environment variables")
        print("   Or configure them in config.py")
        print(f"   API_KEY: {'Set' if API_KEY else 'Missing'} ({'***' + API_KEY[-4:] if API_KEY and len(API_KEY) > 4 else 'empty'})")
        print(f"   API_SECRET: {'Set' if API_SECRET else 'Missing'} ({'***' + API_SECRET[-4:] if API_SECRET and len(API_SECRET) > 4 else 'empty'})")
        print(f"   BASE_URL: {BASE_URL}")
        return -1
    
    # Check if credentials are placeholders
    is_placeholder = (
        'YOUR_PAPER_KEY' in API_KEY or 
        'PKXX' in API_KEY or 
        len(API_KEY) < 10 or
        API_KEY.startswith('PKXX')
    )
    
    if is_placeholder:
        print("❌ Alpaca API credentials appear to be placeholders!")
        print(f"   Current API_KEY: {API_KEY[:15]}...")
        print()
        print("   To fix this, set your Alpaca API credentials:")
        print("   1. Set environment variables:")
        print("      export ALPACA_KEY='your_actual_key'")
        print("      export ALPACA_SECRET='your_actual_secret'")
        print()
        print("   2. Or update config.py with your real credentials")
        print()
        print("   3. Or use Alpaca standard environment variable names:")
        print("      export APCA_API_KEY_ID='your_actual_key'")
        print("      export APCA_API_SECRET_KEY='your_actual_secret'")
        return -1
    
    print(f"✅ Using Alpaca API credentials (Key: {API_KEY[:10]}...{API_KEY[-4:] if len(API_KEY) > 14 else '***'})")
    print(f"   Base URL: {BASE_URL}")
    
    try:
        api = tradeapi.REST(
            API_KEY,
            API_SECRET,
            BASE_URL,
            api_version='v2'
        )
        
        # Get filled orders
        print(f"📊 Fetching filled orders from Alpaca (last {days_back} days, limit {limit})...")
        orders = api.list_orders(status='filled', limit=limit)
        
        if not orders:
            print("✅ No filled orders found in Alpaca")
            return 0
        
        trade_db = TradeDatabase()
        est = pytz.timezone('US/Eastern')
        count = 0
        skipped = 0
        
        print(f"📦 Processing {len(orders)} orders...")
        
        for order in orders:
            try:
                # Convert UTC timestamps to EST
                filled_at_est = convert_utc_to_est(order.filled_at if hasattr(order, 'filled_at') else None)
                submitted_at_est = convert_utc_to_est(order.submitted_at if hasattr(order, 'submitted_at') else None)
                
                # Use filled_at as primary timestamp, fallback to submitted_at
                primary_timestamp = filled_at_est or submitted_at_est
                if not primary_timestamp:
                    primary_timestamp = datetime.now(est).strftime('%Y-%m-%d %H:%M:%S %Z')
                
                # Parse option symbol
                symbol = order.symbol
                
                # Check if it's 0DTE (approximation - check expiration date)
                is_0dte = 0
                if len(symbol) >= 15:
                    try:
                        date_str = symbol[3:9]  # YYMMDD
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
                    'source': 'alpaca_sync',
                    'pnl': 0,  # Cannot determine from single order
                    'is_0dte': is_0dte,
                    'submitted_at': submitted_at_est,
                    'filled_at': filled_at_est
                }
                
                # Save trade (INSERT OR IGNORE prevents duplicates)
                trade_db.save_trade(trade_data)
                count += 1
                
            except Exception as e:
                skipped += 1
                print(f"⚠️ Skipped order {order.id if hasattr(order, 'id') else 'unknown'}: {e}")
        
        print(f"✅ Synced {count} trades to database")
        if skipped > 0:
            print(f"⚠️ Skipped {skipped} orders due to errors")
        
        return count
        
    except Exception as e:
        print(f"❌ Sync error: {e}")
        import traceback
        traceback.print_exc()
        return -1

if __name__ == '__main__':
    print("="*80)
    print("ALPACA TRADE SYNC - UTC to EST Conversion")
    print("="*80)
    print()
    
    result = sync_alpaca_trades(days_back=7, limit=500)
    
    if result >= 0:
        print(f"\n✅ Sync complete: {result} trades synced")
    else:
        print(f"\n❌ Sync failed")

