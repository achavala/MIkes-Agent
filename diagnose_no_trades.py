#!/usr/bin/env python3
"""
Comprehensive diagnostic script to analyze why no trades are executing
"""

import sqlite3
import os
from datetime import datetime, timedelta
import pytz

def analyze_trades():
    """Analyze recent trades and identify patterns"""
    print("="*80)
    print("NO TRADES DIAGNOSTIC ANALYSIS")
    print("="*80)
    print()
    
    # Check database
    if not os.path.exists('trades_database.db'):
        print("❌ Database not found: trades_database.db")
        return
    
    conn = sqlite3.connect('trades_database.db')
    cursor = conn.cursor()
    
    # Get today's date
    est = pytz.timezone('US/Eastern')
    now_est = datetime.now(est)
    today = now_est.strftime('%Y-%m-%d')
    
    print(f"📅 Today's Date: {today}")
    print()
    
    # Check for trades today
    cursor.execute("SELECT COUNT(*) FROM trades WHERE DATE(timestamp) = ?", (today,))
    today_count = cursor.fetchone()[0]
    
    # Get last trade
    cursor.execute("SELECT MAX(timestamp) FROM trades")
    last_trade_time = cursor.fetchone()[0]
    
    # Get recent trades
    cursor.execute("""
        SELECT timestamp, symbol, action, qty, fill_price, pnl, reason
        FROM trades 
        ORDER BY timestamp DESC 
        LIMIT 10
    """)
    recent = cursor.fetchall()
    
    print(f"📊 Trades Today: {today_count}")
    print(f"📊 Last Trade: {last_trade_time}")
    print()
    
    if recent:
        print("Recent Trades (Last 10):")
        for r in recent:
            print(f"  {r[0]} | {r[1]} | {r[2]} | {r[3]} | ${r[4] if r[4] else 'N/A'} | PnL: ${r[5] if r[5] else 'N/A'} | Reason: {r[6] if r[6] else 'N/A'}")
    else:
        print("❌ No trades found in database")
    
    print()
    print("="*80)
    print("POTENTIAL BLOCKING REASONS")
    print("="*80)
    print()
    
    # Check if market is open
    market_open = False
    current_hour = now_est.hour
    current_minute = now_est.minute
    current_time_str = now_est.strftime('%H:%M:%S %Z')
    
    # Market hours: 9:30 AM - 4:00 PM EST
    if now_est.weekday() < 5:  # Monday-Friday
        if (current_hour == 9 and current_minute >= 30) or (9 < current_hour < 16) or (current_hour == 16 and current_minute == 0):
            market_open = True
    
    print(f"🕐 Current Time: {current_time_str}")
    print(f"🕐 Market Open: {'✅ YES' if market_open else '❌ NO'}")
    if not market_open:
        print("   ⚠️  Agent may not be trading outside market hours")
    print()
    
    # Check configuration
    print("⚙️  CONFIGURATION CHECK:")
    print()
    
    try:
        import config
        print(f"  MODE: {getattr(config, 'MODE', 'N/A')}")
        print(f"  TRADING_SYMBOLS: {getattr(config, 'TRADING_SYMBOLS', 'N/A')}")
        print(f"  MIN_ACTION_STRENGTH_THRESHOLD: 0.52 (from code)")
        print()
    except Exception as e:
        print(f"  ⚠️  Could not load config: {e}")
        print()
    
    # Check for common blocking reasons
    print("🔍 COMMON BLOCKING REASONS:")
    print()
    print("  1. Confidence Threshold (0.52)")
    print("     - RL model output must be >= 0.52")
    print("     - If model outputs 0.501, trade is blocked")
    print()
    print("  2. Safeguards")
    print("     - Daily loss limit")
    print("     - Max daily trades")
    print("     - VIX kill switch (>28)")
    print("     - Cooldown periods")
    print()
    print("  3. Market Data Issues")
    print("     - Stale data (older than 5 minutes)")
    print("     - Price outside expected range")
    print("     - Data from wrong date")
    print()
    print("  4. Position Limits")
    print("     - Max positions per symbol")
    print("     - Max notional per trade")
    print("     - Greeks limits (delta/gamma/vega)")
    print()
    print("  5. No Valid Setups")
    print("     - RL model says HOLD (action=0)")
    print("     - No symbols pass filters")
    print("     - All symbols already have positions")
    print()
    
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()
    print("1. Check Fly.io logs for detailed blocking reasons:")
    print("   fly logs --app mike-agent-project | grep -i 'blocked\\|reject\\|confidence'")
    print()
    print("2. Check RL model confidence output:")
    print("   fly logs --app mike-agent-project | grep -i 'action_strength\\|confidence\\|RL'")
    print()
    print("3. Check safeguard status:")
    print("   fly logs --app mike-agent-project | grep -i 'safeguard\\|vix\\|cooldown'")
    print()
    print("4. Check market data validation:")
    print("   fly logs --app mike-agent-project | grep -i 'price\\|data\\|stale\\|validation'")
    print()
    print("5. Check if agent is running:")
    print("   fly status --app mike-agent-project")
    print()
    
    conn.close()

if __name__ == "__main__":
    analyze_trades()


