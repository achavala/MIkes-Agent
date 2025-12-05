#!/usr/bin/env python3
"""
Position Monitoring Script
Tracks positions and explains why they're held despite 40-50%+ profit
"""

import alpaca_trade_api as tradeapi
import config
from datetime import datetime
import time

def monitor_positions():
    """Monitor current positions and analyze why they're held"""
    api = tradeapi.REST(config.ALPACA_KEY, config.ALPACA_SECRET, config.ALPACA_BASE_URL, api_version='v2')
    
    print("=" * 70)
    print(f"POSITION MONITORING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S EST')}")
    print("=" * 70)
    
    positions = api.list_positions()
    opt_positions = [p for p in positions if p.asset_class == 'us_option']
    
    if not opt_positions:
        print("\n✅ No open positions")
        return
    
    for p in opt_positions:
        entry_price = float(p.avg_entry_price)
        market_value = float(p.market_value)
        qty = float(p.qty)
        
        # CORRECT calculation
        current_price_per_share = market_value / (qty * 100) if qty > 0 else 0
        pnl_pct = ((current_price_per_share / entry_price) - 1) * 100 if entry_price > 0 else 0
        
        # WRONG calculation (what agent is doing - line 502)
        wrong_price = market_value / qty if qty > 0 else 0
        wrong_pnl = ((wrong_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        
        print(f"\n📊 {p.symbol}")
        print(f"   Qty: {int(qty)} contracts")
        print(f"   Entry: ${entry_price:.4f} per share")
        print(f"   Current (CORRECT): ${current_price_per_share:.4f} per share")
        print(f"   Current (WRONG - agent calc): ${wrong_price:.2f} per share")
        print(f"   P&L (CORRECT): {pnl_pct:+.2f}%")
        print(f"   P&L (WRONG - agent sees): {wrong_pnl:+.2f}%")
        print(f"   Unrealized P&L: ${float(p.unrealized_pl):+,.2f}")
        
        print(f"\n   🎯 Take-Profit Analysis:")
        print(f"      TP1 (+40%): {'✅ HIT' if pnl_pct >= 40 else '⏳ Waiting'} (Agent sees: {wrong_pnl:.0f}%)")
        print(f"      TP2 (+80%): {'✅ HIT' if pnl_pct >= 80 else '⏳ Waiting'} (Agent sees: {wrong_pnl:.0f}%)")
        print(f"      TP3 (+150%): {'✅ HIT' if pnl_pct >= 150 else '⏳ Waiting'} (Agent sees: {wrong_pnl:.0f}%)")
        
        print(f"\n   🐛 BUG ANALYSIS:")
        if pnl_pct != 0:
            multiplier = wrong_pnl / pnl_pct if pnl_pct != 0 else 0
            print(f"      Agent calculation is {multiplier:.0f}x too high")
        print(f"      This prevents TP1 from triggering correctly")
        print(f"      Fix: Change line 502 to: market_value / (qty * 100)")

if __name__ == "__main__":
    monitor_positions()



