#!/usr/bin/env python3
"""
Stop Loss Diagnostic Tool
Analyzes why stop loss is not triggering for positions
"""
import sys
import os
import alpaca_trade_api as tradeapi
import config
from datetime import datetime
import pytz

def analyze_stop_loss_issue():
    """Detailed analysis of stop loss functionality"""
    
    print("=" * 80)
    print("STOP LOSS DIAGNOSTIC ANALYSIS")
    print("=" * 80)
    print()
    
    # Initialize Alpaca API
    try:
        api = tradeapi.REST(
            config.ALPACA_KEY,
            config.ALPACA_SECRET,
            config.ALPACA_BASE_URL,
            api_version='v2'
        )
        print("✅ Alpaca API connection successful")
    except Exception as e:
        print(f"❌ Failed to connect to Alpaca: {e}")
        return
    
    print()
    
    # Step 1: Check actual positions in Alpaca
    print("STEP 1: CHECKING ACTUAL POSITIONS IN ALPACA")
    print("-" * 80)
    try:
        alpaca_positions = api.list_positions()
        option_positions = []
        
        for pos in alpaca_positions:
            if (hasattr(pos, 'asset_class') and pos.asset_class in ['option', 'us_option']) or \
               (len(pos.symbol) >= 15 and ('C' in pos.symbol[-9:] or 'P' in pos.symbol[-9:])):
                option_positions.append(pos)
        
        if not option_positions:
            print("⚠️  No option positions found in Alpaca")
            return
        
        print(f"✅ Found {len(option_positions)} option position(s):")
        print()
        
        for pos in option_positions:
            symbol = pos.symbol
            qty = float(pos.qty)
            market_value = float(pos.market_value) if hasattr(pos, 'market_value') and pos.market_value else 0.0
            avg_entry = float(pos.avg_entry_price) if hasattr(pos, 'avg_entry_price') and pos.avg_entry_price else 0.0
            
            print(f"Position: {symbol}")
            print(f"  Quantity: {qty}")
            print(f"  Market Value: ${market_value:,.2f}")
            print(f"  Avg Entry Price: ${avg_entry:.4f}")
            
            # Calculate current premium
            if qty > 0:
                current_premium_from_mv = market_value / (qty * 100)
                print(f"  Current Premium (from MV): ${current_premium_from_mv:.4f}")
            
            # Get snapshot
            try:
                snapshot = api.get_option_snapshot(symbol)
                bid_price = float(snapshot.bid_price) if snapshot.bid_price else None
                ask_price = float(snapshot.ask_price) if snapshot.ask_price else None
                
                if bid_price:
                    print(f"  Bid Price: ${bid_price:.4f}")
                if ask_price:
                    print(f"  Ask Price: ${ask_price:.4f}")
                    
                current_premium = bid_price if bid_price else current_premium_from_mv
                
            except Exception as e:
                print(f"  ⚠️  Could not get snapshot: {e}")
                current_premium = current_premium_from_mv
            
            print()
            
            # Step 2: Check if position is tracked
            print("STEP 2: CHECKING POSITION TRACKING")
            print("-" * 80)
            
            # We need to check if the agent has this position tracked
            # Since we can't access risk_mgr directly, let's check the logic
            
            print(f"Analyzing: {symbol}")
            print()
            
            # Step 3: Calculate P&L
            print("STEP 3: CALCULATING P&L")
            print("-" * 80)
            
            if avg_entry > 0 and current_premium > 0:
                pnl_pct = (current_premium - avg_entry) / avg_entry
                pnl_dollar = (current_premium - avg_entry) * qty * 100
                
                print(f"Entry Premium: ${avg_entry:.4f}")
                print(f"Current Premium: ${current_premium:.4f}")
                print(f"PnL Percentage: {pnl_pct:.2%}")
                print(f"PnL Dollar: ${pnl_dollar:,.2f}")
                print()
                
                # Step 4: Check stop loss threshold
                print("STEP 4: CHECKING STOP LOSS CONDITIONS")
                print("-" * 80)
                
                ABSOLUTE_STOP_LOSS = -0.15  # -15%
                EPSILON = 1e-6
                
                print(f"Absolute Stop Loss Threshold: {ABSOLUTE_STOP_LOSS:.2%}")
                print(f"Current PnL: {pnl_pct:.2%}")
                print(f"Should Trigger: {pnl_pct <= (ABSOLUTE_STOP_LOSS + EPSILON)}")
                print()
                
                if pnl_pct <= ABSOLUTE_STOP_LOSS:
                    print("🚨 STOP LOSS SHOULD HAVE TRIGGERED!")
                    print()
                    print("POSSIBLE ISSUES:")
                    print("  1. Position not tracked in risk_mgr.open_positions")
                    print("  2. entry_premium in tracking is different from Alpaca avg_entry")
                    print("  3. check_stop_losses() not being called")
                    print("  4. Exception in check_stop_losses() preventing execution")
                    print("  5. Position was closed externally and removed from tracking")
                    print()
                else:
                    print(f"✅ Position is above stop loss threshold ({pnl_pct:.2%} > {ABSOLUTE_STOP_LOSS:.2%})")
            
            print()
            
    except Exception as e:
        print(f"❌ Error analyzing positions: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 5: Check agent logs
    print("STEP 5: CHECKING RECENT AGENT LOGS")
    print("-" * 80)
    
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).strftime('%Y%m%d')
    log_file = f"logs/mike_agent_safe_{today}.log"
    
    if os.path.exists(log_file):
        print(f"✅ Found log file: {log_file}")
        print()
        print("Last 20 lines related to stop loss:")
        print("-" * 80)
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
            recent_lines = []
            for line in lines[-100:]:
                if 'STOP' in line.upper() or 'PNL' in line.upper() or 'POSITION' in line.upper():
                    recent_lines.append(line)
                elif 'symbol' in locals() and symbol in line:
                    recent_lines.append(line)
            
            for line in recent_lines[-20:]:
                print(line.strip())
    else:
        print(f"⚠️  Log file not found: {log_file}")
    
    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("1. Check if position exists in risk_mgr.open_positions")
    print("2. Verify entry_premium matches Alpaca avg_entry_price")
    print("3. Add debug logging to check_stop_losses() function")
    print("4. Verify check_stop_losses() is being called every iteration")
    print("5. Check for exceptions in stop loss logic")
    print()

if __name__ == "__main__":
    analyze_stop_loss_issue()

