#!/usr/bin/env python3
"""
Detailed Trade Analysis Script
Analyzes today's trades to understand decision-making and losses
"""

import sys
import os
from datetime import datetime, date
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from trade_database import TradeDatabase
    TRADE_DB_AVAILABLE = True
except ImportError:
    TRADE_DB_AVAILABLE = False
    print("❌ Trade database not available")
    sys.exit(1)

def format_currency(value):
    """Format as currency"""
    return f"${value:,.2f}"

def format_percent(value):
    """Format as percentage"""
    return f"{value:+.2f}%"

def analyze_trades(trades_df):
    """Analyze trades and provide detailed insights"""
    
    if trades_df.empty:
        print("No trades found for analysis")
        return
    
    print("=" * 80)
    print("📊 DETAILED TRADE ANALYSIS")
    print("=" * 80)
    print()
    
    # Filter for today's trades
    today = date.today()
    # Handle timestamp parsing - try different formats
    try:
        trades_df['date'] = pd.to_datetime(trades_df['timestamp'], errors='coerce', format='mixed').dt.date
    except:
        trades_df['date'] = pd.to_datetime(trades_df['timestamp'], errors='coerce').dt.date
    
    today_trades = trades_df[trades_df['date'] == today].copy()
    
    if today_trades.empty:
        print(f"❌ No trades found for today ({today})")
        print(f"Total trades in database: {len(trades_df)}")
        if not trades_df.empty:
            # Filter out NaT dates
            valid_dates = trades_df['date'].dropna()
            if len(valid_dates) > 0:
                print(f"Date range: {valid_dates.min()} to {valid_dates.max()}")
                print(f"\n📅 Most recent trades:")
                # Show most recent 10 trades
                recent = trades_df.sort_values('timestamp', ascending=False).head(10)
                for idx, row in recent.iterrows():
                    print(f"   {row.get('timestamp', 'N/A')} - {row.get('action', 'N/A')} {row.get('symbol', 'N/A')} - P&L: ${row.get('pnl', 0):.2f}")
        
        # Ask user if they want to analyze most recent day instead
        print(f"\n💡 Would you like to analyze the most recent trading day instead?")
        print(f"   Run: python3 analyze_trades.py --recent")
        return
    
    print(f"📅 DATE: {today}")
    print(f"📈 TOTAL TRADES TODAY: {len(today_trades)}")
    print()
    
    # Separate BUY and SELL trades
    buy_trades = today_trades[today_trades['action'] == 'BUY'].copy()
    sell_trades = today_trades[today_trades['action'] == 'SELL'].copy()
    
    print(f"✅ BUY Orders: {len(buy_trades)}")
    print(f"🔴 SELL Orders: {len(sell_trades)}")
    print()
    
    # Calculate P&L
    total_pnl = sell_trades['pnl'].sum() if 'pnl' in sell_trades.columns else 0.0
    total_pnl_pct = sell_trades['pnl_pct'].sum() if 'pnl_pct' in sell_trades.columns else 0.0
    
    print("=" * 80)
    print("💰 P&L SUMMARY")
    print("=" * 80)
    print(f"Total P&L: {format_currency(total_pnl)}")
    print(f"Total P&L %: {format_percent(total_pnl_pct)}")
    print()
    
    # Winning vs Losing trades
    winning_trades = sell_trades[sell_trades['pnl'] > 0] if 'pnl' in sell_trades.columns else pd.DataFrame()
    losing_trades = sell_trades[sell_trades['pnl'] < 0] if 'pnl' in sell_trades.columns else pd.DataFrame()
    
    print(f"✅ Winning Trades: {len(winning_trades)}")
    print(f"❌ Losing Trades: {len(losing_trades)}")
    
    if len(winning_trades) > 0:
        avg_win = winning_trades['pnl'].mean()
        total_wins = winning_trades['pnl'].sum()
        print(f"   Average Win: {format_currency(avg_win)}")
        print(f"   Total Wins: {format_currency(total_wins)}")
    
    if len(losing_trades) > 0:
        avg_loss = losing_trades['pnl'].mean()
        total_losses = losing_trades['pnl'].sum()
        print(f"   Average Loss: {format_currency(avg_loss)}")
        print(f"   Total Losses: {format_currency(total_losses)}")
    
    win_rate = (len(winning_trades) / len(sell_trades) * 100) if len(sell_trades) > 0 else 0.0
    print(f"   Win Rate: {win_rate:.1f}%")
    print()
    
    # Detailed trade-by-trade analysis
    print("=" * 80)
    print("📋 DETAILED TRADE BREAKDOWN")
    print("=" * 80)
    print()
    
    # Match BUY and SELL trades
    for idx, sell_trade in sell_trades.iterrows():
        symbol = sell_trade.get('symbol', 'UNKNOWN')
        underlying = sell_trade.get('underlying', '')
        option_type = sell_trade.get('option_type', '').upper()
        strike = sell_trade.get('strike_price', 0)
        
        # Find matching BUY trade
        matching_buy = buy_trades[
            (buy_trades['symbol'] == symbol) | 
            (buy_trades['underlying'] == underlying)
        ]
        
        if not matching_buy.empty:
            buy_trade = matching_buy.iloc[0]
            entry_time = pd.to_datetime(buy_trade['timestamp'])
            exit_time = pd.to_datetime(sell_trade['timestamp'])
            duration = exit_time - entry_time
            
            entry_premium = buy_trade.get('entry_premium', 0)
            exit_premium = sell_trade.get('exit_premium', 0)
            pnl = sell_trade.get('pnl', 0)
            pnl_pct = sell_trade.get('pnl_pct', 0)
            reason = sell_trade.get('reason', 'Unknown')
            regime = sell_trade.get('regime', 'Unknown')
            vix = sell_trade.get('vix', 0)
            
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"TRADE #{idx}")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"Symbol: {symbol}")
            print(f"Underlying: {underlying} | Type: {option_type} | Strike: ${strike:.2f}")
            print()
            print(f"📥 ENTRY:")
            print(f"   Time: {entry_time.strftime('%H:%M:%S')}")
            print(f"   Premium: {format_currency(entry_premium)}")
            print(f"   Regime: {regime.upper()}")
            print(f"   VIX: {vix:.1f}")
            print()
            print(f"📤 EXIT:")
            print(f"   Time: {exit_time.strftime('%H:%M:%S')}")
            print(f"   Premium: {format_currency(exit_premium)}")
            print(f"   Duration: {duration}")
            print(f"   Reason: {reason}")
            print()
            print(f"💰 P&L:")
            pnl_color = "🟢" if pnl > 0 else "🔴"
            print(f"   {pnl_color} {format_currency(pnl)} ({format_percent(pnl_pct)})")
            print()
            
            # Analyze why it lost
            if pnl < 0:
                premium_change = exit_premium - entry_premium
                premium_change_pct = (premium_change / entry_premium * 100) if entry_premium > 0 else 0
                print(f"🔍 LOSS ANALYSIS:")
                print(f"   Premium dropped: {format_currency(abs(premium_change))} ({format_percent(abs(premium_change_pct))})")
                print(f"   Entry premium: {format_currency(entry_premium)}")
                print(f"   Exit premium: {format_currency(exit_premium)}")
                
                # Check if it was a stop-loss
                if 'stop' in reason.lower() or 'sl' in reason.lower():
                    print(f"   ⚠️  Exit reason: STOP-LOSS triggered")
                    print(f"   💡 This trade was automatically stopped to prevent larger loss")
                elif 'take' in reason.lower() or 'tp' in reason.lower():
                    print(f"   ✅ Exit reason: TAKE-PROFIT (partial exit)")
                else:
                    print(f"   ℹ️  Exit reason: {reason}")
                print()
        else:
            # No matching BUY found
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"TRADE #{idx} - SELL ONLY (No matching BUY found)")
            print(f"Symbol: {symbol}")
            print(f"P&L: {format_currency(sell_trade.get('pnl', 0))}")
            print(f"Reason: {sell_trade.get('reason', 'Unknown')}")
            print()
    
    # Pattern Analysis
    print("=" * 80)
    print("🔍 PATTERN ANALYSIS")
    print("=" * 80)
    print()
    
    # Analyze by option type
    if 'option_type' in sell_trades.columns:
        calls = sell_trades[sell_trades['option_type'].str.upper() == 'CALL']
        puts = sell_trades[sell_trades['option_type'].str.upper() == 'PUT']
        
        if len(calls) > 0:
            calls_pnl = calls['pnl'].sum()
            print(f"CALLS: {len(calls)} trades, P&L: {format_currency(calls_pnl)}")
        
        if len(puts) > 0:
            puts_pnl = puts['pnl'].sum()
            print(f"PUTS: {len(puts)} trades, P&L: {format_currency(puts_pnl)}")
        print()
    
    # Analyze by exit reason
    if 'reason' in sell_trades.columns:
        print("Exit Reasons:")
        reason_counts = sell_trades['reason'].value_counts()
        for reason, count in reason_counts.items():
            reason_trades = sell_trades[sell_trades['reason'] == reason]
            reason_pnl = reason_trades['pnl'].sum()
            print(f"   {reason}: {count} trades, P&L: {format_currency(reason_pnl)}")
        print()
    
    # Analyze by regime
    if 'regime' in sell_trades.columns:
        print("Volatility Regime Performance:")
        regime_counts = sell_trades['regime'].value_counts()
        for regime, count in regime_counts.items():
            regime_trades = sell_trades[sell_trades['regime'] == regime]
            regime_pnl = regime_trades['pnl'].sum()
            avg_pnl = regime_trades['pnl'].mean()
            print(f"   {regime.upper()}: {count} trades, Total P&L: {format_currency(regime_pnl)}, Avg: {format_currency(avg_pnl)}")
        print()
    
    # Time-based analysis
    if 'timestamp' in sell_trades.columns:
        sell_trades['hour'] = pd.to_datetime(sell_trades['timestamp']).dt.hour
        print("Performance by Hour:")
        for hour in sorted(sell_trades['hour'].unique()):
            hour_trades = sell_trades[sell_trades['hour'] == hour]
            hour_pnl = hour_trades['pnl'].sum()
            print(f"   {hour:02d}:00 - {len(hour_trades)} trades, P&L: {format_currency(hour_pnl)}")
        print()
    
    # Recommendations
    print("=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    if len(losing_trades) > len(winning_trades):
        print("⚠️  More losing trades than winning trades")
        print("   → Consider increasing MIN_ACTION_STRENGTH_THRESHOLD (currently 0.65)")
        print("   → Review RL model confidence levels")
        print()
    
    if len(losing_trades) > 0:
        avg_loss = abs(losing_trades['pnl'].mean())
        if avg_loss > 100:
            print(f"⚠️  Average loss per trade is high: {format_currency(avg_loss)}")
            print("   → Consider tighter stop-losses")
            print("   → Review position sizing (may be too large)")
            print()
        
        # Check stop-loss effectiveness
        stop_loss_trades = losing_trades[losing_trades['reason'].str.contains('stop|sl', case=False, na=False)]
        if len(stop_loss_trades) > 0:
            sl_avg_loss = abs(stop_loss_trades['pnl'].mean())
            print(f"✅ Stop-losses triggered: {len(stop_loss_trades)} trades")
            print(f"   Average stop-loss: {format_currency(sl_avg_loss)}")
            print("   → Stop-losses are working to limit losses")
            print()
        else:
            print("⚠️  No stop-loss exits found")
            print("   → Consider if stop-losses are being triggered properly")
            print()
    
    # Check if too many trades
    if len(buy_trades) > 20:
        print(f"⚠️  High trade count: {len(buy_trades)} trades today")
        print("   → Consider increasing MIN_TRADE_COOLDOWN_SECONDS")
        print("   → Review MAX_TRADES_PER_SYMBOL (currently 10)")
        print()
    
    # Check win rate
    if win_rate < 40:
        print(f"⚠️  Low win rate: {win_rate:.1f}%")
        print("   → Focus on higher confidence signals only")
        print("   → Consider market conditions (VIX, regime)")
        print()
    
    print("=" * 80)
    print("✅ Analysis Complete")
    print("=" * 80)

def main():
    """Main function"""
    if not TRADE_DB_AVAILABLE:
        print("❌ Trade database not available")
        sys.exit(1)
    
    try:
        trade_db = TradeDatabase()
        
        # Get all trades
        all_trades = trade_db.get_all_trades()
        
        if not all_trades:
            print("❌ No trades found in database")
            sys.exit(1)
        
        # Convert to DataFrame
        trades_df = pd.DataFrame(all_trades)
        
        # Check if --recent flag is used
        analyze_recent = '--recent' in sys.argv or '-r' in sys.argv
        
        if analyze_recent:
            # Analyze most recent trading day
            try:
                trades_df['date'] = pd.to_datetime(trades_df['timestamp'], errors='coerce', format='mixed').dt.date
                valid_dates = trades_df['date'].dropna()
                if len(valid_dates) > 0:
                    most_recent_date = valid_dates.max()
                    print(f"📅 Analyzing most recent trading day: {most_recent_date}")
                    trades_df = trades_df[trades_df['date'] == most_recent_date].copy()
                    if trades_df.empty:
                        print(f"❌ No trades found for {most_recent_date}")
                        return
                else:
                    print("❌ No valid dates found in trades")
                    return
            except Exception as e:
                print(f"❌ Error processing dates: {e}")
                return
        
        # Analyze
        analyze_trades(trades_df)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

