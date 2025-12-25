#!/usr/bin/env python3
"""
5-Day Paper Mode Test Run
Paper trading mode with full risk constraints
"""
import sys
from datetime import datetime, timedelta
from run_30day_backtest import InstitutionalBacktest
import pandas as pd

def main():
    print("="*70)
    print("  5-DAY PAPER MODE TEST RUN")
    print("="*70)
    print("\n📋 Configuration:")
    print("   Mode: Paper (full risk constraints + action nudge, no probe trades)")
    print("   Duration: 5 trading days")
    print("   Symbols: SPY, QQQ (SPX excluded - requires options data)")
    print("   Capital: $100,000")
    print("   Features: Action nudge (0.15 threshold), NO probe trades")
    print("   Risk: Full constraints (gamma/delta 1.0x, VIX kill switch ON)")
    print("   Ensemble: Standard agreement (2+ agents required)")
    
    # Calculate date range
    end_date = datetime.now()
    trading_days = pd.bdate_range(end=end_date, periods=5, freq='B')
    if len(trading_days) > 0:
        start_date = trading_days[0].to_pydatetime()
        end_date = trading_days[-1].to_pydatetime()
        print(f"   Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    else:
        start_date = end_date - timedelta(days=10)
        print(f"   Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (fallback)")
    
    print("\n" + "="*70)
    
    # Initialize backtest in paper mode
    backtest = InstitutionalBacktest(
        symbols=['SPY', 'QQQ'],  # SPX excluded - requires options data, not minute bars
        capital=100000.0,
        mode='paper',  # Paper mode with full risk constraints
        log_dir="logs/5day_paper_test"
    )
    
    # Run backtest
    print("\n🚀 Starting 5-day paper mode backtest...\n")
    try:
        results = backtest.run_backtest(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        print("\n" + "="*70)
        print("  5-DAY PAPER MODE BACKTEST COMPLETE")
        print("="*70)
        
        # Print key results
        if 'trade_activity' in results:
            activity = results['trade_activity']
            print(f"\n📊 Trade Activity:")
            print(f"   Total trades: {activity.get('total_trades', 0)}")
            print(f"   Trading days: {activity.get('trading_days', 0)}")
            print(f"   Avg trades/day: {activity.get('avg_trades_per_day', 0):.2f}")
        
        if 'verdict' in results:
            verdict = results['verdict']
            recommendation = verdict.get('recommendation', {})
            print(f"\n🎯 Verdict: {recommendation.get('decision', 'UNKNOWN')}")
            print(f"   Reason: {recommendation.get('reason', 'N/A')}")
            
            scores = verdict.get('overall_scores', {})
            if scores:
                print(f"\n📊 Scorecards:")
                print(f"   Behavior: {scores.get('behavior', 0):.2f}")
                print(f"   Risk: {scores.get('risk', 0):.2f}")
                print(f"   Execution: {scores.get('execution', 0):.2f}")
                print(f"   Learning: {scores.get('learning', 0):.2f}")
                print(f"   Average: {scores.get('average', 0):.2f}")
        
        print("\n" + "="*70)
        print("✅ Paper mode test complete! Check logs/5day_paper_test/ for detailed logs.")
        print("="*70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())





