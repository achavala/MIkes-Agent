#!/usr/bin/env python3
"""
Phase 0 Backtest Runner

Runs Phase 0 backtest on last week of data and generates detailed analysis.
"""

import os
import sys
from datetime import datetime, timedelta
import pytz

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from phase0_backtest.engine.phase0_loop import Phase0ReplayLoop
from phase0_backtest.metrics.report import Phase0Report

# Import Alpaca API for data fetching
try:
    import alpaca_trade_api as tradeapi
    import config
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠️  Alpaca API not available - will use yfinance fallback")


def init_alpaca():
    """Initialize Alpaca API"""
    if not ALPACA_AVAILABLE:
        return None
    
    try:
        api_key = getattr(config, 'ALPACA_KEY', None) or getattr(config, 'APCA_API_KEY_ID', None)
        api_secret = getattr(config, 'ALPACA_SECRET', None) or getattr(config, 'APCA_API_SECRET_KEY', None)
        base_url = getattr(config, 'ALPACA_BASE_URL', None) or getattr(config, 'APCA_API_BASE_URL', None)
        
        if not api_key or api_key == 'YOUR_PAPER_KEY' or not api_secret:
            print("⚠️  Alpaca credentials not configured")
            return None
        
        api = tradeapi.REST(api_key, api_secret, base_url=base_url, api_version='v2')
        return api
    except Exception as e:
        print(f"⚠️  Error initializing Alpaca: {e}")
        return None


def main():
    """Run Phase 0 backtest"""
    # ========== PREVENT BACKTEST DURING MARKET HOURS ==========
    # Check if live agent is running or if market is open
    est = pytz.timezone('US/Eastern')
    now_est = datetime.now(est)
    
    # Check if market is open (9:30 AM - 4:00 PM EST, Mon-Fri)
    is_weekday = now_est.weekday() < 5
    market_open_time = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_time = now_est.replace(hour=16, minute=0, second=0, microsecond=0)
    market_is_open = is_weekday and (market_open_time <= now_est < market_close_time)
    
    # Check if live agent lock exists
    LIVE_AGENT_LOCK_FILE = "/tmp/mike_agent_live.lock"
    live_agent_running = os.path.exists(LIVE_AGENT_LOCK_FILE)
    
    if market_is_open:
        if live_agent_running:
            print("=" * 80)
            print("🚨 BACKTEST BLOCKED: Live agent is running during market hours")
            print("=" * 80)
            print()
            print("The live trading agent is currently running and has priority.")
            print("Backtests should only run when:")
            print("  1. Market is closed (before 9:30 AM or after 4:00 PM EST)")
            print("  2. Live agent is not running")
            print()
            print("Current time:", now_est.strftime("%Y-%m-%d %H:%M:%S %Z"))
            print("Market status: OPEN")
            print("Live agent: RUNNING")
            print()
            print("To run backtest:")
            print("  1. Wait for market to close (after 4:00 PM EST)")
            print("  2. Or stop the live agent first: pkill -f mike_agent_live_safe")
            print()
            sys.exit(1)
        else:
            print("=" * 80)
            print("⚠️  WARNING: Market is open but live agent is not running!")
            print("=" * 80)
            print()
            print("Starting backtest during market hours is not recommended.")
            print("The live agent should be running instead.")
            print()
            print("Current time:", now_est.strftime("%Y-%m-%d %H:%M:%S %Z"))
            print("Market status: OPEN")
            print("Live agent: NOT RUNNING")
            print()
            response = input("Continue with backtest anyway? (yes/no): ")
            if response.lower() != 'yes':
                print("Backtest cancelled.")
                sys.exit(0)
    
    print("=" * 80)
    print("PHASE 0 BACKTEST - LAST WEEK ANALYSIS")
    print("=" * 80)
    print()
    
    # Calculate date range (last 7 trading days)
    est = pytz.timezone('US/Eastern')
    end_date = datetime.now(est)
    # Go back 10 days to get ~7 trading days (accounting for weekends)
    start_date = end_date - timedelta(days=10)
    
    # For backtest, use specific dates if needed (Dec 16-22, 2025)
    # Uncomment to use specific date range:
    # start_date = datetime(2025, 12, 16, 9, 30, 0, tzinfo=est)
    # end_date = datetime(2025, 12, 22, 16, 0, 0, tzinfo=est)
    
    print(f"📅 Date Range: {start_date.date()} to {end_date.date()}")
    print()
    
    # Model path
    model_path = "models/mike_23feature_model_final.zip"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Initialize API
    api = init_alpaca()
    if api:
        print("✅ Alpaca API initialized")
    else:
        print("⚠️  Using yfinance fallback (may have delays)")
    print()
    
    # Get API credentials for option universe filtering
    api_key = getattr(config, 'ALPACA_KEY', None) or getattr(config, 'APCA_API_KEY_ID', None) or os.getenv('APCA_API_KEY_ID')
    api_secret = getattr(config, 'ALPACA_SECRET', None) or getattr(config, 'APCA_API_SECRET_KEY', None) or os.getenv('APCA_API_SECRET_KEY')
    
    # Initialize replay loop with option universe filtering
    replay_loop = Phase0ReplayLoop(
        model_path,
        starting_equity=10000.0,
        api_key=api_key,
        api_secret=api_secret
    )
    
    # Run backtest
    print("🚀 Starting Phase 0 backtest...")
    print()
    
    results = replay_loop.run_backtest(start_date, end_date, api=api)
    
    # Generate report
    print("\n" + "=" * 80)
    print("📊 GENERATING REPORT")
    print("=" * 80)
    print()
    
    report = Phase0Report(results)
    
    # Print summary
    print(report.generate_summary())
    print()
    
    # Save detailed report
    output_file = f"PHASE0_BACKTEST_REPORT_{datetime.now(est).strftime('%Y%m%d')}.md"
    report.save_report(output_file)
    
    # Print detailed analysis
    print("\n" + "=" * 80)
    print("📋 DETAILED ANALYSIS")
    print("=" * 80)
    print()
    print(report.generate_detailed_analysis())
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("📈 SUMMARY STATISTICS")
    print("=" * 80)
    print()
    
    if results['daily_summaries']:
        total_pnl = sum(d['total_pnl'] for d in results['daily_summaries'])
        total_trades = sum(d['trades_taken'] for d in results['daily_summaries'])
        days_halted = sum(1 for d in results['daily_summaries'] if d['trading_halted'])
        zero_trade_days = sum(1 for d in results['daily_summaries'] if d['trades_taken'] == 0)
        
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"Total Trades: {total_trades}")
        print(f"Days with Zero Trades: {zero_trade_days}/{len(results['daily_summaries'])}")
        print(f"Days Halted: {days_halted}/{len(results['daily_summaries'])}")
        print(f"Average Trades/Day: {total_trades / len(results['daily_summaries']):.2f}")
        print()
        
        # Pass/Fail criteria
        print("PHASE 0 PASS/FAIL CRITERIA:")
        print("-" * 80)
        
        passed = True
        if days_halted > 0:
            print(f"❌ FAIL: {days_halted} day(s) violated hard daily loss limit")
            passed = False
        else:
            print(f"✅ PASS: No days violated hard daily loss limit")
        
        avg_trades = total_trades / len(results['daily_summaries'])
        if avg_trades > 5:
            print(f"❌ FAIL: Average trades/day ({avg_trades:.2f}) > 5")
            passed = False
        else:
            print(f"✅ PASS: Average trades/day ({avg_trades:.2f}) <= 5")
        
        zero_trade_pct = (zero_trade_days / len(results['daily_summaries'])) * 100
        if zero_trade_pct < 20:
            print(f"⚠️  WARNING: Only {zero_trade_pct:.1f}% zero-trade days (expected many)")
        else:
            print(f"✅ PASS: {zero_trade_pct:.1f}% zero-trade days (good discipline)")
        
        print()
        if passed:
            print("✅ PHASE 0 BACKTEST: PASSED")
        else:
            print("❌ PHASE 0 BACKTEST: FAILED")
    
    print("\n" + "=" * 80)
    print("✅ BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

