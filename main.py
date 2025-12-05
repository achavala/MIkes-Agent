#!/usr/bin/env python3
"""
Main entry point for MikeAgent trading system
Supports paper trading and live trading modes
"""
import argparse
import sys
from datetime import datetime
from core.utils.backtest_engine import BacktestEngine
from core.utils.strategy_manager import StrategyManager
from core.strategy.base import Bar
import yfinance as yf


def run_backtest(symbols: list, start_date: str, end_date: str, capital: float = 10000.0):
    """Run backtest mode"""
    engine = BacktestEngine(start_date, end_date, capital)
    engine.run(symbols, "MikeAgent")


def run_paper_trading(symbols: list, capital: float = 10000.0):
    """Run paper trading mode"""
    print(f"\n{'='*60}")
    print("PAPER TRADING MODE")
    print(f"{'='*60}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Initial Capital: ${capital:,.2f}")
    print("Press Ctrl+C to stop\n")
    
    manager = StrategyManager(capital)
    agent = manager.get_agent("MikeAgent")
    
    if not agent:
        print("Error: MikeAgent not found")
        return
    
    try:
        while True:
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d", interval="1m")
                    
                    if len(hist) == 0:
                        continue
                    
                    latest = hist.iloc[-1]
                    bar = Bar(
                        open=latest['Open'],
                        high=latest['High'],
                        low=latest['Low'],
                        close=latest['Close'],
                        volume=int(latest['Volume']),
                        timestamp=hist.index[-1]
                    )
                    
                    signal = agent.on_bar(symbol, bar)
                    
                    if signal:
                        print(f"[{datetime.now()}] {symbol}: {signal.action.value} "
                              f"{signal.option_type} strike={signal.strike} size={signal.size}")
                
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
            
            import time
            time.sleep(60)  # Wait 1 minute between checks
    
    except KeyboardInterrupt:
        print("\n\nPaper trading stopped by user")


def run_live_trading(symbols: list, capital: float = 10000.0):
    """Run live trading mode (requires Alpaca API setup)"""
    print("\n⚠️  LIVE TRADING MODE")
    print("This will execute real trades with real money!")
    print("Make sure you have:")
    print("1. Alpaca API credentials configured")
    print("2. Paper trading tested for at least 1 week")
    print("3. Proper risk management settings")
    
    response = input("\nType 'YES' to continue: ")
    if response != "YES":
        print("Live trading cancelled")
        return
    
    # TODO: Implement Alpaca integration
    print("\nLive trading not yet implemented")
    print("Please use paper trading mode for now")


def main():
    parser = argparse.ArgumentParser(description="MikeAgent Trading System")
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "live"],
        default="backtest",
        help="Trading mode"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="SPY,QQQ",
        help="Comma-separated list of symbols"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="MikeAgent",
        help="Agent to use"
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2025-11-03",
        help="Start date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2025-12-01",
        help="End date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital"
    )
    
    args = parser.parse_args()
    
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    if args.mode == "backtest":
        run_backtest(symbols, args.start_date, args.end_date, args.capital)
    elif args.mode == "paper":
        run_paper_trading(symbols, args.capital)
    elif args.mode == "live":
        run_live_trading(symbols, args.capital)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()

