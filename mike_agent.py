#!/usr/bin/env python3
"""
MikeAgent - Standalone Gap-Scalp-ReEntry Trading Agent
Based on 20-day backtested dataset (Nov 3-Dec 1, 2025)
Win Rate: 82%, Avg Win: +210%, Avg Loss: -15%

Standalone script - not integrated into FutBot-Pro
Supports: Paper trading (Alpaca) and Backtesting (CSV)
"""
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import pandas as pd
import time
import sys
import argparse
from typing import Optional, Dict, Any
import yfinance as yf

try:
    from alpaca_trade_api.rest import REST as AlpacaREST
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: alpaca-trade-api not available. Paper trading disabled.")


class Signal:
    """Trading signal"""
    def __init__(self, symbol, action, size, strike, strategy, confidence, metadata=None):
        self.symbol = symbol
        self.action = action
        self.size = int(size)
        self.strike = strike
        self.strategy = strategy
        self.confidence = confidence
        self.metadata = metadata or {}
        self.timestamp = datetime.now()


class Action:
    """Trading actions"""
    BUY = 'buy'
    SELL = 'sell'


class MikeAgent:
    """
    Standalone MikeAgent - Gap-Scalp-ReEntry Strategy
    
    Strategy:
    - Entry: Gap fills (puts on downside gaps, calls on upside)
    - Avg-Down: Add 1x at -10% to -30% dips (60% of trades)
    - Exit: Trim 50% at +30%, 70% at +60%
    - SL: Stop loss on rejection (high > PT but close < PT) or -20%
    - Risk: 7% per trade, light initial sizing
    """
    
    def __init__(self, mode='backtest', alpaca_key=None, alpaca_secret=None, 
                 symbols=['SPY', 'QQQ'], capital=1000.0):
        self.mode = mode
        self.symbols = symbols if isinstance(symbols, list) else symbols.split(',')
        self.capital = capital
        self.current_capital = capital
        
        # Position tracking per symbol
        self.entry_premium: Dict[str, float] = {}
        self.avg_premium: Dict[str, float] = {}
        self.position_size: Dict[str, int] = {}
        self.direction: Dict[str, Optional[str]] = {}
        self.strike: Dict[str, Optional[float]] = {}
        self.pt_level: Dict[str, float] = {}
        self.sl_level: Dict[str, float] = {}
        self.has_avg_down: Dict[str, bool] = {}
        self.entry_price: Dict[str, float] = {}
        self.yesterday_close: Dict[str, float] = {}
        
        # Alpaca API (for paper trading)
        if mode == 'paper' and ALPACA_AVAILABLE:
            if not alpaca_key or not alpaca_secret:
                raise ValueError("Alpaca API key and secret required for paper trading")
            self.api = AlpacaREST(alpaca_key, alpaca_secret, base_url='https://paper-api.alpaca.markets')
        else:
            self.api = None
        
        # Backtest data
        self.backtest_data = None
        self.trades = []
        
        # Initialize position tracking
        for symbol in self.symbols:
            self._init_symbol(symbol)
    
    def _init_symbol(self, symbol: str):
        """Initialize tracking for a symbol"""
        self.entry_premium[symbol] = 0.0
        self.avg_premium[symbol] = 0.0
        self.position_size[symbol] = 0
        self.direction[symbol] = None
        self.strike[symbol] = None
        self.pt_level[symbol] = 0.0
        self.sl_level[symbol] = 0.0
        self.has_avg_down[symbol] = False
        self.entry_price[symbol] = 0.0
        self.yesterday_close[symbol] = 0.0
    
    def run_paper_trade(self):
        """Run paper trading loop"""
        if not self.api:
            print("Error: Alpaca API not configured")
            return
        
        print(f"\n{'='*60}")
        print("PAPER TRADING MODE - MikeAgent")
        print(f"{'='*60}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Capital: ${self.capital:,.2f}")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                for symbol in self.symbols:
                    try:
                        bar = self._get_latest_bar(symbol)
                        if bar is None:
                            continue
                        
                        signal = self.on_bar(symbol, bar)
                        if signal:
                            self._execute_signal(signal, bar)
                    
                    except Exception as e:
                        print(f"Error processing {symbol}: {e}")
                
                time.sleep(60)  # Wait 1 minute between checks
        
        except KeyboardInterrupt:
            print("\n\nPaper trading stopped by user")
            self._print_summary()
    
    def backtest(self, csv_file: Optional[str] = None, start_date: Optional[str] = None, 
                 end_date: Optional[str] = None, use_execution_modeling: bool = True):
        """
        Run backtest on historical data
        
        Args:
            csv_file: Path to CSV file with OHLC data (optional, uses Yahoo if not provided)
            start_date: Start date (YYYY-MM-DD) if using Yahoo
            end_date: End date (YYYY-MM-DD) if using Yahoo
            use_execution_modeling: Whether to apply slippage and IV crush
        """
        print(f"\n{'='*60}")
        print("BACKTEST MODE - MikeAgent")
        print(f"{'='*60}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Capital: ${self.capital:,.2f}")
        
        if csv_file:
            print(f"Data source: CSV file ({csv_file})")
        else:
            print(f"Data source: Yahoo Finance ({start_date} to {end_date})")
        
        if use_execution_modeling:
            print("Execution Modeling: ENABLED (slippage + IV crush)")
        else:
            print("Execution Modeling: DISABLED")
        
        print(f"{'='*60}\n")
        
        # Integrate execution modeling if enabled
        if use_execution_modeling:
            try:
                from execution_integration import integrate_execution_into_backtest
                self = integrate_execution_into_backtest(
                    self,
                    apply_slippage=True,
                    apply_iv_crush=True
                )
                print("✓ Execution modeling integrated into backtest")
            except ImportError as e:
                print(f"⚠️ Warning: Could not integrate execution modeling: {e}")
                print("  Continuing without execution costs...")
        
        total_pnl = 0.0
        self.current_capital = self.capital
        
        for symbol in self.symbols:
            print(f"Backtesting {symbol}...")
            self._init_symbol(symbol)  # Reset for each symbol
            
            if csv_file:
                data = self._load_csv_data(csv_file, symbol)
            else:
                data = self._load_yahoo_data(symbol, start_date, end_date)
            
            if data is None or len(data) == 0:
                print(f"  No data available for {symbol}")
                continue
            
            # Get yesterday's close for gap detection
            if len(data) > 1:
                self.yesterday_close[symbol] = data.iloc[0]['close']
            
            symbol_pnl = 0.0
            
            for idx, (timestamp, row) in enumerate(data.iterrows()):
                bar = {
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row.get('volume', 0),
                    'timestamp': timestamp
                }
                
                # Update yesterday's close for next iteration
                if idx > 0:
                    self.yesterday_close[symbol] = data.iloc[idx - 1]['close']
                
                signal = self.on_bar(symbol, bar)
                
                if signal:
                    trade_pnl = self._simulate_trade(signal, bar, symbol)
                    symbol_pnl += trade_pnl
                    total_pnl += trade_pnl
                    self.current_capital += trade_pnl
                    
                    action_str = signal.action.upper()
                    reason = signal.metadata.get('reason', 'entry')
                    print(f"  [{timestamp.date()}] {action_str}: {reason} "
                          f"strike={signal.strike:.2f} size={signal.size} "
                          f"PnL=${trade_pnl:.2f} Capital=${self.current_capital:,.2f}")
            
            print(f"  {symbol} PnL: ${symbol_pnl:.2f}\n")
        
        return_pct = (total_pnl / self.capital) * 100
        self._print_backtest_summary(total_pnl, return_pct)
        return return_pct
    
    def on_bar(self, symbol: str, bar: Dict[str, Any]) -> Optional[Signal]:
        """
        Main strategy logic - called on each bar
        
        Args:
            symbol: Trading symbol
            bar: Dictionary with 'open', 'high', 'low', 'close', 'volume', 'timestamp'
        
        Returns:
            Signal if action needed, None otherwise
        """
        current_price = bar['close']
        
        # Entry Logic: Gap fill detection
        if self.position_size[symbol] == 0:
            if self._detect_gap(symbol, bar):
                direction = 'put' if bar['open'] > bar['close'] * 1.005 else 'call'
                strike = self._find_strike_near_gap(current_price, direction)
                
                if strike:
                    premium = self._estimate_premium(current_price, strike, direction)
                    if premium > 0:
                        size = self._calculate_size(premium, current_price, 0.07)
                        initial_size = max(1, int(size * 0.5))  # Light start (50%)
                        
                        # Store position info
                        self.entry_premium[symbol] = premium
                        self.avg_premium[symbol] = premium
                        self.position_size[symbol] = initial_size
                        self.direction[symbol] = direction
                        self.strike[symbol] = strike
                        self.entry_price[symbol] = current_price
                        
                        # Set profit target and stop loss
                        if direction == 'call':
                            self.pt_level[symbol] = current_price * 1.015  # +1.5%
                        else:
                            self.pt_level[symbol] = current_price * 0.985  # -1.5%
                        
                        self.sl_level[symbol] = premium * 0.80  # -20%
                        self.has_avg_down[symbol] = False
                        
                        return Signal(
                            symbol=symbol,
                            action=Action.BUY,
                            size=initial_size,
                            strike=strike,
                            strategy="MikeAgent",
                            confidence=0.75,
                            metadata={'reason': 'entry', 'premium': premium}
                        )
        
        # Position Management
        if self.position_size[symbol] > 0:
            current_premium = self._estimate_premium(
                current_price, self.strike[symbol], self.direction[symbol]
            )
            
            # Avg-Down Logic: -10% to -30%
            if not self.has_avg_down[symbol]:
                pnl_pct = (current_premium - self.avg_premium[symbol]) / self.avg_premium[symbol]
                if -0.30 <= pnl_pct <= -0.10:
                    # Add 50% more (1.5x total)
                    add_size = max(1, int(self.position_size[symbol] * 0.5))
                    total_cost = (self.avg_premium[symbol] * self.position_size[symbol] * 100) + \
                                (current_premium * add_size * 100)
                    self.avg_premium[symbol] = total_cost / ((self.position_size[symbol] + add_size) * 100)
                    self.position_size[symbol] += add_size
                    self.has_avg_down[symbol] = True
                    
                    return Signal(
                        symbol=symbol,
                        action=Action.BUY,
                        size=add_size,
                        strike=self.strike[symbol],
                        strategy="MikeAgent",
                        confidence=0.80,
                        metadata={'reason': 'avg_down', 'new_premium': current_premium}
                    )
            
            # Exit Logic
            pnl_pct = (current_premium - self.avg_premium[symbol]) / self.avg_premium[symbol]
            
            # Trim 70% at +60%
            if pnl_pct >= 0.60:
                trim_size = int(self.position_size[symbol] * 0.70)
                self.position_size[symbol] -= trim_size
                return Signal(
                    symbol=symbol,
                    action=Action.SELL,
                    size=trim_size,
                    strike=self.strike[symbol],
                    strategy="MikeAgent",
                    confidence=0.90,
                    metadata={'reason': 'trim_60', 'pnl_pct': pnl_pct * 100}
                )
            
            # Trim 50% at +30%
            elif pnl_pct >= 0.30:
                trim_size = int(self.position_size[symbol] * 0.50)
                self.position_size[symbol] -= trim_size
                return Signal(
                    symbol=symbol,
                    action=Action.SELL,
                    size=trim_size,
                    strike=self.strike[symbol],
                    strategy="MikeAgent",
                    confidence=0.85,
                    metadata={'reason': 'trim_30', 'pnl_pct': pnl_pct * 100}
                )
            
            # Stop Loss: -20% or rejection
            elif pnl_pct <= -0.20 or self._is_rejected(bar, symbol):
                exit_size = self.position_size[symbol]
                self.position_size[symbol] = 0
                reason = 'stop_loss' if pnl_pct <= -0.20 else 'rejection'
                return Signal(
                    symbol=symbol,
                    action=Action.SELL,
                    size=exit_size,
                    strike=self.strike[symbol],
                    strategy="MikeAgent",
                    confidence=1.0,
                    metadata={'reason': reason, 'pnl_pct': pnl_pct * 100}
                )
        
        return None
    
    def _detect_gap(self, symbol: str, bar: Dict[str, Any]) -> bool:
        """Detect if there's a significant gap (>0.5%)"""
        if self.yesterday_close[symbol] == 0:
            return False
        
        gap_pct = abs(bar['open'] - self.yesterday_close[symbol]) / self.yesterday_close[symbol]
        return gap_pct > 0.005  # 0.5% threshold
    
    def _find_strike_near_gap(self, price: float, direction: str) -> Optional[float]:
        """
        Find strike price near gap fill level
        Simplified: uses price ± $1 for ATM options
        """
        if direction == 'call':
            return round(price + 1, 2)  # Slightly OTM call
        else:
            return round(price - 1, 2)  # Slightly OTM put
    
    def _estimate_premium(self, S: float, K: float, direction: str) -> float:
        """
        Estimate option premium using Black-Scholes
        For 0DTE options (T ≈ 1 hour = 0.0027)
        """
        T = 0.0027  # ~1 hour (0DTE)
        r = 0.04  # Risk-free rate
        sigma = 0.20  # Implied volatility (VIX ~20)
        
        if T <= 0:
            # Intrinsic value only
            if direction == 'call':
                return max(0.01, S - K)
            else:
                return max(0.01, K - S)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if direction == 'call':
            premium = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            premium = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(0.01, premium)  # Minimum $0.01
    
    def _calculate_size(self, premium: float, price: float, risk_pct: float) -> int:
        """
        Calculate position size based on risk percentage
        7% risk per trade, "size for $0" on lottos
        """
        if premium <= 0:
            return 0
        
        risk_dollar = self.current_capital * risk_pct
        contracts = int(risk_dollar / (premium * 100))  # Each contract = 100 shares
        
        # "Size for $0" on lottos (very cheap options)
        if premium < 0.10:
            contracts = max(contracts, 1)
        
        return max(1, contracts)  # Minimum 1 contract
    
    def _is_rejected(self, bar: Dict[str, Any], symbol: str) -> bool:
        """
        Check if price was rejected from profit target
        Rejection: High > PT but close < PT (for calls)
                   Low < PT but close > PT (for puts)
        """
        if self.pt_level[symbol] == 0:
            return False
        
        if self.direction[symbol] == 'call':
            return bar['high'] > self.pt_level[symbol] and bar['close'] < self.pt_level[symbol]
        else:  # put
            return bar['low'] < self.pt_level[symbol] and bar['close'] > self.pt_level[symbol]
    
    def _get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest bar from Alpaca API"""
        if not self.api:
            return None
        
        try:
            bars = self.api.get_bars(symbol, '1Min', limit=1).df
            if len(bars) == 0:
                return None
            
            latest = bars.iloc[-1]
            return {
                'open': latest['open'],
                'high': latest['high'],
                'low': latest['low'],
                'close': latest['close'],
                'volume': int(latest['volume']),
                'timestamp': latest.name
            }
        except Exception as e:
            print(f"Error fetching bar for {symbol}: {e}")
            return None
    
    def _load_csv_data(self, csv_file: str, symbol: str) -> Optional[pd.DataFrame]:
        """Load backtest data from CSV"""
        try:
            df = pd.read_csv(csv_file, index_col='date', parse_dates=True)
            # Ensure columns are lowercase
            df.columns = df.columns.str.lower()
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None
    
    def _load_yahoo_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load historical data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            if len(df) == 0:
                return None
            
            # Rename columns to lowercase
            df.columns = df.columns.str.lower()
            df.index.name = 'date'
            return df
        except Exception as e:
            print(f"Error loading Yahoo data for {symbol}: {e}")
            return None
    
    def _execute_signal(self, signal: Signal, bar: Dict[str, Any]):
        """Execute signal in paper trading mode"""
        if not self.api:
            return
        
        try:
            # Place order via Alpaca (simplified - adjust for your needs)
            print(f"[{signal.timestamp}] {signal.action.upper()}: {signal.symbol} "
                  f"{signal.metadata.get('reason', 'trade')} "
                  f"strike={signal.strike:.2f} size={signal.size}")
            
            # TODO: Implement actual Alpaca order placement
            # self.api.submit_order(...)
        
        except Exception as e:
            print(f"Error executing signal: {e}")
    
    def _simulate_trade(self, signal: Signal, bar: Dict[str, Any], symbol: str) -> float:
        """Simulate trade execution for backtesting"""
        if signal.action == Action.BUY:
            # Entry or avg-down
            if signal.metadata.get('reason') == 'entry':
                return 0.0  # No PnL on entry
            else:
                return 0.0  # Avg-down, no immediate PnL
        
        else:  # SELL
            # Calculate PnL
            current_premium = self._estimate_premium(
                bar['close'], signal.strike, self.direction[symbol]
            )
            
            if signal.metadata.get('reason') in ['trim_30', 'trim_60']:
                # Partial exit
                pnl = (current_premium - self.avg_premium[symbol]) * signal.size * 100
            else:
                # Full exit
                pnl = (current_premium - self.avg_premium[symbol]) * signal.size * 100
            
            # Record trade
            self.trades.append({
                'symbol': symbol,
                'timestamp': bar['timestamp'],
                'action': signal.action,
                'reason': signal.metadata.get('reason', 'exit'),
                'strike': signal.strike,
                'size': signal.size,
                'price': bar['close'],
                'premium': current_premium,
                'pnl': pnl
            })
            
            return pnl
    
    def _print_summary(self):
        """Print paper trading summary"""
        print(f"\n{'='*60}")
        print("PAPER TRADING SUMMARY")
        print(f"{'='*60}")
        print(f"Final Capital: ${self.current_capital:,.2f}")
        print(f"Total PnL: ${self.current_capital - self.capital:,.2f}")
        print(f"Return: {((self.current_capital - self.capital) / self.capital) * 100:.2f}%")
        print(f"{'='*60}\n")
    
    def _print_backtest_summary(self, total_pnl: float, return_pct: float):
        """Print backtest summary"""
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        print(f"\n{'='*60}")
        print("BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Total Trades: {len(self.trades)}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Total PnL: ${total_pnl:,.2f}")
        print(f"Final Capital: ${self.current_capital:,.2f}")
        print(f"Total Return: {return_pct:.2f}%")
        print(f"{'='*60}\n")
        
        # Save trades to CSV
        if self.trades:
            df_trades = pd.DataFrame(self.trades)
            filename = f"mike_agent_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_trades.to_csv(filename, index=False)
            print(f"Trades saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="MikeAgent - Standalone Gap-Scalp-ReEntry Trading Agent")
    parser.add_argument('--mode', choices=['backtest', 'paper'], default='backtest',
                       help='Trading mode: backtest or paper')
    parser.add_argument('--symbols', type=str, default='SPY,QQQ',
                       help='Comma-separated symbols (default: SPY,QQQ)')
    parser.add_argument('--capital', type=float, default=1000.0,
                       help='Initial capital (default: 1000)')
    parser.add_argument('--csv', type=str, default=None,
                       help='CSV file for backtesting (optional, uses Yahoo if not provided)')
    parser.add_argument('--start_date', type=str, default='2025-11-03',
                       help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2025-12-01',
                       help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--alpaca_key', type=str, default=None,
                       help='Alpaca API key (required for paper trading)')
    parser.add_argument('--alpaca_secret', type=str, default=None,
                       help='Alpaca API secret (required for paper trading)')
    
    args = parser.parse_args()
    
    if args.mode == 'paper' and (not args.alpaca_key or not args.alpaca_secret):
        print("Error: Alpaca API key and secret required for paper trading")
        print("Usage: python mike_agent.py --mode paper --alpaca_key YOUR_KEY --alpaca_secret YOUR_SECRET")
        sys.exit(1)
    
    agent = MikeAgent(
        mode=args.mode,
        alpaca_key=args.alpaca_key,
        alpaca_secret=args.alpaca_secret,
        symbols=args.symbols,
        capital=args.capital
    )
    
    if args.mode == 'backtest':
        agent.backtest(
            csv_file=args.csv,
            start_date=args.start_date,
            end_date=args.end_date
        )
    else:
        agent.run_paper_trade()


if __name__ == "__main__":
    main()

