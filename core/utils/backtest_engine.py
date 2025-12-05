"""
Backtesting engine for strategy validation
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict
from core.strategy.base import Bar, Signal
from core.utils.strategy_manager import StrategyManager
import csv


class BacktestEngine:
    """Runs backtests on historical data"""
    
    def __init__(self, start_date: str, end_date: str, initial_capital: float = 10000.0):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades = []
        self.daily_pnl = []
        
    def run(self, symbols: List[str], agent_name: str = "MikeAgent"):
        """Run backtest on symbols"""
        manager = StrategyManager(self.initial_capital)
        agent = manager.get_agent(agent_name)
        
        if not agent:
            print(f"Agent {agent_name} not found")
            return
        
        print(f"\n{'='*60}")
        print(f"Backtesting {agent_name}")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"{'='*60}\n")
        
        for symbol in symbols:
            self._backtest_symbol(symbol, agent, manager)
        
        self._generate_report()
    
    def _backtest_symbol(self, symbol: str, agent, manager):
        """Backtest a single symbol"""
        print(f"Backtesting {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(
                start=self.start_date,
                end=self.end_date + timedelta(days=1)
            )
            
            if len(hist) == 0:
                print(f"No data for {symbol}")
                return
            
            # Reset agent for each symbol
            agent.reset()
            
            position = None
            entry_premium = 0.0
            
            for idx, (timestamp, row) in enumerate(hist.iterrows()):
                bar = Bar(
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    close=row['Close'],
                    volume=int(row['Volume']),
                    timestamp=timestamp
                )
                
                # Get signal from agent
                signal = agent.on_bar(symbol, bar)
                
                if signal:
                    if signal.action.value == "BUY" and position is None:
                        # Entry
                        position = {
                            'symbol': symbol,
                            'strike': signal.strike,
                            'direction': signal.option_type,
                            'size': signal.size,
                            'entry_price': bar.close,
                            'entry_premium': agent.entry_premium,
                            'entry_time': timestamp,
                            'entry_bar': idx
                        }
                        entry_premium = agent.entry_premium
                        print(f"  [{timestamp.date()}] ENTRY: {signal.option_type.upper()} "
                              f"strike={signal.strike:.2f} size={signal.size} "
                              f"premium=${entry_premium:.2f}")
                    
                    elif signal.action.value == "SELL" and position:
                        # Exit
                        exit_premium = agent._estimate_premium(
                            bar.close, position['strike'], position['direction']
                        )
                        pnl = (exit_premium - entry_premium) * position['size'] * 100
                        pnl_pct = ((exit_premium - entry_premium) / entry_premium) * 100
                        
                        self.capital += pnl
                        
                        trade = {
                            'symbol': symbol,
                            'entry_time': position['entry_time'],
                            'exit_time': timestamp,
                            'direction': position['direction'],
                            'strike': position['strike'],
                            'size': position['size'],
                            'entry_premium': entry_premium,
                            'exit_premium': exit_premium,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'reason': signal.metadata.get('reason', 'exit')
                        }
                        self.trades.append(trade)
                        
                        print(f"  [{timestamp.date()}] EXIT: {signal.metadata.get('reason', 'exit')} "
                              f"PnL=${pnl:.2f} ({pnl_pct:+.1f}%) "
                              f"Capital=${self.capital:,.2f}")
                        
                        if signal.metadata.get('reason') in ['trim_30', 'trim_60']:
                            # Partial exit, keep position
                            position['size'] -= signal.size
                            entry_premium = agent.avg_premium
                        else:
                            # Full exit
                            position = None
                            entry_premium = 0.0
                
                # Track daily PnL
                if idx == len(hist) - 1 or hist.index[idx + 1].date() != timestamp.date():
                    daily_pnl = self.capital - self.initial_capital
                    self.daily_pnl.append({
                        'date': timestamp.date(),
                        'symbol': symbol,
                        'pnl': daily_pnl,
                        'capital': self.capital
                    })
        
        except Exception as e:
            print(f"Error backtesting {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_report(self):
        """Generate backtest report"""
        if not self.trades:
            print("\nNo trades executed")
            return
        
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        print(f"\n{'='*60}")
        print("BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Total PnL: ${total_pnl:,.2f}")
        print(f"Final Capital: ${self.capital:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"{'='*60}\n")
        
        # Save trades to CSV
        self._save_trades_csv()
    
    def _save_trades_csv(self):
        """Save trades to CSV file"""
        filename = f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.trades[0].keys())
            writer.writeheader()
            writer.writerows(self.trades)
        print(f"Trades saved to {filename}")

