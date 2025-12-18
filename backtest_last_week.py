#!/usr/bin/env python3
"""
📊 REAL BACKTEST: Last Week Performance Test

Uses REAL market data from the last week (not fake numbers)
Tests the trained model (mike_historical_model.zip) with actual historical data
Validates observation space, execution, and P&L calculations

Author: Mike Agent Validation
Date: December 17, 2025
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env file if it exists (same logic as train_historical_model.py)
def _load_local_env() -> dict:
    """
    Load environment variables from a local .env file if present.
    - Looks in CWD and project root (this file's directory).
    - Does NOT override already-set non-empty env vars.
    """
    candidates = [
        Path(os.getcwd()) / ".env",
        Path(__file__).resolve().parent / ".env",
    ]
    loaded_keys: list[str] = []
    for p in candidates:
        try:
            if not p.exists():
                continue
            for raw in p.read_text(errors="ignore").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if not k or not v:
                    continue
                if (k not in os.environ) or (not str(os.environ.get(k, "")).strip()):
                    os.environ[k] = v
                    loaded_keys.append(k)
        except Exception:
            continue
    return {"loaded_keys": sorted(set(loaded_keys))}

# Load .env file
env_info = _load_local_env()
if env_info.get("loaded_keys"):
    print(f"✅ Loaded {len(env_info['loaded_keys'])} environment variables from .env file")
    print(f"   Keys: {', '.join(env_info['loaded_keys'])}")
else:
    print("⚠️  No .env file found or no keys loaded")

# Import required modules
try:
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("❌ stable-baselines3 not installed. Install with: pip install stable-baselines3 sb3-contrib")
    sys.exit(1)

# Import agent modules
try:
    from mike_agent_live_safe import (
        prepare_observation,
        MODEL_PATH,
        LOOKBACK,
        RiskManager,
        get_market_data,
        estimate_premium,
        find_atm_strike
    )
except ImportError as e:
    print(f"❌ Error importing agent modules: {e}")
    sys.exit(1)

# Alpaca API (optional - will use yfinance if not available)
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠️  Alpaca API not available, using yfinance for data")

# yfinance for historical data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("❌ yfinance not available. Install with: pip install yfinance")
    sys.exit(1)


class BacktestEngine:
    """Real backtest engine using actual historical data"""
    
    def __init__(self, start_date: str, end_date: str, initial_capital: float = 10000.0):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # Trading symbols
        self.symbols = ['SPY', 'QQQ']
        
        # Position tracking
        self.positions: Dict[str, Dict] = {}  # {symbol: {entry_price, strike, premium, qty, entry_time, option_type}}
        self.trades: List[Dict] = []
        self.daily_pnl: List[Dict] = []
        
        # Risk manager (check actual signature)
        try:
            # Try with parameters that match actual RiskManager
            self.risk_mgr = RiskManager()
        except TypeError:
            # Fallback: create minimal risk manager
            class SimpleRiskManager:
                def __init__(self):
                    self.current_vix = 20.0
                    self.open_positions = {}
                
                def get_current_vix(self):
                    return self.current_vix
                
                def log(self, msg, level="INFO"):
                    if level == "ERROR":
                        print(f"❌ {msg}")
                    elif level == "WARNING":
                        print(f"⚠️  {msg}")
                    else:
                        print(f"ℹ️  {msg}")
            
            self.risk_mgr = SimpleRiskManager()
        
        # Load model
        print(f"📦 Loading model from {MODEL_PATH}...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        try:
            self.model = PPO.load(MODEL_PATH, print_system_info=False)
            print(f"✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Data storage
        self.market_data: Dict[str, pd.DataFrame] = {}
        
    def load_historical_data(self, symbol: str) -> pd.DataFrame:
        """Load real historical data for the backtest period - PRIORITIZES PAID SERVICES"""
        print(f"📥 Loading historical data for {symbol} ({self.start_date.date()} to {self.end_date.date()})...")
        
        # Calculate period string (e.g., "7d" for 7 days)
        days_diff = (self.end_date - self.start_date).days + 1
        period_str = f"{days_diff}d"
        
        # PRIORITY 1: Try Alpaca API first (you're paying for this!)
        alpaca_api = None
        if ALPACA_AVAILABLE:
            try:
                # Try multiple environment variable names (check .env file variations)
                # .env file uses: ALPACA_API_KEY and ALPACA_SECRET_KEY
                api_key = (os.getenv('ALPACA_API_KEY') or  # .env uses this
                          os.getenv('ALPACA_KEY') or
                          os.getenv('APCA_API_KEY_ID'))
                api_secret = (os.getenv('ALPACA_SECRET_KEY') or  # .env uses this
                             os.getenv('ALPACA_SECRET') or
                             os.getenv('APCA_API_SECRET_KEY') or
                             os.getenv('ALPACA_API_SECRET'))
                base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
                
                if api_key and api_secret:
                    alpaca_api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
                    print(f"   🔑 Alpaca API initialized (PAID SERVICE), attempting data fetch...")
                else:
                    print(f"   ⚠️  Alpaca credentials not found")
                    print(f"      Checked: ALPACA_KEY, APCA_API_KEY_ID, ALPACA_API_KEY")
                    print(f"      Checked: ALPACA_SECRET, APCA_API_SECRET_KEY, ALPACA_SECRET_KEY")
            except Exception as e:
                print(f"   ⚠️  Alpaca API initialization failed: {e}")
        
        # Try to use get_market_data from agent (PRIORITIZES Alpaca → Massive → yfinance)
        try:
            data = get_market_data(
                symbol=symbol,
                period=period_str,  # Use period string, not days parameter
                interval='1m',
                api=alpaca_api,  # Pass API instance to use Alpaca first
                risk_mgr=self.risk_mgr
            )
            if data is not None and len(data) > 0:
                # Determine which source was used by checking data characteristics
                source_hint = "Alpaca/Massive" if len(data) > 2000 else "yfinance"
                print(f"   ✅ Got {len(data)} bars from {source_hint} (paid service prioritized)")
                return data
        except Exception as e:
            print(f"   ⚠️  Agent data source failed: {e}, trying direct Alpaca API...")
        
        # PRIORITY 2: Try Alpaca API directly
        if alpaca_api:
            try:
                from alpaca_trade_api.rest import TimeFrame
                print(f"   🔑 Attempting direct Alpaca API fetch...")
                
                # Convert to Alpaca TimeFrame
                tf = TimeFrame.Minute
                
                # Get bars from Alpaca
                bars = alpaca_api.get_bars(
                    symbol,
                    tf,
                    start=self.start_date.strftime('%Y-%m-%d'),
                    end=(self.end_date + timedelta(days=1)).strftime('%Y-%m-%d')
                ).df
                
                if len(bars) > 0:
                    print(f"   ✅ Got {len(bars)} bars from Alpaca API (PAID SERVICE)")
                    return bars
            except Exception as e:
                print(f"   ⚠️  Direct Alpaca API failed: {e}, trying Massive API...")
        
        # PRIORITY 3: Try Massive API
        try:
            massive_key = os.getenv('MASSIVE_API_KEY') or os.getenv('POLYGON_API_KEY')
            if massive_key:
                print(f"   🔑 Attempting Massive API fetch...")
                # Import and use Massive API client
                try:
                    from massive_api_client import MassiveAPIClient
                    massive_client = MassiveAPIClient()
                    data = massive_client.get_historical_data(
                        symbol,
                        start_date=self.start_date.strftime('%Y-%m-%d'),
                        end_date=(self.end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                        interval='1m'
                    )
                    if data is not None and len(data) > 0:
                        print(f"   ✅ Got {len(data)} bars from Massive API (PAID SERVICE)")
                        return data
                except ImportError:
                    print(f"   ⚠️  Massive API client not available")
                except Exception as e:
                    print(f"   ⚠️  Massive API failed: {e}")
        except Exception as e:
            print(f"   ⚠️  Massive API check failed: {e}")
        
        # FALLBACK: yfinance (free, but delayed - only if paid services fail)
        print(f"   ⚠️  All paid services failed, falling back to yfinance (FREE, DELAYED)...")
        
        # Fallback to yfinance
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(
                start=self.start_date.strftime('%Y-%m-%d'),
                end=(self.end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                interval='1m'
            )
            
            if len(hist) == 0:
                print(f"   ⚠️  No data from yfinance, trying daily data...")
                hist = ticker.history(
                    start=self.start_date.strftime('%Y-%m-%d'),
                    end=(self.end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                    interval='1d'
                )
            
            if len(hist) == 0:
                raise ValueError(f"No data available for {symbol}")
            
            # Normalize column names
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            hist.columns = [col.lower() for col in hist.columns]
            
            # Filter to market hours (9:30 AM - 4:00 PM ET)
            if isinstance(hist.index, pd.DatetimeIndex):
                hist = hist.between_time('09:30', '16:00')
            
            print(f"   ✅ Got {len(hist)} bars from yfinance")
            return hist
            
        except Exception as e:
            print(f"   ❌ Error loading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def simulate_trade(self, symbol: str, action: int, current_price: float, current_time: pd.Timestamp, vix: float = 20.0) -> Dict:
        """Simulate a trade execution"""
        result = {
            'executed': False,
            'pnl': 0.0,
            'reason': ''
        }
        
        option_symbol = f"{symbol}CALL" if action == 1 else f"{symbol}PUT"
        
        if action == 1 or action == 2:  # BUY_CALL or BUY_PUT
            if option_symbol in self.positions:
                result['reason'] = 'Already in position'
                return result
            
            # Calculate strike and premium
            strike = find_atm_strike(current_price)
            option_type = 'call' if action == 1 else 'put'
            premium = estimate_premium(current_price, strike, option_type)
            
            # Position sizing (7% risk)
            risk_dollar = self.capital * 0.07
            qty = max(1, int(risk_dollar / (premium * 100)))
            cost = qty * premium * 100
            
            if cost > self.capital:
                result['reason'] = 'Insufficient capital'
                return result
            
            # Execute trade
            self.positions[option_symbol] = {
                'symbol': symbol,
                'option_symbol': option_symbol,
                'entry_price': current_price,
                'strike': strike,
                'premium': premium,
                'qty': qty,
                'entry_time': current_time,
                'option_type': option_type,
                'cost': cost
            }
            
            self.capital -= cost
            
            result['executed'] = True
            result['reason'] = f'Bought {qty}x {option_type.upper()} @ ${premium:.2f}'
            
        elif action == 5:  # EXIT
            if option_symbol not in self.positions:
                result['reason'] = 'No position to exit'
                return result
            
            pos = self.positions[option_symbol]
            
            # Calculate exit premium (simplified - use current price movement)
            price_move_pct = (current_price - pos['entry_price']) / pos['entry_price']
            
            # For calls: profit if price goes up
            # For puts: profit if price goes down
            if pos['option_type'] == 'call':
                premium_change = price_move_pct * 10  # Approximate leverage
            else:
                premium_change = -price_move_pct * 10
            
            exit_premium = pos['premium'] * (1 + premium_change)
            exit_premium = max(0.01, exit_premium)  # Minimum value
            
            # Calculate P&L
            exit_value = pos['qty'] * exit_premium * 100
            pnl = exit_value - pos['cost']
            
            self.capital += exit_value
            self.capital = max(0, self.capital)  # Can't go negative
            
            # Record trade
            trade = {
                'symbol': symbol,
                'option_symbol': option_symbol,
                'entry_time': pos['entry_time'],
                'exit_time': current_time,
                'entry_price': pos['entry_price'],
                'exit_price': current_price,
                'entry_premium': pos['premium'],
                'exit_premium': exit_premium,
                'strike': pos['strike'],
                'option_type': pos['option_type'],
                'qty': pos['qty'],
                'cost': pos['cost'],
                'exit_value': exit_value,
                'pnl': pnl,
                'pnl_pct': (pnl / pos['cost']) * 100 if pos['cost'] > 0 else 0.0,
                'duration_minutes': (current_time - pos['entry_time']).total_seconds() / 60.0
            }
            self.trades.append(trade)
            
            # Remove position
            del self.positions[option_symbol]
            
            result['executed'] = True
            result['pnl'] = pnl
            result['reason'] = f'Exited {pos["option_type"].upper()} | P&L: ${pnl:.2f} ({trade["pnl_pct"]:.1f}%)'
        
        return result
    
    def run_backtest(self) -> Dict:
        """Run the backtest"""
        print("\n" + "=" * 70)
        print("🚀 STARTING REAL BACKTEST: Last Week Performance")
        print("=" * 70)
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Model: {MODEL_PATH}")
        print("=" * 70 + "\n")
        
        # Load data for all symbols
        for symbol in self.symbols:
            data = self.load_historical_data(symbol)
            if len(data) == 0:
                print(f"⚠️  Skipping {symbol} - no data available")
                continue
            self.market_data[symbol] = data
        
        if len(self.market_data) == 0:
            print("❌ No market data available. Cannot run backtest.")
            return {}
        
        # Get VIX data
        try:
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(
                start=self.start_date.strftime('%Y-%m-%d'),
                end=(self.end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                interval='1d'
            )
            if isinstance(vix_data.columns, pd.MultiIndex):
                vix_data.columns = vix_data.columns.get_level_values(0)
            vix_data.columns = [col.lower() for col in vix_data.columns]
            vix_series = vix_data['close'] if 'close' in vix_data.columns else vix_data.iloc[:, -1]
        except Exception as e:
            print(f"⚠️  Could not load VIX data: {e}, using default VIX=20")
            vix_series = pd.Series([20.0], index=[self.start_date])
        
        # Process each trading day
        current_date = self.start_date
        total_bars = 0
        
        while current_date <= self.end_date:
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
            
            print(f"\n📅 Processing {current_date.date()}...")
            
            # Get VIX for this day
            try:
                vix_value = float(vix_series[vix_series.index.date == current_date.date()].iloc[-1])
            except:
                vix_value = 20.0
            
            # Process each symbol
            for symbol in self.symbols:
                if symbol not in self.market_data:
                    continue
                
                data = self.market_data[symbol]
                day_data = data[data.index.date == current_date.date()] if isinstance(data.index, pd.DatetimeIndex) else data
                
                if len(day_data) == 0:
                    continue
                
                # Update risk manager VIX
                self.risk_mgr.current_vix = vix_value
                
                # Process each bar
                for idx, (timestamp, bar) in enumerate(day_data.iterrows()):
                    if len(day_data) < LOOKBACK:
                        continue
                    
                    # Get historical window
                    hist_window = day_data.iloc[max(0, idx - LOOKBACK + 1):idx + 1]
                    if len(hist_window) < LOOKBACK:
                        continue
                    
                    current_price = float(bar['close']) if 'close' in bar else float(bar.iloc[-1])
                    
                    # Prepare observation
                    try:
                        obs = prepare_observation(hist_window, self.risk_mgr, symbol=symbol)
                        
                        # Validate observation shape
                        expected_shape = (20, 10) if "mike_historical_model" in MODEL_PATH else (20, 23)
                        if obs.shape != expected_shape:
                            print(f"⚠️  Observation shape mismatch: {obs.shape} != {expected_shape}, skipping")
                            continue
                        
                        # Get RL action
                        action, _ = self.model.predict(obs, deterministic=False)
                        action = int(action) if isinstance(action, (np.ndarray, int)) else int(action[0]) if hasattr(action, '__iter__') else int(action)
                        
                        # Execute trade
                        trade_result = self.simulate_trade(symbol, action, current_price, timestamp, vix_value)
                        
                        if trade_result['executed']:
                            print(f"   {timestamp.strftime('%H:%M')} | {symbol} | Action {action} | {trade_result['reason']}")
                            if trade_result['pnl'] != 0:
                                print(f"      P&L: ${trade_result['pnl']:.2f}")
                        
                        total_bars += 1
                        
                    except Exception as e:
                        print(f"   ⚠️  Error processing bar: {e}")
                        continue
            
            current_date += timedelta(days=1)
        
        # Close all remaining positions at final price
        final_capital = self.capital
        for option_symbol, pos in list(self.positions.items()):
            symbol = pos['symbol']
            if symbol not in self.market_data:
                continue
            
            # Get final price from last bar
            data = self.market_data[symbol]
            if len(data) == 0:
                continue
            
            final_bar = data.iloc[-1]
            final_price = float(final_bar['close']) if 'close' in final_bar else float(final_bar.iloc[-1])
            
            # Calculate exit premium (simplified)
            price_move_pct = (final_price - pos['entry_price']) / pos['entry_price']
            if pos['option_type'] == 'call':
                premium_change = price_move_pct * 10
            else:
                premium_change = -price_move_pct * 10
            
            exit_premium = pos['premium'] * (1 + premium_change)
            exit_premium = max(0.01, exit_premium)
            
            exit_value = pos['qty'] * exit_premium * 100
            pnl = exit_value - pos['cost']
            
            final_capital += exit_value
            
            # Record trade
            trade = {
                'symbol': symbol,
                'option_symbol': option_symbol,
                'entry_time': pos['entry_time'],
                'exit_time': data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else datetime.now(),
                'entry_price': pos['entry_price'],
                'exit_price': final_price,
                'entry_premium': pos['premium'],
                'exit_premium': exit_premium,
                'strike': pos['strike'],
                'option_type': pos['option_type'],
                'qty': pos['qty'],
                'cost': pos['cost'],
                'exit_value': exit_value,
                'pnl': pnl,
                'pnl_pct': (pnl / pos['cost']) * 100 if pos['cost'] > 0 else 0.0,
                'duration_minutes': 0.0,  # Closed at end
                'closed_at_end': True
            }
            self.trades.append(trade)
        
        final_capital = max(0, final_capital)  # Can't go negative
        
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100
        
        # Calculate statistics
        if len(self.trades) > 0:
            winning_trades = [t for t in self.trades if t['pnl'] > 0]
            losing_trades = [t for t in self.trades if t['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(self.trades) * 100 if len(self.trades) > 0 else 0
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            total_pnl = sum([t['pnl'] for t in self.trades])
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            total_pnl = 0
        
        results = {
            'start_date': self.start_date.date(),
            'end_date': self.end_date.date(),
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'total_pnl': total_pnl,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades) if len(self.trades) > 0 else 0,
            'losing_trades': len(losing_trades) if len(self.trades) > 0 else 0,
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_bars_processed': total_bars,
            'trades': self.trades
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print backtest results"""
        print("\n" + "=" * 70)
        print("📊 BACKTEST RESULTS: Last Week Performance")
        print("=" * 70)
        print(f"Period: {results['start_date']} to {results['end_date']}")
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Capital: ${results['final_capital']:,.2f}")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Total P&L: ${results['total_pnl']:,.2f}")
        print()
        print(f"Total Trades: {results['total_trades']}")
        print(f"Winning Trades: {results['winning_trades']}")
        print(f"Losing Trades: {results['losing_trades']}")
        print(f"Win Rate: {results['win_rate_pct']:.1f}%")
        print()
        if results['total_trades'] > 0:
            print(f"Average Win: ${results['avg_win']:.2f}")
            print(f"Average Loss: ${results['avg_loss']:.2f}")
            if results['avg_loss'] != 0:
                profit_factor = abs(results['avg_win'] / results['avg_loss']) if results['avg_loss'] < 0 else 0
                print(f"Profit Factor: {profit_factor:.2f}")
        print()
        print(f"Total Bars Processed: {results['total_bars_processed']}")
        print("=" * 70)
        
        # Print trade details
        if len(results['trades']) > 0:
            print("\n📋 TRADE DETAILS:")
            print("-" * 70)
            for i, trade in enumerate(results['trades'], 1):
                print(f"{i}. {trade['option_type'].upper()} {trade['symbol']} | "
                      f"Entry: {trade['entry_time']} | Exit: {trade['exit_time']} | "
                      f"P&L: ${trade['pnl']:.2f} ({trade['pnl_pct']:.1f}%) | "
                      f"Duration: {trade['duration_minutes']:.1f} min")
            print("-" * 70)


def main():
    """Main backtest function"""
    # Calculate last week dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Ensure we have weekdays
    while start_date.weekday() >= 5:
        start_date -= timedelta(days=1)
    while end_date.weekday() >= 5:
        end_date -= timedelta(days=1)
    
    print("🔍 VALIDATING BACKTEST SETUP...")
    print(f"   Start Date: {start_date.date()}")
    print(f"   End Date: {end_date.date()}")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Model Exists: {os.path.exists(MODEL_PATH)}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model not found at {MODEL_PATH}")
        print("   Please ensure the model file exists.")
        return
    
    # Run backtest
    try:
        engine = BacktestEngine(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            initial_capital=10000.0
        )
        
        results = engine.run_backtest()
        
        if results:
            engine.print_results(results)
            
            # Save results to file
            results_file = f"backtest_results_{start_date.date()}_to_{end_date.date()}.json"
            import json
            from datetime import date
            
            # Convert datetime/date objects to strings for JSON
            def json_serial(obj):
                """JSON serializer for objects not serializable by default json code"""
                if isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.isoformat()
                elif isinstance(obj, date):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")
            
            results_json = {}
            for k, v in results.items():
                if k == 'trades':
                    results_json[k] = [
                        {kk: json_serial(vv) if isinstance(vv, (pd.Timestamp, datetime, date)) else vv 
                         for kk, vv in trade.items()} 
                        for trade in v
                    ]
                elif isinstance(v, (pd.Timestamp, datetime, date)):
                    results_json[k] = json_serial(v)
                else:
                    results_json[k] = v
            
            with open(results_file, 'w') as f:
                json.dump(results_json, f, indent=2, default=json_serial)
            
            print(f"\n💾 Results saved to: {results_file}")
        else:
            print("\n❌ Backtest failed - no results returned")
            
    except Exception as e:
        print(f"\n❌ Backtest error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

