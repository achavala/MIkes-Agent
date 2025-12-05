#!/usr/bin/env python3
"""
Mike Agent v3 - Comprehensive Backtest Engine
Uses actual agent logic with RL model, 5-tier TP system, and all safeguards
"""
import os
import sys
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings
import argparse
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings("ignore")

# Set environment variables BEFORE importing torch/gym
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['OMP_NUM_THREADS'] = '1'

# Import agent components
try:
    from mike_agent_live_safe import (
        VOL_REGIMES, FIXED_STOP_LOSS, MAX_CONCURRENT, TRADING_SYMBOLS,
        estimate_premium, find_atm_strike, get_option_symbol, prepare_observation,
        get_iv_adjusted_risk
    )
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import agent components: {e}")
    print("Using simplified backtest mode...")
    RL_AVAILABLE = False
    
    # Fallback definitions
    VOL_REGIMES = {
        "calm": {"risk": 0.10, "max_pct": 0.30, "sl": -0.15, "tp1": 0.40, "tp2": 0.60, "tp3": 1.00, "tp4": 1.50, "tp5": 2.00, "trail": 1.00},
        "normal": {"risk": 0.07, "max_pct": 0.25, "sl": -0.15, "tp1": 0.40, "tp2": 0.60, "tp3": 1.00, "tp4": 1.50, "tp5": 2.00, "trail": 1.00},
        "storm": {"risk": 0.05, "max_pct": 0.20, "sl": -0.15, "tp1": 0.40, "tp2": 0.60, "tp3": 1.00, "tp4": 1.50, "tp5": 2.00, "trail": 1.00},
        "crash": {"risk": 0.03, "max_pct": 0.15, "sl": -0.15, "tp1": 0.40, "tp2": 0.60, "tp3": 1.00, "tp4": 1.50, "tp5": 2.00, "trail": 1.00}
    }
    FIXED_STOP_LOSS = -0.15
    MAX_CONCURRENT = 10
    TRADING_SYMBOLS = ['SPY', 'QQQ', 'SPX']
    
    def estimate_premium(price, strike, option_type):
        from scipy.stats import norm
        T, r, sigma = 1/365, 0.04, 0.20
        if T <= 0:
            return max(0.01, abs(price - strike) * 0.01)
        d1 = (np.log(price / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return max(0.01, price * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2))
        else:
            return max(0.01, strike * np.exp(-r * T) * norm.cdf(-d2) - price * norm.cdf(-d1))
    
    def find_atm_strike(price):
        return round(price)
    
    def get_option_symbol(underlying, strike, option_type):
        expiry = datetime.now().strftime("%y%m%d")
        direction = "C" if option_type == 'call' else "P"
        return f"{underlying}{expiry}{direction}{int(strike*1000):08d}"
    
    def prepare_observation(hist, risk_mgr):
        """Simplified observation preparation"""
        if len(hist) < 20:
            return np.zeros((20, 5))
        recent = hist.tail(20)
        obs = np.array([
            recent['Open'].values,
            recent['High'].values,
            recent['Low'].values,
            recent['Close'].values,
            recent['Volume'].values
        ]).T
        return obs.astype(np.float32)
    
    def get_iv_adjusted_risk(iv: float) -> float:
        if iv < 20:
            return 0.10
        elif iv < 50:
            return 0.07
        else:
            return 0.04

# ==================== REGIME CLASSIFICATION ====================
def get_regime(vix: float) -> str:
    """Determine volatility regime based on VIX"""
    if vix < 18:
        return "calm"
    elif vix < 25:
        return "normal"
    elif vix < 35:
        return "storm"
    else:
        return "crash"

# ==================== POSITION TRACKING ====================
class Position:
    """Track a single option position"""
    def __init__(self, symbol: str, entry_price: float, entry_premium: float, 
                 strike: float, option_type: str, qty: int, regime: str, timestamp: datetime):
        self.symbol = symbol
        self.entry_price = entry_price
        self.entry_premium = entry_premium
        self.strike = strike
        self.option_type = option_type
        self.qty_remaining = qty
        self.original_qty = qty
        self.regime = regime
        self.entry_time = timestamp
        self.tp1_done = False
        self.tp2_done = False
        self.tp3_done = False
        self.tp4_done = False
        self.tp5_done = False
        self.trail_active = False
        self.trail_price = 0.0
        self.peak_premium = entry_premium
    
    def update_premium(self, current_premium: float):
        """Update current premium and trailing stop"""
        self.peak_premium = max(self.peak_premium, current_premium)
        if self.trail_active:
            self.trail_price = max(self.trail_price, self.entry_premium * (1 + VOL_REGIMES[self.regime]['trail']))
    
    def get_pnl_pct(self, current_premium: float) -> float:
        """Calculate P&L percentage"""
        return (current_premium - self.entry_premium) / self.entry_premium
    
    def get_pnl_dollar(self, current_premium: float) -> float:
        """Calculate P&L in dollars"""
        return (current_premium - self.entry_premium) * self.qty_remaining * 100

# ==================== BACKTEST ENGINE ====================
class BacktestEngine:
    """Comprehensive backtest engine using actual agent logic"""
    
    def __init__(self, start_date: str, end_date: str, initial_capital: float = 10000.0):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.peak_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Dict] = []
        self.equity_curve: List[float] = [initial_capital]
        self.dates: List[datetime] = []
        self.daily_pnl = 0.0
        self.start_of_day_capital = initial_capital
        
        # Load RL model if available
        self.model = None
        if RL_AVAILABLE:
            try:
                self.model = PPO.load("mike_rl_agent.zip")
                print("✓ Loaded RL model")
            except:
                print("⚠️  RL model not found, using random actions")
                self.model = None
        
        # Statistics
        self.regime_stats = {
            "calm": {"trades": 0, "wins": 0, "pnl": 0.0, "days": 0},
            "normal": {"trades": 0, "wins": 0, "pnl": 0.0, "days": 0},
            "storm": {"trades": 0, "wins": 0, "pnl": 0.0, "days": 0},
            "crash": {"trades": 0, "wins": 0, "pnl": 0.0, "days": 0}
        }
    
    def get_current_vix(self, df: pd.DataFrame, current_idx: int) -> float:
        """Get VIX value for current bar"""
        if 'VIX' in df.columns:
            return float(df.iloc[current_idx]['VIX'])
        return 20.0  # Default
    
    def simulate_option_price(self, underlying_price: float, strike: float, 
                             option_type: str, days_to_expiry: int = 1) -> float:
        """Simulate option premium using Black-Scholes"""
        return estimate_premium(underlying_price, strike, option_type)
    
    def check_stop_losses_and_tps(self, df: pd.DataFrame, current_idx: int, symbol: str):
        """Check stop-loss and take-profit levels for a position"""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        current_price = df.iloc[current_idx]['Close']
        current_premium = self.simulate_option_price(current_price, pos.strike, pos.option_type)
        pos.update_premium(current_premium)
        
        pnl_pct = pos.get_pnl_pct(current_premium)
        regime_params = VOL_REGIMES[pos.regime]
        
        # ========== FIXED STOP-LOSS (-15%) - CHECKED FIRST ==========
        if pnl_pct <= FIXED_STOP_LOSS:
            pnl_dollar = pos.get_pnl_dollar(current_premium)
            self.close_position(symbol, current_premium, 'STOP_LOSS', pnl_dollar, pnl_pct)
            return
        
        # ========== 5-TIER TAKE-PROFIT SYSTEM ==========
        # TP5: +200% - Full Exit
        if pnl_pct >= regime_params['tp5'] and not pos.tp5_done:
            pnl_dollar = pos.get_pnl_dollar(current_premium)
            self.close_position(symbol, current_premium, 'TP5', pnl_dollar, pnl_pct)
            return
        
        # TP4: +150% - Sell 10% of remaining
        if pnl_pct >= regime_params['tp4'] and not pos.tp4_done:
            sell_qty = max(1, int(pos.qty_remaining * 0.10))
            if sell_qty < pos.qty_remaining:
                pnl_dollar = (current_premium - pos.entry_premium) * sell_qty * 100
                self.partial_sell(symbol, sell_qty, current_premium, 'TP4', pnl_dollar, pnl_pct)
                pos.tp4_done = True
                pos.trail_active = True
                pos.trail_price = pos.entry_premium * (1 + regime_params['trail'])
            else:
                pnl_dollar = pos.get_pnl_dollar(current_premium)
                self.close_position(symbol, current_premium, 'TP4_FULL', pnl_dollar, pnl_pct)
            return
        
        # TP3: +100% - Sell 10% of remaining
        if pnl_pct >= regime_params['tp3'] and not pos.tp3_done:
            sell_qty = max(1, int(pos.qty_remaining * 0.10))
            if sell_qty < pos.qty_remaining:
                pnl_dollar = (current_premium - pos.entry_premium) * sell_qty * 100
                self.partial_sell(symbol, sell_qty, current_premium, 'TP3', pnl_dollar, pnl_pct)
                pos.tp3_done = True
            else:
                pnl_dollar = pos.get_pnl_dollar(current_premium)
                self.close_position(symbol, current_premium, 'TP3_FULL', pnl_dollar, pnl_pct)
            return
        
        # TP2: +60% - Sell 20% of remaining
        if pnl_pct >= regime_params['tp2'] and not pos.tp2_done:
            sell_qty = max(1, int(pos.qty_remaining * 0.20))
            if sell_qty < pos.qty_remaining:
                pnl_dollar = (current_premium - pos.entry_premium) * sell_qty * 100
                self.partial_sell(symbol, sell_qty, current_premium, 'TP2', pnl_dollar, pnl_pct)
                pos.tp2_done = True
            else:
                pnl_dollar = pos.get_pnl_dollar(current_premium)
                self.close_position(symbol, current_premium, 'TP2_FULL', pnl_dollar, pnl_pct)
            return
        
        # TP1: +40% - Sell 50% of position
        if pnl_pct >= regime_params['tp1'] and not pos.tp1_done:
            sell_qty = max(1, int(pos.qty_remaining * 0.50))
            if sell_qty < pos.qty_remaining:
                pnl_dollar = (current_premium - pos.entry_premium) * sell_qty * 100
                self.partial_sell(symbol, sell_qty, current_premium, 'TP1', pnl_dollar, pnl_pct)
                pos.tp1_done = True
            else:
                # Only 1 contract, skip TP1
                pos.tp1_done = True
            return
        
        # Trailing Stop (after TP4)
        if pos.trail_active and current_premium <= pos.trail_price:
            pnl_dollar = pos.get_pnl_dollar(current_premium)
            self.close_position(symbol, current_premium, 'TRAILING_STOP', pnl_dollar, pnl_pct)
            return
    
    def partial_sell(self, symbol: str, qty: int, exit_premium: float, reason: str, 
                    pnl_dollar: float, pnl_pct: float):
        """Execute partial sell"""
        pos = self.positions[symbol]
        pos.qty_remaining -= qty
        
        # Apply fees (0.1% per contract)
        fees = exit_premium * qty * 100 * 0.001
        net_pnl = pnl_dollar - fees
        
        self.capital += net_pnl
        self.daily_pnl += net_pnl
        self.peak_capital = max(self.peak_capital, self.capital)
        
        self.closed_trades.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'SELL',
            'qty': qty,
            'entry_premium': pos.entry_premium,
            'exit_premium': exit_premium,
            'pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'regime': pos.regime
        })
        
        self.regime_stats[pos.regime]['trades'] += 1
        if net_pnl > 0:
            self.regime_stats[pos.regime]['wins'] += 1
        self.regime_stats[pos.regime]['pnl'] += net_pnl
    
    def close_position(self, symbol: str, exit_premium: float, reason: str, 
                      pnl_dollar: float, pnl_pct: float):
        """Close entire position"""
        pos = self.positions[symbol]
        
        # Apply fees (0.1% per contract)
        fees = exit_premium * pos.qty_remaining * 100 * 0.001
        net_pnl = pnl_dollar - fees
        
        self.capital += net_pnl
        self.daily_pnl += net_pnl
        self.peak_capital = max(self.peak_capital, self.capital)
        
        self.closed_trades.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'SELL',
            'qty': pos.qty_remaining,
            'entry_premium': pos.entry_premium,
            'exit_premium': exit_premium,
            'pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'regime': pos.regime
        })
        
        self.regime_stats[pos.regime]['trades'] += 1
        if net_pnl > 0:
            self.regime_stats[pos.regime]['wins'] += 1
        self.regime_stats[pos.regime]['pnl'] += net_pnl
        
        del self.positions[symbol]
    
    def enter_position(self, symbol: str, underlying_price: float, strike: float, 
                      option_type: str, qty: int, regime: str, timestamp: datetime):
        """Enter a new position"""
        entry_premium = self.simulate_option_price(underlying_price, strike, option_type)
        
        # Calculate cost
        cost = entry_premium * qty * 100
        fees = cost * 0.001  # 0.1% fees
        
        if cost + fees > self.capital:
            return False  # Insufficient capital
        
        self.capital -= (cost + fees)
        
        pos = Position(symbol, underlying_price, entry_premium, strike, option_type, qty, regime, timestamp)
        self.positions[symbol] = pos
        
        self.closed_trades.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'BUY',
            'qty': qty,
            'entry_premium': entry_premium,
            'exit_premium': 0.0,
            'pnl': 0.0,
            'pnl_pct': 0.0,
            'reason': 'ENTRY',
            'regime': regime
        })
        
        return True
    
    def get_rl_action(self, hist: pd.DataFrame) -> int:
        """Get action from RL model or random"""
        if self.model is None:
            # Random action for testing
            return np.random.choice([0, 1, 2], p=[0.7, 0.15, 0.15])
        
        try:
            # Prepare observation
            obs = prepare_observation(hist, None)
            obs_batch = obs.reshape(1, 20, 5)
            action, _ = self.model.predict(obs_batch, deterministic=True)
            return int(action[0])
        except:
            return 0  # HOLD if error
    
    def run(self, symbol: str = 'SPY', interval: str = '5m'):
        """Run backtest"""
        print("=" * 80)
        print(f"MIKE AGENT v3 - COMPREHENSIVE BACKTEST")
        print("=" * 80)
        print(f"Symbol: {symbol}")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Interval: {interval}")
        print("=" * 80)
        
        # Validate dates - first check what data is actually available
        try:
            # Get latest available data date from Yahoo Finance
            test_ticker = yf.Ticker(symbol)
            test_hist = test_ticker.history(period='5d', interval='1d')
            if len(test_hist) > 0:
                latest_available_date = test_hist.index[-1].date()
            else:
                latest_available_date = datetime.now().date()
        except:
            latest_available_date = datetime.now().date()
        
        now = datetime.now()
        today = latest_available_date  # Use latest available date, not system date
        
        # Check if dates are in the future (relative to available data)
        if self.start_date.date() > today:
            print(f"⚠️  WARNING: Start date ({self.start_date.date()}) is after latest available data ({today})!")
            print(f"   Using last 60 days from available data instead...")
            self.start_date = datetime.combine(today, datetime.min.time()) - timedelta(days=60)
            self.end_date = datetime.combine(today, datetime.min.time())
        
        if self.end_date.date() > today:
            print(f"⚠️  WARNING: End date ({self.end_date.date()}) is after latest available data ({today})!")
            print(f"   Using latest available date ({today}) as end date...")
            self.end_date = datetime.combine(today, datetime.min.time())
        
        # Ensure start < end
        if self.start_date >= self.end_date:
            print(f"❌ ERROR: Start date must be before end date")
            return None
        
        print("Downloading data...")
        
        # Check if date range is too old for intraday data
        # Yahoo Finance only provides intraday data (1m, 5m, 15m, etc.) for last 60 days
        now = datetime.now()
        days_ago = (now - self.start_date).days
        is_intraday = interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']
        
        # For intraday data, ensure we're within last 60 days from TODAY
        if is_intraday:
            max_start_date = now - timedelta(days=60)
            if self.start_date < max_start_date or self.start_date > now:
                print(f"⚠️  WARNING: Intraday data ({interval}) only available for last 60 days")
                if self.start_date > now:
                    print(f"   Start date ({self.start_date.date()}) is in the future!")
                else:
                    print(f"   Start date ({self.start_date.date()}) is {days_ago} days ago (beyond 60-day limit)")
                print(f"   Adjusting to last 60 days: {max_start_date.date()} to {now.date()}")
                self.start_date = max_start_date
                self.end_date = now
                print(f"   New period: {self.start_date.date()} to {self.end_date.date()}")
        
        # Ensure start < end after adjustments
        if self.start_date >= self.end_date:
            print(f"❌ ERROR: Start date must be before end date")
            print(f"   Start: {self.start_date.date()}, End: {self.end_date.date()}")
            return None
        
        # Download data
        try:
            ticker = yf.Ticker(symbol)
            
            # Use the adjusted dates (already validated above)
            # For intraday, dates have been adjusted to last 60 days
            # For daily, use original dates
            hist = ticker.history(start=self.start_date, end=self.end_date, interval=interval)
            
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            
            if len(hist) == 0:
                print(f"❌ No data available for {symbol}")
                print(f"   Try using daily (1d) interval for older dates")
                print(f"   Example: --interval 1d")
                return None
            
            # Download VIX (always use daily for VIX)
            vix_ticker = yf.Ticker("^VIX")
            vix_hist = vix_ticker.history(start=self.start_date, end=self.end_date, interval='1d')
            if isinstance(vix_hist.columns, pd.MultiIndex):
                vix_hist.columns = vix_hist.columns.get_level_values(0)
            
            # Merge VIX (forward fill)
            hist['VIX'] = vix_hist['Close'].reindex(hist.index, method='ffill')
            hist = hist.dropna()
            
            if len(hist) == 0:
                print(f"❌ No data available after merging VIX")
                return None
            
            print(f"✓ Downloaded {len(hist)} bars")
            print(f"  Date range: {hist.index[0]} to {hist.index[-1]}")
            print(f"  Actual interval: {interval}")
            print("=" * 80)
            
        except Exception as e:
            error_msg = str(e)
            if "not available" in error_msg.lower() or "60 days" in error_msg.lower():
                print(f"❌ Error: {error_msg}")
                print(f"\n💡 SOLUTION: Yahoo Finance only provides intraday data for the last 60 days")
                print(f"   For older dates, use daily (1d) interval:")
                print(f"   python3 backtest_mike_agent_v3.py --start {self.start_date.date()} --end {self.end_date.date()} --interval 1d")
            else:
                print(f"❌ Error downloading data: {e}")
            return None
        
        # Run simulation
        print("Running backtest...")
        last_date = None
        
        # Adjust LOOKBACK based on data frequency
        # For daily data, we need fewer bars for lookback
        if interval == '1d':
            min_lookback = 5  # 5 days minimum
        else:
            min_lookback = 20  # 20 bars for intraday
        
        start_idx = max(min_lookback, 20)  # Ensure we have enough data
        
        for idx in range(start_idx, len(hist)):  # Start after LOOKBACK period
            current_bar = hist.iloc[idx]
            current_date = current_bar.name.date()
            current_time = current_bar.name
            
            # Reset daily P&L at start of new day
            if last_date is not None and current_date != last_date:
                self.start_of_day_capital = self.capital
                self.daily_pnl = 0.0
            
            # Check daily loss limit
            if self.daily_pnl / self.start_of_day_capital <= -0.15:
                print(f"⚠️  Daily loss limit hit on {current_date}")
                # Close all positions
                for sym in list(self.positions.keys()):
                    current_price = hist.iloc[idx]['Close']
                    pos = self.positions[sym]
                    exit_premium = self.simulate_option_price(current_price, pos.strike, pos.option_type)
                    pnl_dollar = pos.get_pnl_dollar(exit_premium)
                    pnl_pct = pos.get_pnl_pct(exit_premium)
                    self.close_position(sym, exit_premium, 'DAILY_LOSS_LIMIT', pnl_dollar, pnl_pct)
                continue
            
            # Check existing positions
            for sym in list(self.positions.keys()):
                self.check_stop_losses_and_tps(hist, idx, sym)
            
            # Market hours check (9:30 AM - 4:00 PM EST, before 2:30 PM for new entries)
            hour = current_time.hour if hasattr(current_time, 'hour') else current_time.hour
            minute = current_time.minute if hasattr(current_time, 'minute') else 0
            
            can_trade = (9 <= hour < 16) and not (hour == 14 and minute >= 30)
            
            # New entry logic
            if can_trade and len(self.positions) < MAX_CONCURRENT:
                current_vix = self.get_current_vix(hist, idx)
                current_regime = get_regime(current_vix)
                self.regime_stats[current_regime]['days'] += 1
                
                # Get RL action
                action = self.get_rl_action(hist.iloc[:idx+1])
                
                if action in [1, 2]:  # BUY CALL or BUY PUT
                    current_price = current_bar['Close']
                    strike = find_atm_strike(current_price)
                    option_type = 'call' if action == 1 else 'put'
                    option_symbol = get_option_symbol(symbol, strike, option_type)
                    
                    # Skip if already have position in this symbol
                    if any(s.startswith(symbol) for s in self.positions.keys()):
                        continue
                    
                    # Calculate position size
                    regime_params = VOL_REGIMES[current_regime]
                    risk_pct = regime_params['risk']
                    risk_dollar = self.capital * risk_pct
                    estimated_premium = estimate_premium(current_price, strike, option_type)
                    qty = max(1, int(risk_dollar / (estimated_premium * 100)))
                    
                    # Max position size check
                    max_notional = self.capital * regime_params['max_pct']
                    max_contracts = int(max_notional / (estimated_premium * 100))
                    qty = min(qty, max_contracts)
                    
                    if qty >= 1:
                        if self.enter_position(option_symbol, current_price, strike, option_type, 
                                             qty, current_regime, current_time):
                            pass  # Position entered
            
            # Update equity curve
            # Calculate current portfolio value
            portfolio_value = self.capital
            for sym, pos in self.positions.items():
                current_price = hist.iloc[idx]['Close']
                current_premium = self.simulate_option_price(current_price, pos.strike, pos.option_type)
                portfolio_value += current_premium * pos.qty_remaining * 100
            
            self.equity_curve.append(portfolio_value)
            self.dates.append(current_time)
            
            last_date = current_date
            
            # Progress update
            if idx % 100 == 0:
                progress = (idx / len(hist)) * 100
                print(f"  Progress: {progress:.1f}% | Capital: ${self.capital:,.2f} | Positions: {len(self.positions)}")
        
        # Close all remaining positions at end
        final_price = hist.iloc[-1]['Close']
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            exit_premium = self.simulate_option_price(final_price, pos.strike, pos.option_type)
            pnl_dollar = pos.get_pnl_dollar(exit_premium)
            pnl_pct = pos.get_pnl_pct(exit_premium)
            self.close_position(sym, exit_premium, 'END_OF_BACKTEST', pnl_dollar, pnl_pct)
        
        return self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive backtest report"""
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        
        # Overall statistics
        total_return = ((self.capital / self.initial_capital) - 1) * 100
        max_drawdown = ((self.peak_capital - min(self.equity_curve)) / self.peak_capital) * 100
        
        # Calculate returns
        returns = []
        for i in range(1, len(self.equity_curve)):
            ret = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
            returns.append(ret)
        
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-6)) * np.sqrt(252) if returns else 0.0
        
        # Win rate
        total_trades = len([t for t in self.closed_trades if t['action'] == 'SELL'])
        winning_trades = len([t for t in self.closed_trades if t['action'] == 'SELL' and t['pnl'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Profit factor
        wins = [t['pnl'] for t in self.closed_trades if t['action'] == 'SELL' and t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in self.closed_trades if t['action'] == 'SELL' and t['pnl'] < 0]
        profit_factor = (sum(wins) / sum(losses)) if losses else np.inf
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Initial Capital: ${self.initial_capital:,.2f}")
        print(f"  Final Capital: ${self.capital:,.2f}")
        print(f"  Total Return: {total_return:+.2f}%")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Win Rate: {win_rate:.1f}% ({winning_trades}/{total_trades})")
        print(f"  Profit Factor: {profit_factor:.2f}")
        
        print(f"\nREGIME PERFORMANCE:")
        print(f"{'Regime':<10} {'Trades':<8} {'Win Rate':<10} {'Total P&L':<15}")
        print("-" * 50)
        for regime in ['calm', 'normal', 'storm', 'crash']:
            stats = self.regime_stats[regime]
            trades = stats['trades']
            wins = stats['wins']
            win_rate_regime = (wins / trades * 100) if trades > 0 else 0
            pnl = stats['pnl']
            print(f"{regime.upper():<10} {trades:<8} {win_rate_regime:.1f}%{'':<6} ${pnl:+,.2f}")
        
        print(f"\nEXIT REASONS:")
        exit_reasons = {}
        for trade in self.closed_trades:
            if trade['action'] == 'SELL':
                reason = trade['reason']
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count}")
        
        print("=" * 80)
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'regime_stats': self.regime_stats,
            'equity_curve': self.equity_curve,
            'dates': self.dates,
            'closed_trades': self.closed_trades
        }

# ==================== MAIN ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mike Agent v3 Backtest")
    parser.add_argument('--start', type=str, default='2024-11-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-04', help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbol', type=str, default='SPY', help='Symbol to backtest (SPY, QQQ, SPX)')
    parser.add_argument('--interval', type=str, default='5m', help='Data interval (1m, 5m, 15m, 1h, 1d)')
    parser.add_argument('--capital', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--export', type=str, default=None, help='Export results to CSV')
    
    args = parser.parse_args()
    
    engine = BacktestEngine(args.start, args.end, args.capital)
    results = engine.run(args.symbol, args.interval)
    
    if results and args.export:
        # Export equity curve
        equity_df = pd.DataFrame({
            'date': results['dates'],
            'equity': results['equity_curve']
        })
        equity_df.to_csv(args.export, index=False)
        print(f"\n✓ Equity curve exported to {args.export}")

