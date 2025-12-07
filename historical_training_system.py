"""
📚 HISTORICAL TRAINING SYSTEM

Comprehensive RL training on historical data (2002-present) for SPX/SPY/QQQ
Covers all market regimes (good, bad, worst days) with 0DTE options simulation

Author: Mike Agent Institutional Upgrade
Date: December 6, 2025
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# RL imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces

# Import existing modules
try:
    from greeks_calculator import GreeksCalculator
    GREEKS_AVAILABLE = True
except ImportError:
    GREEKS_AVAILABLE = False
    print("Warning: greeks_calculator not available")

try:
    from institutional_features import InstitutionalFeatureEngine
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    print("Warning: institutional_features not available")


class HistoricalDataCollector:
    """
    Collect and cache historical data for SPX, SPY, QQQ from 2002-present
    """
    
    def __init__(self, cache_dir: str = "data/historical"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.symbols = ['SPY', 'QQQ', '^SPX']
        self.symbol_map = {'SPY': 'SPY', 'QQQ': 'QQQ', 'SPX': '^SPX'}
        
    def get_historical_data(
        self,
        symbol: str,
        start_date: str = "2002-01-01",
        end_date: Optional[str] = None,
        interval: str = '1m',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get historical data with caching
        
        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), None = today
            interval: Data interval ('1m', '5m', '1h', '1d')
            use_cache: Use cached data if available
            
        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Map symbol for yfinance
        yf_symbol = self.symbol_map.get(symbol, symbol)
        
        # Cache file path
        cache_file = self.cache_dir / f"{symbol}_{interval}_{start_date}_{end_date}.pkl"
        
        # Load from cache if exists
        if use_cache and cache_file.exists():
            print(f"📂 Loading cached data: {cache_file.name}")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️ Cache load failed: {e}, downloading fresh data")
        
        print(f"📥 Downloading {symbol} data: {start_date} to {end_date} ({interval})...")
        
        try:
            ticker = yf.Ticker(yf_symbol)
            
            # For minute data, fetch in chunks (yfinance limitation)
            if interval == '1m':
                # Fetch in 7-day chunks (yfinance limit: only 8 days of 1m data per request)
                all_data = []
                current_start = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                
                while current_start < end_dt:
                    chunk_end = min(current_start + timedelta(days=7), end_dt)
                    print(f"  Fetching chunk: {current_start.date()} to {chunk_end.date()}")
                    
                    try:
                        hist = ticker.history(
                            start=current_start.strftime('%Y-%m-%d'),
                            end=(chunk_end + timedelta(days=1)).strftime('%Y-%m-%d'),
                            interval=interval
                        )
                        
                        if len(hist) > 0:
                            if isinstance(hist.columns, pd.MultiIndex):
                                hist.columns = hist.columns.get_level_values(0)
                            all_data.append(hist)
                            print(f"    ✅ Got {len(hist)} bars")
                        else:
                            print(f"    ⚠️  No data for this chunk")
                    except Exception as e:
                        print(f"    ⚠️  Error: {str(e)[:100]}")
                    
                    current_start = chunk_end + timedelta(days=1)
                    # Rate limiting
                    import time
                    time.sleep(1.0)  # Increased delay to respect API limits
                
                if all_data:
                    data = pd.concat(all_data, axis=0)
                    data = data.sort_index()
                    data = data.drop_duplicates()
                else:
                    data = pd.DataFrame()
            else:
                # For daily/hourly data, fetch all at once
                hist = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval
                )
                
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                data = hist
            
            # Normalize column names
            data.columns = [col.lower() for col in data.columns]
            
            # Ensure required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            for col in required:
                if col not in data.columns:
                    if col == 'volume':
                        data['volume'] = 0
                    else:
                        data[col] = data.get('close', 0)
            
            # Filter to trading hours only (9:30 AM - 4:00 PM ET) for minute data
            if interval == '1m' and len(data) > 0:
                data = data.between_time('09:30', '16:00')
            
            # Remove duplicates and sort
            data = data.drop_duplicates().sort_index()
            
            # Save to cache
            if use_cache and len(data) > 0:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(data, f)
                    print(f"💾 Cached {len(data)} bars to {cache_file.name}")
                except Exception as e:
                    print(f"⚠️ Cache save failed: {e}")
            
            return data
            
        except Exception as e:
            print(f"❌ Error downloading {symbol}: {e}")
            return pd.DataFrame()
    
    def get_vix_data(
        self,
        start_date: str = "2002-01-01",
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.Series:
        """Get historical VIX data"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        cache_file = self.cache_dir / f"VIX_daily_{start_date}_{end_date}.pkl"
        
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        print(f"📥 Downloading VIX data: {start_date} to {end_date}...")
        
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(start=start_date, end=end_date, interval='1d')
            
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            
            vix_series = hist['Close'] if 'Close' in hist.columns else hist.iloc[:, -1]
            vix_series.name = 'VIX'
            
            # Cache
            if use_cache:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(vix_series, f)
                except:
                    pass
            
            return vix_series
            
        except Exception as e:
            print(f"❌ Error downloading VIX: {e}")
            return pd.Series()


class OptionsSimulator:
    """
    Simulate 0DTE options pricing from historical underlying data
    Since historical options chain data is hard to obtain, we simulate it
    """
    
    def __init__(self):
        from scipy.stats import norm
        self.norm = norm
        
    def estimate_premium(
        self,
        S: float,  # Underlying price
        K: float,  # Strike price
        T: float,  # Time to expiration (years, ~1/252/6.5 for 0DTE)
        r: float = 0.04,  # Risk-free rate
        sigma: float = None,  # Implied volatility (estimated if None)
        option_type: str = 'call',
        vix: float = None  # VIX for IV estimation
    ) -> float:
        """
        Estimate 0DTE option premium using Black-Scholes
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Implied volatility (estimated from VIX if None)
            option_type: 'call' or 'put'
            vix: VIX level for IV estimation
            
        Returns:
            Estimated premium
        """
        if T <= 0:
            # Expired - intrinsic value only
            if option_type == 'call':
                return max(0.01, S - K)
            else:
                return max(0.01, K - S)
        
        # Estimate IV if not provided
        if sigma is None:
            if vix is not None:
                # VIX is annualized, for 0DTE use higher IV
                sigma = (vix / 100.0) * 1.3  # 0DTE typically 30% higher IV
            else:
                sigma = 0.20  # Default 20% IV
        
        # Black-Scholes calculation
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            premium = S * self.norm.cdf(d1) - K * np.exp(-r * T) * self.norm.cdf(d2)
        else:  # put
            premium = K * np.exp(-r * T) * self.norm.cdf(-d2) - S * self.norm.cdf(-d1)
        
        # Minimum premium for 0DTE
        return max(0.01, premium)
    
    def simulate_option_price(
        self,
        entry_premium: float,
        underlying_return: float,
        time_decay: float = 0.0,
        iv_change: float = 0.0
    ) -> float:
        """
        Simulate option price change based on underlying move
        
        Args:
            entry_premium: Entry premium
            underlying_return: Underlying price change (%)
            time_decay: Time decay factor (0.0 to 1.0)
            iv_change: IV change (%)
            
        Returns:
            New premium
        """
        # Simplified model: premium change ≈ delta * underlying_change - theta_decay
        # For 0DTE, gamma effects are large
        delta = 0.5  # Approximate delta for ATM 0DTE
        gamma_effect = 0.1 * (underlying_return ** 2)  # Gamma convexity
        
        # Time decay (theta)
        theta_decay = entry_premium * time_decay * 0.05  # ~5% per hour
        
        # IV impact (vega)
        vega_effect = entry_premium * iv_change * 0.1  # ~10% per 1% IV change
        
        # Total premium change
        price_change_pct = (delta * underlying_return + gamma_effect - theta_decay / entry_premium + vega_effect / entry_premium)
        
        new_premium = entry_premium * (1 + price_change_pct)
        
        return max(0.01, new_premium)


class HistoricalTradingEnv(gym.Env):
    """
    Enhanced RL environment for historical training with 0DTE options simulation
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        vix_data: pd.Series,
        symbol: str = 'SPY',
        window_size: int = 20,
        initial_capital: float = 100000.0,
        use_greeks: bool = True,
        use_features: bool = False
    ):
        super().__init__()
        
        self.data = data.reset_index(drop=True) if not isinstance(data.index, pd.DatetimeIndex) else data
        self.vix_data = vix_data
        self.symbol = symbol
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # Options simulator
        self.options_sim = OptionsSimulator()
        
        # Greeks calculator
        self.greeks_calc = GreeksCalculator() if GREEKS_AVAILABLE else None
        
        # Feature engine
        self.use_features = use_features
        self.use_greeks = use_greeks
        self.feature_engine = InstitutionalFeatureEngine(lookback_minutes=window_size) if FEATURES_AVAILABLE and use_features else None
        
        # Position tracking
        self.position = None  # {symbol, qty, entry_premium, entry_price, strike, option_type, entry_time}
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.trade_history = []
        
        # Observation space (enhanced)
        if use_features and self.feature_engine:
            # Use institutional features (500+ features)
            obs_shape = (window_size, 130)  # Approximate feature count
        elif use_greeks and self.greeks_calc:
            # OHLCV + VIX + Greeks + Position
            obs_shape = (window_size, 10)  # 5 OHLCV + 1 VIX + 4 Greeks
        else:
            # Basic: OHLCV
            obs_shape = (window_size, 5)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )
        
        # Action space: 0=HOLD, 1=BUY_CALL, 2=BUY_PUT, 3=TRIM_50%, 4=TRIM_70%, 5=EXIT
        self.action_space = spaces.Discrete(6)
        
        self.current_step = 0
        self.current_bar = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size  # Start after window
        self.capital = self.initial_capital
        self.position = None
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.trade_history = []
        return self._get_obs(), {}
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation"""
        if self.current_step < self.window_size:
            # Pad with first bar
            pad_data = [self.data.iloc[0]] * (self.window_size - self.current_step)
            window_data = pd.concat([pd.DataFrame(pad_data), self.data.iloc[:self.current_step]])
        else:
            window_data = self.data.iloc[self.current_step - self.window_size:self.current_step]
        
        if len(window_data) < self.window_size:
            pad_data = [window_data.iloc[-1]] * (self.window_size - len(window_data))
            window_data = pd.concat([window_data, pd.DataFrame(pad_data)])
        
        # Ensure we have OHLCV
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in window_data.columns:
                window_data[col] = window_data.get('close', 0)
        
        # Get current bar
        self.current_bar = window_data.iloc[-1]
        current_price = float(self.current_bar['close'])
        current_time = window_data.index[-1] if isinstance(window_data.index, pd.DatetimeIndex) else None
        
        # Build observation
        if hasattr(self, 'use_features') and self.use_features and self.feature_engine:
            # Use institutional features
            features, _ = self.feature_engine.extract_all_features(
                window_data,
                symbol=self.symbol,
                risk_mgr=None
            )
            obs = features[-self.window_size:].astype(np.float32)
        else:
            # Basic OHLCV
            ohlcv = window_data[required_cols].values.astype(np.float32)
            
            # Check if we need to add VIX and Greeks
            if self.use_greeks and self.greeks_calc:
                # Always return consistent shape: OHLCV + VIX + Greeks
                # Get VIX
                vix = self._get_vix(current_time)
                vix_normalized = np.full((self.window_size, 1), vix / 50.0 if vix else 0.4, dtype=np.float32)
                
                if self.position:
                    # Calculate actual Greeks when position exists
                    strike = self.position['strike']
                    option_type = self.position['option_type']
                    T = 1.0 / (252 * 6.5)  # ~1 hour for 0DTE
                    sigma = (vix / 100.0) * 1.3 if vix else 0.20
                    
                    greeks = self.greeks_calc.calculate_greeks(
                        S=current_price,
                        K=strike,
                        T=T,
                        sigma=sigma,
                        option_type=option_type
                    )
                    
                    greeks_array = np.full((self.window_size, 4), [
                        greeks['delta'],
                        greeks['gamma'],
                        greeks['theta'],
                        greeks['vega']
                    ], dtype=np.float32)
                else:
                    # No position: use zeros for Greeks
                    greeks_array = np.zeros((self.window_size, 4), dtype=np.float32)
                
                # Combine OHLCV + VIX + Greeks (always 10 features)
                obs = np.concatenate([ohlcv, vix_normalized, greeks_array], axis=1)
            else:
                # Just OHLCV (5 features)
                obs = ohlcv
        
        return obs
    
    def _get_vix(self, timestamp) -> Optional[float]:
        """Get VIX for current timestamp"""
        if len(self.vix_data) == 0:
            return 20.0  # Default
        
        if timestamp is None:
            return float(self.vix_data.iloc[-1]) if len(self.vix_data) > 0 else 20.0
        
        # Find closest VIX value
        try:
            if isinstance(self.vix_data.index, pd.DatetimeIndex):
                date_only = timestamp.date() if hasattr(timestamp, 'date') else pd.to_datetime(timestamp).date()
                vix_values = self.vix_data[self.vix_data.index.date == date_only]
                if len(vix_values) > 0:
                    return float(vix_values.iloc[-1])
            
            # Fallback to latest
            return float(self.vix_data.iloc[-1])
        except:
            return 20.0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Actions:
            0: HOLD
            1: BUY_CALL
            2: BUY_PUT
            3: TRIM_50%
            4: TRIM_70%
            5: EXIT
        """
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Get current price and bar
        if self.current_step < len(self.data):
            current_bar = self.data.iloc[self.current_step]
            current_price = float(current_bar['close'])
            current_time = current_bar.name if hasattr(current_bar, 'name') else None
        else:
            # End of data
            reward = self._calculate_final_reward()
            return self._get_obs(), reward, True, False, {}
        
        # Get VIX
        vix = self._get_vix(current_time)
        
        # Update position if exists
        if self.position:
            self._update_position(current_price, vix)
        
        # Execute action
        reward = 0.0
        info = {}
        
        if action == 0:  # HOLD
            reward = self._calculate_reward()
        
        elif action == 1:  # BUY_CALL
            if self.position is None:
                reward = self._execute_buy_call(current_price, vix)
            else:
                reward = -0.001  # Penalty for buying when already in position
        
        elif action == 2:  # BUY_PUT
            if self.position is None:
                reward = self._execute_buy_put(current_price, vix)
            else:
                reward = -0.001  # Penalty for buying when already in position
        
        elif action == 3:  # TRIM_50%
            if self.position:
                reward = self._execute_trim(0.5, current_price, vix)
            else:
                reward = -0.001  # Penalty for trim without position
        
        elif action == 4:  # TRIM_70%
            if self.position:
                reward = self._execute_trim(0.7, current_price, vix)
            else:
                reward = -0.001  # Penalty for trim without position
        
        elif action == 5:  # EXIT
            if self.position:
                reward = self._execute_exit(current_price, vix)
            else:
                reward = 0.0  # No penalty for exit without position
        
        return self._get_obs(), reward, done, False, info
    
    def _execute_buy_call(self, price: float, vix: float) -> float:
        """Execute buy call order"""
        strike = round(price)  # ATM strike
        T = 1.0 / (252 * 6.5)  # ~1 hour for 0DTE
        
        # Estimate premium
        premium = self.options_sim.estimate_premium(
            S=price,
            K=strike,
            T=T,
            vix=vix,
            option_type='call'
        )
        
        # Calculate position size (7% risk)
        risk_dollar = self.capital * 0.07
        qty = max(1, int(risk_dollar / (premium * 100)))
        
        # Cost
        cost = qty * premium * 100
        
        if cost > self.capital:
            return -0.01  # Penalty for insufficient capital
        
        # Open position
        self.position = {
            'symbol': f"{self.symbol}CALL{strike:.0f}",
            'qty': qty,
            'entry_premium': premium,
            'entry_price': price,
            'strike': strike,
            'option_type': 'call',
            'entry_time': self.current_step,
            'cost': cost
        }
        
        self.capital -= cost
        return 0.0  # No immediate reward
    
    def _execute_buy_put(self, price: float, vix: float) -> float:
        """Execute buy put order"""
        strike = round(price)  # ATM strike
        T = 1.0 / (252 * 6.5)  # ~1 hour for 0DTE
        
        # Estimate premium
        premium = self.options_sim.estimate_premium(
            S=price,
            K=strike,
            T=T,
            vix=vix,
            option_type='put'
        )
        
        # Calculate position size (7% risk)
        risk_dollar = self.capital * 0.07
        qty = max(1, int(risk_dollar / (premium * 100)))
        
        # Cost
        cost = qty * premium * 100
        
        if cost > self.capital:
            return -0.01  # Penalty for insufficient capital
        
        # Open position
        self.position = {
            'symbol': f"{self.symbol}PUT{strike:.0f}",
            'qty': qty,
            'entry_premium': premium,
            'entry_price': price,
            'strike': strike,
            'option_type': 'put',
            'entry_time': self.current_step,
            'cost': cost
        }
        
        self.capital -= cost
        return 0.0  # No immediate reward
    
    def _execute_trim(self, trim_pct: float, price: float, vix: float) -> float:
        """Trim position (partial exit)"""
        if not self.position:
            return 0.0
        
        # Calculate new premium
        underlying_return = (price - self.position['entry_price']) / self.position['entry_price']
        time_elapsed = (self.current_step - self.position['entry_time']) / (252 * 6.5)  # Hours
        time_decay = min(1.0, time_elapsed / 6.5)  # Full decay over 6.5 hours
        
        current_premium = self.options_sim.simulate_option_price(
            entry_premium=self.position['entry_premium'],
            underlying_return=underlying_return,
            time_decay=time_decay
        )
        
        # Calculate P&L
        pnl_per_contract = (current_premium - self.position['entry_premium']) * 100
        pnl_pct = pnl_per_contract / (self.position['entry_premium'] * 100)
        
        # Trim quantity
        trim_qty = max(1, int(self.position['qty'] * trim_pct))
        proceeds = trim_qty * current_premium * 100
        
        # Update position
        self.position['qty'] -= trim_qty
        self.capital += proceeds
        
        # Calculate reward
        reward = pnl_pct * trim_pct  # Reward based on P&L percentage
        
        if self.position['qty'] <= 0:
            self.position = None
        
        return reward
    
    def _execute_exit(self, price: float, vix: float) -> float:
        """Exit entire position"""
        if not self.position:
            return 0.0
        
        # Calculate final premium
        underlying_return = (price - self.position['entry_price']) / self.position['entry_price']
        time_elapsed = (self.current_step - self.position['entry_time']) / (252 * 6.5)
        time_decay = min(1.0, time_elapsed / 6.5)
        
        current_premium = self.options_sim.simulate_option_price(
            entry_premium=self.position['entry_premium'],
            underlying_return=underlying_return,
            time_decay=time_decay
        )
        
        # Calculate P&L
        proceeds = self.position['qty'] * current_premium * 100
        cost = self.position['cost']
        pnl = proceeds - cost
        pnl_pct = pnl / cost
        
        # Update capital
        self.capital += proceeds
        
        # Record trade
        self.trade_history.append({
            'entry_premium': self.position['entry_premium'],
            'exit_premium': current_premium,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'duration': self.current_step - self.position['entry_time']
        })
        
        self.realized_pnl += pnl
        
        # Clear position
        self.position = None
        
        # Reward based on P&L
        return pnl_pct * 0.1  # Scale reward
    
    def _update_position(self, price: float, vix: float):
        """Update unrealized P&L for current position"""
        if not self.position:
            return
        
        underlying_return = (price - self.position['entry_price']) / self.position['entry_price']
        time_elapsed = (self.current_step - self.position['entry_time']) / (252 * 6.5)
        time_decay = min(1.0, time_elapsed / 6.5)
        
        current_premium = self.options_sim.simulate_option_price(
            entry_premium=self.position['entry_premium'],
            underlying_return=underlying_return,
            time_decay=time_decay
        )
        
        # Update unrealized P&L
        current_value = self.position['qty'] * current_premium * 100
        cost = self.position['cost']
        self.unrealized_pnl = current_value - cost
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current state"""
        if self.position:
            # Reward based on unrealized P&L
            return (self.unrealized_pnl / self.position['cost']) * 0.01
        else:
            # Small negative reward for holding cash (encourage trading)
            return -0.0001
    
    def _calculate_final_reward(self) -> float:
        """Calculate final reward at episode end"""
        # Close any open position
        if self.position:
            # Estimate final value (likely worthless for 0DTE)
            final_value = self.position['qty'] * 0.01 * 100  # Minimum value
            pnl = final_value - self.position['cost']
            self.realized_pnl += pnl
            self.capital += final_value
        
        # Total return
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        return total_return * 0.1  # Scale reward


def create_regime_aware_training_data(
    collector: HistoricalDataCollector,
    symbols: List[str] = ['SPY', 'QQQ'],
    start_date: str = "2002-01-01",
    end_date: Optional[str] = None,
    min_daily_bars: int = 300  # Minimum bars per day (9:30 AM - 4:00 PM = ~390 bars)
) -> Dict[str, pd.DataFrame]:
    """
    Create training data ensuring coverage of all market regimes
    
    Returns dictionary with dataframes for each symbol
    """
    print("=" * 70)
    print("CREATING REGIME-AWARE TRAINING DATASET")
    print("=" * 70)
    print(f"Symbols: {symbols}")
    print(f"Date Range: {start_date} to {end_date or 'today'}")
    print()
    
    # Get VIX data for regime classification
    vix_data = collector.get_vix_data(start_date, end_date)
    
    all_data = {}
    
    for symbol in symbols:
        print(f"\n📊 Processing {symbol}...")
        
        # Get historical data
        data = collector.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval='1m'
        )
        
        if len(data) == 0:
            print(f"⚠️ No data for {symbol}")
            continue
        
        # Filter to trading days only (9:30 AM - 4:00 PM)
        data = data.between_time('09:30', '16:00')
        
        # Group by date and filter days with enough data
        if isinstance(data.index, pd.DatetimeIndex):
            data['date'] = data.index.date
            daily_counts = data.groupby('date').size()
            valid_dates = daily_counts[daily_counts >= min_daily_bars].index
            
            data = data[data['date'].isin(valid_dates)]
            data = data.drop('date', axis=1)
        
        print(f"✅ {symbol}: {len(data)} bars across {len(valid_dates) if 'valid_dates' in locals() else 'N/A'} trading days")
        
        all_data[symbol] = data
    
    print("\n" + "=" * 70)
    print("TRAINING DATA PREPARATION COMPLETE")
    print("=" * 70)
    
    return all_data, vix_data


# This is Part 1 - Data Collection and Environment
# Part 2 will be the actual training pipeline

if __name__ == "__main__":
    print("Historical Training System - Data Collection Module")
    print("Use this module to collect and prepare historical data")

