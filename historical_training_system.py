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
        self.symbols = ['SPY', 'QQQ', '^GSPC']
        # yfinance uses ^GSPC for S&P 500 index (more reliable than ^SPX)
        self.symbol_map = {'SPY': 'SPY', 'QQQ': 'QQQ', 'SPX': '^GSPC'}
        # Polygon/Massive ticker mapping (for true intraday minute bars)
        # Indices on Polygon use the "I:" prefix (e.g., I:SPX).
        self.massive_symbol_map = {'SPY': 'SPY', 'QQQ': 'QQQ', 'IWM': 'IWM', 'SPX': 'I:SPX'}

    def get_historical_data_alpaca(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1m",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data from Alpaca API (PRIORITY 1 for paid members).
        
        Requires ALPACA_KEY and ALPACA_SECRET env vars.
        Caches results to disk.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Cache file path
        safe_symbol = symbol.replace(":", "_")
        cache_file = self.cache_dir / f"{safe_symbol}_{interval}_{start_date}_{end_date}_alpaca.pkl"
        
        if use_cache and cache_file.exists():
            print(f"📂 Loading cached Alpaca data: {cache_file.name}")
            try:
                with open(cache_file, "rb") as f:
                    df = pickle.load(f)
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    print(f"✅ Loaded {len(df)} bars from Alpaca cache")
                    return df
            except Exception as e:
                print(f"⚠️ Alpaca cache load failed: {e}, downloading fresh data")
        
        # Check for Alpaca credentials
        api_key = os.getenv('ALPACA_KEY') or os.getenv('ALPACA_API_KEY') or os.getenv('APCA_API_KEY_ID')
        api_secret = os.getenv('ALPACA_SECRET') or os.getenv('ALPACA_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY')
        
        if not api_key or not api_secret:
            print(f"⚠️ Alpaca credentials not found. Skipping Alpaca data source.")
            return pd.DataFrame()
        
        try:
            import alpaca_trade_api as tradeapi
            from alpaca_trade_api.rest import TimeFrame
            
            # Use paper trading URL (or live if configured)
            base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
            api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
            
            # Convert interval to Alpaca TimeFrame
            if interval == '1m':
                tf = TimeFrame.Minute
            elif interval == '5m':
                tf = TimeFrame(5, TimeFrame.Minute)
            elif interval == '1h':
                tf = TimeFrame.Hour
            elif interval == '1d':
                tf = TimeFrame.Day
            else:
                tf = TimeFrame.Minute
            
            print(f"📥 Alpaca download: {symbol} {start_date}→{end_date} ({interval})")
            
            # Get bars from Alpaca
            bars = api.get_bars(
                symbol,
                tf,
                start=start_date,
                end=end_date
            ).df
            
            if len(bars) == 0:
                print(f"⚠️ Alpaca returned 0 bars for {symbol}")
                return pd.DataFrame()
            
            # Normalize column names
            bars.columns = [col.lower() for col in bars.columns]
            
            # Ensure required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            for col in required:
                if col not in bars.columns:
                    if col == 'volume':
                        bars['volume'] = 0
                    else:
                        bars[col] = bars.get('close', 0)
            
            # Filter to trading hours for minute data
            if interval == '1m' and len(bars) > 0:
                bars = bars.between_time('09:30', '16:00')
            
            # Remove duplicates and sort
            bars = bars.drop_duplicates().sort_index()
            
            # Cache
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(bars, f)
                print(f"💾 Cached {len(bars)} bars from Alpaca to {cache_file.name}")
            except Exception as e:
                print(f"⚠️ Could not write Alpaca cache: {e}")
            
            return bars
            
        except ImportError:
            print(f"⚠️ alpaca-trade-api not installed. Skipping Alpaca data source.")
            return pd.DataFrame()
        except Exception as e:
            print(f"⚠️ Alpaca API error: {e}")
            return pd.DataFrame()

    def get_historical_data_massive(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1m",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data with PRIORITY: Alpaca → Massive → yfinance
        
        This ensures paid data sources (Alpaca/Massive) are used before free yfinance.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # PRIORITY 1: Try Alpaca first (you're paying for this!)
        print(f"🔑 Priority 1: Attempting Alpaca API for {symbol}...")
        alpaca_data = self.get_historical_data_alpaca(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            use_cache=use_cache
        )
        
        if alpaca_data is not None and len(alpaca_data) > 0:
            print(f"✅ SUCCESS: Got {len(alpaca_data):,} bars from Alpaca API (PAID SERVICE)")
            print(f"   Date range: {alpaca_data.index.min()} to {alpaca_data.index.max()}")
            return alpaca_data
        
        # PRIORITY 2: Try Massive API
        print(f"🔑 Priority 2: Attempting Massive API for {symbol}...")
        safe_symbol = symbol.replace(":", "_")
        cache_file = self.cache_dir / f"{safe_symbol}_{interval}_{start_date}_{end_date}_massive.pkl"
        
        if use_cache and cache_file.exists():
            print(f"📂 Loading cached Massive data: {cache_file.name}")
            try:
                with open(cache_file, "rb") as f:
                    df = pickle.load(f)
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    print(f"✅ SUCCESS: Loaded {len(df):,} bars from Massive cache (PAID SERVICE)")
                    return df
            except Exception as e:
                print(f"⚠️ Massive cache load failed: {e}, downloading fresh data")
        
        # Lazy import to keep base env light
        try:
            from massive_api_client import MassiveAPIClient
        except Exception as e:
            print(f"⚠️ massive_api_client.py not available: {e}")
            # Fall through to yfinance
        else:
            try:
                massive_symbol = self.massive_symbol_map.get(symbol, symbol)
                client = MassiveAPIClient()
                print(f"📥 Massive/Polygon download: {symbol} ({massive_symbol}) {start_date}→{end_date} ({interval})")
                df = client.get_historical_data(massive_symbol, start_date=start_date, end_date=end_date, interval=interval)
                
                if df is not None and len(df) > 0:
                    # Basic hygiene
                    if isinstance(df.index, pd.DatetimeIndex):
                        df = df.sort_index()
                        df = df[~df.index.duplicated(keep="last")]
                    
                    # Cache
                    try:
                        with open(cache_file, "wb") as f:
                            pickle.dump(df, f)
                    except Exception as e:
                        print(f"⚠️ Could not write Massive cache: {e}")
                    
                    print(f"✅ SUCCESS: Got {len(df):,} bars from Massive API (PAID SERVICE)")
                    print(f"   Date range: {df.index.min()} to {df.index.max()}")
                    return df
            except Exception as e:
                print(f"⚠️ Massive API error: {e}")
        
        # PRIORITY 3: Fallback to yfinance (FREE, but limited)
        print(f"⚠️ WARNING: Paid services (Alpaca/Massive) failed or unavailable")
        print(f"   Falling back to yfinance (FREE, LIMITED - max 7 days for 1m data)")
        return self.get_historical_data(symbol, start_date, end_date, interval, use_cache)
        
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
        use_features: bool = False,
        human_momentum_mode: bool = False
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
        self.human_momentum_mode = human_momentum_mode
        
        # Position tracking
        self.position = None  # {symbol, qty, entry_premium, entry_price, strike, option_type, entry_time}
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.trade_history = []
        
        # Observation space (enhanced)
        if use_features and self.feature_engine:
            # Use institutional features (500+ features)
            obs_shape = (window_size, 130)  # Approximate feature count
        elif self.human_momentum_mode:
            # Human-style momentum/context features (scalp focused)
            # OHLC returns (4) + volume (1) + VIX (1) + VIX delta (1)
            # + EMA diff (1) + VWAP dist (1) + RSI (1) + MACD hist (1) + ATR (1)
            # + candle body ratio (1) + wick ratio (1) + pullback % (1) + breakout score (1)
            # + trend slope (1) + momentum burst (1) + trend strength (1)
            # + Greeks (4) = 23
            obs_shape = (window_size, 23)
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

    def action_masks(self) -> np.ndarray:
        """
        Action masking support (for MaskablePPO).
        True = allowed, False = masked out.
        """
        # 0=HOLD, 1=BUY_CALL, 2=BUY_PUT, 3=TRIM_50, 4=TRIM_70, 5=EXIT
        if self.position is None:
            return np.array([True, True, True, False, False, False], dtype=bool)
        # When in a position: allow HOLD/TRIM/EXIT; disallow opening another position
        return np.array([True, False, False, True, True, True], dtype=bool)
    
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

            # Human momentum mode: compute technical/context features per bar
            if self.human_momentum_mode:
                closes = window_data['close'].astype(float).values
                highs = window_data['high'].astype(float).values
                lows = window_data['low'].astype(float).values
                opens = window_data['open'].astype(float).values
                vols = window_data['volume'].astype(float).values

                # Normalize OHLC as % change from first close in window (stable scale)
                base = float(closes[0]) if float(closes[0]) != 0.0 else 1.0
                o = (opens - base) / base * 100.0
                h = (highs - base) / base * 100.0
                l = (lows - base) / base * 100.0
                c = (closes - base) / base * 100.0

                # Volume normalized to max in window
                maxv = float(vols.max()) if float(vols.max()) > 0 else 1.0
                v = vols / maxv

                # VIX (constant across window)
                vix = self._get_vix(current_time)
                vix_norm = np.full(self.window_size, (vix / 50.0) if vix else 0.4, dtype=np.float32)
                # VIX delta (day-over-day) normalized
                try:
                    vix_prev = float(self.vix_data.iloc[-2]) if len(self.vix_data) > 1 else float(vix)
                    vix_delta = float(vix) - float(vix_prev)
                except Exception:
                    vix_delta = 0.0
                vix_delta_norm = np.full(self.window_size, np.tanh(vix_delta / 5.0), dtype=np.float32)

                # EMA 9/20 diff (scaled)
                def ema(arr, span: int):
                    return pd.Series(arr).ewm(span=span, adjust=False).mean().values

                ema9 = ema(closes, 9)
                ema20 = ema(closes, 20)
                ema_diff = (ema9 - ema20) / base * 100.0  # % of price
                ema_diff = np.tanh(ema_diff / 0.5)  # squash

                # VWAP distance (%)
                # typical price * volume cumulative / cum volume
                tp = (highs + lows + closes) / 3.0
                cumv = np.cumsum(vols)
                cumv[cumv == 0] = 1.0
                vwap = np.cumsum(tp * vols) / cumv
                vwap_dist = (closes - vwap) / base * 100.0
                vwap_dist = np.tanh(vwap_dist / 0.5)

                # RSI(14) scaled to [-1,1]
                delta = np.diff(closes, prepend=closes[0])
                up = np.where(delta > 0, delta, 0.0)
                down = np.where(delta < 0, -delta, 0.0)
                roll = 14
                up_ema = pd.Series(up).ewm(alpha=1/roll, adjust=False).mean().values
                down_ema = pd.Series(down).ewm(alpha=1/roll, adjust=False).mean().values
                rs = up_ema / np.maximum(down_ema, 1e-9)
                rsi = 100.0 - (100.0 / (1.0 + rs))
                rsi_scaled = (rsi - 50.0) / 50.0

                # MACD histogram (12,26,9) scaled
                macd = ema(closes, 12) - ema(closes, 26)
                signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
                macd_hist = (macd - signal) / base * 100.0
                macd_hist = np.tanh(macd_hist / 0.3)

                # ATR(14) normalized by price
                prev_close = np.roll(closes, 1)
                prev_close[0] = closes[0]
                tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
                atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
                atr_pct = (atr / base) * 100.0
                atr_scaled = np.tanh(atr_pct / 1.0)

                # Candle structure: body ratio + wick ratio (per bar)
                rng = np.maximum(highs - lows, 1e-9)
                body = np.abs(closes - opens)
                body_ratio = body / rng
                wick_ratio = (rng - body) / rng

                # Pullback % from rolling high (negative values = pullback)
                roll_high = pd.Series(highs).rolling(self.window_size, min_periods=1).max().values
                pullback = (closes - roll_high) / np.maximum(roll_high, 1e-9) * 100.0
                pullback = np.tanh(pullback / 0.5)

                # Breakout score: close vs prior rolling high, normalized by ATR
                prior_high = pd.Series(highs).rolling(10, min_periods=1).max().shift(1).fillna(highs[0]).values
                breakout = (closes - prior_high) / np.maximum(atr, 1e-9)
                breakout = np.tanh(breakout / 1.5)

                # Trend slope (linear regression) scaled
                x = np.arange(self.window_size)
                # slope per window, but broadcast as constant series for stability
                try:
                    slope = np.polyfit(x, closes, 1)[0]
                except Exception:
                    slope = 0.0
                slope_pct = (slope / base) * 100.0
                trend_slope = np.full(self.window_size, np.tanh(slope_pct / 0.05), dtype=np.float32)

                # Momentum burst: volume spike * return impulse (proxy)
                vol_z = (v - np.mean(v)) / (np.std(v) + 1e-9)
                ret = np.diff(closes, prepend=closes[0]) / base * 100.0
                impulse = np.abs(ret)
                burst = np.tanh((vol_z * impulse) / 2.0)

                # Trend strength score: combine EMA diff + MACD hist + VWAP dist (squashed)
                trend_strength = np.tanh((np.abs(ema_diff) + np.abs(macd_hist) + np.abs(vwap_dist)) / 1.5)

                # Greeks (scaled/squashed, same across window)
                greeks_array = np.zeros((self.window_size, 4), dtype=np.float32)
                if self.use_greeks and self.greeks_calc and self.position:
                    strike = self.position['strike']
                    option_type = self.position['option_type']
                    T = 1.0 / (252 * 6.5)
                    sigma = (vix / 100.0) * 1.3 if vix else 0.20
                    g = self.greeks_calc.calculate_greeks(S=current_price, K=strike, T=T, sigma=sigma, option_type=option_type)
                    d = float(np.clip(g.get('delta', 0.0), -1.0, 1.0))
                    gam = float(np.tanh(float(g.get('gamma', 0.0)) * 100.0))
                    the = float(np.tanh(float(g.get('theta', 0.0)) / 10.0))
                    veg = float(np.tanh(float(g.get('vega', 0.0)) / 10.0))
                    greeks_array[:] = np.array([d, gam, the, veg], dtype=np.float32)

                obs = np.column_stack([
                    o, h, l, c, v,
                    vix_norm,
                    vix_delta_norm,
                    ema_diff,
                    vwap_dist,
                    rsi_scaled,
                    macd_hist,
                    atr_scaled,
                    body_ratio,
                    wick_ratio,
                    pullback,
                    breakout,
                    trend_slope,
                    burst,
                    trend_strength,
                    greeks_array[:, 0],  # delta
                    greeks_array[:, 1],  # gamma
                    greeks_array[:, 2],  # theta
                    greeks_array[:, 3],  # vega
                ]).astype(np.float32)

                # Final safety clamp
                obs = np.clip(obs, -10.0, 10.0)
                return obs
            
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

            # ==================== HARD -15% "SEATBELT" STOP (MATCH LIVE RISK RULE) ====================
            # In live trading, risk controls must force an exit regardless of RL preference.
            # Teach the model this reality: once drawdown exceeds -15%, the position is closed.
            if self.human_momentum_mode and self.position and self.position.get('cost', 0) > 0:
                try:
                    pnl_pct_now = float(self.unrealized_pnl) / float(self.position['cost'])
                except Exception:
                    pnl_pct_now = 0.0
                if pnl_pct_now <= -0.15:
                    info = {
                        "forced_stop_loss": True,
                        "forced_stop_pnl_pct": float(pnl_pct_now),
                    }
                    # Force full exit at current bar pricing (simulated premium)
                    reward = float(self._execute_exit(current_price, vix))
                    # Add steep penalty so PPO strongly avoids reaching this zone
                    reward -= 0.8
                    return self._get_obs(), reward, done, False, info
        
        # Execute action
        reward = 0.0
        info = {}

        # ==================== HUMAN MOMENTUM ENTRY QUALITY / OPPORTUNITY SHAPING ====================
        # When flat: compute a simple "setup score" proxy from recent window to:
        # - reward GOOD buys (+bonus)
        # - penalize missed obvious setups (HOLD when setup strong)
        # - penalize chasing bad buys (e.g. RSI very high + rejection wick)
        pre_action_flat = self.position is None
        setup_score = 0.0
        chase_penalty = 0.0
        strong_setup = False
        chase_setup = False
        if self.human_momentum_mode and self.position is None and self.current_step >= self.window_size:
            try:
                window = self.data.iloc[self.current_step - self.window_size:self.current_step]
                closes = window['close'].astype(float).values
                highs = window['high'].astype(float).values
                lows = window['low'].astype(float).values
                opens = window['open'].astype(float).values
                vols = window['volume'].astype(float).values
                base = float(closes[0]) if float(closes[0]) != 0.0 else 1.0

                # EMA9/EMA20 on close
                ema9 = pd.Series(closes).ewm(span=9, adjust=False).mean().values
                ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().values
                ema_up = ema9[-1] > ema20[-1]

                # VWAP reclaim proxy: close above vwap
                tp = (highs + lows + closes) / 3.0
                cumv = np.cumsum(vols); cumv[cumv == 0] = 1.0
                vwap = (np.cumsum(tp * vols) / cumv)[-1]
                vwap_up = closes[-1] > vwap

                # RSI scaled
                delta = np.diff(closes, prepend=closes[0])
                up = np.where(delta > 0, delta, 0.0)
                down = np.where(delta < 0, -delta, 0.0)
                roll = 14
                up_ema = pd.Series(up).ewm(alpha=1/roll, adjust=False).mean().values
                down_ema = pd.Series(down).ewm(alpha=1/roll, adjust=False).mean().values
                rs = up_ema[-1] / max(down_ema[-1], 1e-9)
                rsi = 100.0 - (100.0 / (1.0 + rs))

                # MACD hist sign
                macd = pd.Series(closes).ewm(span=12, adjust=False).mean().values - pd.Series(closes).ewm(span=26, adjust=False).mean().values
                sig = pd.Series(macd).ewm(span=9, adjust=False).mean().values
                macd_pos = (macd[-1] - sig[-1]) > 0

                # Candle rejection proxy (last candle)
                rng = max(highs[-1] - lows[-1], 1e-9)
                body = abs(closes[-1] - opens[-1])
                wick_ratio = (rng - body) / rng
                rejection = wick_ratio > 0.65

                # Setup score (0..4)
                setup_score = float(ema_up) + float(vwap_up) + float(macd_pos) + float(rsi > 50)

                # Chasing penalty: RSI very high + rejection candle
                # Keep small to avoid over-punishing (which would drive model back to HOLD)
                if rsi > 80 and rejection:
                    chase_penalty = -0.03  # Reduced from -0.07 to -0.03 to avoid over-punishment
                    chase_setup = True

                strong_setup = setup_score >= 3.0
            except Exception:
                setup_score = 0.0
                chase_penalty = 0.0
                strong_setup = False
                chase_setup = False

        # Expose telemetry for training diagnostics
        if self.human_momentum_mode:
            info.update({
                "is_flat_pre": bool(pre_action_flat),
                "setup_score": float(setup_score),
                "strong_setup": bool(strong_setup),
                "chase_setup": bool(chase_setup),
            })
        
        if action == 0:  # HOLD
            reward = self._calculate_reward()
            # Opportunity missed penalty (only when FLAT) - STRENGTHENED further to prevent slow recollapse
            if self.human_momentum_mode and self.position is None and setup_score >= 3.0:
                reward -= 0.06  # Increased from -0.05 to -0.06 (sweet spot for scalper-style agent)
                info["missed_opportunity"] = True
            elif self.human_momentum_mode:
                info["missed_opportunity"] = False
                # Small per-step "time tax" on HOLD while flat (anti-collapse measure)
                # Reduced from -0.001 to -0.0005 to avoid overwhelming PPO early
                reward -= 0.0005
        
        elif action == 1:  # BUY_CALL
            if self.position is None:
                reward = self._execute_buy_call(current_price, vix)
                # Buy-strength bonus for good momentum entries - STRENGTHENED further to prevent slow recollapse
                if self.human_momentum_mode and setup_score >= 3.0:
                    reward += 0.12  # Increased from +0.10 to +0.12 to maintain strong BUY bias
                    info["good_buy_bonus"] = True
                elif self.human_momentum_mode:
                    info["good_buy_bonus"] = False
                # Penalize chasing bad buys (keep small to avoid over-punishing)
                reward += chase_penalty
                if self.human_momentum_mode:
                    info["bad_chase_penalty"] = bool(chase_penalty != 0.0)
                
                # FIX #3: Penalize low-advantage actions (encourage selectivity over frequency)
                # If setup_score is low (< 2.0), penalize the action to discourage overtrading
                if self.human_momentum_mode and setup_score < 2.0:
                    reward -= 0.15  # Significant penalty for low-confidence trades
                    info["low_advantage_penalty"] = True
            else:
                reward = -0.001  # Penalty for buying when already in position
        
        elif action == 2:  # BUY_PUT
            if self.position is None:
                reward = self._execute_buy_put(current_price, vix)
                # For puts, reuse the same setup_score proxy as a weak placeholder.
                # (In a future version we'd compute bearish equivalents: EMA9<EMA20, VWAP reject, RSI<50, MACD<0)
                # STRENGTHENED further to prevent slow recollapse
                if self.human_momentum_mode and setup_score >= 3.0:
                    reward += 0.10  # Increased from +0.08 to +0.10 to maintain strong BUY bias
                    info["good_buy_bonus"] = True
                elif self.human_momentum_mode:
                    info["good_buy_bonus"] = False
                reward += chase_penalty
                if self.human_momentum_mode:
                    info["bad_chase_penalty"] = bool(chase_penalty != 0.0)
                
                # FIX #3: Penalize low-advantage actions (encourage selectivity over frequency)
                # If setup_score is low (< 2.0), penalize the action to discourage overtrading
                if self.human_momentum_mode and setup_score < 2.0:
                    reward -= 0.15  # Significant penalty for low-confidence trades
                    info["low_advantage_penalty"] = True
            else:
                reward = -0.001  # Penalty for buying when already in position
        
        elif action == 3:  # TRIM_50%
            if self.position:
                reward = self._execute_trim(0.5, current_price, vix)
            else:
                # Stronger penalty in human_momentum_mode to prevent policy collapse into TRIM while flat
                reward = -0.02 if self.human_momentum_mode else -0.001
        
        elif action == 4:  # TRIM_70%
            if self.position:
                reward = self._execute_trim(0.7, current_price, vix)
            else:
                reward = -0.02 if self.human_momentum_mode else -0.001
        
        elif action == 5:  # EXIT
            if self.position:
                reward = self._execute_exit(current_price, vix)
            else:
                reward = -0.01 if self.human_momentum_mode else 0.0  # discourage useless exits while flat in scalp mode
        
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
        # Human scalp mode: small exploration/participation reward for taking entries
        # (encourages higher trade frequency early in training)
        return 0.02 if self.human_momentum_mode else 0.0
    
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
        return 0.02 if self.human_momentum_mode else 0.0
    
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
        if not self.human_momentum_mode:
            reward = pnl_pct * trim_pct  # Reward based on P&L percentage
        else:
            # Human scalp tiers: reward exiting into strength, punish trimming losers
            # pnl_pct here is per-contract return on premium (approx)
            tier = 0.0
            if pnl_pct >= 2.0:
                tier = 2.0
            elif pnl_pct >= 1.0:
                tier = 1.2
            elif pnl_pct >= 0.7:
                tier = 1.0
            elif pnl_pct >= 0.5:
                tier = 0.7
            elif pnl_pct >= 0.3:
                tier = 0.5
            elif pnl_pct >= 0.2:
                tier = 0.3
            # losses: enforce -15% as a "hard unacceptable" zone
            elif pnl_pct <= -0.15:
                tier = -0.9
            elif pnl_pct <= -0.4:
                tier = -1.0
            elif pnl_pct <= -0.3:
                tier = -0.7
            elif pnl_pct <= -0.2:
                tier = -0.4
            elif pnl_pct < 0:
                tier = -0.2
            reward = tier * float(trim_pct)
        
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
        if not self.human_momentum_mode:
            return pnl_pct * 0.1  # Scale reward

        # Human scalp mode: tiered reward table (fast scalps + runners)
        if pnl_pct >= 2.0:
            return 2.0
        if pnl_pct >= 1.0:
            return 1.2
        if pnl_pct >= 0.7:
            return 1.0
        if pnl_pct >= 0.5:
            return 0.7
        if pnl_pct >= 0.3:
            return 0.5
        if pnl_pct >= 0.2:
            return 0.3
        # losses
        # Treat -15% as a "hard stop" zone (align with live max-loss rule)
        if pnl_pct <= -0.15:
            return -0.9
        if pnl_pct <= -0.4:
            return -1.0
        if pnl_pct <= -0.3:
            return -0.7
        if pnl_pct <= -0.2:
            return -0.4
        if pnl_pct < 0:
            return -0.2
        return 0.0
    
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
        """
        Calculate reward for current state
        
        FIX #2: Added execution penalties to account for real-world trading costs:
        - Spread cost: -0.05 per trade (5-20% spread in 0DTE options)
        - Slippage: -0.01 per minute held (IV crush, poor fills)
        - Holding time penalty: -0.01 per minute after 30 min (theta decay)
        """
        if self.position:
            # Reward based on unrealized P&L
            pnl_pct = (self.unrealized_pnl / self.position['cost']) if self.position.get('cost', 0) else 0.0
            duration = max(1, self.current_step - self.position.get('entry_time', self.current_step))
            
            if not self.human_momentum_mode:
                base_reward = pnl_pct * 0.01
                # Add execution penalties
                spread_penalty = -0.05  # Spread cost per trade
                slippage_penalty = -0.01 * min(duration, 60) / 60.0  # Slippage increases with time
                holding_penalty = -0.01 * max(0, (duration - 30)) / 60.0 if duration > 30 else 0.0
                return base_reward + spread_penalty + slippage_penalty + holding_penalty

            # Human scalp mode: encourage quick winners, punish slow losers
            # small positive if green, negative if red
            shaped = pnl_pct * 0.05
            
            # FIX #2: Add execution penalties
            spread_penalty = -0.05  # Spread cost per trade
            slippage_penalty = -0.01 * min(duration, 60) / 60.0  # Slippage increases with time
            holding_penalty = -0.01 * max(0, (duration - 30)) / 60.0 if duration > 30 else 0.0
            shaped += spread_penalty + slippage_penalty + holding_penalty
            
            # hard-loss zone: add steep ongoing penalty before forced stop triggers
            if pnl_pct <= -0.15:
                shaped -= 0.2
            # time penalty: after ~30 minutes, holding costs reward
            if duration > 30:
                shaped -= min(0.05, (duration - 30) * 0.001)
            # punish big drawdowns quickly
            if pnl_pct <= -0.30:
                shaped -= 0.3
            return float(shaped)
        else:
            # Small negative reward for holding cash (encourage trading)
            # FIX #3: Reduced penalty to encourage selectivity over frequency
            return -0.0001 if self.human_momentum_mode else -0.00005
    
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

