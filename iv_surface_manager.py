"""
🔬 IMPLIED VOLATILITY SURFACE MANAGER

Real-time IV surface tracking and analysis for institutional-grade options trading.
Replaces VIX proxy with actual market IV data from Polygon.io.

Features:
- Real-time ATM IV per symbol
- IV skew calculation (put/call)
- IV term structure
- IV percentile (historical context)
- Cache management (avoid API rate limits)

Author: Mike Agent Institutional Upgrade
Date: December 11, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import time

try:
    from massive_api_client import MassiveAPIClient
    MASSIVE_AVAILABLE = True
except ImportError:
    MASSIVE_AVAILABLE = False
    print("⚠️ Massive API not available - IV surface disabled")


class IVSurfaceManager:
    """
    Manages real-time implied volatility surface for multiple symbols
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_duration_seconds: int = 60,
        history_length: int = 100
    ):
        """
        Initialize IV Surface Manager
        
        Args:
            api_key: Polygon.io API key
            cache_duration_seconds: How long to cache IV data (default 60s)
            history_length: Number of historical IV samples to keep
        """
        self.api_key = api_key
        self.cache_duration = cache_duration_seconds
        self.history_length = history_length
        
        # Initialize client
        if MASSIVE_AVAILABLE and api_key:
            try:
                self.client = MassiveAPIClient(api_key)
                self.enabled = True
            except Exception as e:
                print(f"⚠️ Failed to initialize Massive API: {e}")
                self.client = None
                self.enabled = False
        else:
            self.client = None
            self.enabled = False
        
        # Cache structure: {symbol: {'iv': float, 'timestamp': float, 'skew': dict}}
        self.cache = {}
        
        # Historical IV tracking: {symbol: deque of (timestamp, iv) tuples}
        self.iv_history = {}
    
    def get_atm_iv(
        self,
        symbol: str,
        expiration_date: str,
        spot_price: Optional[float] = None,
        use_cache: bool = True
    ) -> float:
        """
        Get at-the-money implied volatility for a symbol
        
        Args:
            symbol: Underlying symbol (SPY, QQQ, SPX)
            expiration_date: Expiry date (YYYY-MM-DD)
            spot_price: Current spot price (optional)
            use_cache: Whether to use cached data
            
        Returns:
            ATM IV (annualized, e.g., 0.20 = 20%)
        """
        if not self.enabled:
            # Fallback to VIX proxy if API not available
            return self._fallback_iv_from_vix()
        
        # Check cache
        cache_key = f"{symbol}_{expiration_date}"
        if use_cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            age = time.time() - cached['timestamp']
            if age < self.cache_duration:
                return cached['iv']
        
        # Fetch fresh data
        try:
            call_iv, put_iv = self.client.get_atm_iv(symbol, expiration_date, spot_price)
            
            # Average call and put IV
            atm_iv = (call_iv + put_iv) / 2.0 if call_iv and put_iv else 0.0
            
            # Update cache
            self.cache[cache_key] = {
                'iv': atm_iv,
                'timestamp': time.time(),
                'skew': None  # Fetch separately if needed
            }
            
            # Update history
            if symbol not in self.iv_history:
                self.iv_history[symbol] = deque(maxlen=self.history_length)
            self.iv_history[symbol].append((time.time(), atm_iv))
            
            return atm_iv
        
        except Exception as e:
            print(f"⚠️ Error fetching IV for {symbol}: {e}")
            return self._fallback_iv_from_vix()
    
    def get_iv_skew(
        self,
        symbol: str,
        expiration_date: str,
        spot_price: Optional[float] = None,
        use_cache: bool = True
    ) -> Dict[str, float]:
        """
        Get IV skew metrics for a symbol
        
        Returns:
            Dictionary with: atm_iv, otm_put_iv, otm_call_iv, put_skew, call_skew, total_skew
        """
        if not self.enabled:
            return {
                'atm_iv': self._fallback_iv_from_vix(),
                'otm_put_iv': 0.0,
                'otm_call_iv': 0.0,
                'put_skew': 0.0,
                'call_skew': 0.0,
                'total_skew': 0.0
            }
        
        # Check cache
        cache_key = f"{symbol}_{expiration_date}"
        if use_cache and cache_key in self.cache and self.cache[cache_key]['skew'] is not None:
            cached = self.cache[cache_key]
            age = time.time() - cached['timestamp']
            if age < self.cache_duration:
                return cached['skew']
        
        # Fetch fresh data
        try:
            skew = self.client.get_iv_skew(symbol, expiration_date, spot_price)
            
            # Update cache
            if cache_key in self.cache:
                self.cache[cache_key]['skew'] = skew
            else:
                self.cache[cache_key] = {
                    'iv': skew.get('atm_iv', 0.0),
                    'timestamp': time.time(),
                    'skew': skew
                }
            
            return skew
        
        except Exception as e:
            print(f"⚠️ Error fetching IV skew for {symbol}: {e}")
            return {
                'atm_iv': self._fallback_iv_from_vix(),
                'otm_put_iv': 0.0,
                'otm_call_iv': 0.0,
                'put_skew': 0.0,
                'call_skew': 0.0,
                'total_skew': 0.0
            }
    
    def get_iv_percentile(self, symbol: str, lookback_days: int = 30) -> float:
        """
        Calculate IV percentile (where current IV sits in historical range)
        
        Args:
            symbol: Symbol to check
            lookback_days: Number of days to look back
            
        Returns:
            IV percentile (0.0 to 1.0, where 1.0 = highest IV in period)
        """
        if symbol not in self.iv_history or len(self.iv_history[symbol]) < 10:
            return 0.50  # Default to median if insufficient data
        
        # Get current IV
        history = list(self.iv_history[symbol])
        current_iv = history[-1][1]
        
        # Filter to lookback period
        cutoff_time = time.time() - (lookback_days * 86400)
        recent_ivs = [iv for ts, iv in history if ts >= cutoff_time and iv > 0]
        
        if len(recent_ivs) < 5:
            return 0.50
        
        # Calculate percentile
        percentile = sum(1 for iv in recent_ivs if iv <= current_iv) / len(recent_ivs)
        return percentile
    
    def get_iv_rank(self, symbol: str, lookback_days: int = 252) -> float:
        """
        Calculate IV rank (1-year)
        
        IV Rank = (Current IV - Min IV) / (Max IV - Min IV)
        
        Returns:
            IV rank (0.0 to 1.0)
        """
        if symbol not in self.iv_history or len(self.iv_history[symbol]) < 10:
            return 0.50
        
        history = list(self.iv_history[symbol])
        current_iv = history[-1][1]
        
        # Filter to lookback period
        cutoff_time = time.time() - (lookback_days * 86400)
        recent_ivs = [iv for ts, iv in history if ts >= cutoff_time and iv > 0]
        
        if len(recent_ivs) < 5:
            return 0.50
        
        min_iv = min(recent_ivs)
        max_iv = max(recent_ivs)
        
        if max_iv == min_iv:
            return 0.50
        
        iv_rank = (current_iv - min_iv) / (max_iv - min_iv)
        return iv_rank
    
    def get_enhanced_observation_features(
        self,
        symbol: str,
        expiration_date: str,
        spot_price: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Get all IV-related features for RL observation space
        
        Returns:
            Dictionary with enhanced IV features
        """
        atm_iv = self.get_atm_iv(symbol, expiration_date, spot_price)
        skew = self.get_iv_skew(symbol, expiration_date, spot_price)
        iv_percentile = self.get_iv_percentile(symbol)
        iv_rank = self.get_iv_rank(symbol)
        
        return {
            'atm_iv': atm_iv,
            'iv_percentile': iv_percentile,
            'iv_rank': iv_rank,
            'put_skew': skew.get('put_skew', 0.0),
            'call_skew': skew.get('call_skew', 0.0),
            'total_skew': skew.get('total_skew', 0.0),
            'otm_put_iv': skew.get('otm_put_iv', 0.0),
            'otm_call_iv': skew.get('otm_call_iv', 0.0)
        }
    
    def _fallback_iv_from_vix(self) -> float:
        """
        Fallback to VIX-based IV estimation if API unavailable
        
        Returns:
            Estimated IV from VIX
        """
        try:
            import yfinance as yf
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="1d")
            if not vix_data.empty:
                vix = vix_data['Close'].iloc[-1]
                # Convert VIX to volatility (VIX is annualized)
                return (vix / 100.0) * 1.3  # Same as current proxy
        except:
            pass
        
        return 0.20  # Default 20% IV
    
    def clear_cache(self):
        """Clear all cached IV data"""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about cache usage"""
        return {
            'num_cached_symbols': len(self.cache),
            'cache_duration': self.cache_duration,
            'enabled': self.enabled,
            'history_length': {sym: len(hist) for sym, hist in self.iv_history.items()}
        }


# Global instance (initialized by agent)
_iv_manager: Optional[IVSurfaceManager] = None


def initialize_iv_manager(api_key: Optional[str] = None):
    """Initialize global IV surface manager"""
    global _iv_manager
    _iv_manager = IVSurfaceManager(api_key=api_key)
    return _iv_manager


def get_iv_manager() -> Optional[IVSurfaceManager]:
    """Get global IV surface manager instance"""
    return _iv_manager






