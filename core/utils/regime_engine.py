"""
Market regime detection engine
"""
from typing import Dict
import yfinance as yf
import pandas as pd


class RegimeEngine:
    """Detects market regime (neutral, trending, volatile)"""
    
    def __init__(self):
        self._regime_cache = {}
        
    def is_neutral(self, symbol: str) -> bool:
        """
        Check if market is in neutral regime.
        Mike strategy works best in neutral/gap-fill regimes.
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="20d")
            
            if len(hist) < 20:
                return True  # Default to neutral if insufficient data
            
            # Simple neutral check: low volatility, no strong trend
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std()
            
            # Neutral if volatility < 0.02 (2% daily)
            return volatility < 0.02
        except Exception as e:
            print(f"Error checking regime for {symbol}: {e}")
            return True  # Default to neutral
    
    def has_catalyst(self, symbol: str) -> bool:
        """
        Check for catalyst events (Fed, AI news, etc.)
        Simplified: returns True for backtesting
        """
        # In production, integrate with news feed
        return True

