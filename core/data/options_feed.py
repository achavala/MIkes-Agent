"""
Options data feed interface
"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd


class OptionsFeed:
    """Provides options chain and historical data"""
    
    def __init__(self):
        self._yesterday_closes = {}
        self._cache = {}
        
    def get_yesterday_close(self, symbol: str) -> Optional[float]:
        """Get yesterday's closing price"""
        if symbol not in self._yesterday_closes:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                if len(hist) > 1:
                    self._yesterday_closes[symbol] = hist['Close'].iloc[-2]
                else:
                    return None
            except Exception as e:
                print(f"Error fetching yesterday close for {symbol}: {e}")
                return None
        return self._yesterday_closes[symbol]
    
    def get_chain(self, symbol: str, expiration: Optional[datetime] = None) -> List[Dict]:
        """
        Get options chain for symbol.
        Returns simplified chain structure for backtesting.
        """
        try:
            ticker = yf.Ticker(symbol)
            if expiration is None:
                # Get nearest expiration (0DTE)
                expirations = ticker.options
                if not expirations:
                    return []
                expiration = expirations[0]
            
            chain = ticker.option_chain(expiration)
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
            
            # Combine calls and puts
            options = []
            
            # Calls
            for _, row in chain.calls.iterrows():
                options.append({
                    'strike_price': row['strike'],
                    'option_type': 'call',
                    'bid': row.get('bid', 0),
                    'ask': row.get('ask', 0),
                    'lastPrice': row.get('lastPrice', 0),
                    'impliedVolatility': row.get('impliedVolatility', 0.2),
                    'expiration': expiration
                })
            
            # Puts
            for _, row in chain.puts.iterrows():
                options.append({
                    'strike_price': row['strike'],
                    'option_type': 'put',
                    'bid': row.get('bid', 0),
                    'ask': row.get('ask', 0),
                    'lastPrice': row.get('lastPrice', 0),
                    'impliedVolatility': row.get('impliedVolatility', 0.2),
                    'expiration': expiration
                })
            
            return options
        except Exception as e:
            print(f"Error fetching options chain for {symbol}: {e}")
            return []
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if len(hist) > 0:
                return hist['Close'].iloc[-1]
        except Exception as e:
            print(f"Error fetching current price for {symbol}: {e}")
        return None
    
    def update_cache(self, symbol: str):
        """Update cache for symbol"""
        self._yesterday_closes.pop(symbol, None)

