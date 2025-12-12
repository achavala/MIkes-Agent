#!/usr/bin/env python3
"""
Massive API Client (Polygon.io)
REST API wrapper using requests (no websockets dependency)
"""

import os
import requests
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
from pathlib import Path

class MassiveAPIClient:
    """
    Massive API Client (Polygon.io rebranded)
    
    Uses REST API with requests library (no websockets dependency)
    to avoid conflicts with alpaca-trade-api
    """
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Massive API client
        
        Args:
            api_key: Polygon.io API key (or set MASSIVE_API_KEY env var)
        """
        # Resolve API key from (1) explicit param, (2) env var, (3) .env file, (4) config.py
        self.api_key = api_key or os.getenv('MASSIVE_API_KEY', '') or ''
        if not self.api_key:
            self._try_load_local_env()
            self.api_key = os.getenv('MASSIVE_API_KEY', '') or ''
        if not self.api_key:
            self._try_load_from_config()
            self.api_key = os.getenv('MASSIVE_API_KEY', '') or ''
        if not self.api_key:
            raise ValueError(
                "API key required. Set MASSIVE_API_KEY env var (or add it to .env / config.py)."
            )
        
        self.session = requests.Session()
        self.session.params = {"apiKey": self.api_key}

    def _try_load_local_env(self) -> None:
        """
        Minimal .env loader (no external dependencies).
        Looks for a '.env' file in:
          - current working directory
          - the project root (directory of this file)
        """
        candidates = [
            Path(os.getcwd()) / ".env",
            Path(__file__).resolve().parent / ".env",
        ]
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
                    if k and v:
                        if (k not in os.environ) or (not str(os.environ.get(k, "")).strip()):
                            os.environ[k] = v
            except Exception:
                continue

    def _try_load_from_config(self) -> None:
        """
        Optionally load MASSIVE_API_KEY from config.py (if user stores it there).
        We set os.environ['MASSIVE_API_KEY'] so downstream code is consistent.
        """
        try:
            import config  # type: ignore
            key = getattr(config, "MASSIVE_API_KEY", None)
            if isinstance(key, str) and key.strip():
                if ("MASSIVE_API_KEY" not in os.environ) or (not str(os.environ.get("MASSIVE_API_KEY", "")).strip()):
                    os.environ["MASSIVE_API_KEY"] = key.strip()
        except Exception:
            return

    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, retries: int = 3) -> Dict:
        """Make API request with retry logic"""
        url = f"{self.BASE_URL}{endpoint}"
        request_params = {"apiKey": self.api_key}
        if params:
            request_params.update(params)
        
        for attempt in range(retries):
            try:
                response = self.session.get(url, params=request_params, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limit - wait and retry
                    wait_time = (attempt + 1) * 2
                    print(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 401:
                    raise ValueError("Authentication failed - check API key")
                elif response.status_code == 403:
                    raise ValueError("Forbidden - check subscription access")
                else:
                    response.raise_for_status()
            except requests.exceptions.RequestException as e:
                if attempt == retries - 1:
                    raise
                time.sleep(1)
        
        raise Exception("Request failed after retries")
    
    # ==================== STOCKS DATA ====================
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1m',
        limit: int = 50000
    ) -> pd.DataFrame:
        """
        Get historical stock data
        
        Args:
            symbol: Stock symbol (e.g., 'SPY')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: '1m', '5m', '1h', '1d'
            limit: Max results per request
            
        Returns:
            DataFrame with OHLCV data
        """
        # Map interval to Polygon timespan
        timespan_map = {
            '1m': 'minute',
            '5m': 'minute',
            '15m': 'minute',
            '1h': 'hour',
            '1d': 'day'
        }
        timespan = timespan_map.get(interval, 'minute')
        
        # Determine multiplier
        if interval == '1m':
            multiplier = 1
        elif interval == '5m':
            multiplier = 5
        elif interval == '15m':
            multiplier = 15
        elif interval == '1h':
            multiplier = 1
        elif interval == '1d':
            multiplier = 1
        else:
            multiplier = 1
        
        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        
        all_results = []
        next_url = endpoint
        
        while next_url:
            params = {"limit": limit}
            data = self._make_request(next_url, params)
            
            # Polygon can return status like "OK" or "DELAYED" depending on plan.
            # If results are present, accept them.
            if 'results' in data and isinstance(data.get('results'), list) and len(data['results']) > 0:
                all_results.extend(data['results'])
            
            # Check for pagination
            next_url = data.get('next_url')
            if next_url:
                # Remove base URL if present
                next_url = next_url.replace(self.BASE_URL, '')
                # Remove apiKey param (we add it in _make_request)
                if '?' in next_url:
                    next_url = next_url.split('?')[0]
            
            # Rate limiting
            time.sleep(0.1)
        
        if not all_results:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Rename columns (Polygon format: v=volume, o=open, h=high, l=low, c=close, t=timestamp)
        column_map = {
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap',
            'n': 'transactions'
        }
        
        df = df.rename(columns=column_map)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
        
        # Keep only OHLCV columns
        keep_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in keep_cols if c in df.columns]]
        
        return df
    
    def get_real_time_price(self, symbol: str) -> Optional[float]:
        """
        Get real-time stock price
        
        Args:
            symbol: Stock symbol (SPY, QQQ, SPX, VIX.X, etc.)
            
        Returns:
            Current price or None
        """
        # For VIX, use different endpoint
        if symbol == 'VIX.X' or symbol.startswith('VIX'):
            endpoint = f"/v2/aggs/ticker/VIX/prev"
        else:
            endpoint = f"/v2/aggs/ticker/{symbol}/prev"
        
        try:
            data = self._make_request(endpoint)
            if 'results' in data and isinstance(data.get('results'), list) and len(data['results']) > 0:
                return data['results'][0].get('c')  # Close price
        except Exception as e:
            # Silent fail - fallback to yfinance
            pass
        return None
    
    # ==================== OPTIONS DATA ====================
    
    def get_options_contracts(
        self,
        underlying: str,
        expiration_date: Optional[str] = None,
        contract_type: Optional[str] = None,
        strike_price: Optional[float] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Get options contracts
        
        Args:
            underlying: Underlying symbol (e.g., 'SPY')
            expiration_date: Expiry date (YYYY-MM-DD, optional)
            contract_type: 'call' or 'put' (optional)
            strike_price: Strike price (optional)
            limit: Max results
            
        Returns:
            List of option contracts
        """
        endpoint = "/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying,
            "limit": limit
        }
        
        if expiration_date:
            params["expiration_date"] = expiration_date
        if contract_type:
            params["contract_type"] = contract_type
        if strike_price:
            params["strike_price"] = strike_price
        
        data = self._make_request(endpoint, params)
        
        if data.get('status') == 'OK':
            return data.get('results', [])
        return []
    
    def get_options_chain_with_iv(
        self,
        underlying: str,
        expiration_date: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get full options chain with implied volatility for a given expiration
        
        Args:
            underlying: Underlying symbol (e.g., 'SPY')
            expiration_date: Expiry date (YYYY-MM-DD)
            limit: Max contracts per type (call/put)
            
        Returns:
            DataFrame with columns: strike, type, bid, ask, mid, last, iv, delta, gamma, vega, theta, volume, open_interest
        """
        # Get snapshot of all options for this underlying
        endpoint = f"/v3/snapshot/options/{underlying}"
        params = {"limit": limit * 2}  # Get both calls and puts
        
        try:
            data = self._make_request(endpoint, params)
            
            if data.get('status') != 'OK' or 'results' not in data:
                return pd.DataFrame()
            
            chain_data = []
            for contract in data.get('results', []):
                details = contract.get('details', {})
                # Filter by expiration date
                if details.get('expiration_date') != expiration_date:
                    continue
                
                last_quote = contract.get('last_quote', {})
                last_trade = contract.get('last_trade', {})
                greeks = contract.get('greeks', {})
                
                bid = last_quote.get('bid', 0.0)
                ask = last_quote.get('ask', 0.0)
                mid = (bid + ask) / 2.0 if bid and ask else 0.0
                
                chain_data.append({
                    'strike': details.get('strike_price', 0.0),
                    'type': details.get('contract_type', ''),
                    'bid': bid,
                    'ask': ask,
                    'mid': mid,
                    'last': last_trade.get('price', 0.0),
                    'iv': greeks.get('implied_volatility', 0.0),  # Real-time IV!
                    'delta': greeks.get('delta', 0.0),
                    'gamma': greeks.get('gamma', 0.0),
                    'vega': greeks.get('vega', 0.0),
                    'theta': greeks.get('theta', 0.0),
                    'volume': contract.get('volume', 0),
                    'open_interest': contract.get('open_interest', 0),
                    'ticker': contract.get('ticker', '')
                })
            
            return pd.DataFrame(chain_data)
        
        except Exception as e:
            print(f"Error fetching options chain: {e}")
            return pd.DataFrame()
    
    def get_iv_surface(
        self,
        underlying: str,
        expiration_dates: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Get implied volatility surface across multiple expirations
        
        Args:
            underlying: Underlying symbol
            expiration_dates: List of expiration dates (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping expiration_date -> DataFrame with IV data
        """
        surface = {}
        
        for exp_date in expiration_dates:
            chain = self.get_options_chain_with_iv(underlying, exp_date)
            if not chain.empty:
                surface[exp_date] = chain
        
        return surface
    
    def get_atm_iv(
        self,
        underlying: str,
        expiration_date: str,
        spot_price: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Get at-the-money implied volatility for calls and puts
        
        Args:
            underlying: Underlying symbol
            expiration_date: Expiry date
            spot_price: Current spot price (fetched if not provided)
            
        Returns:
            Tuple of (call_iv, put_iv) for ATM strikes
        """
        if spot_price is None:
            spot_price = self.get_real_time_price(underlying)
            if spot_price is None:
                return (0.0, 0.0)
        
        chain = self.get_options_chain_with_iv(underlying, expiration_date)
        if chain.empty:
            return (0.0, 0.0)
        
        # Find closest strike to spot
        chain['distance'] = abs(chain['strike'] - spot_price)
        atm_strike = chain.loc[chain['distance'].idxmin(), 'strike']
        
        # Get IV for ATM call and put
        atm_chain = chain[chain['strike'] == atm_strike]
        call_iv = atm_chain[atm_chain['type'] == 'call']['iv'].values
        put_iv = atm_chain[atm_chain['type'] == 'put']['iv'].values
        
        call_iv = call_iv[0] if len(call_iv) > 0 else 0.0
        put_iv = put_iv[0] if len(put_iv) > 0 else 0.0
        
        return (call_iv, put_iv)
    
    def get_iv_skew(
        self,
        underlying: str,
        expiration_date: str,
        spot_price: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate implied volatility skew metrics
        
        Args:
            underlying: Underlying symbol
            expiration_date: Expiry date
            spot_price: Current spot price
            
        Returns:
            Dictionary with skew metrics: atm_iv, otm_put_iv, otm_call_iv, put_skew, call_skew
        """
        if spot_price is None:
            spot_price = self.get_real_time_price(underlying)
            if spot_price is None:
                return {}
        
        chain = self.get_options_chain_with_iv(underlying, expiration_date)
        if chain.empty:
            return {}
        
        # ATM IV
        atm_call_iv, atm_put_iv = self.get_atm_iv(underlying, expiration_date, spot_price)
        atm_iv = (atm_call_iv + atm_put_iv) / 2.0 if atm_call_iv and atm_put_iv else 0.0
        
        # OTM strikes (10% away from spot)
        otm_put_strike = spot_price * 0.90
        otm_call_strike = spot_price * 1.10
        
        # Find closest OTM strikes
        puts = chain[chain['type'] == 'put']
        calls = chain[chain['type'] == 'call']
        
        if not puts.empty:
            otm_put_iv = puts.iloc[(puts['strike'] - otm_put_strike).abs().argmin()]['iv']
        else:
            otm_put_iv = 0.0
        
        if not calls.empty:
            otm_call_iv = calls.iloc[(calls['strike'] - otm_call_strike).abs().argmin()]['iv']
        else:
            otm_call_iv = 0.0
        
        # Skew = OTM IV - ATM IV (positive = smile, negative = reverse skew)
        put_skew = (otm_put_iv - atm_iv) if atm_iv else 0.0
        call_skew = (otm_call_iv - atm_iv) if atm_iv else 0.0
        
        return {
            'atm_iv': atm_iv,
            'otm_put_iv': otm_put_iv,
            'otm_call_iv': otm_call_iv,
            'put_skew': put_skew,  # Typically positive for equity options
            'call_skew': call_skew,
            'total_skew': put_skew - call_skew  # Put skew - call skew
        }
    
    def get_option_details(self, contract_ticker: str) -> Optional[Dict]:
        """
        Get details for specific option contract
        
        Args:
            contract_ticker: Option contract ticker (e.g., 'O:SPY251208C00680000')
            
        Returns:
            Option details with price, Greeks, etc.
        """
        # Get last quote
        endpoint = f"/v2/last/trade/{contract_ticker}"
        try:
            trade_data = self._make_request(endpoint)
        except:
            trade_data = None
        
        # Get last quote for bid/ask
        endpoint = f"/v2/last/nbbo/{contract_ticker}"
        try:
            quote_data = self._make_request(endpoint)
        except:
            quote_data = None
        
        # Get contract details
        endpoint = "/v3/reference/options/contracts"
        params = {"ticker": contract_ticker}
        try:
            contract_data = self._make_request(endpoint, params)
            contracts = contract_data.get('results', [])
            if contracts:
                contract_info = contracts[0]
                
                # Combine all data
                result = {
                    **contract_info,
                    "last_trade": trade_data.get('results', [{}])[0] if trade_data else {},
                    "last_quote": quote_data.get('results', [{}])[0] if quote_data else {}
                }
                return result
        except:
            pass
        
        return None
    
    def get_option_price(
        self,
        underlying: str,
        strike: float,
        expiry: str,
        option_type: str
    ) -> Optional[Dict[str, float]]:
        """
        Get option price and Greeks
        
        Args:
            underlying: Underlying symbol
            strike: Strike price
            expiry: Expiry date (YYYY-MM-DD)
            option_type: 'call' or 'put'
            
        Returns:
            Dict with 'bid', 'ask', 'mid', 'delta', 'gamma', 'theta', 'vega', 'iv'
        """
        # Find contract
        contracts = self.get_options_contracts(
            underlying=underlying,
            expiration_date=expiry,
            contract_type=option_type
        )
        
        # Find matching strike (within $0.50)
        for contract in contracts:
            contract_strike = contract.get('strike_price', 0)
            if abs(contract_strike - strike) < 0.50:
                contract_ticker = contract.get('ticker')
                if contract_ticker:
                    details = self.get_option_details(contract_ticker)
                    if details:
                        last_quote = details.get('last_quote', {})
                        bid = last_quote.get('p', 0)  # Price
                        ask = last_quote.get('p', 0)
                        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else bid or ask
                        
                        # Extract Greeks if available
                        greeks = details.get('greeks', {})
                        
                        return {
                            'bid': bid,
                            'ask': ask,
                            'mid': mid,
                            'delta': greeks.get('delta', 0),
                            'gamma': greeks.get('gamma', 0),
                            'theta': greeks.get('theta', 0),
                            'vega': greeks.get('vega', 0),
                            'iv': greeks.get('implied_volatility', 0),
                            'volume': details.get('volume', 0),
                            'open_interest': details.get('open_interest', 0)
                        }
        
        return None
    
    # ==================== HELPER METHODS ====================
    
    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            endpoint = "/v2/aggs/ticker/SPY/prev"
            data = self._make_request(endpoint)
            return data.get('status') == 'OK'
        except:
            return False

