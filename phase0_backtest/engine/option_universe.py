"""
Phase 0 Option Universe Filter

Restricts option universe BEFORE RL sees it.
This is how real desks work - filter by liquidity, spread, and strike proximity.
"""

from typing import List, Dict, Optional, Tuple
from datetime import date
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionSnapshotRequest


class OptionUniverseFilter:
    """
    Phase 0 Option Universe Filter
    
    Purpose: Filter option universe by liquidity, spread, and strike proximity BEFORE RL selection.
    Philosophy: RL should only see tradeable options, not the entire chain.
    """
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize option universe filter
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
        """
        self.client = OptionHistoricalDataClient(api_key=api_key, secret_key=api_secret)
        
        # Filter parameters (Phase 0 conservative)
        self.MAX_SPREAD_PCT = 20.0  # 20% max spread
        self.MAX_STRIKE_DISTANCE = 10.0  # $10 max distance from spot
        self.MIN_BID = 0.01  # Minimum bid price (reject ask-only contracts)
        self.MIN_SIZE = 1  # Minimum size on bid/ask
    
    def generate_option_symbols(
        self,
        underlying: str,
        current_price: float,
        trading_day: date,
        option_type: str = 'call',
        strike_range: float = 10.0
    ) -> List[str]:
        """
        Generate option symbols for a given underlying and price
        
        Args:
            underlying: Underlying symbol (SPY, QQQ)
            current_price: Current underlying price
            trading_day: Trading date (for expiration)
            option_type: 'call' or 'put'
            strike_range: Range of strikes to generate (±$10 from current)
        
        Returns:
            List of option symbols
        """
        exp_date_str = trading_day.strftime('%y%m%d')
        
        # Generate strikes around current price
        # For 0DTE, typically $1 increments
        min_strike = int((current_price - strike_range) / 1.0) * 1.0
        max_strike = int((current_price + strike_range) / 1.0) * 1.0
        
        option_symbols = []
        strike = min_strike
        
        while strike <= max_strike:
            strike_str = f"{int(strike * 1000):08d}"
            type_char = 'C' if option_type.lower() == 'call' else 'P'
            symbol = f"{underlying}{exp_date_str}{type_char}{strike_str}"
            option_symbols.append(symbol)
            strike += 1.0
        
        return option_symbols
    
    def filter_by_liquidity(
        self,
        option_symbols: List[str],
        expected_move: Optional[float] = None
    ) -> Tuple[List[str], Dict[str, dict]]:
        """
        Filter option symbols by liquidity (spread, bid, size)
        
        Args:
            option_symbols: List of option symbols to filter
            expected_move: Expected move in dollars (optional, for strike filtering)
        
        Returns:
            (filtered_symbols, snapshot_data)
        """
        if not option_symbols:
            return [], {}
        
        # Fetch snapshots in batches (API may have limits)
        batch_size = 50
        all_snapshots = {}
        
        for i in range(0, len(option_symbols), batch_size):
            batch = option_symbols[i:i + batch_size]
            try:
                req = OptionSnapshotRequest(symbol_or_symbols=batch)
                snapshots = self.client.get_option_snapshot(req)
                all_snapshots.update(snapshots)
            except Exception as e:
                # Log error but continue with other batches
                print(f"⚠️  Error fetching snapshots for batch: {e}")
                continue
        
        # Filter by liquidity criteria
        filtered_symbols = []
        snapshot_data = {}
        
        for symbol in option_symbols:
            if symbol not in all_snapshots:
                continue  # Skip if snapshot not available
            
            snapshot = all_snapshots[symbol]
            
            # Extract quote data
            if not snapshot.latest_quote:
                continue  # No quote = no liquidity
            
            quote = snapshot.latest_quote
            bid = quote.bid_price
            ask = quote.ask_price
            
            # GATE 1: Reject ask-only contracts (bid = 0 or very small)
            if bid < self.MIN_BID:
                continue  # Untradeable
            
            # GATE 2: Reject contracts with insufficient size
            if quote.bid_size < self.MIN_SIZE or quote.ask_size < self.MIN_SIZE:
                continue  # Too thin
            
            # GATE 3: Calculate spread
            mid = (bid + ask) / 2
            spread = ask - bid
            spread_pct = (spread / mid) * 100 if mid > 0 else 100
            
            # GATE 4: Reject wide spreads
            if spread_pct > self.MAX_SPREAD_PCT:
                continue  # Spread too wide
            
            # All gates passed - add to filtered list
            filtered_symbols.append(symbol)
            snapshot_data[symbol] = {
                'bid': bid,
                'ask': ask,
                'mid': mid,
                'spread': spread,
                'spread_pct': spread_pct,
                'bid_size': quote.bid_size,
                'ask_size': quote.ask_size,
                'snapshot': snapshot
            }
        
        return filtered_symbols, snapshot_data
    
    def get_tradeable_options(
        self,
        underlying: str,
        current_price: float,
        trading_day: date,
        expected_move: Optional[float] = None
    ) -> Dict[str, dict]:
        """
        Get tradeable options for an underlying (calls and puts)
        
        Args:
            underlying: Underlying symbol
            current_price: Current underlying price
            trading_day: Trading date
            expected_move: Expected move in dollars (for strike filtering)
        
        Returns:
            Dictionary mapping option symbols to snapshot data
        """
        # Generate option symbols for calls and puts
        call_symbols = self.generate_option_symbols(
            underlying, current_price, trading_day, 'call', self.MAX_STRIKE_DISTANCE
        )
        put_symbols = self.generate_option_symbols(
            underlying, current_price, trading_day, 'put', self.MAX_STRIKE_DISTANCE
        )
        
        # Filter by liquidity
        tradeable_calls, call_data = self.filter_by_liquidity(call_symbols, expected_move)
        tradeable_puts, put_data = self.filter_by_liquidity(put_symbols, expected_move)
        
        # Combine results
        all_tradeable = {}
        all_tradeable.update(call_data)
        all_tradeable.update(put_data)
        
        return all_tradeable
    
    def get_best_strike(
        self,
        tradeable_options: Dict[str, dict],
        current_price: float,
        option_type: str = 'call',
        target_delta: float = 0.50
    ) -> Optional[Tuple[str, dict]]:
        """
        Get best strike from tradeable options (closest to ATM)
        
        Args:
            tradeable_options: Dictionary of tradeable options
            current_price: Current underlying price
            option_type: 'call' or 'put'
            target_delta: Target delta (not used in Phase 0, but kept for future)
        
        Returns:
            (option_symbol, snapshot_data) or None
        """
        # Filter by option type
        type_char = 'C' if option_type.lower() == 'call' else 'P'
        filtered = {
            sym: data for sym, data in tradeable_options.items()
            if sym[6] == type_char  # 7th character is C or P
        }
        
        if not filtered:
            return None
        
        # Find strike closest to current price
        best_symbol = None
        best_distance = float('inf')
        
        for symbol, data in filtered.items():
            # Extract strike from symbol (positions 7-14 are strike)
            try:
                strike_str = symbol[7:15]
                strike = float(strike_str) / 1000.0
                
                distance = abs(strike - current_price)
                if distance < best_distance:
                    best_distance = distance
                    best_symbol = symbol
            except (ValueError, IndexError):
                continue
        
        if best_symbol:
            return best_symbol, filtered[best_symbol]
        
        return None


