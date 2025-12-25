"""
Phase 0 Fill Model - Conservative Execution Model

Assumes worst reasonable execution to test discipline, not microstructure alpha.
"""

from typing import Optional


class FillModel:
    """
    Phase 0 Fill Model
    
    Purpose: Simulate realistic execution costs.
    Philosophy: Assume worst reasonable execution.
    """
    
    def __init__(self, fixed_slippage: float = 0.05, spread_penalty: float = 0.25):
        """
        Initialize fill model
        
        Args:
            fixed_slippage: Fixed slippage per contract ($0.05)
            spread_penalty: Proportional spread penalty (25% of spread)
        """
        self.fixed_slippage = fixed_slippage
        self.spread_penalty = spread_penalty
    
    def execute_entry(
        self,
        mid_price: float,
        premium: float,
        spread_estimate: Optional[float] = None,
        qty: int = 1,
        real_bid: Optional[float] = None,
        real_ask: Optional[float] = None
    ) -> float:
        """
        Execute entry (buy) order
        
        🔴 RED-TEAM FIX: Use real bid/ask when available (ground truth)
        
        Args:
            mid_price: Mid price of option
            premium: Premium estimate
            spread_estimate: Bid/ask spread estimate (optional, fallback)
            qty: Number of contracts
            real_bid: Real bid price from option snapshot (preferred)
            real_ask: Real ask price from option snapshot (preferred)
        
        Returns:
            Fill price per contract
        """
        # Use real bid/ask if available (ground truth)
        if real_bid is not None and real_ask is not None:
            # Entry (buy): we pay the ask price
            # Add slippage penalty for size
            fill_price = real_ask + self.fixed_slippage
            return fill_price
        
        # Fallback to estimated spread (for backtests without real quotes)
        if spread_estimate is not None:
            spread = spread_estimate
        else:
            # Conservative estimate: 15-30% of premium for 0DTE
            spread = premium * 0.20  # 20% of premium (conservative)
        
        # Entry fill: mid + max(fixed_slippage, spread_penalty * spread)
        fill_price = mid_price + max(
            self.fixed_slippage,
            self.spread_penalty * spread
        )
        
        return fill_price
    
    def execute_exit(
        self,
        mid_price: float,
        premium: float,
        spread_estimate: Optional[float] = None,
        qty: int = 1,
        real_bid: Optional[float] = None,
        real_ask: Optional[float] = None
    ) -> float:
        """
        Execute exit (sell) order
        
        🔴 RED-TEAM FIX: Use real bid/ask when available (ground truth)
        
        Args:
            mid_price: Mid price of option
            premium: Premium estimate
            spread_estimate: Bid/ask spread estimate (optional, fallback)
            qty: Number of contracts
            real_bid: Real bid price from option snapshot (preferred)
            real_ask: Real ask price from option snapshot (preferred)
        
        Returns:
            Fill price per contract
        """
        # Use real bid/ask if available (ground truth)
        if real_bid is not None and real_ask is not None:
            # Exit (sell): we receive the bid price
            # Subtract slippage penalty for size
            fill_price = real_bid - self.fixed_slippage
            return max(0.01, fill_price)  # Minimum $0.01
        
        # Fallback to estimated spread (for backtests without real quotes)
        if spread_estimate is not None:
            spread = spread_estimate
        else:
            # Conservative estimate: 15-30% of premium for 0DTE
            spread = premium * 0.20  # 20% of premium (conservative)
        
        # Exit fill: mid - max(fixed_slippage, spread_penalty * spread)
        fill_price = mid_price - max(
            self.fixed_slippage,
            self.spread_penalty * spread
        )
        
        return max(0.01, fill_price)  # Minimum $0.01 (can't go negative)
    
    def estimate_spread(self, premium: float) -> float:
        """
        Estimate bid/ask spread if not available
        
        Args:
            premium: Option premium
        
        Returns:
            Estimated spread
        """
        # Conservative: 20% of premium for 0DTE
        return premium * 0.20

