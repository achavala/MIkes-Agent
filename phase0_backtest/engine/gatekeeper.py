"""
Phase 0 Gatekeeper - Hard Vetoes (No Exceptions)

This module implements the non-negotiable trade gating rules.
These are HARD VETOES - if any gate fails, trade is blocked.

NO EXCEPTIONS. NO OVERRIDES.
"""

from typing import Optional, Tuple
from datetime import datetime
import pytz


class Gatekeeper:
    """
    Phase 0 Trade Gatekeeper
    
    Purpose: Prevent catastrophic behavior by enforcing hard rules.
    Philosophy: Better zero trades than wrong trades.
    """
    
    def __init__(self):
        """Initialize gatekeeper with Phase 0 rules"""
        # Phase 0 constants (from red-team report)
        self.MIN_CONFIDENCE = 0.60  # 60% minimum confidence
        self.MAX_SPREAD_PCT = 20.0  # 20% of premium max spread
        self.ALLOWED_SYMBOLS = {'SPY', 'QQQ'}  # IWM disabled
        self.NO_TRADE_AFTER = 1430  # 14:30 EST (2:30 PM) - market close protection
        
        self.est = pytz.timezone('US/Eastern')
    
    def allow_trade(
        self,
        action: int,
        confidence: float,
        symbol: str,
        current_price: float,
        strike: float,
        option_type: str,
        premium: float,
        spread_estimate: Optional[float] = None,
        expected_move: Optional[float] = None,
        breakeven_move: Optional[float] = None,
        time_of_day: Optional[int] = None,
        vix: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Check if trade is allowed (HARD VETOES)
        
        Args:
            action: RL action (0=HOLD, 1=BUY_CALL, 2=BUY_PUT, 3=TRIM, 4=TRIM, 5=EXIT)
            confidence: Action strength/confidence (0.0-1.0)
            symbol: Underlying symbol (SPY, QQQ, IWM)
            current_price: Current underlying price
            strike: Option strike price
            option_type: 'call' or 'put'
            premium: Option premium estimate
            spread_estimate: Bid/ask spread estimate (optional)
            expected_move: Expected move in dollars (optional)
            breakeven_move: Breakeven move needed in dollars (optional)
            time_of_day: Current time as integer (e.g., 930 = 9:30 AM)
            vix: Current VIX level (optional)
        
        Returns:
            (is_allowed, reason_if_blocked)
        """
        
        # ========== GATE 1: Action Must Be BUY (1 or 2) ==========
        if action not in [1, 2]:  # Only BUY_CALL (1) or BUY_PUT (2) allowed
            return False, f"Action {action} is not BUY (must be 1=BUY_CALL or 2=BUY_PUT)"
        
        # ========== GATE 2: Confidence Threshold ==========
        if confidence < self.MIN_CONFIDENCE:
            return False, f"Confidence {confidence:.3f} < {self.MIN_CONFIDENCE:.3f} threshold"
        
        # ========== GATE 3: Symbol Restriction ==========
        if symbol not in self.ALLOWED_SYMBOLS:
            return False, f"Symbol {symbol} not in allowed list {self.ALLOWED_SYMBOLS}"
        
        # ========== GATE 4: Time Restriction ==========
        if time_of_day is not None and time_of_day >= self.NO_TRADE_AFTER:
            return False, f"Time {time_of_day} >= {self.NO_TRADE_AFTER} (no trade after 14:30 EST)"
        
        # ========== GATE 5: Spread Check (REAL QUOTES) ==========
        # 🔴 RED-TEAM FIX: Use real bid/ask from option snapshots, not estimates
        # This is ground truth, not heuristic
        if spread_estimate is not None and premium > 0:
            spread_pct = (spread_estimate / premium) * 100
            if spread_pct > self.MAX_SPREAD_PCT:
                return False, f"Spread {spread_pct:.1f}% > {self.MAX_SPREAD_PCT}% of premium ${premium:.2f} (REAL QUOTE)"
        
        # Additional check: Reject "ask-only" contracts (bid = 0)
        # These are untradeable and indicate liquidity collapse
        if spread_estimate is not None:
            # If spread_estimate represents (ask - bid) and bid is effectively 0,
            # this is a structural liquidity issue
            # We'll check this in the option universe filter, but also here as safety
            if spread_estimate >= premium * 0.95:  # Spread is ~95%+ of premium = bid is near zero
                return False, f"Bid is effectively zero (ask-only contract) - untradeable"
        
        # ========== GATE 6: Expected Move vs Breakeven (CRITICAL) ==========
        # This single rule eliminates most losses per red-team report
        if expected_move is not None and breakeven_move is not None:
            if expected_move < abs(breakeven_move):
                return False, f"Expected move ${expected_move:.2f} < Breakeven move ${abs(breakeven_move):.2f}"
        
        # ========== GATE 7: Calculate Expected Move if Not Provided ==========
        if expected_move is None and vix is not None and current_price > 0:
            # Simplified expected move: VIX/16 * sqrt(days_to_expiry) * price
            days_to_expiry = 1.0 / (252 * 6.5)  # 0DTE: ~1 trading day
            expected_move_pct = (vix / 16.0) * (days_to_expiry ** 0.5) * 100
            expected_move = current_price * (expected_move_pct / 100)
        
        # ========== GATE 8: Calculate Breakeven if Not Provided ==========
        if breakeven_move is None and current_price > 0 and strike > 0 and premium > 0:
            if option_type.lower() == 'call':
                breakeven_price = strike + premium
                breakeven_move = breakeven_price - current_price
            else:  # put
                breakeven_price = strike - premium
                breakeven_move = current_price - breakeven_price
            
            # Re-check expected move vs breakeven
            if expected_move is not None and expected_move < abs(breakeven_move):
                return False, f"Expected move ${expected_move:.2f} < Breakeven move ${abs(breakeven_move):.2f} (calculated)"
        
        # ========== ALL GATES PASSED ==========
        return True, "All gates passed"
    
    def get_gate_summary(self) -> dict:
        """Get summary of all gates for reporting"""
        return {
            'min_confidence': self.MIN_CONFIDENCE,
            'max_spread_pct': self.MAX_SPREAD_PCT,
            'allowed_symbols': list(self.ALLOWED_SYMBOLS),
            'no_trade_after': self.NO_TRADE_AFTER,
            'total_gates': 8
        }

