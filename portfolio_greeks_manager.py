"""
📊 PORTFOLIO GREEKS MANAGER

Portfolio-level Greek risk management for institutional options trading.
Ensures net directional exposure and time decay stay within acceptable limits.

Features:
- Portfolio Delta limit (net directional exposure)
- Portfolio Gamma limit (convexity risk)
- Portfolio Theta limit (time decay budget)
- Portfolio Vega limit (volatility exposure)
- Real-time position aggregation
- Dynamic limit adjustment based on account size

Author: Mike Agent Institutional Upgrade
Date: December 11, 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd


class PortfolioGreeksManager:
    """
    Manages portfolio-level Greek limits and risk exposure
    """
    
    def __init__(
        self,
        account_size: float = 1000.0,
        max_delta_pct: float = 0.20,
        max_gamma_pct: float = 0.10,
        max_theta_dollar: float = 100.0,
        max_vega_pct: float = 0.15
    ):
        """
        Initialize Portfolio Greeks Manager
        
        Args:
            account_size: Total account value ($)
            max_delta_pct: Max portfolio delta as % of account (e.g., 0.20 = ±20%)
            max_gamma_pct: Max portfolio gamma as % of account
            max_theta_dollar: Max daily theta decay ($) - negative for long positions
            max_vega_pct: Max portfolio vega as % of account
        """
        self.account_size = account_size
        self.max_delta_pct = max_delta_pct
        self.max_gamma_pct = max_gamma_pct
        self.max_theta_dollar = max_theta_dollar
        self.max_vega_pct = max_vega_pct
        
        # Current portfolio greeks
        self.portfolio_delta = 0.0
        self.portfolio_gamma = 0.0
        self.portfolio_theta = 0.0
        self.portfolio_vega = 0.0
        
        # Position tracking: {symbol: {'qty': int, 'delta': float, 'gamma': float, ...}}
        self.positions = {}
        
        # History for monitoring
        self.greek_history = []
    
    def update_account_size(self, new_size: float):
        """Update account size and recalculate dynamic limits"""
        self.account_size = new_size
    
    def add_position(
        self,
        symbol: str,
        qty: int,
        delta: float,
        gamma: float,
        theta: float,
        vega: float,
        option_price: float
    ):
        """
        Add or update a position in portfolio
        
        Args:
            symbol: Option contract symbol
            qty: Number of contracts (positive = long, negative = short)
            delta: Per-contract delta
            gamma: Per-contract gamma
            theta: Per-contract theta (per day)
            vega: Per-contract vega
            option_price: Current option price
        """
        self.positions[symbol] = {
            'qty': qty,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'price': option_price,
            'notional': qty * option_price * 100,  # $100 multiplier
            'timestamp': datetime.now()
        }
        
        # Recalculate portfolio Greeks
        self._recalculate_portfolio_greeks()
    
    def remove_position(self, symbol: str):
        """Remove a position from portfolio"""
        if symbol in self.positions:
            del self.positions[symbol]
            self._recalculate_portfolio_greeks()
    
    def update_position_greeks(
        self,
        symbol: str,
        delta: float,
        gamma: float,
        theta: float,
        vega: float,
        option_price: float
    ):
        """
        Update Greeks for existing position (due to price/time changes)
        """
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos['delta'] = delta
            pos['gamma'] = gamma
            pos['theta'] = theta
            pos['vega'] = vega
            pos['price'] = option_price
            pos['notional'] = pos['qty'] * option_price * 100
            pos['timestamp'] = datetime.now()
            
            self._recalculate_portfolio_greeks()
    
    def _recalculate_portfolio_greeks(self):
        """Recalculate total portfolio Greeks from all positions"""
        self.portfolio_delta = 0.0
        self.portfolio_gamma = 0.0
        self.portfolio_theta = 0.0
        self.portfolio_vega = 0.0
        
        for symbol, pos in self.positions.items():
            qty = pos['qty']
            # Multiply by qty and $100 contract multiplier
            self.portfolio_delta += qty * pos['delta'] * 100
            self.portfolio_gamma += qty * pos['gamma'] * 100
            self.portfolio_theta += qty * pos['theta'] * 100
            self.portfolio_vega += qty * pos['vega'] * 100
        
        # Record in history
        self.greek_history.append({
            'timestamp': datetime.now(),
            'delta': self.portfolio_delta,
            'gamma': self.portfolio_gamma,
            'theta': self.portfolio_theta,
            'vega': self.portfolio_vega,
            'num_positions': len(self.positions)
        })
        
        # Keep last 1000 history entries
        if len(self.greek_history) > 1000:
            self.greek_history = self.greek_history[-1000:]
    
    def check_delta_limit(self, proposed_delta: float = 0.0) -> Tuple[bool, str]:
        """
        Check if portfolio delta is within limits
        
        Args:
            proposed_delta: Additional delta to add (for pre-trade check)
            
        Returns:
            (is_within_limit, reason)
        """
        # Calculate dollar limit
        max_delta_dollar = self.account_size * self.max_delta_pct
        
        # Current + proposed delta
        total_delta = self.portfolio_delta + proposed_delta
        abs_delta = abs(total_delta)
        
        if abs_delta > max_delta_dollar:
            return (
                False,
                f"Portfolio Delta limit exceeded: ${abs_delta:.2f} > ${max_delta_dollar:.2f} "
                f"(current: ${self.portfolio_delta:.2f}, proposed: ${proposed_delta:.2f})"
            )
        
        return (True, "OK")
    
    def check_gamma_limit(self, proposed_gamma: float = 0.0) -> Tuple[bool, str]:
        """Check if portfolio gamma is within limits"""
        max_gamma_dollar = self.account_size * self.max_gamma_pct
        total_gamma = self.portfolio_gamma + proposed_gamma
        abs_gamma = abs(total_gamma)
        
        if abs_gamma > max_gamma_dollar:
            return (
                False,
                f"Portfolio Gamma limit exceeded: ${abs_gamma:.2f} > ${max_gamma_dollar:.2f}"
            )
        
        return (True, "OK")
    
    def check_theta_limit(self, proposed_theta: float = 0.0) -> Tuple[bool, str]:
        """
        Check if portfolio theta is within limits
        
        Note: Theta is typically negative for long options (time decay)
        """
        total_theta = self.portfolio_theta + proposed_theta
        
        # For long options, theta is negative - limit the daily burn
        if total_theta < -self.max_theta_dollar:
            return (
                False,
                f"Portfolio Theta limit exceeded: ${total_theta:.2f}/day < -${self.max_theta_dollar:.2f}/day "
                f"(daily decay too high)"
            )
        
        return (True, "OK")
    
    def check_vega_limit(self, proposed_vega: float = 0.0) -> Tuple[bool, str]:
        """Check if portfolio vega is within limits"""
        max_vega_dollar = self.account_size * self.max_vega_pct
        total_vega = self.portfolio_vega + proposed_vega
        abs_vega = abs(total_vega)
        
        if abs_vega > max_vega_dollar:
            return (
                False,
                f"Portfolio Vega limit exceeded: ${abs_vega:.2f} > ${max_vega_dollar:.2f}"
            )
        
        return (True, "OK")
    
    def check_all_limits(
        self,
        proposed_delta: float = 0.0,
        proposed_gamma: float = 0.0,
        proposed_theta: float = 0.0,
        proposed_vega: float = 0.0
    ) -> Tuple[bool, str]:
        """
        Check all portfolio Greek limits at once
        
        Returns:
            (all_ok, reason) - reason contains first violation
        """
        # Check Delta
        ok, reason = self.check_delta_limit(proposed_delta)
        if not ok:
            return (False, f"[DELTA] {reason}")
        
        # Check Gamma
        ok, reason = self.check_gamma_limit(proposed_gamma)
        if not ok:
            return (False, f"[GAMMA] {reason}")
        
        # Check Theta
        ok, reason = self.check_theta_limit(proposed_theta)
        if not ok:
            return (False, f"[THETA] {reason}")
        
        # Check Vega
        ok, reason = self.check_vega_limit(proposed_vega)
        if not ok:
            return (False, f"[VEGA] {reason}")
        
        return (True, "All portfolio Greek limits OK")
    
    def get_current_exposure(self) -> Dict[str, float]:
        """
        Get current portfolio Greek exposure
        
        Returns:
            Dictionary with current Greeks and % utilization
        """
        max_delta_dollar = self.account_size * self.max_delta_pct
        max_gamma_dollar = self.account_size * self.max_gamma_pct
        max_vega_dollar = self.account_size * self.max_vega_pct
        
        return {
            'portfolio_delta': self.portfolio_delta,
            'portfolio_gamma': self.portfolio_gamma,
            'portfolio_theta': self.portfolio_theta,
            'portfolio_vega': self.portfolio_vega,
            'delta_utilization_pct': abs(self.portfolio_delta) / max_delta_dollar * 100 if max_delta_dollar else 0,
            'gamma_utilization_pct': abs(self.portfolio_gamma) / max_gamma_dollar * 100 if max_gamma_dollar else 0,
            'theta_daily_burn': self.portfolio_theta,
            'vega_utilization_pct': abs(self.portfolio_vega) / max_vega_dollar * 100 if max_vega_dollar else 0,
            'num_positions': len(self.positions),
            'account_size': self.account_size
        }
    
    def get_position_summary(self) -> pd.DataFrame:
        """Get DataFrame of all positions with Greeks"""
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for symbol, pos in self.positions.items():
            data.append({
                'symbol': symbol,
                'qty': pos['qty'],
                'price': pos['price'],
                'notional': pos['notional'],
                'delta': pos['delta'],
                'gamma': pos['gamma'],
                'theta': pos['theta'],
                'vega': pos['vega'],
                'portfolio_delta_contribution': pos['qty'] * pos['delta'] * 100,
                'portfolio_theta_contribution': pos['qty'] * pos['theta'] * 100
            })
        
        return pd.DataFrame(data)
    
    def suggest_hedge(self) -> Dict[str, float]:
        """
        Suggest a hedge to neutralize portfolio Greeks
        
        Returns:
            Dictionary with suggested hedge quantities
        """
        # Simple delta hedge suggestion (more sophisticated hedging possible)
        return {
            'delta_hedge_needed': -self.portfolio_delta,
            'current_delta': self.portfolio_delta,
            'hedge_contracts': -self.portfolio_delta / 100,  # Assuming delta-1 hedge
            'hedge_direction': 'BUY' if self.portfolio_delta < 0 else 'SELL'
        }
    
    def reset(self):
        """Reset all positions and Greeks"""
        self.positions.clear()
        self.portfolio_delta = 0.0
        self.portfolio_gamma = 0.0
        self.portfolio_theta = 0.0
        self.portfolio_vega = 0.0


# Global instance
_portfolio_greeks_manager: Optional[PortfolioGreeksManager] = None


def initialize_portfolio_greeks(account_size: float = 1000.0):
    """Initialize global portfolio Greeks manager"""
    global _portfolio_greeks_manager
    _portfolio_greeks_manager = PortfolioGreeksManager(account_size=account_size)
    return _portfolio_greeks_manager


def get_portfolio_greeks_manager() -> Optional[PortfolioGreeksManager]:
    """Get global portfolio Greeks manager instance"""
    return _portfolio_greeks_manager






