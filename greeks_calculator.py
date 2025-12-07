"""
🏦 GREEKS CALCULATOR MODULE

Institutional-grade options Greeks calculation using Black-Scholes model.
This addresses the #1 "alpha killer" identified in the architect review:
"0DTE PnL ≈ 100·Γ·(ΔS)²/2 — without live Greeks, RL confounds momentum with convexity"

Provides:
- Delta (directional exposure)
- Gamma (convexity/acceleration)
- Theta (time decay)
- Vega (volatility sensitivity)

Author: Mike Agent Institutional Upgrade
Date: December 6, 2025
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional, Tuple
import pandas as pd


class GreeksCalculator:
    """
    Professional options Greeks calculator using Black-Scholes model
    """
    
    def __init__(self, risk_free_rate: float = 0.04):
        """
        Initialize Greeks calculator
        
        Args:
            risk_free_rate: Risk-free interest rate (default 4% annual)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: Optional[float] = None,
        sigma: float = 0.20,
        option_type: str = 'call'
    ) -> Dict[str, float]:
        """
        Calculate all Greeks for an option using Black-Scholes model
        
        Args:
            S: Current stock price (spot price)
            K: Strike price
            T: Time to expiration (in years, e.g., 0DTE = ~1 hour = 1/252/6.5)
            r: Risk-free rate (defaults to instance default)
            sigma: Implied volatility (annualized)
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with:
            - delta: Price sensitivity to underlying change
            - gamma: Delta sensitivity to underlying change (convexity)
            - theta: Time decay (per day, negative for long positions)
            - vega: Volatility sensitivity (per 1% IV change)
            - rho: Interest rate sensitivity (optional)
        """
        if r is None:
            r = self.risk_free_rate
        
        # Handle edge cases
        if T <= 0:
            # Option expired - only intrinsic value
            if option_type == 'call':
                intrinsic = max(0, S - K)
            else:
                intrinsic = max(0, K - S)
            
            return {
                'delta': 1.0 if intrinsic > 0 else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0,
                'intrinsic': intrinsic
            }
        
        # Calculate d1 and d2 (Black-Scholes parameters)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate Greeks
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (
                -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2)
            ) / 365.0  # Convert to per day
            vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100.0  # Per 1% IV change
            rho = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100.0  # Per 1% rate change
        else:  # put
            delta = -norm.cdf(-d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (
                -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
            ) / 365.0  # Convert to per day
            vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100.0  # Per 1% IV change
            rho = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100.0  # Per 1% rate change
        
        # Ensure no NaN or Inf values
        delta = np.clip(delta, -1.0, 1.0) if not np.isnan(delta) else 0.0
        gamma = max(0.0, gamma) if not np.isnan(gamma) else 0.0
        theta = theta if not np.isnan(theta) else 0.0
        vega = max(0.0, vega) if not np.isnan(vega) else 0.0
        rho = rho if not np.isnan(rho) else 0.0
        
        return {
            'delta': float(delta),
            'gamma': float(gamma),
            'theta': float(theta),
            'vega': float(vega),
            'rho': float(rho)
        }
    
    def calculate_portfolio_greeks(
        self,
        positions: Dict[str, Dict],
        current_prices: Dict[str, float],
        time_to_expiry: float = 1.0 / (252 * 6.5)  # ~1 hour for 0DTE
    ) -> Dict[str, float]:
        """
        Calculate portfolio-level aggregated Greeks
        
        Args:
            positions: Dictionary of positions, keyed by symbol
                Each position dict should have:
                - 'strike': Strike price
                - 'qty': Number of contracts
                - 'option_type': 'call' or 'put'
                - 'entry_premium': Entry premium (for IV estimation)
            current_prices: Current underlying prices, keyed by underlying symbol
            time_to_expiry: Time to expiration in years (default ~1 hour for 0DTE)
            
        Returns:
            Dictionary with aggregated Greeks:
            - net_delta: Net directional exposure
            - net_gamma: Net convexity exposure
            - net_theta: Net time decay (per day)
            - net_vega: Net volatility exposure
            - position_count: Number of positions
        """
        net_delta = 0.0
        net_gamma = 0.0
        net_theta = 0.0
        net_vega = 0.0
        position_count = 0
        
        for symbol, pos_data in positions.items():
            try:
                # Extract underlying symbol (e.g., "SPY251205C00450000" -> "SPY")
                underlying = self._extract_underlying_symbol(symbol)
                
                # Get current price
                S = current_prices.get(underlying, 0.0)
                if S <= 0:
                    continue
                
                # Extract option details
                strike = pos_data.get('strike', 0)
                qty = pos_data.get('qty_remaining', pos_data.get('qty', 0))
                option_type = pos_data.get('option_type', 'call')
                
                # Estimate IV from entry premium if available
                entry_premium = pos_data.get('entry_premium', 0.0)
                if entry_premium > 0:
                    # Rough IV estimation: premium / (S * sqrt(T))
                    sigma = min(0.5, max(0.10, entry_premium / (S * np.sqrt(time_to_expiry))))
                else:
                    sigma = 0.20  # Default IV
                
                # Calculate Greeks for this position
                greeks = self.calculate_greeks(
                    S=S,
                    K=strike,
                    T=time_to_expiry,
                    r=self.risk_free_rate,
                    sigma=sigma,
                    option_type=option_type
                )
                
                # Aggregate (multiply by contract multiplier and quantity)
                contract_multiplier = 100  # Standard options contract size
                net_delta += greeks['delta'] * qty * contract_multiplier
                net_gamma += greeks['gamma'] * qty * contract_multiplier * S  # Gamma in $ terms
                net_theta += greeks['theta'] * qty
                net_vega += greeks['vega'] * qty
                position_count += 1
                
            except Exception as e:
                # Skip positions with calculation errors
                continue
        
        return {
            'net_delta': float(net_delta),
            'net_gamma': float(net_gamma),
            'net_theta': float(net_theta),
            'net_vega': float(net_vega),
            'position_count': position_count
        }
    
    def check_greeks_limits(
        self,
        portfolio_greeks: Dict[str, float],
        equity: float,
        max_delta_pct: float = 0.50,
        max_gamma_risk_pct: float = 0.02,
        max_theta_pct: float = 0.05
    ) -> Tuple[bool, str]:
        """
        Check if portfolio Greeks exceed risk limits
        
        Args:
            portfolio_greeks: Output from calculate_portfolio_greeks()
            equity: Account equity
            max_delta_pct: Maximum Delta exposure as % of equity (default 50%)
            max_gamma_risk_pct: Maximum Gamma risk as % of equity (default 2%)
            max_theta_pct: Maximum Theta decay as % of equity per day (default 5%)
            
        Returns:
            Tuple of (is_safe, warning_message)
        """
        net_delta = portfolio_greeks.get('net_delta', 0.0)
        net_gamma = portfolio_greeks.get('net_gamma', 0.0)
        net_theta = abs(portfolio_greeks.get('net_theta', 0.0))
        
        # Delta check (directional exposure)
        delta_pct = abs(net_delta) / equity if equity > 0 else 0.0
        if delta_pct > max_delta_pct:
            return False, f"Portfolio Delta ({delta_pct:.1%}) > {max_delta_pct:.0%} limit"
        
        # Gamma check (convexity risk)
        gamma_risk_pct = net_gamma / equity if equity > 0 else 0.0
        if gamma_risk_pct > max_gamma_risk_pct:
            return False, f"Portfolio Gamma risk ({gamma_risk_pct:.2%}) > {max_gamma_risk_pct:.2%} limit"
        
        # Theta check (time decay)
        theta_pct = net_theta / equity if equity > 0 else 0.0
        if theta_pct > max_theta_pct:
            return False, f"Portfolio Theta ({theta_pct:.2%}) > {max_theta_pct:.2%} limit"
        
        return True, "OK"
    
    def _extract_underlying_symbol(self, option_symbol: str) -> str:
        """
        Extract underlying symbol from option symbol
        e.g., "SPY251205C00450000" -> "SPY"
        """
        # Common patterns
        for symbol in ['SPY', 'QQQ', 'SPX']:
            if option_symbol.startswith(symbol):
                return symbol
        
        # Fallback: extract first 3 characters (assumes standard format)
        return option_symbol[:3] if len(option_symbol) > 3 else option_symbol


# Factory function for easy instantiation
def create_greeks_calculator(risk_free_rate: float = 0.04) -> GreeksCalculator:
    """
    Create a GreeksCalculator instance
    
    Args:
        risk_free_rate: Risk-free interest rate (default 4%)
        
    Returns:
        GreeksCalculator instance
    """
    return GreeksCalculator(risk_free_rate=risk_free_rate)


# Convenience function for single option calculation
def calculate_greeks(
    S: float,
    K: float,
    T: float,
    r: float = 0.04,
    sigma: float = 0.20,
    option_type: str = 'call'
) -> Dict[str, float]:
    """
    Convenience function to calculate Greeks for a single option
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (default 4%)
        sigma: Implied volatility (default 20%)
        option_type: 'call' or 'put'
        
    Returns:
        Dictionary with delta, gamma, theta, vega, rho
    """
    calc = GreeksCalculator(risk_free_rate=r)
    return calc.calculate_greeks(S, K, T, r, sigma, option_type)

