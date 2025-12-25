"""
📊 VALUE AT RISK (VaR) CALCULATOR

Institutional-grade risk measurement using Value at Risk and Expected Shortfall.
Provides multiple VaR calculation methods for portfolio risk assessment.

Features:
- Historical VaR (empirical distribution)
- Parametric VaR (assumes normal distribution)
- Monte Carlo VaR (simulation-based)
- Expected Shortfall / CVaR (tail risk)
- Greeks-based VaR (options portfolio)
- Multi-day VaR scaling

Author: Mike Agent Institutional Upgrade
Date: December 11, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
from scipy import stats


class VaRCalculator:
    """
    Calculate Value at Risk using multiple methodologies
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        historical_window: int = 252,
        n_simulations: int = 10000
    ):
        """
        Initialize VaR Calculator
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 = 95%)
            historical_window: Days of historical data for VaR
            n_simulations: Number of Monte Carlo simulations
        """
        self.confidence_level = confidence_level
        self.historical_window = historical_window
        self.n_simulations = n_simulations
        
        # Historical return tracking: {symbol: deque of daily returns}
        self.return_history = {}
        
        # Portfolio return history
        self.portfolio_returns = deque(maxlen=historical_window)
    
    def update_return(self, symbol: str, daily_return: float, timestamp: Optional[datetime] = None):
        """
        Add new daily return observation
        
        Args:
            symbol: Symbol to update
            daily_return: Daily return (e.g., 0.02 = 2%)
            timestamp: Optional timestamp
        """
        if symbol not in self.return_history:
            self.return_history[symbol] = deque(maxlen=self.historical_window)
        
        self.return_history[symbol].append({
            'return': daily_return,
            'timestamp': timestamp or datetime.now()
        })
    
    def update_portfolio_return(self, portfolio_return: float, timestamp: Optional[datetime] = None):
        """Add portfolio-level return"""
        self.portfolio_returns.append({
            'return': portfolio_return,
            'timestamp': timestamp or datetime.now()
        })
    
    # ==================== HISTORICAL VaR ====================
    
    def calculate_historical_var(
        self,
        returns: Optional[List[float]] = None,
        portfolio_value: float = 1000.0,
        horizon_days: int = 1
    ) -> Tuple[float, float]:
        """
        Calculate Historical VaR (empirical quantile method)
        
        Args:
            returns: List of returns (if None, uses portfolio returns)
            portfolio_value: Current portfolio value ($)
            horizon_days: VaR horizon (1 = 1-day VaR)
            
        Returns:
            (VaR_dollar, VaR_percentage)
        """
        if returns is None:
            if len(self.portfolio_returns) < 30:
                return (0.0, 0.0)  # Not enough data
            returns = [r['return'] for r in self.portfolio_returns]
        
        if len(returns) < 30:
            return (0.0, 0.0)
        
        returns = np.array(returns)
        
        # Scale for multi-day horizon (square-root-of-time rule)
        if horizon_days > 1:
            returns = returns * np.sqrt(horizon_days)
        
        # Calculate percentile (1 - confidence_level = loss threshold)
        var_percentile = np.percentile(returns, (1 - self.confidence_level) * 100)
        
        # VaR in dollars (negative = loss)
        var_dollar = abs(var_percentile * portfolio_value)
        var_percentage = abs(var_percentile)
        
        return (var_dollar, var_percentage)
    
    # ==================== PARAMETRIC VaR ====================
    
    def calculate_parametric_var(
        self,
        returns: Optional[List[float]] = None,
        portfolio_value: float = 1000.0,
        horizon_days: int = 1
    ) -> Tuple[float, float]:
        """
        Calculate Parametric VaR (assumes normal distribution)
        
        Args:
            returns: List of returns
            portfolio_value: Current portfolio value ($)
            horizon_days: VaR horizon
            
        Returns:
            (VaR_dollar, VaR_percentage)
        """
        if returns is None:
            if len(self.portfolio_returns) < 30:
                return (0.0, 0.0)
            returns = [r['return'] for r in self.portfolio_returns]
        
        if len(returns) < 30:
            return (0.0, 0.0)
        
        returns = np.array(returns)
        
        # Calculate mean and std
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Scale for horizon
        if horizon_days > 1:
            mean_return = mean_return * horizon_days
            std_return = std_return * np.sqrt(horizon_days)
        
        # Z-score for confidence level (e.g., 95% = 1.645, 99% = 2.326)
        z_score = stats.norm.ppf(self.confidence_level)
        
        # VaR = mean - z * std (worst case at confidence level)
        var_percentage = abs(mean_return - z_score * std_return)
        var_dollar = var_percentage * portfolio_value
        
        return (var_dollar, var_percentage)
    
    # ==================== MONTE CARLO VaR ====================
    
    def calculate_monte_carlo_var(
        self,
        returns: Optional[List[float]] = None,
        portfolio_value: float = 1000.0,
        horizon_days: int = 1,
        n_simulations: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Calculate Monte Carlo VaR (simulation-based)
        
        Args:
            returns: Historical returns for parameter estimation
            portfolio_value: Current portfolio value ($)
            horizon_days: VaR horizon
            n_simulations: Number of simulations (default: self.n_simulations)
            
        Returns:
            (VaR_dollar, VaR_percentage)
        """
        if returns is None:
            if len(self.portfolio_returns) < 30:
                return (0.0, 0.0)
            returns = [r['return'] for r in self.portfolio_returns]
        
        if len(returns) < 30:
            return (0.0, 0.0)
        
        returns = np.array(returns)
        n_simulations = n_simulations or self.n_simulations
        
        # Estimate parameters from historical data
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Simulate returns
        simulated_returns = np.random.normal(
            loc=mean_return,
            scale=std_return,
            size=(n_simulations, horizon_days)
        )
        
        # Cumulative returns over horizon
        cumulative_returns = np.sum(simulated_returns, axis=1)
        
        # VaR = percentile of simulated losses
        var_percentile = np.percentile(cumulative_returns, (1 - self.confidence_level) * 100)
        var_percentage = abs(var_percentile)
        var_dollar = var_percentage * portfolio_value
        
        return (var_dollar, var_percentage)
    
    # ==================== EXPECTED SHORTFALL (CVaR) ====================
    
    def calculate_expected_shortfall(
        self,
        returns: Optional[List[float]] = None,
        portfolio_value: float = 1000.0,
        horizon_days: int = 1
    ) -> Tuple[float, float]:
        """
        Calculate Expected Shortfall (CVaR) - average loss beyond VaR
        
        This measures tail risk: "What's the average loss when things go really bad?"
        
        Args:
            returns: Historical returns
            portfolio_value: Current portfolio value ($)
            horizon_days: Horizon
            
        Returns:
            (ES_dollar, ES_percentage)
        """
        if returns is None:
            if len(self.portfolio_returns) < 30:
                return (0.0, 0.0)
            returns = [r['return'] for r in self.portfolio_returns]
        
        if len(returns) < 30:
            return (0.0, 0.0)
        
        returns = np.array(returns)
        
        # Scale for horizon
        if horizon_days > 1:
            returns = returns * np.sqrt(horizon_days)
        
        # Find VaR threshold
        var_percentile = np.percentile(returns, (1 - self.confidence_level) * 100)
        
        # Expected Shortfall = average of all losses worse than VaR
        tail_losses = returns[returns <= var_percentile]
        
        if len(tail_losses) == 0:
            es_percentage = abs(var_percentile)
        else:
            es_percentage = abs(np.mean(tail_losses))
        
        es_dollar = es_percentage * portfolio_value
        
        return (es_dollar, es_percentage)
    
    # ==================== GREEKS-BASED VaR (OPTIONS PORTFOLIO) ====================
    
    def calculate_greeks_var(
        self,
        portfolio_delta: float,
        portfolio_gamma: float,
        portfolio_vega: float,
        underlying_price: float,
        underlying_volatility: float,
        price_shock_std: float = 2.0,
        vol_shock_std: float = 0.05,
        correlation: float = -0.5
    ) -> Dict[str, float]:
        """
        Calculate VaR for options portfolio using Greeks approximation
        
        P&L ≈ Delta * ΔS + 0.5 * Gamma * (ΔS)² + Vega * Δσ
        
        Args:
            portfolio_delta: Total portfolio delta
            portfolio_gamma: Total portfolio gamma
            portfolio_vega: Total portfolio vega
            underlying_price: Current underlying price
            underlying_volatility: Current IV
            price_shock_std: Standard deviations for price move (e.g., 2 = 2σ)
            vol_shock_std: Vol shock in absolute terms (e.g., 0.05 = 5% IV increase)
            correlation: Correlation between price and vol (typically negative)
            
        Returns:
            Dictionary with VaR estimates for different scenarios
        """
        # Price shock (e.g., -2σ move)
        price_down_shock = -price_shock_std * underlying_price * underlying_volatility / np.sqrt(252)
        price_up_shock = price_shock_std * underlying_price * underlying_volatility / np.sqrt(252)
        
        # Vol shock (e.g., +5% IV)
        vol_up_shock = vol_shock_std
        vol_down_shock = -vol_shock_std
        
        # Scenario 1: Price down, Vol up (typical crash scenario)
        pnl_crash = (
            portfolio_delta * price_down_shock +
            0.5 * portfolio_gamma * price_down_shock ** 2 +
            portfolio_vega * vol_up_shock * 100  # Vega is per 1% IV
        )
        
        # Scenario 2: Price up, Vol down (rally)
        pnl_rally = (
            portfolio_delta * price_up_shock +
            0.5 * portfolio_gamma * price_up_shock ** 2 +
            portfolio_vega * vol_down_shock * 100
        )
        
        # Scenario 3: Price down, Vol down
        pnl_down_low_vol = (
            portfolio_delta * price_down_shock +
            0.5 * portfolio_gamma * price_down_shock ** 2 +
            portfolio_vega * vol_down_shock * 100
        )
        
        # Scenario 4: Price up, Vol up
        pnl_up_high_vol = (
            portfolio_delta * price_up_shock +
            0.5 * portfolio_gamma * price_up_shock ** 2 +
            portfolio_vega * vol_up_shock * 100
        )
        
        # Worst case scenario
        worst_pnl = min(pnl_crash, pnl_rally, pnl_down_low_vol, pnl_up_high_vol)
        
        return {
            'greeks_var_dollar': abs(worst_pnl),
            'scenario_crash': pnl_crash,
            'scenario_rally': pnl_rally,
            'scenario_down_low_vol': pnl_down_low_vol,
            'scenario_up_high_vol': pnl_up_high_vol,
            'worst_scenario': 'crash' if worst_pnl == pnl_crash else 'rally' if worst_pnl == pnl_rally else 'other'
        }
    
    # ==================== COMPREHENSIVE VAR REPORT ====================
    
    def get_var_report(
        self,
        portfolio_value: float,
        portfolio_delta: Optional[float] = None,
        portfolio_gamma: Optional[float] = None,
        portfolio_vega: Optional[float] = None,
        underlying_price: Optional[float] = None,
        underlying_vol: Optional[float] = None,
        horizon_days: int = 1
    ) -> Dict[str, any]:
        """
        Get comprehensive VaR report using all methodologies
        
        Returns:
            Dictionary with VaR from multiple methods
        """
        report = {
            'portfolio_value': portfolio_value,
            'confidence_level': self.confidence_level,
            'horizon_days': horizon_days,
            'timestamp': datetime.now()
        }
        
        # Historical VaR
        hist_var_dollar, hist_var_pct = self.calculate_historical_var(
            portfolio_value=portfolio_value,
            horizon_days=horizon_days
        )
        report['historical_var_dollar'] = hist_var_dollar
        report['historical_var_pct'] = hist_var_pct
        
        # Parametric VaR
        param_var_dollar, param_var_pct = self.calculate_parametric_var(
            portfolio_value=portfolio_value,
            horizon_days=horizon_days
        )
        report['parametric_var_dollar'] = param_var_dollar
        report['parametric_var_pct'] = param_var_pct
        
        # Monte Carlo VaR
        mc_var_dollar, mc_var_pct = self.calculate_monte_carlo_var(
            portfolio_value=portfolio_value,
            horizon_days=horizon_days
        )
        report['monte_carlo_var_dollar'] = mc_var_dollar
        report['monte_carlo_var_pct'] = mc_var_pct
        
        # Expected Shortfall
        es_dollar, es_pct = self.calculate_expected_shortfall(
            portfolio_value=portfolio_value,
            horizon_days=horizon_days
        )
        report['expected_shortfall_dollar'] = es_dollar
        report['expected_shortfall_pct'] = es_pct
        
        # Greeks-based VaR (if Greeks provided)
        if all(x is not None for x in [portfolio_delta, portfolio_gamma, portfolio_vega, underlying_price, underlying_vol]):
            greeks_var = self.calculate_greeks_var(
                portfolio_delta=portfolio_delta,
                portfolio_gamma=portfolio_gamma,
                portfolio_vega=portfolio_vega,
                underlying_price=underlying_price,
                underlying_volatility=underlying_vol
            )
            report.update(greeks_var)
        
        # Average VaR across methods
        var_estimates = [hist_var_dollar, param_var_dollar, mc_var_dollar]
        var_estimates = [v for v in var_estimates if v > 0]
        report['average_var_dollar'] = np.mean(var_estimates) if var_estimates else 0.0
        
        return report


# Global instance
_var_calculator: Optional[VaRCalculator] = None


def initialize_var_calculator(confidence_level: float = 0.95):
    """Initialize global VaR calculator"""
    global _var_calculator
    _var_calculator = VaRCalculator(confidence_level=confidence_level)
    return _var_calculator


def get_var_calculator() -> Optional[VaRCalculator]:
    """Get global VaR calculator instance"""
    return _var_calculator






