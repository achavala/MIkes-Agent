"""
🔗 INSTITUTIONAL FEATURES INTEGRATION MODULE

Complete integration layer for all institutional features into live trading agent.
This module provides drop-in integration with zero breaking changes to existing code.

Features Integrated:
1. Real-time IV surface (strike/expiry interpolation)
2. Portfolio Greeks limits (entry/exit gating)
3. Limit order execution (smart routing)
4. VaR-based position sizing

Author: Mike Agent - Final Institutional Integration
Date: December 11, 2025
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Import all institutional modules
try:
    from iv_surface_interpolator import initialize_iv_surface_interpolator, get_iv_surface_interpolator
    IV_SURFACE_AVAILABLE = True
except ImportError:
    IV_SURFACE_AVAILABLE = False
    print("⚠️ IV Surface Interpolator not available")

try:
    from portfolio_greeks_manager import initialize_portfolio_greeks, get_portfolio_greeks_manager
    PORTFOLIO_GREEKS_AVAILABLE = True
except ImportError:
    PORTFOLIO_GREEKS_AVAILABLE = False
    print("⚠️ Portfolio Greeks Manager not available")

try:
    from var_calculator import initialize_var_calculator, get_var_calculator
    VAR_CALC_AVAILABLE = True
except ImportError:
    VAR_CALC_AVAILABLE = False
    print("⚠️ VaR Calculator not available")

try:
    from advanced_execution import initialize_execution_engine, get_execution_engine
    EXECUTION_ENGINE_AVAILABLE = True
except ImportError:
    EXECUTION_ENGINE_AVAILABLE = False
    print("⚠️ Advanced Execution Engine not available")

try:
    from volatility_forecasting import initialize_volatility_forecaster, get_volatility_forecaster
    VOL_FORECASTER_AVAILABLE = True
except ImportError:
    VOL_FORECASTER_AVAILABLE = False
    print("⚠️ Volatility Forecaster not available")


class InstitutionalIntegration:
    """
    Integration layer for all institutional features
    """
    
    def __init__(
        self,
        account_size: float = 1000.0,
        polygon_api_key: Optional[str] = None,
        enable_iv_surface: bool = True,
        enable_portfolio_greeks: bool = True,
        enable_var: bool = True,
        enable_limit_orders: bool = True,
        enable_vol_forecasting: bool = True
    ):
        """
        Initialize institutional integration
        
        Args:
            account_size: Account size for risk calculations
            polygon_api_key: Polygon.io API key for IV surface
            enable_iv_surface: Enable IV surface interpolation
            enable_portfolio_greeks: Enable portfolio Greek limits
            enable_var: Enable VaR calculations
            enable_limit_orders: Enable limit order execution
            enable_vol_forecasting: Enable volatility forecasting
        """
        self.account_size = account_size
        self.polygon_api_key = polygon_api_key or os.getenv('MASSIVE_API_KEY')
        
        # Initialize all components
        self.iv_surface = None
        self.portfolio_greeks = None
        self.var_calc = None
        self.execution_engine = None
        self.vol_forecaster = None
        
        # Feature flags
        self.features_enabled = {
            'iv_surface': False,
            'portfolio_greeks': False,
            'var': False,
            'limit_orders': False,
            'vol_forecasting': False
        }
        
        # Initialize components
        if enable_iv_surface and IV_SURFACE_AVAILABLE:
            try:
                self.iv_surface = initialize_iv_surface_interpolator(self.polygon_api_key)
                self.features_enabled['iv_surface'] = True
                print("✅ IV Surface Interpolator: ENABLED")
            except Exception as e:
                print(f"⚠️ IV Surface initialization failed: {e}")
        
        if enable_portfolio_greeks and PORTFOLIO_GREEKS_AVAILABLE:
            try:
                self.portfolio_greeks = initialize_portfolio_greeks(account_size=account_size)
                self.features_enabled['portfolio_greeks'] = True
                print("✅ Portfolio Greeks Manager: ENABLED")
            except Exception as e:
                print(f"⚠️ Portfolio Greeks initialization failed: {e}")
        
        if enable_var and VAR_CALC_AVAILABLE:
            try:
                self.var_calc = initialize_var_calculator(confidence_level=0.95)
                self.features_enabled['var'] = True
                print("✅ VaR Calculator: ENABLED")
            except Exception as e:
                print(f"⚠️ VaR Calculator initialization failed: {e}")
        
        if enable_limit_orders and EXECUTION_ENGINE_AVAILABLE:
            try:
                self.execution_engine = initialize_execution_engine()
                self.features_enabled['limit_orders'] = True
                print("✅ Advanced Execution Engine: ENABLED")
            except Exception as e:
                print(f"⚠️ Execution Engine initialization failed: {e}")
        
        if enable_vol_forecasting and VOL_FORECASTER_AVAILABLE:
            try:
                self.vol_forecaster = initialize_volatility_forecaster()
                self.features_enabled['vol_forecasting'] = True
                print("✅ Volatility Forecaster: ENABLED")
            except Exception as e:
                print(f"⚠️ Volatility Forecaster initialization failed: {e}")
    
    # ==================== IV SURFACE INTEGRATION ====================
    
    def get_iv_for_option(
        self,
        symbol: str,
        strike: float,
        expiry_date: str,
        spot_price: float,
        option_type: str = 'call',
        fallback_to_vix: bool = True
    ) -> float:
        """
        Get interpolated IV for any option
        
        Args:
            symbol: Underlying symbol
            strike: Strike price
            expiry_date: Expiry date (YYYY-MM-DD)
            spot_price: Current spot price
            option_type: 'call' or 'put'
            fallback_to_vix: Whether to fallback to VIX proxy
            
        Returns:
            IV (annualized)
        """
        if self.features_enabled['iv_surface'] and self.iv_surface:
            try:
                iv = self.iv_surface.get_iv_for_strike_expiry(
                    underlying=symbol,
                    strike=strike,
                    expiry_date=expiry_date,
                    spot_price=spot_price,
                    option_type=option_type
                )
                
                if iv > 0.01:  # Valid IV
                    return iv
            except Exception as e:
                print(f"⚠️ IV surface lookup failed: {e}")
        
        # Fallback to VIX proxy
        if fallback_to_vix:
            return self._fallback_iv_from_vix()
        
        return 0.20  # Default 20%
    
    def _fallback_iv_from_vix(self) -> float:
        """Fallback to VIX-based IV"""
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
            return (vix / 100.0) * 1.3
        except:
            return 0.20
    
    # ==================== PORTFOLIO GREEKS INTEGRATION ====================
    
    def check_portfolio_greek_limits_before_entry(
        self,
        proposed_delta: float,
        proposed_gamma: float,
        proposed_theta: float,
        proposed_vega: float
    ) -> Tuple[bool, str]:
        """
        Check if new position would exceed portfolio Greek limits
        
        Args:
            proposed_delta: New Delta to add (per contract * qty * 100)
            proposed_gamma: New Gamma to add
            proposed_theta: New Theta to add
            proposed_vega: New Vega to add
            
        Returns:
            (allowed, reason)
        """
        if not self.features_enabled['portfolio_greeks'] or not self.portfolio_greeks:
            return (True, "Portfolio Greeks limits disabled")
        
        try:
            ok, reason = self.portfolio_greeks.check_all_limits(
                proposed_delta=proposed_delta,
                proposed_gamma=proposed_gamma,
                proposed_theta=proposed_theta,
                proposed_vega=proposed_vega
            )
            
            return (ok, reason)
        
        except Exception as e:
            print(f"⚠️ Portfolio Greeks check failed: {e}")
            return (True, f"Check failed: {e}")
    
    def update_portfolio_greeks(
        self,
        symbol: str,
        qty: int,
        delta: float,
        gamma: float,
        theta: float,
        vega: float,
        option_price: float,
        action: str = 'add'
    ):
        """
        Update portfolio Greeks after trade execution
        
        Args:
            symbol: Option contract symbol
            qty: Number of contracts
            delta: Per-contract delta
            gamma: Per-contract gamma
            theta: Per-contract theta
            vega: Per-contract vega
            option_price: Option price
            action: 'add' or 'remove'
        """
        if not self.features_enabled['portfolio_greeks'] or not self.portfolio_greeks:
            return
        
        try:
            if action == 'add':
                self.portfolio_greeks.add_position(
                    symbol=symbol,
                    qty=qty,
                    delta=delta,
                    gamma=gamma,
                    theta=theta,
                    vega=vega,
                    option_price=option_price
                )
            elif action == 'remove':
                self.portfolio_greeks.remove_position(symbol)
        
        except Exception as e:
            print(f"⚠️ Portfolio Greeks update failed: {e}")
    
    def get_portfolio_exposure(self) -> Dict:
        """Get current portfolio Greek exposure"""
        if not self.features_enabled['portfolio_greeks'] or not self.portfolio_greeks:
            return {}
        
        try:
            return self.portfolio_greeks.get_current_exposure()
        except Exception as e:
            print(f"⚠️ Portfolio exposure check failed: {e}")
            return {}
    
    # ==================== VAR-BASED POSITION SIZING ====================
    
    def calculate_var_adjusted_position_size(
        self,
        base_position_size: float,
        current_portfolio_value: float,
        proposed_delta: float,
        proposed_gamma: float,
        proposed_vega: float,
        underlying_price: float,
        underlying_vol: float,
        max_var_pct: float = 0.05
    ) -> Tuple[float, Dict]:
        """
        Adjust position size based on VaR limits
        
        Args:
            base_position_size: Base position size ($)
            current_portfolio_value: Current portfolio value
            proposed_delta: Proposed Delta
            proposed_gamma: Proposed Gamma
            proposed_vega: Proposed Vega
            underlying_price: Current underlying price
            underlying_vol: Current volatility
            max_var_pct: Max VaR as % of portfolio (e.g., 0.05 = 5%)
            
        Returns:
            (adjusted_position_size, var_info)
        """
        if not self.features_enabled['var'] or not self.var_calc:
            return (base_position_size, {'status': 'var_disabled'})
        
        try:
            # Calculate Greeks-based VaR for proposed position
            greeks_var = self.var_calc.calculate_greeks_var(
                portfolio_delta=proposed_delta,
                portfolio_gamma=proposed_gamma,
                portfolio_vega=proposed_vega,
                underlying_price=underlying_price,
                underlying_volatility=underlying_vol
            )
            
            var_dollar = greeks_var['greeks_var_dollar']
            
            # Max VaR in dollars
            max_var_dollar = current_portfolio_value * max_var_pct
            
            # If VaR exceeds limit, scale down position
            if var_dollar > max_var_dollar:
                scale_factor = max_var_dollar / var_dollar
                adjusted_size = base_position_size * scale_factor
                
                return (adjusted_size, {
                    'status': 'scaled_down',
                    'var_dollar': var_dollar,
                    'max_var_dollar': max_var_dollar,
                    'scale_factor': scale_factor,
                    'original_size': base_position_size,
                    'adjusted_size': adjusted_size
                })
            else:
                return (base_position_size, {
                    'status': 'within_limits',
                    'var_dollar': var_dollar,
                    'max_var_dollar': max_var_dollar,
                    'utilization_pct': (var_dollar / max_var_dollar) * 100
                })
        
        except Exception as e:
            print(f"⚠️ VaR calculation failed: {e}")
            return (base_position_size, {'status': 'error', 'error': str(e)})
    
    def update_var_history(self, portfolio_return: float):
        """Update VaR calculator with new return"""
        if self.features_enabled['var'] and self.var_calc:
            try:
                self.var_calc.update_portfolio_return(portfolio_return)
            except Exception as e:
                print(f"⚠️ VaR history update failed: {e}")
    
    # ==================== LIMIT ORDER EXECUTION ====================
    
    def execute_smart_limit_order(
        self,
        api,
        symbol: str,
        qty: int,
        side: str,
        bid: float,
        ask: float,
        aggressive: float = 0.6,
        timeout_seconds: int = 5
    ) -> Dict:
        """
        Execute limit order with smart pricing
        
        Args:
            api: Alpaca API instance
            symbol: Option symbol
            qty: Quantity
            side: 'buy' or 'sell'
            bid: Current bid
            ask: Current ask
            aggressive: Aggressiveness (0.5 = mid, 1.0 = cross spread)
            timeout_seconds: Max time to wait for fill
            
        Returns:
            Execution result
        """
        if not self.features_enabled['limit_orders'] or not self.execution_engine:
            # Fallback to market order
            try:
                order = api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                return {
                    'success': True,
                    'order_id': order.id,
                    'execution_type': 'market'
                }
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        # Use limit order execution
        try:
            result = self.execution_engine.execute_limit_order(
                api=api,
                symbol=symbol,
                qty=qty,
                side=side,
                bid=bid,
                ask=ask,
                aggressive=aggressive,
                timeout_seconds=timeout_seconds
            )
            return result
        
        except Exception as e:
            print(f"⚠️ Limit order execution failed: {e}, falling back to market")
            try:
                order = api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                return {
                    'success': True,
                    'order_id': order.id,
                    'execution_type': 'market_fallback',
                    'error': str(e)
                }
            except Exception as e2:
                return {'success': False, 'error': str(e2)}
    
    # ==================== VOLATILITY FORECASTING ====================
    
    def get_volatility_forecast(self, symbol: str, price: float) -> Dict:
        """Get volatility forecast for symbol"""
        if not self.features_enabled['vol_forecasting'] or not self.vol_forecaster:
            return {'status': 'vol_forecasting_disabled'}
        
        try:
            forecast = self.vol_forecaster.update_and_forecast(
                symbol=symbol,
                price=price,
                refit_interval_minutes=60
            )
            return forecast
        except Exception as e:
            print(f"⚠️ Volatility forecast failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    # ==================== STATUS & DIAGNOSTICS ====================
    
    def get_integration_status(self) -> Dict:
        """Get status of all institutional features"""
        return {
            'features_enabled': self.features_enabled,
            'account_size': self.account_size,
            'polygon_api_key_set': bool(self.polygon_api_key),
            'components': {
                'iv_surface': 'enabled' if self.iv_surface else 'disabled',
                'portfolio_greeks': 'enabled' if self.portfolio_greeks else 'disabled',
                'var_calc': 'enabled' if self.var_calc else 'disabled',
                'execution_engine': 'enabled' if self.execution_engine else 'disabled',
                'vol_forecaster': 'enabled' if self.vol_forecaster else 'disabled'
            }
        }
    
    def get_comprehensive_risk_snapshot(self) -> Dict:
        """Get comprehensive risk snapshot across all systems"""
        snapshot = {
            'timestamp': datetime.now(),
            'portfolio_greeks': self.get_portfolio_exposure() if self.portfolio_greeks else {},
            'features_status': self.features_enabled
        }
        
        # Add VaR if available
        if self.features_enabled['var'] and self.var_calc:
            try:
                var_report = self.var_calc.get_var_report(portfolio_value=self.account_size)
                snapshot['var'] = {
                    'average_var_dollar': var_report.get('average_var_dollar', 0),
                    'expected_shortfall_dollar': var_report.get('expected_shortfall_dollar', 0)
                }
            except:
                snapshot['var'] = {}
        
        return snapshot


# Global instance
_institutional_integration: Optional[InstitutionalIntegration] = None


def initialize_institutional_integration(
    account_size: float = 1000.0,
    polygon_api_key: Optional[str] = None
) -> InstitutionalIntegration:
    """Initialize global institutional integration"""
    global _institutional_integration
    _institutional_integration = InstitutionalIntegration(
        account_size=account_size,
        polygon_api_key=polygon_api_key
    )
    return _institutional_integration


def get_institutional_integration() -> Optional[InstitutionalIntegration]:
    """Get global institutional integration instance"""
    return _institutional_integration





