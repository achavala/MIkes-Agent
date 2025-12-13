"""
Helper function for executing orders with execution modeling and Greeks checks
"""
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Type hint only - avoid circular import
    from typing import Any as RiskManager
else:
    # Runtime: use string annotation or Any
    RiskManager = None

def execute_order_with_checks(
    api,
    symbol: str,
    qty: int,
    side: str,
    strike: float,
    option_type: str,
    current_price: float,
    risk_mgr,  # RiskManager type (avoid import for flexibility)
    use_execution_modeling: bool = True,
    check_greeks: bool = True
) -> Dict:
    """
    Execute order with execution modeling and Greeks limit checks
    
    Args:
        api: Alpaca API instance
        symbol: Option symbol
        qty: Quantity
        side: 'buy' or 'sell'
        strike: Strike price
        option_type: 'call' or 'put'
        current_price: Current underlying price
        risk_mgr: RiskManager instance
        use_execution_modeling: Whether to use execution engine
        check_greeks: Whether to check portfolio Greeks limits
        
    Returns:
        Order result dictionary
    """
    # 1. Check portfolio Greeks limits if enabled
    if check_greeks and PORTFOLIO_GREEKS_AVAILABLE:
        try:
            greeks_mgr = get_portfolio_greeks_manager()
            if greeks_mgr and GREEKS_CALCULATOR_AVAILABLE and greeks_calc:
                # Calculate proposed trade Greeks
                T = 1.0 / (252 * 6.5)  # 0DTE: ~1 hour
                vix_value = risk_mgr.get_current_vix() if risk_mgr else 20.0
                sigma = (vix_value / 100.0) * 1.3 if vix_value else 0.20
                
                trade_greeks = greeks_calc.calculate_greeks(
                    S=current_price,
                    K=strike,
                    T=T,
                    sigma=sigma,
                    option_type=option_type
                )
                
                # Calculate portfolio impact (multiply by qty and $100 contract multiplier)
                proposed_delta = qty * trade_greeks.get('delta', 0) * 100
                proposed_gamma = qty * trade_greeks.get('gamma', 0) * 100
                proposed_theta = qty * trade_greeks.get('theta', 0) * 100
                proposed_vega = qty * trade_greeks.get('vega', 0) * 100
                
                # Check all limits
                ok, reason = greeks_mgr.check_all_limits(
                    proposed_delta=proposed_delta,
                    proposed_gamma=proposed_gamma,
                    proposed_theta=proposed_theta,
                    proposed_vega=proposed_vega
                )
                
                if not ok:
                    risk_mgr.log(f"🚫 Greeks limit check FAILED: {reason} | Trade rejected: {symbol} qty={qty}", "WARNING")
                    return {
                        'success': False,
                        'error': f"Greeks limit exceeded: {reason}",
                        'rejected_by': 'greeks_manager'
                    }
                
                risk_mgr.log(f"✅ Greeks check passed: {symbol} | Proposed: Δ={proposed_delta:.1f}, Γ={proposed_gamma:.1f}, Θ={proposed_theta:.1f}, ν={proposed_vega:.1f}", "INFO")
        except Exception as e:
            risk_mgr.log(f"⚠️ Error checking Greeks limits: {e}", "WARNING")
            # Continue with trade if check fails (fail-open for safety)
    
    # 2. Execute order with execution modeling
    if use_execution_modeling and EXECUTION_MODELING_AVAILABLE:
        try:
            result = integrate_execution_into_live(
                api=api,
                symbol=symbol,
                qty=qty,
                side=side,
                use_limit_orders=True,
                aggressive=0.6
            )
            
            # 3. Update portfolio Greeks if order succeeded
            if result.get('success') and PORTFOLIO_GREEKS_AVAILABLE:
                try:
                    greeks_mgr = get_portfolio_greeks_manager()
                    if greeks_mgr and GREEKS_CALCULATOR_AVAILABLE and greeks_calc:
                        # Get filled price (or estimate)
                        filled_price = result.get('filled_price', 0.0)
                        if filled_price <= 0:
                            # Estimate premium if not available
                            filled_price = estimate_premium(current_price, strike, option_type)
                        
                        # Calculate Greeks at fill
                        T = 1.0 / (252 * 6.5)
                        vix_value = risk_mgr.get_current_vix() if risk_mgr else 20.0
                        sigma = (vix_value / 100.0) * 1.3 if vix_value else 0.20
                        
                        fill_greeks = greeks_calc.calculate_greeks(
                            S=current_price,
                            K=strike,
                            T=T,
                            sigma=sigma,
                            option_type=option_type
                        )
                        
                        # Add position to portfolio Greeks manager
                        # qty is positive for long, negative for short
                        position_qty = qty if side == 'buy' else -qty
                        
                        greeks_mgr.add_position(
                            symbol=symbol,
                            qty=position_qty,
                            delta=fill_greeks.get('delta', 0),
                            gamma=fill_greeks.get('gamma', 0),
                            theta=fill_greeks.get('theta', 0),
                            vega=fill_greeks.get('vega', 0),
                            option_price=filled_price
                        )
                        
                        risk_mgr.log(f"✅ Portfolio Greeks updated: {symbol} | Portfolio: Δ={greeks_mgr.portfolio_delta:.1f}, Γ={greeks_mgr.portfolio_gamma:.1f}", "INFO")
                except Exception as e:
                    risk_mgr.log(f"⚠️ Error updating portfolio Greeks: {e}", "WARNING")
            
            return result
        except Exception as e:
            risk_mgr.log(f"⚠️ Execution modeling failed, falling back to market order: {e}", "WARNING")
    
    # Fallback: Simple market order
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
            'type': 'market',
            'execution_engine': False
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'execution_engine': False
        }

