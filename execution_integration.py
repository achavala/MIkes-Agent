"""
EXECUTION MODELING INTEGRATION
Integrates AdvancedExecutionEngine with backtester and live agent
"""

import sys
import os
from typing import Dict, Optional, Tuple
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_execution import AdvancedExecutionEngine, get_execution_engine, initialize_execution_engine
    EXECUTION_AVAILABLE = True
except ImportError:
    EXECUTION_AVAILABLE = False
    print("Warning: advanced_execution module not found")

try:
    from advanced_backtesting import AdvancedBacktester
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False


def apply_execution_costs(
    premium: float,
    qty: int,
    side: str = 'buy',
    execution_engine: Optional[AdvancedExecutionEngine] = None,
    volume: int = 0,
    apply_slippage: bool = True,
    apply_iv_crush: bool = False,
    time_in_day: float = 0.0,
    has_event: bool = False
) -> Tuple[float, Dict]:
    """
    Apply execution costs (slippage + IV crush) to option premium
    
    Args:
        premium: Base option premium
        qty: Order quantity
        side: 'buy' or 'sell'
        execution_engine: AdvancedExecutionEngine instance (optional)
        volume: Recent volume for slippage calculation
        apply_slippage: Whether to apply slippage
        apply_iv_crush: Whether to apply IV crush adjustment
        time_in_day: Time in trading day (0.0 = open, 1.0 = close)
        has_event: Whether there's an earnings event
        
    Returns:
        (adjusted_premium, execution_details)
    """
    execution_details = {
        'base_premium': premium,
        'slippage': 0.0,
        'iv_crush_adjustment': 0.0,
        'final_premium': premium
    }
    
    adjusted_premium = premium
    
    # Get execution engine
    if execution_engine is None:
        execution_engine = get_execution_engine()
        if execution_engine is None:
            execution_engine = initialize_execution_engine()
    
    # 1. Apply slippage
    if apply_slippage and EXECUTION_AVAILABLE and execution_engine:
        # Estimate bid-ask spread (assume 5% spread for options)
        spread_pct = 0.05  # 5% typical for 0DTE options
        bid = premium * (1 - spread_pct / 2)
        ask = premium * (1 + spread_pct / 2)
        
        slippage = execution_engine.estimate_slippage(
            symbol='',  # Not needed for estimation
            qty=qty,
            bid=bid,
            ask=ask,
            volume=volume
        )
        
        execution_details['slippage'] = slippage
        
        # Apply slippage: buyers pay more, sellers receive less
        if side == 'buy':
            adjusted_premium += slippage
        else:  # sell
            adjusted_premium -= slippage
    
    # 2. Apply IV crush adjustment
    if apply_iv_crush and BACKTESTING_AVAILABLE:
        try:
            # Create temporary backtester for IV crush calculation
            backtester = AdvancedBacktester()
            
            # Estimate initial IV from premium (rough approximation)
            # For 0DTE options, IV is typically high (20-40%)
            initial_iv = 0.25  # 25% default
            
            # Apply IV crush
            crushed_iv = backtester.apply_iv_crush(
                initial_iv=initial_iv,
                time_in_day=time_in_day,
                has_event=has_event
            )
            
            # Adjust premium based on IV change
            iv_change_pct = (crushed_iv - initial_iv) / initial_iv
            # Vega sensitivity: premium changes roughly proportionally to IV
            # For 0DTE, vega is lower, so use 0.5x sensitivity
            iv_crush_adjustment = premium * iv_change_pct * 0.5
            
            execution_details['iv_crush_adjustment'] = iv_crush_adjustment
            adjusted_premium += iv_crush_adjustment
            
        except Exception as e:
            print(f"Warning: Could not apply IV crush: {e}")
    
    execution_details['final_premium'] = adjusted_premium
    
    return adjusted_premium, execution_details


def integrate_execution_into_backtest(
    agent_instance,
    apply_slippage: bool = True,
    apply_iv_crush: bool = False
):
    """
    Monkey-patch agent's _simulate_trade method to include execution costs
    
    Args:
        agent_instance: MikeAgent or similar instance
        apply_slippage: Whether to apply slippage
        apply_iv_crush: Whether to apply IV crush
    """
    original_simulate = agent_instance._simulate_trade
    
    def simulate_with_execution(signal, bar, symbol):
        """Wrapper that adds execution costs"""
        # Call original simulation
        pnl = original_simulate(signal, bar, symbol)
        
        # If this is an entry (BUY), apply execution costs to entry premium
        if signal.action.value == 'BUY' and signal.metadata.get('reason') == 'entry':
            # Get estimated premium
            current_premium = agent_instance._estimate_premium(
                bar['close'],
                signal.strike,
                agent_instance.direction.get(symbol, 'call')
            )
            
            # Apply execution costs
            adjusted_premium, exec_details = apply_execution_costs(
                premium=current_premium,
                qty=signal.size,
                side='buy',
                volume=bar.get('volume', 0),
                apply_slippage=apply_slippage,
                apply_iv_crush=apply_iv_crush,
                time_in_day=0.5  # Assume midday execution
            )
            
            # Update entry premium with execution costs
            if hasattr(agent_instance, 'entry_premium'):
                agent_instance.entry_premium[symbol] = adjusted_premium
            if hasattr(agent_instance, 'avg_premium'):
                agent_instance.avg_premium[symbol] = adjusted_premium
        
        # If this is an exit (SELL), apply execution costs
        elif signal.action.value == 'SELL':
            # Get current premium
            current_premium = agent_instance._estimate_premium(
                bar['close'],
                signal.strike,
                agent_instance.direction.get(symbol, 'call')
            )
            
            # Apply execution costs (sellers receive less)
            adjusted_premium, exec_details = apply_execution_costs(
                premium=current_premium,
                qty=signal.size,
                side='sell',
                volume=bar.get('volume', 0),
                apply_slippage=apply_slippage,
                apply_iv_crush=apply_iv_crush,
                time_in_day=0.8  # Assume late-day exit
            )
            
            # Recalculate PnL with execution costs
            entry_premium = agent_instance.avg_premium.get(symbol, current_premium)
            adjusted_pnl = (adjusted_premium - entry_premium) * signal.size * 100
            
            return adjusted_pnl
        
        return pnl
    
    # Replace method
    agent_instance._simulate_trade = simulate_with_execution
    
    return agent_instance


def integrate_execution_into_live(
    api,
    symbol: str,
    qty: int,
    side: str,
    use_limit_orders: bool = True,
    aggressive: float = 0.6
) -> Dict:
    """
    Integrate execution engine into live trading
    
    Args:
        api: Alpaca API instance
        symbol: Option symbol
        qty: Quantity
        side: 'buy' or 'sell'
        use_limit_orders: Whether to use limit orders (vs market)
        aggressive: Aggressiveness (0.5 = mid, 1.0 = cross spread)
        
    Returns:
        Order result dictionary
    """
    if not EXECUTION_AVAILABLE:
        # Fallback to simple market order
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
    
    execution_engine = get_execution_engine()
    if execution_engine is None:
        execution_engine = initialize_execution_engine()
    
    # Get current bid/ask (simplified - in production, get from market data)
    # For now, estimate from option pricing
    try:
        # Try to get quote from Alpaca
        quote = api.get_latest_quote(symbol)
        bid = float(quote.bp) if hasattr(quote, 'bp') else 0.0
        ask = float(quote.ap) if hasattr(quote, 'ap') else 0.0
    except:
        # Fallback: estimate 5% spread
        # This is a simplification - in production, always get real quotes
        bid = 0.0
        ask = 0.0
    
    if bid <= 0 or ask <= 0:
        # No quote available - use market order
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
                'execution_engine': False,
                'reason': 'no_quote_available'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_engine': False
            }
    
    # Use execution engine for limit order
    if use_limit_orders:
        result = execution_engine.execute_limit_order(
            api=api,
            symbol=symbol,
            qty=qty,
            side=side,
            bid=bid,
            ask=ask,
            aggressive=aggressive
        )
        result['execution_engine'] = True
        return result
    else:
        # Market order fallback
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





