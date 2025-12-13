"""
REALISTIC FILL MODELING FOR 0DTE OPTIONS
Implements market-maker behavior, gamma squeezes, IV collapse, theta explosion, hidden liquidity
"""
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import random


class RealisticFillModel:
    """
    Realistic fill modeling for 0DTE options
    
    Formula: realistic_fill = mid ± (spread * randomness) * liquidity_factor
    
    Factors:
    - Market-maker quoting behavior
    - Gamma squeezes
    - IV collapse after news
    - Time-to-expiry (Theta explosion)
    - Hidden liquidity
    - Spread width
    """
    
    def __init__(self):
        self.fill_history = []
    
    def calculate_realistic_fill(
        self,
        mid: float,
        bid: float,
        ask: float,
        qty: int,
        side: str,
        time_to_expiry: float,  # Hours remaining (0DTE = ~6.5 hours)
        vix: float = 20.0,
        volume: int = 0,
        has_news: bool = False,
        gamma_exposure: float = 0.0,  # Net gamma exposure in market
        hidden_liquidity_pct: float = 0.1  # 10% hidden liquidity
    ) -> Tuple[float, Dict]:
        """
        Calculate realistic fill price
        
        Args:
            mid: Midpoint price
            bid: Current bid
            ask: Current ask
            qty: Order quantity
            side: 'buy' or 'sell'
            time_to_expiry: Hours to expiration (0DTE = ~6.5 hours)
            vix: Current VIX level
            volume: Recent volume
            has_news: Whether there's news/earnings
            gamma_exposure: Net gamma exposure (positive = long gamma squeeze risk)
            hidden_liquidity_pct: Percentage of hidden liquidity
            
        Returns:
            (realistic_fill_price, fill_details)
        """
        if bid <= 0 or ask <= 0 or mid <= 0:
            return mid, {'error': 'Invalid prices'}
        
        spread = ask - bid
        spread_pct = spread / mid if mid > 0 else 0.0
        
        # 1. Base randomness (market-maker behavior)
        # Market makers quote wider spreads when uncertain
        mm_uncertainty = self._market_maker_uncertainty(vix, time_to_expiry, has_news)
        # Ensure mm_uncertainty is non-negative (scale must be >= 0 for np.random.normal)
        mm_uncertainty = max(0.0, abs(mm_uncertainty))  # Use absolute value, ensure non-negative
        randomness = np.random.normal(0, mm_uncertainty)  # Random walk
        
        # 2. Liquidity factor
        liquidity_factor = self._calculate_liquidity_factor(
            qty, volume, spread_pct, hidden_liquidity_pct
        )
        
        # 3. Gamma squeeze impact
        gamma_impact = self._calculate_gamma_squeeze_impact(
            gamma_exposure, side, time_to_expiry
        )
        
        # 4. IV collapse after news
        iv_collapse_impact = self._calculate_iv_collapse_impact(
            has_news, time_to_expiry, vix
        )
        
        # 5. Theta explosion (time decay acceleration)
        theta_impact = self._calculate_theta_explosion_impact(
            time_to_expiry, side
        )
        
        # Combine all factors
        # Formula: realistic_fill = mid ± (spread * randomness) * liquidity_factor
        spread_component = spread * randomness * liquidity_factor
        
        # Additional adjustments
        total_adjustment = (
            spread_component +
            gamma_impact +
            iv_collapse_impact +
            theta_impact
        )
        
        # For buyers: pay more (positive adjustment)
        # For sellers: receive less (negative adjustment)
        if side == 'buy':
            realistic_fill = mid + abs(total_adjustment)
        else:  # sell
            realistic_fill = mid - abs(total_adjustment)
        
        # Clamp to bid-ask range (with some slippage allowed)
        if side == 'buy':
            realistic_fill = max(bid, min(ask * 1.1, realistic_fill))  # Can pay up to 10% above ask
        else:
            realistic_fill = max(bid * 0.9, min(ask, realistic_fill))  # Can receive down to 10% below bid
        
        fill_details = {
            'mid': mid,
            'bid': bid,
            'ask': ask,
            'spread': spread,
            'spread_pct': spread_pct,
            'realistic_fill': realistic_fill,
            'slippage': abs(realistic_fill - mid),
            'slippage_pct': abs(realistic_fill - mid) / mid if mid > 0 else 0.0,
            'liquidity_factor': liquidity_factor,
            'gamma_impact': gamma_impact,
            'iv_collapse_impact': iv_collapse_impact,
            'theta_impact': theta_impact,
            'mm_uncertainty': mm_uncertainty,
            'randomness': randomness,
            'time_to_expiry': time_to_expiry,
            'vix': vix,
            'has_news': has_news
        }
        
        self.fill_history.append(fill_details)
        
        return realistic_fill, fill_details
    
    def _market_maker_uncertainty(
        self,
        vix: float,
        time_to_expiry: float,
        has_news: bool
    ) -> float:
        """
        Market-maker uncertainty affects spread width and quoting behavior
        
        Higher uncertainty = wider spreads = more randomness
        """
        base_uncertainty = 0.3  # Base 30% randomness
        
        # VIX impact (high VIX = more uncertainty)
        vix_factor = 1.0 + (vix - 20.0) / 100.0  # +1% per VIX point above 20
        
        # Time-to-expiry impact (closer to expiry = more uncertainty)
        # 0DTE: 6.5 hours = 1.0, 1 hour = 0.15, 0.5 hours = 0.08
        time_factor = 1.0 + (6.5 - time_to_expiry) / 6.5  # More uncertainty as expiry approaches
        
        # News impact (earnings/news = much more uncertainty)
        news_factor = 2.0 if has_news else 1.0
        
        uncertainty = base_uncertainty * vix_factor * time_factor * news_factor
        
        # Ensure non-negative and cap at 100%
        # Note: time_factor can be negative if time_to_expiry > 6.5, so we need to ensure non-negative
        return max(0.0, min(uncertainty, 1.0))  # Ensure non-negative, cap at 100%
    
    def _calculate_liquidity_factor(
        self,
        qty: int,
        volume: int,
        spread_pct: float,
        hidden_liquidity_pct: float
    ) -> float:
        """
        Calculate liquidity factor
        
        Low liquidity = higher slippage
        Hidden liquidity can help (some orders filled better than expected)
        """
        base_factor = 1.0
        
        # Volume impact
        if volume > 0:
            # Large order relative to volume = low liquidity
            order_size_ratio = qty / volume
            volume_impact = min(order_size_ratio * 2.0, 2.0)  # Cap at 2x
        else:
            volume_impact = 1.5  # No volume data = assume low liquidity
        
        # Spread impact (wide spreads = low liquidity)
        spread_impact = 1.0 + spread_pct * 5.0  # 5% spread = 1.25x impact
        
        # Hidden liquidity (sometimes better fills)
        hidden_liquidity_boost = 1.0 - (hidden_liquidity_pct * random.uniform(0.5, 1.0))
        
        liquidity_factor = base_factor * volume_impact * spread_impact * hidden_liquidity_boost
        
        return max(0.5, min(liquidity_factor, 3.0))  # Between 0.5x and 3x
    
    def _calculate_gamma_squeeze_impact(
        self,
        gamma_exposure: float,
        side: str,
        time_to_expiry: float
    ) -> float:
        """
        Calculate gamma squeeze impact
        
        High positive gamma exposure = market makers hedge = price moves faster
        Closer to expiry = stronger gamma effects
        """
        if abs(gamma_exposure) < 0.01:
            return 0.0
        
        # Gamma squeeze is stronger closer to expiry
        time_factor = 1.0 + (6.5 - time_to_expiry) / 6.5
        
        # Positive gamma = long gamma squeeze risk
        # If buying calls in high gamma environment = pay more (squeeze)
        # If selling in high gamma environment = receive less
        if gamma_exposure > 0:
            # Long gamma squeeze
            if side == 'buy':
                impact = gamma_exposure * 0.1 * time_factor  # Pay more
            else:
                impact = -gamma_exposure * 0.1 * time_factor  # Receive less
        else:
            # Short gamma (less impact)
            impact = abs(gamma_exposure) * 0.05 * time_factor
        
        return impact
    
    def _calculate_iv_collapse_impact(
        self,
        has_news: bool,
        time_to_expiry: float,
        vix: float
    ) -> float:
        """
        Calculate IV collapse impact after news
        
        After earnings/news, IV collapses rapidly
        Sellers benefit (receive less), buyers hurt (pay more initially, then IV crushes)
        """
        if not has_news:
            return 0.0
        
        # IV collapse is stronger closer to expiry
        time_factor = 1.0 + (6.5 - time_to_expiry) / 6.5
        
        # High VIX = more IV to collapse
        vix_factor = vix / 20.0  # Normalize to VIX=20
        
        # IV collapse typically 20-40% after news
        iv_collapse_pct = 0.30 * time_factor * vix_factor
        
        # Impact: negative for both (IV collapse hurts option prices)
        # But sellers are hurt more (they receive less)
        impact = -iv_collapse_pct * 0.1  # 10% of IV collapse as price impact
        
        return impact
    
    def _calculate_theta_explosion_impact(
        self,
        time_to_expiry: float,
        side: str
    ) -> float:
        """
        Calculate theta explosion impact (time decay acceleration)
        
        Theta explodes in last hour (exponential decay)
        Sellers benefit (time decay), buyers hurt (paying for decaying asset)
        """
        if time_to_expiry > 1.0:
            # Not close enough for theta explosion
            return 0.0
        
        # Theta explosion: exponential as expiry approaches
        # Last hour: theta is 10x normal
        theta_factor = 10.0 * (1.0 - time_to_expiry)  # 0.5 hours = 5x, 0.1 hours = 9x
        
        # Theta hurts buyers (time decay), helps sellers (collecting premium)
        if side == 'buy':
            impact = -theta_factor * 0.02  # Buyers pay for decay
        else:
            impact = theta_factor * 0.01  # Sellers benefit from decay (smaller)
        
        return impact
    
    def get_fill_statistics(self) -> Dict:
        """Get statistics on fill quality"""
        if not self.fill_history:
            return {'avg_slippage': 0.0, 'max_slippage': 0.0, 'num_fills': 0}
        
        slippages = [f['slippage_pct'] for f in self.fill_history]
        
        return {
            'avg_slippage_pct': np.mean(slippages),
            'max_slippage_pct': np.max(slippages),
            'min_slippage_pct': np.min(slippages),
            'num_fills': len(self.fill_history),
            'avg_liquidity_factor': np.mean([f['liquidity_factor'] for f in self.fill_history]),
            'avg_gamma_impact': np.mean([f['gamma_impact'] for f in self.fill_history]),
            'avg_iv_collapse_impact': np.mean([f['iv_collapse_impact'] for f in self.fill_history]),
            'avg_theta_impact': np.mean([f['theta_impact'] for f in self.fill_history])
        }


# Global instance
_realistic_fill_model: Optional[RealisticFillModel] = None


def get_realistic_fill_model() -> RealisticFillModel:
    """Get global realistic fill model instance"""
    global _realistic_fill_model
    if _realistic_fill_model is None:
        _realistic_fill_model = RealisticFillModel()
    return _realistic_fill_model


def calculate_realistic_fill(
    mid: float,
    bid: float,
    ask: float,
    qty: int,
    side: str,
    time_to_expiry: float,
    vix: float = 20.0,
    volume: int = 0,
    has_news: bool = False,
    gamma_exposure: float = 0.0,
    hidden_liquidity_pct: float = 0.1
) -> Tuple[float, Dict]:
    """Calculate realistic fill price"""
    model = get_realistic_fill_model()
    return model.calculate_realistic_fill(
        mid, bid, ask, qty, side, time_to_expiry,
        vix, volume, has_news, gamma_exposure, hidden_liquidity_pct
    )

