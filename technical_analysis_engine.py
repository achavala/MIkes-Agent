#!/usr/bin/env python3
"""
Technical Analysis Engine - Replicates Mike's Pattern Recognition
Detects: False breakouts, Gap fills, Trendline breaks, Structure confirmations, Rejections
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TechnicalAnalysisEngine:
    """
    Technical Analysis Engine that replicates Mike's pattern recognition
    
    Patterns Detected:
    - False breakouts (trendline, support/resistance)
    - Gap fills (upward gaps, downward gaps)
    - Trendline breaks (with confirmation)
    - Structure confirmations (upside, downside)
    - Rejection patterns (double, triple rejections)
    """
    
    def __init__(self, lookback_bars: int = 50):
        """
        Initialize Technical Analysis Engine
        
        Args:
            lookback_bars: Number of bars to look back for pattern detection
        """
        self.lookback_bars = lookback_bars
        self.pattern_cache = {}  # Cache detected patterns
    
    def _normalize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        CRITICAL FIX: Normalize column names to lowercase for consistent access.
        
        This handles data from different sources:
        - Alpaca returns: Open, High, Low, Close, Volume (capitalized after our mapping)
        - Massive API: may return lowercase or capitalized
        - yfinance: typically returns capitalized
        
        We normalize ALL columns to lowercase for consistent access in TA calculations.
        """
        if data is None or len(data) == 0:
            return data
        
        # Create a copy to avoid modifying original
        df = data.copy()
        
        # Standard column mapping (handle all variations)
        column_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['open', 'o']:
                column_map[col] = 'open'
            elif col_lower in ['high', 'h']:
                column_map[col] = 'high'
            elif col_lower in ['low', 'l']:
                column_map[col] = 'low'
            elif col_lower in ['close', 'c', 'adj close', 'adjclose']:
                column_map[col] = 'close'
            elif col_lower in ['volume', 'v', 'vol']:
                column_map[col] = 'volume'
        
        if column_map:
            df = df.rename(columns=column_map)
        
        return df
    
    def detect_false_breakout(
        self, 
        data: pd.DataFrame, 
        trendline_level: Optional[float] = None,
        invalidation_level: Optional[float] = None
    ) -> Dict:
        """
        Detect false trendline breakout (Mike's pattern)
        
        Pattern: Price breaks above/below trendline but then rejects back
        Example: "False trend-line breakout, invalidated if $678.6 reclaims"
        
        Args:
            data: DataFrame with OHLCV data (must have 'high', 'low', 'close', 'open')
            trendline_level: Optional trendline level to check
            invalidation_level: Optional invalidation level (e.g., $678.6)
        
        Returns:
            Dict with pattern info:
            {
                'detected': bool,
                'pattern_type': 'false_breakout',
                'direction': 'bullish' or 'bearish',
                'breakout_level': float,
                'invalidation_level': float,
                'confidence': float (0-1),
                'reason': str
            }
        """
        if len(data) < 20:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        # CRITICAL: Normalize columns to lowercase for consistent access
        data = self._normalize_columns(data)
        
        # Get recent price action
        recent = data.tail(20)
        current_price = recent['close'].iloc[-1]
        current_high = recent['high'].iloc[-1]
        current_low = recent['low'].iloc[-1]
        prev_close = recent['close'].iloc[-2]
        
        # Calculate trendline if not provided
        if trendline_level is None:
            # Use recent highs/lows to estimate trendline
            highs = recent['high'].values
            lows = recent['low'].values
            
            # For PUTS (bearish false breakout): Price breaks above trendline then rejects
            # For CALLS (bullish false breakout): Price breaks below trendline then rejects
            
            # Detect bearish false breakout (PUT setup)
            # Pattern: Price breaks above resistance, then closes below
            resistance_level = np.max(highs[-10:])  # Recent high
            if current_high > resistance_level * 1.002 and current_price < resistance_level:
                invalidation = invalidation_level if invalidation_level else resistance_level * 1.01
                return {
                    'detected': True,
                    'pattern_type': 'false_breakout',
                    'direction': 'bearish',  # PUT setup
                    'breakout_level': resistance_level,
                    'invalidation_level': invalidation,
                    'confidence': 0.75,
                    'reason': f'False breakout above ${resistance_level:.2f}, invalidated if ${invalidation:.2f} reclaims'
                }
            
            # Detect bullish false breakout (CALL setup)
            # Pattern: Price breaks below support, then closes above
            support_level = np.min(lows[-10:])  # Recent low
            if current_low < support_level * 0.998 and current_price > support_level:
                invalidation = invalidation_level if invalidation_level else support_level * 0.99
                return {
                    'detected': True,
                    'pattern_type': 'false_breakout',
                    'direction': 'bullish',  # CALL setup
                    'breakout_level': support_level,
                    'invalidation_level': invalidation,
                    'confidence': 0.75,
                    'reason': f'False breakout below ${support_level:.2f}, invalidated if ${invalidation:.2f} breaks'
                }
        
        return {'detected': False, 'reason': 'No false breakout pattern detected'}
    
    def detect_gap_fill(
        self, 
        data: pd.DataFrame,
        gap_level: Optional[float] = None
    ) -> Dict:
        """
        Detect gap fill setup (Mike's pattern)
        
        Pattern: Price gaps up/down, then fills the gap
        Example: "Looking for gap fill" (SPY $672 PUTS)
        
        Args:
            data: DataFrame with OHLCV data
            gap_level: Optional gap level to check
        
        Returns:
            Dict with pattern info:
            {
                'detected': bool,
                'pattern_type': 'gap_fill',
                'direction': 'bullish' or 'bearish',
                'gap_level': float,
                'target': float,
                'confidence': float (0-1),
                'reason': str
            }
        """
        if len(data) < 5:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        # CRITICAL: Normalize columns to lowercase for consistent access
        data = self._normalize_columns(data)
        
        # Get recent price action
        recent = data.tail(10)
        current_price = recent['close'].iloc[-1]
        prev_close = recent['close'].iloc[-2]
        
        # Detect gap up (PUT setup - gap fill down)
        # Pattern: Price gaps up, then fills gap by moving down
        if len(recent) >= 2:
            gap_up = recent['open'].iloc[-1] > recent['close'].iloc[-2] * 1.005  # 0.5% gap up
            if gap_up:
                gap_level_calc = recent['close'].iloc[-2]  # Previous close
                gap_top = recent['open'].iloc[-1]  # Gap top
                
                # Check if price is moving down to fill gap
                if current_price < gap_top and current_price > gap_level_calc * 0.995:
                    target = gap_level_calc  # Fill to previous close
                    return {
                        'detected': True,
                        'pattern_type': 'gap_fill',
                        'direction': 'bearish',  # PUT setup
                        'gap_level': gap_level_calc,
                        'gap_top': gap_top,
                        'target': target,
                        'confidence': 0.80,
                        'reason': f'Gap fill setup: Price gapped to ${gap_top:.2f}, filling toward ${target:.2f}'
                    }
            
            # Detect gap down (CALL setup - gap fill up)
            # Pattern: Price gaps down, then fills gap by moving up
            gap_down = recent['open'].iloc[-1] < recent['close'].iloc[-2] * 0.995  # 0.5% gap down
            if gap_down:
                gap_level_calc = recent['close'].iloc[-2]  # Previous close
                gap_bottom = recent['open'].iloc[-1]  # Gap bottom
                
                # Check if price is moving up to fill gap
                if current_price > gap_bottom and current_price < gap_level_calc * 1.005:
                    target = gap_level_calc  # Fill to previous close
                    return {
                        'detected': True,
                        'pattern_type': 'gap_fill',
                        'direction': 'bullish',  # CALL setup
                        'gap_level': gap_level_calc,
                        'gap_bottom': gap_bottom,
                        'target': target,
                        'confidence': 0.80,
                        'reason': f'Gap fill setup: Price gapped to ${gap_bottom:.2f}, filling toward ${target:.2f}'
                    }
        
        return {'detected': False, 'reason': 'No gap fill pattern detected'}
    
    def detect_trendline_break(
        self,
        data: pd.DataFrame,
        timeframe: str = '15M',
        require_confirmation: bool = True
    ) -> Dict:
        """
        Detect trendline break with confirmation (Mike's pattern)
        
        Pattern: Price breaks trendline, candle body confirms on specified timeframe
        Example: "Candle body close above on 15M confirming trendline break"
        
        Args:
            data: DataFrame with OHLCV data
            timeframe: Timeframe for confirmation ('15M', '5M', etc.)
            require_confirmation: Whether to require candle body confirmation
        
        Returns:
            Dict with pattern info:
            {
                'detected': bool,
                'pattern_type': 'trendline_break',
                'direction': 'bullish' or 'bearish',
                'break_level': float,
                'confirmed': bool,
                'confidence': float (0-1),
                'reason': str
            }
        """
        if len(data) < 30:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        # CRITICAL: Normalize columns to lowercase for consistent access
        data = self._normalize_columns(data)
        
        # Get recent price action
        recent = data.tail(30)
        current_price = recent['close'].iloc[-1]
        current_open = recent['open'].iloc[-1]
        current_high = recent['high'].iloc[-1]
        current_low = recent['low'].iloc[-1]
        prev_close = recent['close'].iloc[-2]
        
        # Calculate trendline (simplified: use recent highs/lows)
        highs = recent['high'].values[-20:]
        lows = recent['low'].values[-20:]
        
        # Detect bullish trendline break (CALL setup)
        # Pattern: Price breaks above downtrend, candle body confirms
        resistance_trend = np.max(highs[-10:-5])  # Previous resistance
        if current_high > resistance_trend * 1.003:  # Break above
            # Check candle body confirmation
            candle_body_above = current_close > current_open and current_close > resistance_trend
            if candle_body_above or not require_confirmation:
                return {
                    'detected': True,
                    'pattern_type': 'trendline_break',
                    'direction': 'bullish',  # CALL setup
                    'break_level': resistance_trend,
                    'confirmed': candle_body_above,
                    'confidence': 0.85 if candle_body_above else 0.70,
                    'reason': f'Trendline break above ${resistance_trend:.2f}' + 
                             (f', confirmed by candle body on {timeframe}' if candle_body_above else '')
                }
        
        # Detect bearish trendline break (PUT setup)
        # Pattern: Price breaks below uptrend, candle body confirms
        support_trend = np.min(lows[-10:-5])  # Previous support
        if current_low < support_trend * 0.997:  # Break below
            # Check candle body confirmation
            candle_body_below = current_close < current_open and current_close < support_trend
            if candle_body_below or not require_confirmation:
                return {
                    'detected': True,
                    'pattern_type': 'trendline_break',
                    'direction': 'bearish',  # PUT setup
                    'break_level': support_trend,
                    'confirmed': candle_body_below,
                    'confidence': 0.85 if candle_body_below else 0.70,
                    'reason': f'Trendline break below ${support_trend:.2f}' +
                             (f', confirmed by candle body on {timeframe}' if candle_body_below else '')
                }
        
        return {'detected': False, 'reason': 'No trendline break detected'}
    
    def confirm_structure(
        self,
        data: pd.DataFrame,
        structure_type: str = 'upside'
    ) -> Dict:
        """
        Confirm structure (upside/downside) - Mike's pattern
        
        Pattern: Structure is confirmed by price action
        Example: "Upside structure confirmed, looking for squeeze"
        
        Args:
            data: DataFrame with OHLCV data
            structure_type: 'upside' or 'downside'
        
        Returns:
            Dict with pattern info:
            {
                'detected': bool,
                'pattern_type': 'structure_confirmation',
                'structure_type': str,
                'confidence': float (0-1),
                'reason': str
            }
        """
        if len(data) < 20:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        recent = data.tail(20)
        current_price = recent['close'].iloc[-1]
        
        # Calculate structure
        highs = recent['high'].values
        lows = recent['low'].values
        
        # Upside structure: Higher highs, higher lows
        if structure_type == 'upside':
            higher_highs = np.all(np.diff(highs[-5:]) >= 0) or np.sum(np.diff(highs[-5:]) > 0) >= 3
            higher_lows = np.all(np.diff(lows[-5:]) >= 0) or np.sum(np.diff(lows[-5:]) > 0) >= 3
            
            if higher_highs or higher_lows:
                return {
                    'detected': True,
                    'pattern_type': 'structure_confirmation',
                    'structure_type': 'upside',
                    'confidence': 0.80,
                    'reason': 'Upside structure confirmed (higher highs/lows)'
                }
        
        # Downside structure: Lower highs, lower lows
        elif structure_type == 'downside':
            lower_highs = np.all(np.diff(highs[-5:]) <= 0) or np.sum(np.diff(highs[-5:]) < 0) >= 3
            lower_lows = np.all(np.diff(lows[-5:]) <= 0) or np.sum(np.diff(lows[-5:]) < 0) >= 3
            
            if lower_highs or lower_lows:
                return {
                    'detected': True,
                    'pattern_type': 'structure_confirmation',
                    'structure_type': 'downside',
                    'confidence': 0.80,
                    'reason': 'Downside structure confirmed (lower highs/lows)'
                }
        
        return {'detected': False, 'reason': f'No {structure_type} structure confirmation'}
    
    def detect_rejection(
        self,
        data: pd.DataFrame,
        rejection_level: Optional[float] = None,
        rejection_type: str = 'double'
    ) -> Dict:
        """
        Detect rejection pattern (Mike's pattern)
        
        Pattern: Price rejects from a level multiple times
        Example: "Double rejection on SPY"
        
        Args:
            data: DataFrame with OHLCV data
            rejection_level: Optional level to check for rejection
            rejection_type: 'double' or 'triple'
        
        Returns:
            Dict with pattern info:
            {
                'detected': bool,
                'pattern_type': 'rejection',
                'rejection_type': str,
                'rejection_level': float,
                'direction': 'bullish' or 'bearish',
                'confidence': float (0-1),
                'reason': str
            }
        """
        if len(data) < 20:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        # CRITICAL: Normalize columns to lowercase for consistent access
        data = self._normalize_columns(data)
        
        recent = data.tail(30)
        highs = recent['high'].values
        lows = recent['low'].values
        closes = recent['close'].values
        
        # Detect bearish rejection (PUT setup)
        # Pattern: Price hits resistance multiple times, then rejects down
        if rejection_level is None:
            resistance_level = np.max(highs[-20:])
        else:
            resistance_level = rejection_level
        
        # Count rejections at resistance
        rejections = 0
        for i in range(len(highs) - 5, len(highs)):
            if highs[i] >= resistance_level * 0.998 and closes[i] < resistance_level * 0.995:
                rejections += 1
        
        if rejections >= 2:  # Double or more rejections
            return {
                'detected': True,
                'pattern_type': 'rejection',
                'rejection_type': 'double' if rejections == 2 else 'triple',
                'rejection_level': resistance_level,
                'direction': 'bearish',  # PUT setup
                'confidence': 0.85 if rejections >= 3 else 0.75,
                'reason': f'{rejection_type.capitalize()} rejection at ${resistance_level:.2f}'
            }
        
        # Detect bullish rejection (CALL setup)
        # Pattern: Price hits support multiple times, then rejects up
        if rejection_level is None:
            support_level = np.min(lows[-20:])
        else:
            support_level = rejection_level
        
        # Count rejections at support
        rejections = 0
        for i in range(len(lows) - 5, len(lows)):
            if lows[i] <= support_level * 1.002 and closes[i] > support_level * 1.005:
                rejections += 1
        
        if rejections >= 2:  # Double or more rejections
            return {
                'detected': True,
                'pattern_type': 'rejection',
                'rejection_type': 'double' if rejections == 2 else 'triple',
                'rejection_level': support_level,
                'direction': 'bullish',  # CALL setup
                'confidence': 0.85 if rejections >= 3 else 0.75,
                'reason': f'{rejection_type.capitalize()} rejection at ${support_level:.2f}'
            }
        
        return {'detected': False, 'reason': 'No rejection pattern detected'}
    
    def detect_structure_based_entry(
        self,
        data: pd.DataFrame,
        direction: str = 'bearish'
    ) -> Dict:
        """
        Detect structure-based entry (Mike's pattern) - PROACTIVE ENTRY
        
        Pattern: Lower lows + lower highs = PUT signal (bearish structure)
        Pattern: Higher highs + higher lows = CALL signal (bullish structure)
        
        Mike enters on structure, not waiting for breakdown
        
        Args:
            data: DataFrame with OHLCV data
            direction: 'bearish' (PUT) or 'bullish' (CALL)
        
        Returns:
            Dict with pattern info
        """
        if len(data) < 20:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        # CRITICAL: Normalize columns to lowercase for consistent access
        data = self._normalize_columns(data)
        
        recent = data.tail(30)
        highs = recent['high'].values
        lows = recent['low'].values
        closes = recent['close'].values
        
        # Check for bearish structure (PUT setup)
        if direction == 'bearish':
            # Lower lows: Recent lows are decreasing
            recent_lows_10 = lows[-10:]
            lower_lows = np.sum(np.diff(recent_lows_10) < 0) >= 3  # At least 3 lower lows
            
            # Lower highs: Recent highs are decreasing
            recent_highs_10 = highs[-10:]
            lower_highs = np.sum(np.diff(recent_highs_10) < 0) >= 3  # At least 3 lower highs
            
            # Bearish structure: Lower lows AND lower highs
            if lower_lows and lower_highs:
                # Calculate support level (recent low)
                support_level = np.min(lows[-20:])
                current_price = closes[-1]
                
                # Check if price is near support (within 1%)
                distance_to_support = (current_price - support_level) / current_price
                
                return {
                    'detected': True,
                    'pattern_type': 'structure_based_entry',
                    'direction': 'bearish',
                    'support_level': support_level,
                    'distance_to_support': distance_to_support * 100,
                    'confidence': 0.75 if distance_to_support < 0.01 else 0.70,
                    'reason': f'Bearish structure: Lower lows + lower highs, support at ${support_level:.2f}'
                }
        
        # Check for bullish structure (CALL setup)
        elif direction == 'bullish':
            # Higher highs: Recent highs are increasing
            recent_highs_10 = highs[-10:]
            higher_highs = np.sum(np.diff(recent_highs_10) > 0) >= 3  # At least 3 higher highs
            
            # Higher lows: Recent lows are increasing
            recent_lows_10 = lows[-10:]
            higher_lows = np.sum(np.diff(recent_lows_10) > 0) >= 3  # At least 3 higher lows
            
            # Bullish structure: Higher highs AND higher lows
            if higher_highs and higher_lows:
                # Calculate resistance level (recent high)
                resistance_level = np.max(highs[-20:])
                current_price = closes[-1]
                
                # Check if price is near resistance (within 1%)
                distance_to_resistance = (resistance_level - current_price) / current_price
                
                return {
                    'detected': True,
                    'pattern_type': 'structure_based_entry',
                    'direction': 'bullish',
                    'resistance_level': resistance_level,
                    'distance_to_resistance': distance_to_resistance * 100,
                    'confidence': 0.75 if distance_to_resistance < 0.01 else 0.70,
                    'reason': f'Bullish structure: Higher highs + higher lows, resistance at ${resistance_level:.2f}'
                }
        
        return {'detected': False, 'reason': 'No structure-based entry pattern'}
    
    def detect_target_based_entry(
        self,
        data: pd.DataFrame,
        target_level: float,
        direction: str = 'bearish'
    ) -> Dict:
        """
        Detect target-based entry (Mike's pattern)
        
        Pattern: Price approaching target level with matching structure
        Example: "$676.6-$675 PT range" - enters when price approaches target
        
        Args:
            data: DataFrame with OHLCV data
            target_level: Target price level
            direction: 'bearish' (PUT) or 'bullish' (CALL)
        
        Returns:
            Dict with pattern info
        """
        if len(data) < 20:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        # CRITICAL: Normalize columns to lowercase for consistent access
        data = self._normalize_columns(data)
        
        recent = data.tail(30)
        current_price = recent['close'].iloc[-1]
        highs = recent['high'].values
        lows = recent['low'].values
        
        # Calculate distance to target
        distance_to_target = abs(current_price - target_level) / current_price
        
        # Check if price is approaching target (within 2%)
        if distance_to_target > 0.02:
            return {'detected': False, 'reason': f'Price too far from target (${target_level:.2f})'}
        
        # Check structure matches direction
        structure_match = False
        
        if direction == 'bearish':
            # Check for bearish structure (lower lows/highs)
            recent_lows_10 = lows[-10:]
            recent_highs_10 = highs[-10:]
            lower_lows = np.sum(np.diff(recent_lows_10) < 0) >= 2
            lower_highs = np.sum(np.diff(recent_highs_10) < 0) >= 2
            
            if lower_lows or lower_highs:
                structure_match = True
        else:  # bullish
            # Check for bullish structure (higher highs/lows)
            recent_highs_10 = highs[-10:]
            recent_lows_10 = lows[-10:]
            higher_highs = np.sum(np.diff(recent_highs_10) > 0) >= 2
            higher_lows = np.sum(np.diff(recent_lows_10) > 0) >= 2
            
            if higher_highs or higher_lows:
                structure_match = True
        
        if structure_match:
            return {
                'detected': True,
                'pattern_type': 'target_based_entry',
                'direction': direction,
                'target_level': target_level,
                'distance_to_target': distance_to_target * 100,
                'confidence': 0.80 if distance_to_target < 0.01 else 0.75,
                'reason': f'Price approaching target ${target_level:.2f} ({distance_to_target*100:.2f}% away) with matching structure'
            }
        
        return {'detected': False, 'reason': 'Structure does not match target direction'}
    
    def detect_lod_sweep(
        self,
        data: pd.DataFrame
    ) -> Dict:
        """
        Detect LOD (Low of Day) sweep pattern (Mike's pattern)
        
        Pattern: Price sweeps below LOD, then recovers
        Example: "LOD sweep into $674/$673"
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            Dict with pattern info
        """
        if len(data) < 30:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        # CRITICAL: Normalize columns to lowercase for consistent access
        data = self._normalize_columns(data)
        
        recent = data.tail(50)  # Need more bars to find LOD
        lows = recent['low'].values
        closes = recent['close'].values
        highs = recent['high'].values
        
        # Find LOD (lowest low of the day so far)
        lod = np.min(lows)
        lod_index = np.argmin(lows)
        
        # Check if price swept below LOD recently (last 10 bars)
        recent_lows = lows[-10:]
        swept_below = np.any(recent_lows < lod * 0.998)  # Swept 0.2% below LOD
        
        if swept_below:
            # Check if price recovered (closed above LOD)
            current_price = closes[-1]
            recovered = current_price > lod * 1.002  # Recovered 0.2% above LOD
            
            # Calculate target (LOD or slightly below)
            target = lod
            
            return {
                'detected': True,
                'pattern_type': 'lod_sweep',
                'direction': 'bearish',  # PUT setup
                'lod_level': lod,
                'target': target,
                'recovered': recovered,
                'confidence': 0.75 if not recovered else 0.70,  # Higher confidence if not recovered yet
                'reason': f'LOD sweep: Price swept below LOD ${lod:.2f}, target ${target:.2f}'
            }
        
        return {'detected': False, 'reason': 'No LOD sweep detected'}
    
    def detect_v_shape_recovery(
        self,
        data: pd.DataFrame
    ) -> Dict:
        """
        Detect V-shape recovery pattern (Mike's pattern)
        
        Pattern: Price drops, then recovers sharply (V shape)
        Example: "EOD V shape recovery back to $678/680 range"
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            Dict with pattern info
        """
        if len(data) < 30:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        # CRITICAL: Normalize columns to lowercase for consistent access
        data = self._normalize_columns(data)
        
        recent = data.tail(50)
        closes = recent['close'].values
        lows = recent['low'].values
        highs = recent['high'].values
        
        # Find recent low (bottom of V)
        recent_low = np.min(lows[-20:])
        low_index = np.argmin(lows[-20:])
        
        # Check if price dropped then recovered
        if low_index < len(closes) - 5:  # Low was at least 5 bars ago
            price_before_low = closes[-20 + low_index - 5] if low_index >= 5 else closes[-20]
            price_after_low = closes[-1]
            
            # Calculate drop and recovery
            drop_pct = (price_before_low - recent_low) / price_before_low
            recovery_pct = (price_after_low - recent_low) / recent_low
            
            # V-shape: Significant drop (>0.3%) followed by recovery (>0.2%)
            if drop_pct > 0.003 and recovery_pct > 0.002:
                    # Calculate target (resistance or previous high)
                    target = np.max(highs[-30:])
                    
                    return {
                        'detected': True,
                        'pattern_type': 'v_shape_recovery',
                        'direction': 'bullish',  # CALL setup
                        'v_bottom': recent_low_30,
                        'target': target,
                        'drop_pct': drop_pct * 100,
                        'recovery_pct': recovery_pct * 100,
                        'confidence': 0.80 if recovery_pct > 0.003 else 0.75,  # Higher confidence for stronger recovery
                        'reason': f'V-shape recovery: Dropped {drop_pct*100:.2f}% to ${recent_low_30:.2f}, recovering {recovery_pct*100:.2f}%, target ${target:.2f}'
                    }
        
        # Fallback: Check if price is recovering from recent low (simpler check)
        if len(closes) >= 10:
            recent_low_simple = np.min(lows[-10:])
            low_index_simple = np.argmin(lows[-10:])
            current_price = closes[-1]
            
            # If low was at least 3 bars ago and price is recovering
            if low_index_simple < len(closes) - 3 and current_price > recent_low_simple * 1.001:
                # Check if there was a drop before
                if low_index_simple >= 3:
                    price_before = closes[-10 + low_index_simple - 3]
                    drop = (price_before - recent_low_simple) / price_before
                    recovery = (current_price - recent_low_simple) / recent_low_simple
                    
                    if drop > 0.0003 and recovery > 0.0003:  # Very low threshold
                        target = np.max(highs[-10:])
                        return {
                            'detected': True,
                            'pattern_type': 'v_shape_recovery',
                            'direction': 'bullish',
                            'v_bottom': recent_low_simple,
                            'target': target,
                            'drop_pct': drop * 100,
                            'recovery_pct': recovery * 100,
                            'confidence': 0.70,
                            'reason': f'V-shape recovery (simple): Recovering from ${recent_low_simple:.2f}, target ${target:.2f}'
                        }
        
        return {'detected': False, 'reason': 'No V-shape recovery detected'}
    
    def calculate_price_targets(
        self,
        data: pd.DataFrame,
        pattern: Dict,
        current_price: float
    ) -> Dict:
        """
        Calculate price targets based on detected pattern (Mike's approach)
        
        Examples:
        - False breakout: Target = breakout level
        - Gap fill: Target = gap level
        - Trendline break: Target = break level ± extension
        - Rejection: Target = rejection level
        
        Args:
            data: DataFrame with OHLCV data
            pattern: Detected pattern dict
            current_price: Current price
        
        Returns:
            Dict with targets:
            {
                'target1': float,
                'target2': float,
                'strike_suggestion': float,
                'reason': str
            }
        """
        pattern_type = pattern.get('pattern_type')
        direction = pattern.get('direction', 'bearish')
        
        if pattern_type == 'false_breakout':
            breakout_level = pattern.get('breakout_level', current_price)
            if direction == 'bearish':  # PUT
                target1 = breakout_level * 0.995  # Slightly below breakout
                target2 = breakout_level * 0.99   # Further down
                strike = breakout_level * 0.995   # Strike at target
            else:  # CALL
                target1 = breakout_level * 1.005  # Slightly above breakout
                target2 = breakout_level * 1.01   # Further up
                strike = breakout_level * 1.005   # Strike at target
            
            return {
                'target1': target1,
                'target2': target2,
                'strike_suggestion': strike,
                'reason': f'Targets based on false breakout at ${breakout_level:.2f}'
            }
        
        elif pattern_type == 'gap_fill':
            gap_level = pattern.get('gap_level', current_price)
            target = gap_level
            if direction == 'bearish':  # PUT
                strike = gap_level * 0.995  # Slightly below gap
            else:  # CALL
                strike = gap_level * 1.005  # Slightly above gap
            
            return {
                'target1': target,
                'target2': target * (0.99 if direction == 'bearish' else 1.01),
                'strike_suggestion': strike,
                'reason': f'Target based on gap fill at ${gap_level:.2f}'
            }
        
        elif pattern_type == 'trendline_break':
            break_level = pattern.get('break_level', current_price)
            if direction == 'bullish':  # CALL
                target1 = break_level * 1.01   # $680/$682 range
                target2 = break_level * 1.015
                strike = break_level * 1.0015  # Between targets ($681)
            else:  # PUT
                target1 = break_level * 0.99
                target2 = break_level * 0.985
                strike = break_level * 0.995
            
            return {
                'target1': target1,
                'target2': target2,
                'strike_suggestion': strike,
                'reason': f'Targets based on trendline break at ${break_level:.2f}'
            }
        
        elif pattern_type == 'rejection':
            rejection_level = pattern.get('rejection_level', current_price)
            if direction == 'bearish':  # PUT
                target1 = rejection_level * 0.995
                target2 = rejection_level * 0.99
                strike = rejection_level * 0.995
            else:  # CALL
                target1 = rejection_level * 1.005
                target2 = rejection_level * 1.01
                strike = rejection_level * 1.005
            
            return {
                'target1': target1,
                'target2': target2,
                'strike_suggestion': strike,
                'reason': f'Targets based on rejection at ${rejection_level:.2f}'
            }
        
        elif pattern_type == 'structure_breakdown':
            breakdown_level = pattern.get('breakdown_level', current_price)
            if direction == 'bearish':  # PUT
                # Mike's logic: Target = key support level (rounded), Strike = target - $2-5
                # Mike's targets are usually round numbers: $675, $670, $669, etc.
                # The breakdown_level might be $680.30, but target should be $675 (rounded down)
                
                # Calculate target: Round breakdown level down to nearest $5 or key level
                # If breakdown is $680.30, target should be $675 (round down to $5 increment)
                # If breakdown is $678.57, target should be $675 (round down)
                # If breakdown is $673.50, target should be $670 (round down to $5)
                
                if breakdown_level >= 100:
                    # Round down to nearest $5 for key levels (e.g., $675, $670, $665)
                    target1 = (breakdown_level // 5) * 5  # Round down to nearest $5
                    # But if breakdown is close to a round number, use that
                    if abs(breakdown_level - round(breakdown_level)) < 0.5:
                        target1 = round(breakdown_level)
                    # Ensure target is not higher than breakdown
                    if target1 > breakdown_level:
                        target1 = target1 - 5
                else:
                    # For lower prices, round to nearest $0.50
                    target1 = round(breakdown_level * 2) / 2
                    if target1 > breakdown_level:
                        target1 = target1 - 0.5
                
                # Ensure target is reasonable (not too far from current price)
                # Target should be at or below current price for PUTS
                if target1 > current_price:
                    # If target is above current price, use current price rounded down
                    if current_price >= 100:
                        target1 = (current_price // 5) * 5
                    else:
                        target1 = round((current_price - 2) * 2) / 2
                
                target2 = target1 * 0.995  # Secondary target slightly below
                
                # Strike calculation: Target - $2-5 (Mike's approach)
                # Mike uses: $675 target → $672 strike ($3 below)
                # Mike uses: $670 target → $669 strike ($1 below)
                # Default: $3 below target
                strike_offset = 3.0
                
                # For specific targets, adjust offset
                if target1 == 670:
                    strike_offset = 1.0  # $670 → $669
                elif target1 == 675:
                    strike_offset = 3.0  # $675 → $672
                elif target1 < 100:
                    strike_offset = 2.0
                
                strike = target1 - strike_offset
                
                # Round to nearest $0.50 or $1.00 increment
                if target1 >= 100:
                    strike = round(strike)  # Round to nearest $1.00
                else:
                    strike = round(strike * 2) / 2  # Round to nearest $0.50
                
                # Ensure strike is reasonable (not too far from target)
                if strike < target1 - 5:
                    strike = target1 - 3  # Cap at $5 below
                    if target1 >= 100:
                        strike = round(strike)
                    else:
                        strike = round(strike * 2) / 2
            else:  # CALL
                target1 = breakdown_level
                target2 = breakdown_level * 1.005
                # For CALLS: Strike = target + $1-3
                if breakdown_level >= 100:
                    strike = breakdown_level + 2.0
                    strike = round(strike)
                else:
                    strike = breakdown_level + 1.5
                    strike = round(strike * 2) / 2
            
            return {
                'target1': target1,
                'target2': target2,
                'strike_suggestion': strike,
                'reason': f'Targets: ${target1:.2f} / ${target2:.2f}, Strike: ${strike:.2f} (target - ${strike_offset if direction == "bearish" else 2.0:.1f})'
            }
        
        elif pattern_type == 'momentum_shift':
            # Use target-based logic for momentum shifts
            # For downward momentum: Target = current price - extension, Strike = target - $2-5
            if direction == 'bearish':  # PUT
                # Target: Extend current move down by 0.5-1%
                target1 = current_price * 0.995  # 0.5% down
                target2 = current_price * 0.99   # 1% down
                # Strike: Target - $2-5
                if current_price >= 100:
                    strike = target1 - 3.0  # $3 below target
                    strike = round(strike)
                else:
                    strike = target1 - 2.0
                    strike = round(strike * 2) / 2
            else:  # CALL
                target1 = current_price * 1.005  # 0.5% up
                target2 = current_price * 1.01   # 1% up
                if current_price >= 100:
                    strike = target1 + 2.0  # $2 above target
                    strike = round(strike)
                else:
                    strike = target1 + 1.5
                    strike = round(strike * 2) / 2
            
            return {
                'target1': target1,
                'target2': target2,
                'strike_suggestion': strike,
                'reason': f'Targets based on momentum shift: ${target1:.2f} / ${target2:.2f}'
            }
        
        # Default: Use current price ± small offset
        if direction == 'bearish':  # PUT
            strike = current_price * 0.995
            target1 = current_price * 0.99
        else:  # CALL
            strike = current_price * 1.005
            target1 = current_price * 1.01
        
        return {
            'target1': target1,
            'target2': target1 * (0.99 if direction == 'bearish' else 1.01),
            'strike_suggestion': strike,
            'reason': 'Default targets (no specific pattern)'
        }
    
    def detect_structure_breakdown(
        self,
        data: pd.DataFrame,
        breakdown_level: Optional[float] = None
    ) -> Dict:
        """
        Detect structure breakdown (Mike's pattern)
        
        Pattern: Price breaks below key structure level (e.g., $673.5 breakdown)
        Example: "$673.5 breakdown is essential for downside"
        
        Args:
            data: DataFrame with OHLCV data
            breakdown_level: Optional breakdown level to check
        
        Returns:
            Dict with pattern info
        """
        if len(data) < 30:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        # CRITICAL: Normalize columns to lowercase for consistent access
        data = self._normalize_columns(data)
        
        recent = data.tail(30)
        current_price = recent['close'].iloc[-1]
        current_low = recent['low'].iloc[-1]
        current_high = recent['high'].iloc[-1]
        
        # Calculate structure levels
        highs = recent['high'].values
        lows = recent['low'].values
        closes = recent['close'].values
        
        # Find key support levels (recent lows that held)
        # Look for levels where price bounced multiple times
        support_levels = []
        for i in range(len(lows) - 10, len(lows) - 1):
            level = lows[i]
            touches = sum(1 for j in range(i, len(lows)) if abs(lows[j] - level) < 0.5)
            if touches >= 2:
                support_levels.append(level)
        
        if support_levels:
            key_support = max(support_levels)  # Highest support level
        else:
            key_support = np.min(lows[-20:])
        
        # Check if price is breaking below key support
        if breakdown_level:
            key_level = breakdown_level
        else:
            key_level = key_support
        
        # Check for breakdown
        if current_low < key_level * 0.998 and current_price < key_level * 1.002:
            # Price is breaking below key level
            return {
                'detected': True,
                'pattern_type': 'structure_breakdown',
                'direction': 'bearish',  # PUT setup
                'breakdown_level': key_level,
                'confidence': 0.75,
                'reason': f'Structure breakdown below ${key_level:.2f}'
            }
        
        # Check for momentum shift (price starting to move down)
        price_change_5 = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        price_change_10 = (closes[-1] - closes[-10]) / closes[-10] if len(closes) >= 10 else 0
        
        # If price is declining and approaching support
        if price_change_5 < -0.001 and price_change_10 < -0.002:  # Downward momentum
            if current_price < key_level * 1.01:  # Near key level
                return {
                    'detected': True,
                    'pattern_type': 'structure_breakdown',
                    'direction': 'bearish',
                    'breakdown_level': key_level,
                    'confidence': 0.70,
                    'reason': f'Momentum shift down, approaching breakdown at ${key_level:.2f}'
                }
        
        return {'detected': False, 'reason': 'No structure breakdown detected'}
    
    def detect_momentum_shift(
        self,
        data: pd.DataFrame,
        direction: str = 'down'
    ) -> Dict:
        """
        Detect momentum shift (Mike's pattern)
        
        Pattern: Price momentum shifting in a direction
        Example: "Breakdown after 10:30" - momentum shifting down
        
        Args:
            data: DataFrame with OHLCV data
            direction: 'up' or 'down'
        
        Returns:
            Dict with pattern info
        """
        if len(data) < 20:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        # CRITICAL: Normalize columns to lowercase for consistent access
        data = self._normalize_columns(data)
        
        recent = data.tail(20)
        closes = recent['close'].values
        
        # Calculate momentum
        momentum_5 = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        momentum_10 = (closes[-1] - closes[-10]) / closes[-10] if len(closes) >= 10 else 0
        momentum_20 = (closes[-1] - closes[-20]) / closes[-20] if len(closes) >= 20 else 0
        
        if direction == 'down':
            # Check for downward momentum shift - LOWERED THRESHOLD (0.01% instead of 0.1%)
            if momentum_5 < -0.0001 and momentum_10 < -0.0002:
                return {
                    'detected': True,
                    'pattern_type': 'momentum_shift',
                    'direction': 'bearish',
                    'momentum_5': momentum_5 * 100,
                    'momentum_10': momentum_10 * 100,
                    'confidence': 0.70,
                    'reason': f'Downward momentum shift: {momentum_5*100:.2f}% (5 bars), {momentum_10*100:.2f}% (10 bars)'
                }
        else:  # up
            # Check for upward momentum shift - LOWERED THRESHOLD (0.01% instead of 0.1%)
            if momentum_5 > 0.0001 and momentum_10 > 0.0002:
                return {
                    'detected': True,
                    'pattern_type': 'momentum_shift',
                    'direction': 'bullish',
                    'momentum_5': momentum_5 * 100,
                    'momentum_10': momentum_10 * 100,
                    'confidence': 0.70,
                    'reason': f'Upward momentum shift: {momentum_5*100:.2f}% (5 bars), {momentum_10*100:.2f}% (10 bars)'
                }
        
        return {'detected': False, 'reason': 'No momentum shift detected'}
    
    def analyze_symbol(
        self,
        data: pd.DataFrame,
        symbol: str,
        current_price: float,
        target_levels: Optional[List[float]] = None
    ) -> Dict:
        """
        Complete technical analysis for a symbol
        
        Detects all patterns and calculates targets
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol name
            current_price: Current price
        
        Returns:
            Dict with all detected patterns and recommendations:
            {
                'patterns': List[Dict],
                'best_pattern': Dict,
                'targets': Dict,
                'strike_suggestion': float,
                'confidence_boost': float,
                'recommended_action': 'CALL' or 'PUT' or 'HOLD'
            }
        """
        # CRITICAL FIX: Normalize columns ONCE at entry point
        # This ensures all detection methods receive lowercase columns
        data = self._normalize_columns(data)
        
        if data is None or len(data) == 0:
            return {
                'patterns': [],
                'best_pattern': None,
                'targets': {},
                'strike_suggestion': None,
                'confidence_boost': 0.0,
                'recommended_action': 'HOLD',
                'reason': 'No data available for analysis'
            }
        
        patterns = []
        
        # Detect all patterns
        false_breakout = self.detect_false_breakout(data)
        if false_breakout.get('detected'):
            patterns.append(false_breakout)
        
        gap_fill = self.detect_gap_fill(data)
        if gap_fill.get('detected'):
            patterns.append(gap_fill)
        
        trendline_break = self.detect_trendline_break(data)
        if trendline_break.get('detected'):
            patterns.append(trendline_break)
        
        rejection = self.detect_rejection(data)
        if rejection.get('detected'):
            patterns.append(rejection)
        
        # NEW: Structure breakdown detection
        structure_breakdown = self.detect_structure_breakdown(data)
        if structure_breakdown.get('detected'):
            patterns.append(structure_breakdown)
        
        # NEW: Momentum shift detection
        momentum_shift_down = self.detect_momentum_shift(data, direction='down')
        if momentum_shift_down.get('detected'):
            patterns.append(momentum_shift_down)
        
        momentum_shift_up = self.detect_momentum_shift(data, direction='up')
        if momentum_shift_up.get('detected'):
            patterns.append(momentum_shift_up)
        
        # NEW: V-shape recovery detection (check first - higher priority)
        v_shape = self.detect_v_shape_recovery(data)
        if v_shape.get('detected'):
            patterns.append(v_shape)
        
        # NEW: LOD sweep detection (check before structure - specialized pattern)
        lod_sweep = self.detect_lod_sweep(data)
        if lod_sweep.get('detected'):
            patterns.append(lod_sweep)
        
        # NEW: Structure-based entry detection (PROACTIVE)
        # Only detect if no specialized patterns found
        if not v_shape.get('detected') and not lod_sweep.get('detected'):
            structure_entry_bearish = self.detect_structure_based_entry(data, direction='bearish')
            if structure_entry_bearish.get('detected'):
                patterns.append(structure_entry_bearish)
            
            structure_entry_bullish = self.detect_structure_based_entry(data, direction='bullish')
            if structure_entry_bullish.get('detected'):
                patterns.append(structure_entry_bullish)
        else:
            # Still check structure, but with lower priority
            structure_entry_bearish = self.detect_structure_based_entry(data, direction='bearish')
            if structure_entry_bearish.get('detected'):
                patterns.append(structure_entry_bearish)
            
            structure_entry_bullish = self.detect_structure_based_entry(data, direction='bullish')
            if structure_entry_bullish.get('detected'):
                patterns.append(structure_entry_bullish)
        
        # NEW: Target-based entry detection (if target levels provided)
        # Note: Target levels can be passed, but we also auto-detect from structure
        # For now, we'll detect structure-based entries which are proactive
        
        # Find best pattern (prioritize specialized patterns over general structure)
        best_pattern = None
        if patterns:
            # Priority order: Specialized patterns > Structure-based > Others
            pattern_priority = {
                'v_shape_recovery': 10,  # Highest priority
                'lod_sweep': 9,
                'target_based_entry': 8,
                'structure_breakdown': 7,
                'trendline_break': 6,
                'false_breakout': 5,
                'gap_fill': 4,
                'rejection': 3,
                'structure_based_entry': 2,  # Lower priority (general)
                'momentum_shift': 1
            }
            
            # Sort by priority first, then confidence
            def pattern_score(p):
                pattern_type = p.get('pattern_type', '')
                priority = pattern_priority.get(pattern_type, 0)
                confidence = p.get('confidence', 0)
                return (priority, confidence)
            
            best_pattern = max(patterns, key=pattern_score)
        
        # Calculate targets if pattern detected
        targets = None
        strike_suggestion = None
        if best_pattern:
            targets = self.calculate_price_targets(data, best_pattern, current_price)
            strike_suggestion = targets.get('strike_suggestion')
        
        # Calculate confidence boost
        confidence_boost = 0.0
        if best_pattern:
            # Boost confidence based on pattern
            pattern_confidence = best_pattern.get('confidence', 0)
            confidence_boost = (pattern_confidence - 0.50) * 0.6  # Scale to 0-0.30 boost
        
        # Determine recommended action
        recommended_action = 'HOLD'
        if best_pattern:
            direction = best_pattern.get('direction')
            if direction == 'bullish':
                recommended_action = 'CALL'
            elif direction == 'bearish':
                recommended_action = 'PUT'
        
        return {
            'patterns': patterns,
            'best_pattern': best_pattern,
            'targets': targets,
            'strike_suggestion': strike_suggestion,
            'confidence_boost': confidence_boost,
            'recommended_action': recommended_action,
            'analysis_time': datetime.now()
        }

