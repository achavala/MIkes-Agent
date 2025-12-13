"""
MULTI-AGENT ENSEMBLE SYSTEM

Specialized agents for different market regimes:
1. Trend Agent - Follows momentum and trends
2. Reversal Agent - Mean reversion and contrarian signals
3. Volatility Breakout Agent - Volatility expansion and breakouts
4. Meta-Policy Router - Combines agent signals with dynamic weighting

Author: Mike Agent Institutional Upgrade
Date: December 13, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


class AgentType(Enum):
    """Types of specialized agents"""
    TREND = "trend"
    REVERSAL = "reversal"
    VOLATILITY = "volatility"
    GAMMA_MODEL = "gamma_model"
    DELTA_HEDGING = "delta_hedging"
    MACRO = "macro"


class AgentSignal:
    """Signal from a specialized agent"""
    def __init__(
        self,
        agent_type: AgentType,
        action: int,
        confidence: float,
        strength: float,
        reasoning: str = ""
    ):
        self.agent_type = agent_type
        self.action = action  # 0=HOLD, 1=BUY_CALL, 2=BUY_PUT
        self.confidence = confidence  # 0.0 to 1.0
        self.strength = strength  # Signal strength (-1.0 to 1.0)
        self.reasoning = reasoning
        self.timestamp = datetime.now()
    
    def __repr__(self):
        return f"{self.agent_type.value.upper()}: action={self.action}, conf={self.confidence:.2f}, strength={self.strength:.2f}"


class TrendAgent:
    """
    Trend-following agent that identifies and follows momentum
    
    Strategy:
    - Strong uptrend → BUY CALL
    - Strong downtrend → BUY PUT
    - Weak/no trend → HOLD
    
    Indicators:
    - EMA crossovers (9/20, 20/50)
    - MACD momentum
    - Price above/below moving averages
    - Trend strength (ADX-like)
    """
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.name = "TrendAgent"
    
    def analyze(self, data: pd.DataFrame, vix: float = 20.0) -> AgentSignal:
        """
        Analyze market data for trend signals
        
        Args:
            data: DataFrame with OHLCV data (last lookback bars)
            vix: Current VIX level
            
        Returns:
            AgentSignal with action and confidence
        """
        if len(data) < self.lookback:
            return AgentSignal(AgentType.TREND, 0, 0.0, 0.0, "Insufficient data")
        
        # Get recent data
        recent = data.tail(self.lookback).copy()
        closes = recent['close'].values if 'close' in recent.columns else recent['Close'].values
        highs = recent['high'].values if 'high' in recent.columns else recent['High'].values
        lows = recent['low'].values if 'low' in recent.columns else recent['Low'].values
        
        # Calculate EMAs
        def ema(arr, span):
            return pd.Series(arr).ewm(span=span, adjust=False).mean().values
        
        ema9 = ema(closes, 9)
        ema20 = ema(closes, 20)
        ema50 = ema(closes, 50) if len(closes) >= 50 else ema20
        
        # Trend indicators
        current_price = closes[-1]
        price_above_ema9 = current_price > ema9[-1]
        price_above_ema20 = current_price > ema20[-1]
        ema9_above_ema20 = ema9[-1] > ema20[-1]
        ema20_above_ema50 = ema20[-1] > ema50[-1]
        
        # MACD
        macd_line = ema(closes, 12) - ema(closes, 26)
        signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
        macd_hist = macd_line - signal_line
        
        # Trend strength (ADX-like)
        high_low = highs - lows
        high_close = np.abs(highs - np.roll(closes, 1))
        low_close = np.abs(lows - np.roll(closes, 1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = pd.Series(true_range).rolling(14, min_periods=1).mean().values[-1]
        
        # Price momentum
        price_change = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0.0
        momentum = price_change * 100  # Percentage
        
        # Trend score calculation
        bullish_signals = 0
        bearish_signals = 0
        
        if price_above_ema9:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if price_above_ema20:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if ema9_above_ema20:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if ema20_above_ema50:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if macd_hist[-1] > 0:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if momentum > 0.5:  # Strong upward momentum
            bullish_signals += 2
        elif momentum < -0.5:  # Strong downward momentum
            bearish_signals += 2
        
        # Calculate trend strength
        total_signals = bullish_signals + bearish_signals
        trend_strength = (bullish_signals - bearish_signals) / max(total_signals, 1)
        
        # Determine action
        if trend_strength > 0.3:  # Bullish trend
            action = 1  # BUY CALL
            confidence = min(abs(trend_strength), 0.95)
            reasoning = f"Bullish trend: {bullish_signals}/{total_signals} signals, momentum={momentum:.2f}%"
        elif trend_strength < -0.3:  # Bearish trend
            action = 2  # BUY PUT
            confidence = min(abs(trend_strength), 0.95)
            reasoning = f"Bearish trend: {bearish_signals}/{total_signals} signals, momentum={momentum:.2f}%"
        else:  # No clear trend
            action = 0  # HOLD
            confidence = 1.0 - abs(trend_strength)  # Low confidence in trend
            reasoning = f"No clear trend: strength={trend_strength:.2f}"
        
        return AgentSignal(
            agent_type=AgentType.TREND,
            action=action,
            confidence=confidence,
            strength=trend_strength,
            reasoning=reasoning
        )


class ReversalAgent:
    """
    Mean-reversion agent that identifies overbought/oversold conditions
    
    Strategy:
    - Overbought (RSI > 70, price far above MA) → BUY PUT (expect reversal down)
    - Oversold (RSI < 30, price far below MA) → BUY CALL (expect reversal up)
    - Neutral → HOLD
    
    Indicators:
    - RSI (14-period)
    - Price distance from moving averages
    - Bollinger Bands
    - Stochastic oscillator
    """
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.name = "ReversalAgent"
    
    def analyze(self, data: pd.DataFrame, vix: float = 20.0) -> AgentSignal:
        """
        Analyze market data for reversal signals
        
        Args:
            data: DataFrame with OHLCV data
            vix: Current VIX level
            
        Returns:
            AgentSignal with action and confidence
        """
        if len(data) < self.lookback:
            return AgentSignal(AgentType.REVERSAL, 0, 0.0, 0.0, "Insufficient data")
        
        recent = data.tail(self.lookback).copy()
        closes = recent['close'].values if 'close' in recent.columns else recent['Close'].values
        highs = recent['high'].values if 'high' in recent.columns else recent['High'].values
        lows = recent['low'].values if 'low' in recent.columns else recent['Low'].values
        
        # RSI calculation
        delta = np.diff(closes, prepend=closes[0])
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        
        def ema(arr, span):
            return pd.Series(arr).ewm(alpha=1/span, adjust=False).mean().values
        
        avg_up = ema(up, 14)
        avg_down = ema(down, 14)
        rs = avg_up / (avg_down + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi[-1]
        
        # Moving averages
        sma20 = pd.Series(closes).rolling(20, min_periods=1).mean().values[-1]
        sma50 = pd.Series(closes).rolling(min(50, len(closes)), min_periods=1).mean().values[-1]
        
        current_price = closes[-1]
        
        # Price distance from MAs
        dist_from_sma20 = (current_price - sma20) / sma20 * 100
        dist_from_sma50 = (current_price - sma50) / sma50 * 100 if sma50 > 0 else 0
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2.0
        sma_bb = pd.Series(closes).rolling(bb_period, min_periods=1).mean().values[-1]
        std_bb = pd.Series(closes).rolling(bb_period, min_periods=1).std().values[-1]
        bb_upper = sma_bb + (bb_std * std_bb)
        bb_lower = sma_bb - (bb_std * std_bb)
        
        # Stochastic oscillator
        period = 14
        lowest_low = pd.Series(lows).rolling(period, min_periods=1).min().values[-1]
        highest_high = pd.Series(highs).rolling(period, min_periods=1).max().values[-1]
        stoch_k = ((current_price - lowest_low) / (highest_high - lowest_low + 1e-9)) * 100
        
        # Reversal signals
        oversold_signals = 0
        overbought_signals = 0
        
        # RSI signals
        if current_rsi < 30:
            oversold_signals += 2  # Strong oversold
        elif current_rsi < 40:
            oversold_signals += 1  # Mild oversold
        
        if current_rsi > 70:
            overbought_signals += 2  # Strong overbought
        elif current_rsi > 60:
            overbought_signals += 1  # Mild overbought
        
        # Price distance signals
        if dist_from_sma20 < -2.0:  # Price 2% below SMA20
            oversold_signals += 1
        elif dist_from_sma20 > 2.0:  # Price 2% above SMA20
            overbought_signals += 1
        
        if dist_from_sma50 < -3.0:  # Price 3% below SMA50
            oversold_signals += 1
        elif dist_from_sma50 > 3.0:  # Price 3% above SMA50
            overbought_signals += 1
        
        # Bollinger Bands
        if current_price < bb_lower:
            oversold_signals += 1
        elif current_price > bb_upper:
            overbought_signals += 1
        
        # Stochastic
        if stoch_k < 20:
            oversold_signals += 1
        elif stoch_k > 80:
            overbought_signals += 1
        
        # Calculate reversal strength
        total_signals = oversold_signals + overbought_signals
        if total_signals == 0:
            reversal_strength = 0.0
        else:
            reversal_strength = (oversold_signals - overbought_signals) / max(total_signals, 1)
        
        # Determine action
        if reversal_strength > 0.4:  # Oversold - expect reversal up
            action = 1  # BUY CALL
            confidence = min(abs(reversal_strength) * 1.2, 0.95)
            reasoning = f"Oversold reversal: RSI={current_rsi:.1f}, signals={oversold_signals}/{total_signals}"
        elif reversal_strength < -0.4:  # Overbought - expect reversal down
            action = 2  # BUY PUT
            confidence = min(abs(reversal_strength) * 1.2, 0.95)
            reasoning = f"Overbought reversal: RSI={current_rsi:.1f}, signals={overbought_signals}/{total_signals}"
        else:  # Neutral
            action = 0  # HOLD
            confidence = 1.0 - abs(reversal_strength)
            reasoning = f"No reversal signal: RSI={current_rsi:.1f}, strength={reversal_strength:.2f}"
        
        return AgentSignal(
            agent_type=AgentType.REVERSAL,
            action=action,
            confidence=confidence,
            strength=reversal_strength,
            reasoning=reasoning
        )


class VolatilityBreakoutAgent:
    """
    Volatility breakout agent that identifies expansion and breakouts
    
    Strategy:
    - Volatility expansion + upward breakout → BUY CALL
    - Volatility expansion + downward breakout → BUY PUT
    - Low volatility / no breakout → HOLD
    
    Indicators:
    - ATR (Average True Range)
    - Bollinger Band width
    - Price breakouts from ranges
    - VIX levels
    - Volume spikes
    """
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.name = "VolatilityBreakoutAgent"
    
    def analyze(self, data: pd.DataFrame, vix: float = 20.0) -> AgentSignal:
        """
        Analyze market data for volatility breakout signals
        
        Args:
            data: DataFrame with OHLCV data
            vix: Current VIX level
            
        Returns:
            AgentSignal with action and confidence
        """
        if len(data) < self.lookback:
            return AgentSignal(AgentType.VOLATILITY, 0, 0.0, 0.0, "Insufficient data")
        
        recent = data.tail(self.lookback).copy()
        closes = recent['close'].values if 'close' in recent.columns else recent['Close'].values
        highs = recent['high'].values if 'high' in recent.columns else recent['High'].values
        lows = recent['low'].values if 'low' in recent.columns else recent['Low'].values
        volumes = recent['volume'].values if 'volume' in recent.columns else recent['Volume'].values
        
        current_price = closes[-1]
        
        # ATR calculation
        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]
        high_low = highs - lows
        high_close = np.abs(highs - prev_close)
        low_close = np.abs(lows - prev_close)
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = pd.Series(true_range).rolling(14, min_periods=1).mean().values
        current_atr = atr[-1]
        avg_atr = np.mean(atr[-10:]) if len(atr) >= 10 else current_atr
        
        # Volatility expansion
        atr_expansion = current_atr / (avg_atr + 1e-9) if avg_atr > 0 else 1.0
        
        # Bollinger Bands width
        bb_period = 20
        sma_bb = pd.Series(closes).rolling(bb_period, min_periods=1).mean().values
        std_bb = pd.Series(closes).rolling(bb_period, min_periods=1).std().values
        bb_width = (std_bb * 4) / sma_bb * 100  # Width as % of price
        current_bb_width = bb_width[-1]
        avg_bb_width = np.mean(bb_width[-10:]) if len(bb_width) >= 10 else current_bb_width
        
        # Range identification
        lookback_range = 20
        range_high = pd.Series(highs).rolling(lookback_range, min_periods=1).max().values[-1]
        range_low = pd.Series(lows).rolling(lookback_range, min_periods=1).min().values[-1]
        range_mid = (range_high + range_low) / 2
        range_size = range_high - range_low
        
        # Breakout detection
        breakout_up = current_price > range_high * 1.001  # 0.1% above range high
        breakout_down = current_price < range_low * 0.999  # 0.1% below range low
        
        # Volume analysis
        avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else volumes[-1]
        volume_spike = volumes[-1] / (avg_volume + 1e-9) if avg_volume > 0 else 1.0
        
        # VIX analysis
        vix_expansion = vix > 25  # High VIX = volatility expansion
        
        # Breakout signals
        bullish_breakout_signals = 0
        bearish_breakout_signals = 0
        
        # Volatility expansion
        if atr_expansion > 1.3:  # 30% above average ATR
            bullish_breakout_signals += 1
            bearish_breakout_signals += 1  # Volatility can go either way
        elif atr_expansion < 0.7:  # Low volatility
            # No breakout signals in low volatility
            pass
        
        # BB width expansion
        if current_bb_width > avg_bb_width * 1.2:  # 20% wider than average
            bullish_breakout_signals += 1
            bearish_breakout_signals += 1
        
        # Breakout direction
        if breakout_up:
            bullish_breakout_signals += 3  # Strong upward breakout
        elif breakout_down:
            bearish_breakout_signals += 3  # Strong downward breakout
        
        # Volume confirmation
        if volume_spike > 1.5:  # 50% above average volume
            if breakout_up:
                bullish_breakout_signals += 2
            elif breakout_down:
                bearish_breakout_signals += 2
        
        # VIX confirmation
        if vix_expansion:
            bullish_breakout_signals += 1
            bearish_breakout_signals += 1
        
        # Calculate breakout strength
        total_signals = bullish_breakout_signals + bearish_breakout_signals
        if total_signals == 0:
            breakout_strength = 0.0
        else:
            breakout_strength = (bullish_breakout_signals - bearish_breakout_signals) / max(total_signals, 1)
        
        # Determine action
        if breakout_strength > 0.3 and atr_expansion > 1.2:  # Upward breakout with volatility
            action = 1  # BUY CALL
            confidence = min(abs(breakout_strength) * 1.1, 0.95)
            reasoning = f"Upward breakout: ATR expansion={atr_expansion:.2f}x, signals={bullish_breakout_signals}/{total_signals}"
        elif breakout_strength < -0.3 and atr_expansion > 1.2:  # Downward breakout with volatility
            action = 2  # BUY PUT
            confidence = min(abs(breakout_strength) * 1.1, 0.95)
            reasoning = f"Downward breakout: ATR expansion={atr_expansion:.2f}x, signals={bearish_breakout_signals}/{total_signals}"
        else:  # No clear breakout
            action = 0  # HOLD
            confidence = 1.0 - abs(breakout_strength)
            reasoning = f"No breakout: ATR expansion={atr_expansion:.2f}x, strength={breakout_strength:.2f}"
        
        return AgentSignal(
            agent_type=AgentType.VOLATILITY,
            action=action,
            confidence=confidence,
            strength=breakout_strength,
            reasoning=reasoning
        )


class GammaModelAgent:
    """
    Gamma model agent that analyzes gamma exposure and convexity
    
    Strategy:
    - High gamma + positive momentum → BUY (gamma acceleration)
    - High gamma + negative momentum → BUY PUT (gamma acceleration)
    - Low gamma → HOLD (not enough convexity)
    
    Indicators:
    - Gamma exposure (GEX)
    - Gamma profile analysis
    - Convexity opportunities
    - Strike concentration
    """
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.name = "GammaModelAgent"
    
    def analyze(
        self,
        data: pd.DataFrame,
        vix: float = 20.0,
        current_price: float = None,
        strike: float = None
    ) -> AgentSignal:
        """
        Analyze gamma exposure and convexity
        
        Args:
            data: Market data DataFrame
            vix: Current VIX level
            current_price: Current underlying price (optional)
            strike: Option strike price (optional)
            
        Returns:
            AgentSignal with action and confidence
        """
        if len(data) < self.lookback:
            return AgentSignal(AgentType.GAMMA_MODEL, 0, 0.0, 0.0, "Insufficient data")
        
        recent = data.tail(self.lookback).copy()
        closes = recent['close'].values if 'close' in recent.columns else recent['Close'].values
        
        if current_price is None:
            current_price = closes[-1]
        if strike is None:
            strike = round(current_price)  # ATM
        
        # Calculate time to expiration (0DTE = ~1 hour)
        T = 1.0 / (252 * 6.5)  # ~1 hour
        
        # Estimate IV from VIX
        sigma = (vix / 100.0) * 1.3 if vix > 0 else 0.20
        
        # Calculate Greeks using Black-Scholes
        from scipy.stats import norm
        r = 0.04
        
        d1 = (np.log(current_price / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (current_price * sigma * np.sqrt(T))
        
        # Gamma exposure proxy (simplified)
        # High gamma = more convexity = better for momentum trades
        gamma_threshold_high = 0.05  # High gamma threshold
        gamma_threshold_low = 0.01   # Low gamma threshold
        
        # Price momentum
        price_change = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0.0
        momentum = price_change * 100  # Percentage
        
        # Gamma-based signals
        if gamma > gamma_threshold_high:
            # High gamma = high convexity
            if momentum > 0.3:  # Positive momentum
                action = 1  # BUY CALL (gamma acceleration on upside)
                confidence = min(gamma / 0.1, 0.95)  # Scale by gamma
                strength = momentum / 2.0  # Momentum strength
                reasoning = f"High gamma ({gamma:.4f}) + positive momentum ({momentum:.2f}%) → Gamma acceleration"
            elif momentum < -0.3:  # Negative momentum
                action = 2  # BUY PUT (gamma acceleration on downside)
                confidence = min(gamma / 0.1, 0.95)
                strength = abs(momentum) / 2.0
                reasoning = f"High gamma ({gamma:.4f}) + negative momentum ({momentum:.2f}%) → Gamma acceleration"
            else:
                action = 0  # HOLD (high gamma but no momentum)
                confidence = 0.3
                strength = 0.0
                reasoning = f"High gamma ({gamma:.4f}) but no clear momentum ({momentum:.2f}%)"
        elif gamma < gamma_threshold_low:
            # Low gamma = low convexity
            action = 0  # HOLD
            confidence = 0.7
            strength = 0.0
            reasoning = f"Low gamma ({gamma:.4f}) → Insufficient convexity for gamma trade"
        else:
            # Medium gamma
            if abs(momentum) > 0.5:  # Strong momentum
                action = 1 if momentum > 0 else 2
                confidence = 0.6
                strength = abs(momentum) / 2.0
                reasoning = f"Medium gamma ({gamma:.4f}) + strong momentum ({momentum:.2f}%)"
            else:
                action = 0
                confidence = 0.5
                strength = 0.0
                reasoning = f"Medium gamma ({gamma:.4f}), weak momentum ({momentum:.2f}%)"
        
        return AgentSignal(
            agent_type=AgentType.GAMMA_MODEL,
            action=action,
            confidence=confidence,
            strength=strength,
            reasoning=reasoning
        )


class DeltaHedgingAgent:
    """
    Delta hedging agent that manages directional exposure
    
    Strategy:
    - Portfolio delta too high → Hedge with opposite direction
    - Portfolio delta neutral → No hedge needed
    - Delta imbalance → Adjust position
    
    Indicators:
    - Portfolio delta
    - Delta exposure limits
    - Hedging opportunities
    """
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.name = "DeltaHedgingAgent"
    
    def analyze(
        self,
        data: pd.DataFrame,
        vix: float = 20.0,
        portfolio_delta: float = 0.0,
        delta_limit: float = 2000.0,
        current_price: float = None,
        strike: float = None
    ) -> AgentSignal:
        """
        Analyze delta exposure and suggest hedging
        
        Args:
            data: Market data DataFrame
            vix: Current VIX level
            portfolio_delta: Current portfolio delta exposure
            delta_limit: Maximum allowed delta exposure
            current_price: Current underlying price
            strike: Option strike price
            
        Returns:
            AgentSignal with hedging recommendation
        """
        if len(data) < self.lookback:
            return AgentSignal(AgentType.DELTA_HEDGING, 0, 0.0, 0.0, "Insufficient data")
        
        recent = data.tail(self.lookback).copy()
        closes = recent['close'].values if 'close' in recent.columns else recent['Close'].values
        
        if current_price is None:
            current_price = closes[-1]
        if strike is None:
            strike = round(current_price)
        
        # Calculate option delta
        from scipy.stats import norm
        T = 1.0 / (252 * 6.5)
        r = 0.04
        sigma = (vix / 100.0) * 1.3 if vix > 0 else 0.20
        
        d1 = (np.log(current_price / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        call_delta = norm.cdf(d1)
        put_delta = norm.cdf(d1) - 1.0
        
        # Delta exposure analysis
        abs_portfolio_delta = abs(portfolio_delta)
        delta_utilization = abs_portfolio_delta / delta_limit if delta_limit > 0 else 0.0
        
        # Hedging signals
        if delta_utilization > 0.8:  # High delta exposure
            # Need to hedge
            if portfolio_delta > 0:  # Long delta → Hedge with PUT
                action = 2  # BUY PUT to hedge
                confidence = min(delta_utilization, 0.95)
                strength = delta_utilization - 0.8  # Strength based on over-limit
                reasoning = f"High long delta ({portfolio_delta:.0f}, {delta_utilization:.0%} of limit) → Hedge with PUT"
            else:  # Short delta → Hedge with CALL
                action = 1  # BUY CALL to hedge
                confidence = min(delta_utilization, 0.95)
                strength = delta_utilization - 0.8
                reasoning = f"High short delta ({portfolio_delta:.0f}, {delta_utilization:.0%} of limit) → Hedge with CALL"
        elif delta_utilization > 0.5:  # Medium delta exposure
            # Consider hedging
            if portfolio_delta > 0:
                action = 2  # Consider PUT hedge
            else:
                action = 1  # Consider CALL hedge
            confidence = 0.4
            strength = (delta_utilization - 0.5) * 2  # Scale 0.5-0.8 to 0-0.6
            reasoning = f"Medium delta exposure ({portfolio_delta:.0f}, {delta_utilization:.0%} of limit)"
        else:  # Low delta exposure
            # No hedge needed
            action = 0  # HOLD
            confidence = 0.6
            strength = 0.0
            reasoning = f"Low delta exposure ({portfolio_delta:.0f}, {delta_utilization:.0%} of limit) → No hedge needed"
        
        return AgentSignal(
            agent_type=AgentType.DELTA_HEDGING,
            action=action,
            confidence=confidence,
            strength=strength,
            reasoning=reasoning
        )


class MacroAgent:
    """
    Risk-on/Risk-off macro agent that analyzes market regime
    
    Strategy:
    - Risk-on (VIX low, SPY up, bonds down) → BUY CALL
    - Risk-off (VIX high, SPY down, bonds up) → BUY PUT
    - Neutral → HOLD
    
    Indicators:
    - VIX level
    - SPY momentum
    - Market breadth
    - Risk sentiment
    """
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.name = "MacroAgent"
    
    def analyze(
        self,
        data: pd.DataFrame,
        vix: float = 20.0,
        spy_data: pd.DataFrame = None
    ) -> AgentSignal:
        """
        Analyze risk-on/risk-off regime
        
        Args:
            data: Market data DataFrame (can be SPY or other)
            vix: Current VIX level
            spy_data: SPY data if available (for cross-validation)
            
        Returns:
            AgentSignal with macro regime signal
        """
        if len(data) < self.lookback:
            return AgentSignal(AgentType.MACRO, 0, 0.0, 0.0, "Insufficient data")
        
        recent = data.tail(self.lookback).copy()
        closes = recent['close'].values if 'close' in recent.columns else recent['Close'].values
        
        # VIX-based regime
        vix_low = vix < 18  # Risk-on
        vix_high = vix > 25  # Risk-off
        vix_neutral = 18 <= vix <= 25
        
        # Price momentum
        price_change = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0.0
        momentum = price_change * 100
        
        # Trend strength
        ema9 = pd.Series(closes).ewm(span=9, adjust=False).mean().values[-1]
        ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().values[-1]
        trend_up = ema9 > ema20
        trend_strength = abs((ema9 - ema20) / ema20) if ema20 > 0 else 0.0
        
        # Risk-on/risk-off signals
        risk_on_signals = 0
        risk_off_signals = 0
        
        # VIX signals
        if vix_low:
            risk_on_signals += 2  # Strong risk-on
        elif vix_high:
            risk_off_signals += 2  # Strong risk-off
        
        # Momentum signals
        if momentum > 0.5 and trend_up:
            risk_on_signals += 2  # Strong upward momentum
        elif momentum < -0.5 and not trend_up:
            risk_off_signals += 2  # Strong downward momentum
        
        # Trend signals
        if trend_up and trend_strength > 0.01:
            risk_on_signals += 1
        elif not trend_up and trend_strength > 0.01:
            risk_off_signals += 1
        
        # Calculate macro strength
        total_signals = risk_on_signals + risk_off_signals
        if total_signals == 0:
            macro_strength = 0.0
        else:
            macro_strength = (risk_on_signals - risk_off_signals) / max(total_signals, 1)
        
        # Determine action
        if macro_strength > 0.3:  # Risk-on
            action = 1  # BUY CALL
            confidence = min(abs(macro_strength) * 1.2, 0.95)
            reasoning = f"Risk-ON: VIX={vix:.1f}, momentum={momentum:.2f}%, signals={risk_on_signals}/{total_signals}"
        elif macro_strength < -0.3:  # Risk-off
            action = 2  # BUY PUT
            confidence = min(abs(macro_strength) * 1.2, 0.95)
            reasoning = f"Risk-OFF: VIX={vix:.1f}, momentum={momentum:.2f}%, signals={risk_off_signals}/{total_signals}"
        else:  # Neutral
            action = 0  # HOLD
            confidence = 1.0 - abs(macro_strength)
            reasoning = f"Neutral macro: VIX={vix:.1f}, momentum={momentum:.2f}%, strength={macro_strength:.2f}"
        
        return AgentSignal(
            agent_type=AgentType.MACRO,
            action=action,
            confidence=confidence,
            strength=macro_strength,
            reasoning=reasoning
        )


class MetaPolicyRouter:
    """
    Meta-policy router that combines signals from multiple specialized agents
    
    Strategy:
    - Weighted voting based on agent confidence and historical performance
    - Dynamic regime detection (trending vs mean-reverting vs volatile)
    - Confidence-weighted action selection
    - Conflict resolution (when agents disagree)
    - Hierarchical overrides (Risk > Macro > Volatility > Gamma > Trend > Reversal > RL)
    - Interaction rules between agents
    """
    
    def __init__(self):
        self.trend_agent = TrendAgent()
        self.reversal_agent = ReversalAgent()
        self.volatility_agent = VolatilityBreakoutAgent()
        self.gamma_agent = GammaModelAgent()
        self.delta_agent = DeltaHedgingAgent()
        self.macro_agent = MacroAgent()
        
        # Agent weights (can be adjusted based on market regime)
        self.base_weights = {
            AgentType.TREND: 0.20,
            AgentType.REVERSAL: 0.15,
            AgentType.VOLATILITY: 0.20,
            AgentType.GAMMA_MODEL: 0.20,
            AgentType.DELTA_HEDGING: 0.15,
            AgentType.MACRO: 0.10
        }
        
        # Hierarchical priority (higher = more important)
        # Risk > Macro > Volatility > Gamma > Trend > Reversal > RL
        self.hierarchy = {
            AgentType.DELTA_HEDGING: 6,  # Highest (Risk)
            AgentType.MACRO: 5,
            AgentType.VOLATILITY: 4,
            AgentType.GAMMA_MODEL: 3,
            AgentType.TREND: 2,
            AgentType.REVERSAL: 1  # Lowest
        }
        
        # Confidence thresholds
        self.min_confidence_threshold = 0.3  # Below this, signal is weak
        self.strong_confidence_threshold = 0.7  # Above this, signal is strong
        
        # Drift detection
        self.signal_history = []
        self.max_history = 100
        
        # Performance tracking
        self.agent_performance = {
            AgentType.TREND: {'wins': 0, 'losses': 0, 'total': 0},
            AgentType.REVERSAL: {'wins': 0, 'losses': 0, 'total': 0},
            AgentType.VOLATILITY: {'wins': 0, 'losses': 0, 'total': 0},
            AgentType.GAMMA_MODEL: {'wins': 0, 'losses': 0, 'total': 0},
            AgentType.DELTA_HEDGING: {'wins': 0, 'losses': 0, 'total': 0},
            AgentType.MACRO: {'wins': 0, 'losses': 0, 'total': 0}
        }
        
        # Regime detection
        self.current_regime = "neutral"  # "trending", "mean_reverting", "volatile", "neutral"
    
    def detect_regime(self, data: pd.DataFrame, vix: float) -> str:
        """
        Detect current market regime
        
        Returns:
            "trending", "mean_reverting", "volatile", or "neutral"
        """
        if len(data) < 20:
            return "neutral"
        
        recent = data.tail(20).copy()
        closes = recent['close'].values if 'close' in recent.columns else recent['Close'].values
        
        # Trend strength
        ema9 = pd.Series(closes).ewm(span=9, adjust=False).mean().values[-1]
        ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().values[-1]
        trend_strength = abs((ema9 - ema20) / ema20) if ema20 > 0 else 0
        
        # Volatility
        returns = pd.Series(closes).pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized %
        
        # Mean reversion indicator (RSI)
        delta = np.diff(closes, prepend=closes[0])
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        avg_up = pd.Series(up).ewm(alpha=1/14, adjust=False).mean().values[-1]
        avg_down = pd.Series(down).ewm(alpha=1/14, adjust=False).mean().values[-1]
        rsi = 100 - (100 / (1 + avg_up / (avg_down + 1e-9)))
        
        # Regime classification
        if trend_strength > 0.02 and volatility < 30:  # Strong trend, low vol
            return "trending"
        elif volatility > 40 or vix > 30:  # High volatility
            return "volatile"
        elif 30 < rsi < 70 and trend_strength < 0.01:  # Neutral RSI, weak trend
            return "mean_reverting"
        else:
            return "neutral"
    
    def get_dynamic_weights(self, regime: str) -> Dict[AgentType, float]:
        """
        Get dynamic weights based on market regime
        
        Args:
            regime: Current market regime
            
        Returns:
            Dictionary of agent weights
        """
        weights = self.base_weights.copy()
        
        if regime == "trending":
            weights[AgentType.TREND] = 0.30
            weights[AgentType.REVERSAL] = 0.10
            weights[AgentType.VOLATILITY] = 0.20
            weights[AgentType.GAMMA_MODEL] = 0.25
            weights[AgentType.DELTA_HEDGING] = 0.10
            weights[AgentType.MACRO] = 0.05
        elif regime == "mean_reverting":
            weights[AgentType.TREND] = 0.10
            weights[AgentType.REVERSAL] = 0.35
            weights[AgentType.VOLATILITY] = 0.15
            weights[AgentType.GAMMA_MODEL] = 0.15
            weights[AgentType.DELTA_HEDGING] = 0.15
            weights[AgentType.MACRO] = 0.10
        elif regime == "volatile":
            weights[AgentType.TREND] = 0.15
            weights[AgentType.REVERSAL] = 0.15
            weights[AgentType.VOLATILITY] = 0.30
            weights[AgentType.GAMMA_MODEL] = 0.20
            weights[AgentType.DELTA_HEDGING] = 0.10
            weights[AgentType.MACRO] = 0.10
        else:  # neutral
            weights = self.base_weights.copy()
        
        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
    
    def route(
        self,
        data: pd.DataFrame,
        vix: float = 20.0,
        symbol: str = "SPY",
        current_price: float = None,
        strike: float = None,
        portfolio_delta: float = 0.0,
        delta_limit: float = 2000.0
    ) -> Tuple[int, float, Dict[str, any]]:
        """
        Route decision by combining all agent signals
        
        Args:
            data: Market data DataFrame
            vix: Current VIX level
            symbol: Trading symbol
            current_price: Current underlying price (for gamma/delta agents)
            strike: Option strike price (for gamma/delta agents)
            portfolio_delta: Current portfolio delta (for delta hedging agent)
            delta_limit: Maximum delta limit (for delta hedging agent)
            
        Returns:
            (final_action, final_confidence, details_dict)
        """
        # Get current price if not provided
        if current_price is None:
            closes = data['close'].values if 'close' in data.columns else data['Close'].values
            current_price = closes[-1] if len(closes) > 0 else 500.0
        
        if strike is None:
            strike = round(current_price)
        
        # Get signals from all agents
        trend_signal = self.trend_agent.analyze(data, vix)
        reversal_signal = self.reversal_agent.analyze(data, vix)
        volatility_signal = self.volatility_agent.analyze(data, vix)
        gamma_signal = self.gamma_agent.analyze(data, vix, current_price, strike)
        delta_signal = self.delta_agent.analyze(data, vix, portfolio_delta, delta_limit, current_price, strike)
        macro_signal = self.macro_agent.analyze(data, vix)
        
        signals = [trend_signal, reversal_signal, volatility_signal, gamma_signal, delta_signal, macro_signal]
        
        # Normalize confidence values to [0, 1]
        for signal in signals:
            signal.confidence = max(0.0, min(1.0, signal.confidence))  # Clamp to [0, 1]
        
        # Apply interaction rules
        signals = self._apply_interaction_rules(signals, vix)
        
        # Detect regime
        regime = self.detect_regime(data, vix)
        self.current_regime = regime
        
        # Get dynamic weights
        weights = self.get_dynamic_weights(regime)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Apply hierarchical overrides
        signals = self._apply_hierarchical_overrides(signals, weights)
        
        # Weighted voting with normalized confidence
        action_scores = {0: 0.0, 1: 0.0, 2: 0.0}  # HOLD, BUY_CALL, BUY_PUT
        
        for signal in signals:
            weight = weights.get(signal.agent_type, 0.10)
            # Normalized confidence (already clamped to [0, 1])
            confidence_weight = signal.confidence
            combined_weight = weight * confidence_weight
            
            action_scores[signal.action] += combined_weight
        
        # Find winning action
        final_action = max(action_scores, key=action_scores.get)
        final_confidence = action_scores[final_action]
        
        # Normalize confidence using softmax for better scaling
        total_score = sum(action_scores.values())
        if total_score > 0:
            # Softmax normalization for smoother confidence
            import math
            exp_scores = {k: math.exp(v * 2) for k, v in action_scores.items()}  # Scale by 2 for sharper
            sum_exp = sum(exp_scores.values())
            final_confidence = exp_scores[final_action] / sum_exp if sum_exp > 0 else 0.0
        else:
            final_confidence = 0.0
        
        # Apply confidence override rules
        # Note: min_confidence_threshold is typically 0.3, but we want to allow signals at 0.3
        # So we only override if confidence is significantly below threshold
        if final_confidence < max(0.1, self.min_confidence_threshold * 0.5):
            # Very low confidence: default to HOLD
            final_action = 0
            final_confidence = 0.3  # Low confidence HOLD (but still return 0.3 to allow threshold checks)
        
        # Record signal for drift detection
        self.record_signal(final_action, final_confidence, regime)
        
        # Build details
        details = {
            'regime': regime,
            'signals': {
                'trend': {
                    'action': trend_signal.action,
                    'confidence': trend_signal.confidence,
                    'strength': trend_signal.strength,
                    'reasoning': trend_signal.reasoning,
                    'weight': weights.get(AgentType.TREND, 0.20)
                },
                'reversal': {
                    'action': reversal_signal.action,
                    'confidence': reversal_signal.confidence,
                    'strength': reversal_signal.strength,
                    'reasoning': reversal_signal.reasoning,
                    'weight': weights.get(AgentType.REVERSAL, 0.15)
                },
                'volatility': {
                    'action': volatility_signal.action,
                    'confidence': volatility_signal.confidence,
                    'strength': volatility_signal.strength,
                    'reasoning': volatility_signal.reasoning,
                    'weight': weights.get(AgentType.VOLATILITY, 0.20)
                },
                'gamma_model': {
                    'action': gamma_signal.action,
                    'confidence': gamma_signal.confidence,
                    'strength': gamma_signal.strength,
                    'reasoning': gamma_signal.reasoning,
                    'weight': weights.get(AgentType.GAMMA_MODEL, 0.20)
                },
                'delta_hedging': {
                    'action': delta_signal.action,
                    'confidence': delta_signal.confidence,
                    'strength': delta_signal.strength,
                    'reasoning': delta_signal.reasoning,
                    'weight': weights.get(AgentType.DELTA_HEDGING, 0.15)
                },
                'macro': {
                    'action': macro_signal.action,
                    'confidence': macro_signal.confidence,
                    'strength': macro_signal.strength,
                    'reasoning': macro_signal.reasoning,
                    'weight': weights.get(AgentType.MACRO, 0.10)
                }
            },
            'action_scores': action_scores,
            'final_action': final_action,
            'final_confidence': final_confidence
        }
        
        return final_action, final_confidence, details
    
    def _apply_interaction_rules(self, signals: List[AgentSignal], vix: float) -> List[AgentSignal]:
        """
        Apply interaction rules between agents
        
        Rules:
        - If macro says RISK-OFF and gamma/trend say BUY → suppress gamma/trend
        - If trend + volatility both say BUY → boost confidence
        - If reversal disagrees in trending market → suppress reversal
        - If delta hedging says hedge → prioritize over others
        """
        # Create signal dict for easier access
        signal_dict = {s.agent_type: s for s in signals}
        
        # Rule 1: Macro RISK-OFF overrides bullish signals
        if signal_dict[AgentType.MACRO].action == 2 and signal_dict[AgentType.MACRO].confidence > 0.7:
            # Risk-off: suppress bullish signals
            for agent_type in [AgentType.TREND, AgentType.GAMMA_MODEL, AgentType.VOLATILITY]:
                if signal_dict[agent_type].action == 1:  # BUY CALL
                    signal_dict[agent_type].confidence *= 0.5  # Reduce confidence
                    signal_dict[agent_type].reasoning += " [Suppressed by RISK-OFF]"
        
        # Rule 2: Macro RISK-ON overrides bearish signals
        if signal_dict[AgentType.MACRO].action == 1 and signal_dict[AgentType.MACRO].confidence > 0.7:
            # Risk-on: suppress bearish signals
            for agent_type in [AgentType.TREND, AgentType.GAMMA_MODEL, AgentType.VOLATILITY]:
                if signal_dict[agent_type].action == 2:  # BUY PUT
                    signal_dict[agent_type].confidence *= 0.5
                    signal_dict[agent_type].reasoning += " [Suppressed by RISK-ON]"
        
        # Rule 3: Trend + Volatility agreement boosts confidence
        if (signal_dict[AgentType.TREND].action == signal_dict[AgentType.VOLATILITY].action and
            signal_dict[AgentType.TREND].action != 0):  # Both agree on non-HOLD
            boost = 1.2  # 20% boost
            signal_dict[AgentType.TREND].confidence = min(0.95, signal_dict[AgentType.TREND].confidence * boost)
            signal_dict[AgentType.VOLATILITY].confidence = min(0.95, signal_dict[AgentType.VOLATILITY].confidence * boost)
            signal_dict[AgentType.TREND].reasoning += " [Boosted by Volatility agreement]"
        
        # Rule 4: Reversal in trending market gets suppressed
        if self.current_regime == "trending":
            if signal_dict[AgentType.REVERSAL].action != signal_dict[AgentType.TREND].action:
                signal_dict[AgentType.REVERSAL].confidence *= 0.6  # Suppress contrarian in trends
                signal_dict[AgentType.REVERSAL].reasoning += " [Suppressed in trending market]"
        
        # Rule 5: Delta hedging high confidence overrides others
        if (signal_dict[AgentType.DELTA_HEDGING].confidence > 0.8 and
            signal_dict[AgentType.DELTA_HEDGING].action != 0):
            # High-confidence hedging signal: boost it
            signal_dict[AgentType.DELTA_HEDGING].confidence = min(0.95, signal_dict[AgentType.DELTA_HEDGING].confidence * 1.3)
            signal_dict[AgentType.DELTA_HEDGING].reasoning += " [Priority: Risk management]"
        
        return list(signal_dict.values())
    
    def _apply_hierarchical_overrides(self, signals: List[AgentSignal], weights: Dict[AgentType, float]) -> List[AgentSignal]:
        """
        Apply hierarchical overrides: Risk > Macro > Volatility > Gamma > Trend > Reversal > RL
        
        Higher priority agents can override lower priority ones
        """
        # Sort signals by hierarchy (highest first)
        sorted_signals = sorted(signals, key=lambda s: self.hierarchy.get(s.agent_type, 0), reverse=True)
        
        # Apply overrides
        for i, high_signal in enumerate(sorted_signals):
            if high_signal.confidence < self.strong_confidence_threshold:
                continue  # Only override if high confidence
            
            # Override lower priority conflicting signals
            for j, low_signal in enumerate(sorted_signals[i+1:], start=i+1):
                if (high_signal.action != low_signal.action and
                    high_signal.action != 0 and  # High priority has non-HOLD action
                    low_signal.action != 0):  # Low priority has non-HOLD action
                    
                    # Suppress lower priority signal
                    low_signal.confidence *= 0.4  # Strong suppression
                    low_signal.reasoning += f" [Overridden by {high_signal.agent_type.value}]"
        
        return signals
    
    def check_drift(self) -> Dict[str, any]:
        """
        Check for model drift in ensemble signals
        
        Returns:
            Dictionary with drift metrics
        """
        if len(self.signal_history) < 10:
            return {'drift_detected': False, 'reason': 'Insufficient history'}
        
        recent_signals = self.signal_history[-20:]  # Last 20 signals
        
        # Check for sudden regime changes
        regimes = [s.get('regime') for s in recent_signals if 'regime' in s]
        if len(set(regimes[-5:])) > 2:  # More than 2 regimes in last 5
            return {
                'drift_detected': True,
                'reason': 'Rapid regime changes',
                'regimes': regimes[-5:]
            }
        
        # Check for confidence degradation
        confidences = [s.get('confidence', 0) for s in recent_signals]
        if len(confidences) >= 10:
            recent_avg = np.mean(confidences[-5:])
            earlier_avg = np.mean(confidences[-10:-5])
            if recent_avg < earlier_avg * 0.7:  # 30% drop
                return {
                    'drift_detected': True,
                    'reason': 'Confidence degradation',
                    'recent_avg': recent_avg,
                    'earlier_avg': earlier_avg
                }
        
        return {'drift_detected': False, 'reason': 'No drift detected'}
    
    def update_performance(self, agent_type: AgentType, pnl: float):
        """Update agent performance tracking"""
        if pnl > 0:
            self.agent_performance[agent_type]['wins'] += 1
        else:
            self.agent_performance[agent_type]['losses'] += 1
        self.agent_performance[agent_type]['total'] += 1
    
    def record_signal(self, action: int, confidence: float, regime: str):
        """Record signal for drift detection"""
        self.signal_history.append({
            'action': action,
            'confidence': confidence,
            'regime': regime,
            'timestamp': datetime.now()
        })
        # Keep only last N signals
        if len(self.signal_history) > self.max_history:
            self.signal_history = self.signal_history[-self.max_history:]


# Global instance
_meta_router: Optional[MetaPolicyRouter] = None


def initialize_meta_router() -> MetaPolicyRouter:
    """Initialize global meta-policy router"""
    global _meta_router
    _meta_router = MetaPolicyRouter()
    return _meta_router


def get_meta_router() -> Optional[MetaPolicyRouter]:
    """Get global meta-policy router instance"""
    return _meta_router

