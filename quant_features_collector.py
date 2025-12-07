#!/usr/bin/env python3
"""
🏦 QUANT FEATURES COLLECTOR

Institutional-grade feature collection for historical data:
- IV (Implied Volatility) from VIX
- Greeks (Delta, Gamma, Vega, Theta) calculated
- Theta decay model
- Market microstructure (order flow imbalance)
- Cross-asset correlations (SPY-QQQ-VIX-SPX)
- Volatility regime classification
- TPO/Market Profile signals

This module enriches historical OHLCV data with all quant features
needed for professional trading system training.

Author: Mike Agent Institutional Upgrade
Date: December 7, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import existing modules
try:
    from greeks_calculator import GreeksCalculator
    GREEKS_AVAILABLE = True
except ImportError:
    GREEKS_AVAILABLE = False
    print("⚠️ GreeksCalculator not found. Greeks will be calculated inline.")

try:
    from institutional_features import InstitutionalFeatureEngine
    INSTITUTIONAL_FEATURES_AVAILABLE = True
except ImportError:
    INSTITUTIONAL_FEATURES_AVAILABLE = False
    print("⚠️ InstitutionalFeatureEngine not found. Using basic features.")


class QuantFeaturesCollector:
    """
    Comprehensive quant feature collector for historical data
    
    Calculates and stores:
    - Options Greeks (Delta, Gamma, Vega, Theta)
    - Implied Volatility (from VIX)
    - Theta decay models
    - Market microstructure features
    - Cross-asset correlations
    - Volatility regime classification
    - Market Profile/TPO signals
    """
    
    def __init__(self, risk_free_rate: float = 0.04):
        """
        Initialize quant features collector
        
        Args:
            risk_free_rate: Risk-free interest rate (default 4%)
        """
        self.risk_free_rate = risk_free_rate
        if GREEKS_AVAILABLE:
            self.greeks_calc = GreeksCalculator(risk_free_rate)
        else:
            self.greeks_calc = None
        
        # Volatility regime thresholds (from mike_agent_live_safe.py)
        self.regime_thresholds = {
            'calm': 18,
            'normal': 25,
            'storm': 35,
            'crash': float('inf')
        }
    
    def collect_all_features(
        self,
        price_data: pd.DataFrame,
        vix_data: pd.Series,
        symbol: str = 'SPY',
        strike_prices: Optional[List[float]] = None,
        option_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Collect all quant features for historical data
        
        Args:
            price_data: DataFrame with OHLCV data (datetime index)
            vix_data: Series with VIX values (datetime index)
            symbol: Trading symbol (SPY, QQQ, SPX)
            strike_prices: Optional list of strike prices to calculate Greeks for
            option_types: Optional list of option types ('call' or 'put')
            
        Returns:
            DataFrame with all quant features added
        """
        if len(price_data) == 0:
            return price_data
        
        # Make a copy to avoid modifying original
        features_df = price_data.copy()
        
        # Ensure datetime index
        if not isinstance(features_df.index, pd.DatetimeIndex):
            features_df.index = pd.to_datetime(features_df.index)
        
        # Ensure lowercase column names
        features_df.columns = [col.lower() for col in features_df.columns]
        
        # 1. IMPLIED VOLATILITY (IV) from VIX
        features_df = self._add_implied_volatility(features_df, vix_data, symbol)
        
        # 2. OPTIONS GREEKS (Delta, Gamma, Vega, Theta)
        if strike_prices:
            features_df = self._add_greeks(features_df, symbol, strike_prices, option_types)
        else:
            # Use ATM strikes (current price rounded)
            features_df = self._add_greeks_atm(features_df, symbol)
        
        # 3. THETA DECAY MODEL
        features_df = self._add_theta_decay(features_df)
        
        # 4. MARKET MICROSTRUCTURE (Order Flow Imbalance)
        features_df = self._add_microstructure_features(features_df)
        
        # 5. VOLATILITY REGIME CLASSIFICATION
        features_df = self._add_volatility_regime(features_df, vix_data)
        
        # 6. CORRELATIONS (SPY-QQQ-VIX-SPX) - will be added when all symbols available
        # This is handled separately in collect_cross_asset_features
        
        # 7. MARKET PROFILE / TPO SIGNALS
        features_df = self._add_market_profile(features_df)
        
        # 8. REALIZED VOLATILITY FEATURES
        features_df = self._add_realized_volatility(features_df)
        
        # 9. REGIME TRANSITION SIGNALS
        features_df = self._add_regime_transitions(features_df)
        
        return features_df
    
    def _add_implied_volatility(
        self,
        df: pd.DataFrame,
        vix_data: pd.Series,
        symbol: str
    ) -> pd.DataFrame:
        """Add implied volatility from VIX"""
        # VIX is already in percentage form (e.g., 20 = 20%)
        # Convert to decimal for IV (e.g., 20% = 0.20)
        df['vix'] = vix_data.reindex(df.index, method='ffill') / 100.0
        
        # IV from VIX (VIX is 30-day forward IV, scale for 0DTE)
        # 0DTE IV ≈ VIX * sqrt(T_0DTE / T_VIX) where T_0DTE ≈ 1/252, T_VIX = 30/252
        df['iv_0dte'] = df['vix'] * np.sqrt(1.0 / 30.0)
        
        # Also store raw VIX-normalized IV
        df['iv_from_vix'] = df['vix']
        
        return df
    
    def _add_greeks_atm(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """Add Greeks for ATM (At-The-Money) options"""
        # Use current price as strike (ATM)
        df['strike_atm'] = df['close'].round(0)  # Round to nearest dollar
        
        # Calculate Greeks for both calls and puts
        for option_type in ['call', 'put']:
            for greek in ['delta', 'gamma', 'vega', 'theta']:
                col_name = f'{option_type}_{greek}'
                df[col_name] = 0.0
        
        # Calculate Greeks using current price, ATM strike, and IV
        if 'iv_from_vix' not in df.columns:
            df['iv_from_vix'] = 0.20  # Default IV
        
        # Time to expiration for 0DTE (1 day = 1/252 years)
        T_0DTE = 1.0 / 252.0
        
        # Calculate for calls and puts
        for idx in df.index:
            S = df.loc[idx, 'close']
            K = df.loc[idx, 'strike_atm']
            sigma = df.loc[idx, 'iv_from_vix']
            T = T_0DTE
            
            if self.greeks_calc:
                # Use Greeks calculator
                call_greeks = self.greeks_calc.calculate_greeks(
                    S=S, K=K, T=T, sigma=sigma, option_type='call'
                )
                put_greeks = self.greeks_calc.calculate_greeks(
                    S=S, K=K, T=T, sigma=sigma, option_type='put'
                )
            else:
                # Calculate inline (simplified)
                call_greeks = self._calculate_greeks_inline(S, K, T, sigma, 'call')
                put_greeks = self._calculate_greeks_inline(S, K, T, sigma, 'put')
            
            df.loc[idx, 'call_delta'] = call_greeks['delta']
            df.loc[idx, 'call_gamma'] = call_greeks['gamma']
            df.loc[idx, 'call_vega'] = call_greeks['vega']
            df.loc[idx, 'call_theta'] = call_greeks['theta']
            
            df.loc[idx, 'put_delta'] = put_greeks['delta']
            df.loc[idx, 'put_gamma'] = put_greeks['gamma']
            df.loc[idx, 'put_vega'] = put_greeks['vega']
            df.loc[idx, 'put_theta'] = put_greeks['theta']
        
        return df
    
    def _add_greeks(
        self,
        df: pd.DataFrame,
        symbol: str,
        strike_prices: List[float],
        option_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Add Greeks for specific strikes"""
        if option_types is None:
            option_types = ['call', 'put']
        
        # Add Greeks for each strike/type combination
        for strike in strike_prices:
            for opt_type in option_types:
                col_prefix = f'{opt_type}_k{strike:.0f}'
                
                for greek in ['delta', 'gamma', 'vega', 'theta']:
                    df[f'{col_prefix}_{greek}'] = 0.0
        
        # Calculate Greeks for each strike
        if 'iv_from_vix' not in df.columns:
            df['iv_from_vix'] = 0.20
        
        T_0DTE = 1.0 / 252.0
        
        for idx in df.index:
            S = df.loc[idx, 'close']
            sigma = df.loc[idx, 'iv_from_vix']
            T = T_0DTE
            
            for strike in strike_prices:
                K = strike
                
                for opt_type in option_types:
                    if self.greeks_calc:
                        greeks = self.greeks_calc.calculate_greeks(
                            S=S, K=K, T=T, sigma=sigma, option_type=opt_type
                        )
                    else:
                        greeks = self._calculate_greeks_inline(S, K, T, sigma, opt_type)
                    
                    col_prefix = f'{opt_type}_k{strike:.0f}'
                    df.loc[idx, f'{col_prefix}_delta'] = greeks['delta']
                    df.loc[idx, f'{col_prefix}_gamma'] = greeks['gamma']
                    df.loc[idx, f'{col_prefix}_vega'] = greeks['vega']
                    df.loc[idx, f'{col_prefix}_theta'] = greeks['theta']
        
        return df
    
    def _calculate_greeks_inline(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str
    ) -> Dict[str, float]:
        """Calculate Greeks inline (simplified Black-Scholes)"""
        from scipy.stats import norm
        
        if T <= 0:
            return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}
        
        r = self.risk_free_rate
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100.0
            theta = (
                -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2)
            ) / 365.0
        else:  # put
            delta = -norm.cdf(-d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100.0
            theta = (
                -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
            ) / 365.0
        
        return {
            'delta': float(delta),
            'gamma': float(gamma),
            'vega': float(vega),
            'theta': float(theta)
        }
    
    def _add_theta_decay(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add theta decay model for 0DTE options"""
        # Time to expiration (in days)
        df['time_to_exp'] = 1.0  # 0DTE = 1 day remaining
        
        # Theta decay rate (exponential decay)
        # Theta decay ≈ Premium * Theta * dt
        if 'call_theta' in df.columns:
            df['theta_decay_rate_call'] = -df['call_theta']  # Negative theta = decay
            df['theta_decay_rate_put'] = -df['put_theta']
        else:
            # Estimate theta decay from IV and time
            if 'iv_from_vix' in df.columns:
                # Simplified: theta ≈ -0.5 * IV^2 * S^2 / (2 * T)
                df['theta_decay_rate_call'] = -0.5 * df['iv_from_vix']**2 * df['close']**2 / (2 * df['time_to_exp'])
                df['theta_decay_rate_put'] = df['theta_decay_rate_call']
            else:
                df['theta_decay_rate_call'] = 0.0
                df['theta_decay_rate_put'] = 0.0
        
        # Expected theta decay over next hour (for 0DTE)
        df['theta_decay_1h'] = df['theta_decay_rate_call'] / 6.5  # Trading hours per day
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features (order flow imbalance)"""
        # Order Flow Imbalance (OFI) approximation
        # OFI ≈ (buy volume - sell volume) / total volume
        # Approximate using price movement and volume
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Approximate buy/sell pressure from price movement
        df['buy_pressure'] = np.where(
            df['returns'] > 0,
            df['volume'],
            0.0
        )
        df['sell_pressure'] = np.where(
            df['returns'] < 0,
            df['volume'],
            0.0
        )
        
        # Order Flow Imbalance
        df['ofi'] = (df['buy_pressure'] - df['sell_pressure']) / (df['volume'] + 1e-10)
        
        # Volume-weighted price change
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3.0
        df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
        
        # Price impact (returns per unit volume)
        df['price_impact'] = df['returns'].abs() / (df['volume'] + 1e-10)
        
        # Spread proxy (high-low range as % of close)
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def _add_volatility_regime(
        self,
        df: pd.DataFrame,
        vix_data: pd.Series
    ) -> pd.DataFrame:
        """Add volatility regime classification"""
        # Get VIX values
        vix_values = vix_data.reindex(df.index, method='ffill')
        
        # Classify regime based on VIX
        df['vix_level'] = vix_values
        
        def classify_regime(vix):
            if pd.isna(vix):
                return 'normal'
            if vix < self.regime_thresholds['calm']:
                return 'calm'
            elif vix < self.regime_thresholds['normal']:
                return 'normal'
            elif vix < self.regime_thresholds['storm']:
                return 'storm'
            else:
                return 'crash'
        
        df['vol_regime'] = df['vix_level'].apply(classify_regime)
        
        # Encode regime as numeric
        regime_map = {'calm': 0, 'normal': 1, 'storm': 2, 'crash': 3}
        df['vol_regime_encoded'] = df['vol_regime'].map(regime_map).fillna(1)
        
        return df
    
    def _add_market_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Market Profile / TPO signals"""
        # Market Profile: Identify value areas (high volume nodes)
        # TPO (Time Price Opportunity): Price levels where most time was spent
        
        # Value Area (70% of volume)
        window = min(20, len(df))
        if window < 5:
            df['value_area_high'] = df['high']
            df['value_area_low'] = df['low']
            df['poc'] = df['close']  # Point of Control
            return df
        
        # Calculate volume-weighted price distribution
        df['price_range'] = df['high'] - df['low']
        df['volume_density'] = df['volume'] / (df['price_range'] + 1e-10)
        
        # Rolling value area (simplified)
        df['value_area_high'] = df['high'].rolling(window=window, min_periods=1).max()
        df['value_area_low'] = df['low'].rolling(window=window, min_periods=1).min()
        
        # Point of Control (POC) - price with most volume (simplified as VWAP)
        df['poc'] = df['vwap'] if 'vwap' in df.columns else df['close']
        
        # Distance from value area
        df['distance_from_value_area'] = np.where(
            df['close'] > df['value_area_high'],
            (df['close'] - df['value_area_high']) / df['close'],
            np.where(
                df['close'] < df['value_area_low'],
                (df['close'] - df['value_area_low']) / df['close'],
                0.0
            )
        )
        
        return df
    
    def _add_realized_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realized volatility features"""
        # Calculate returns if not already present
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        if 'log_returns' not in df.columns:
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Realized Volatility (multiple methods and periods)
        for period in [5, 10, 20, 30]:
            # Returns-based RV (annualized)
            rv_returns = df['returns'].rolling(window=period, min_periods=1).std() * np.sqrt(252)
            df[f'rv_{period}d'] = rv_returns
            
            # Log returns-based RV
            rv_log = df['log_returns'].rolling(window=period, min_periods=1).std() * np.sqrt(252)
            df[f'rv_log_{period}d'] = rv_log
            
            # Parkinson (High-Low) volatility estimator
            if 'high' in df.columns and 'low' in df.columns:
                hl_vol = np.sqrt((1/(4*np.log(2))) * (np.log(df['high'] / df['low'])**2))
                rv_parkinson = hl_vol.rolling(window=period, min_periods=1).mean() * np.sqrt(252)
                df[f'rv_parkinson_{period}d'] = rv_parkinson
        
        # RV-IV Spread (if IV available)
        if 'iv_from_vix' in df.columns:
            # Use 20-day RV for comparison
            if 'rv_20d' in df.columns:
                df['rv_iv_spread'] = df['rv_20d'] - (df['iv_from_vix'] * 100)  # Convert IV to percentage
                df['rv_iv_ratio'] = df['rv_20d'] / ((df['iv_from_vix'] * 100) + 1e-6)
        
        # Volatility of Volatility (vol of vol)
        if 'rv_20d' in df.columns:
            df['vol_of_vol'] = df['rv_20d'].rolling(window=20, min_periods=1).std()
            df['vol_of_vol_ma'] = df['vol_of_vol'].rolling(window=10, min_periods=1).mean()
        
        # HAR-RV (Heterogeneous AutoRegressive Realized Volatility)
        # HAR-RV uses daily, weekly, monthly components
        if 'rv_1d' not in df.columns:
            df['rv_1d'] = df['returns'].rolling(window=1, min_periods=1).std() * np.sqrt(252)
        
        if len(df) > 5:
            # Daily component (already have rv_1d)
            # Weekly component (5-day)
            if 'rv_5d' in df.columns:
                df['har_rv_weekly'] = df['rv_5d']
            # Monthly component (20-day)
            if 'rv_20d' in df.columns:
                df['har_rv_monthly'] = df['rv_20d']
            
            # HAR-RV model: RV_t = α + β_d * RV_daily + β_w * RV_weekly + β_m * RV_monthly
            # Simplified: use weighted combination
            if 'rv_1d' in df.columns and 'rv_5d' in df.columns and 'rv_20d' in df.columns:
                df['har_rv'] = (
                    0.1 * df['rv_1d'] +
                    0.3 * df['rv_5d'] +
                    0.6 * df['rv_20d']
                )
        
        # ATR-based volatility (if not already present)
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            for period in [5, 10, 20]:
                if f'atr_{period}d' not in df.columns:
                    prev_close = df['close'].shift(1).fillna(df['close'])
                    tr = pd.Series([
                        max(high - low, abs(high - prev_close.iloc[i]), abs(low - prev_close.iloc[i]))
                        for i, (high, low) in enumerate(zip(df['high'], df['low']))
                    ], index=df.index)
                    atr = tr.rolling(window=period, min_periods=1).mean()
                    df[f'atr_{period}d'] = atr / df['close']  # Normalized ATR
        
        return df
    
    def _add_regime_transitions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime transition signals"""
        if 'vol_regime' not in df.columns:
            # Can't add transitions without regime data
            return df
        
        # Regime change indicator (1 when regime changes, 0 otherwise)
        df['regime_change'] = (df['vol_regime'] != df['vol_regime'].shift(1)).astype(int)
        df.loc[df.index[0], 'regime_change'] = 0  # First row is not a change
        
        # Time in current regime (days since last change)
        regime_durations = []
        current_duration = 0
        for i, is_change in enumerate(df['regime_change']):
            if is_change:
                current_duration = 1
            else:
                current_duration += 1
            regime_durations.append(current_duration)
        df['time_in_regime'] = regime_durations
        
        # Regime transition types
        df['regime_to_calm'] = 0
        df['regime_to_normal'] = 0
        df['regime_to_storm'] = 0
        df['regime_to_crash'] = 0
        
        # Previous regime
        prev_regime = df['vol_regime'].shift(1)
        
        # Transition indicators (only on change days)
        change_mask = df['regime_change'] == 1
        if change_mask.any():
            df.loc[change_mask & (df['vol_regime'] == 'calm'), 'regime_to_calm'] = 1
            df.loc[change_mask & (df['vol_regime'] == 'normal'), 'regime_to_normal'] = 1
            df.loc[change_mask & (df['vol_regime'] == 'storm'), 'regime_to_storm'] = 1
            df.loc[change_mask & (df['vol_regime'] == 'crash'), 'regime_to_crash'] = 1
        
        # Regime transition direction (improving vs worsening)
        # Improving: crash -> storm -> normal -> calm
        # Worsening: calm -> normal -> storm -> crash
        regime_order = {'calm': 0, 'normal': 1, 'storm': 2, 'crash': 3}
        
        def get_transition_direction(current, previous):
            if pd.isna(previous):
                return 0
            current_order = regime_order.get(current, 1)
            prev_order = regime_order.get(previous, 1)
            if current_order > prev_order:
                return 1  # Worsening
            elif current_order < prev_order:
                return -1  # Improving
            else:
                return 0  # No change
        
        df['regime_transition_direction'] = [
            get_transition_direction(curr, prev)
            for curr, prev in zip(df['vol_regime'], prev_regime)
        ]
        
        # Regime stability (inverse of time in regime, normalized)
        max_duration = df['time_in_regime'].max() if df['time_in_regime'].max() > 0 else 1
        df['regime_stability'] = 1.0 - (df['time_in_regime'] / max_duration)
        
        # Probability of regime change (based on time in regime)
        # Longer time in regime = higher probability of change
        df['regime_change_probability'] = np.minimum(df['time_in_regime'] / 60.0, 1.0)  # Cap at 60 days
        
        return df
    
    def collect_cross_asset_features(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        vix_data: pd.Series
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect cross-asset features (correlations between SPY, QQQ, VIX, SPX)
        
        Args:
            symbol_data: Dict of {symbol: DataFrame} with OHLCV data
            vix_data: Series with VIX values
            
        Returns:
            Dict of {symbol: DataFrame} with cross-asset features added
        """
        symbols = list(symbol_data.keys())
        
        # Calculate returns for all symbols
        returns = {}
        for symbol, df in symbol_data.items():
            if 'close' in df.columns:
                returns[symbol] = df['close'].pct_change()
        
        # Add VIX returns
        returns['VIX'] = vix_data.pct_change()
        
        # Calculate rolling correlations (30-day window)
        window = 30
        
        for symbol in symbols:
            df = symbol_data[symbol].copy()
            
            # Correlations with other symbols
            for other_symbol in symbols:
                if other_symbol != symbol and other_symbol in returns:
                    # Rolling correlation
                    corr_values = []
                    for i in range(len(df)):
                        start_idx = max(0, i - window + 1)
                        if start_idx < len(returns[symbol]) and start_idx < len(returns[other_symbol]):
                            corr = returns[symbol].iloc[start_idx:i+1].corr(
                                returns[other_symbol].iloc[start_idx:i+1]
                            )
                            corr_values.append(corr if not np.isnan(corr) else 0.0)
                        else:
                            corr_values.append(0.0)
                    
                    df[f'corr_{other_symbol.lower()}'] = corr_values
            
            # Correlation with VIX
            if 'VIX' in returns:
                corr_values = []
                for i in range(len(df)):
                    start_idx = max(0, i - window + 1)
                    if start_idx < len(returns[symbol]) and start_idx < len(returns['VIX']):
                        corr = returns[symbol].iloc[start_idx:i+1].corr(
                            returns['VIX'].iloc[start_idx:i+1]
                        )
                        corr_values.append(corr if not np.isnan(corr) else -1.0)  # Usually negative
                    else:
                        corr_values.append(0.0)
                
                df['corr_vix'] = corr_values
            
            symbol_data[symbol] = df
        
        return symbol_data


def collect_quant_features_for_symbol(
    price_data: pd.DataFrame,
    vix_data: pd.Series,
    symbol: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to collect all quant features for a single symbol
    
    Args:
        price_data: DataFrame with OHLCV data
        vix_data: Series with VIX values
        symbol: Trading symbol
        output_path: Optional path to save enriched data
        
    Returns:
        DataFrame with all quant features
    """
    collector = QuantFeaturesCollector()
    
    enriched_data = collector.collect_all_features(
        price_data=price_data,
        vix_data=vix_data,
        symbol=symbol
    )
    
    if output_path:
        enriched_data.to_pickle(output_path)
        print(f"✅ Saved enriched data to {output_path}")
    
    return enriched_data

