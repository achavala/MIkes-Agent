"""
🎓 ADVANCED RL TRAINING ENVIRONMENT

Comprehensive training environment for 0DTE options trading.
Handles all market regimes, all trading activities, and full 23-year historical data.

Features:
- Realistic 0DTE options simulation
- Market regime awareness
- Full trading activity (buy, sell, trim, exit, avg-down)
- Historical data integration
- Reward shaping for options trading

Author: Mike Agent Training System
Date: December 6, 2025
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Tuple, List
from scipy.stats import norm
from datetime import datetime, timedelta


class Advanced0DTETradingEnv(gym.Env):
    """
    Advanced 0DTE options trading environment for RL training
    
    Handles:
    - All market regimes (calm, normal, storm, crash)
    - All trading activities (buy, sell, trim, exit, avg-down)
    - Realistic options pricing and Greeks
    - Position management
    - Risk controls
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000.0,
        lookback: int = 20,
        include_greeks: bool = True
    ):
        """
        Initialize training environment
        
        Args:
            data: Historical data with OHLCV + VIX + regime
            initial_capital: Starting capital
            lookback: Number of bars to look back
            include_greeks: Whether to include Greeks in observation
        """
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.lookback = lookback
        self.initial_capital = initial_capital
        self.include_greeks = include_greeks
        
        # Action space: Continuous Box(-1, 1) for flexible action mapping
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space
        if include_greeks:
            # OHLCV (5) + VIX (1) + Regime (4) + Position (3) + Greeks (4) = 17 features
            obs_dim = 17
        else:
            # OHLCV (5) + VIX (1) + Regime (4) + Position (3) = 13 features
            obs_dim = 13
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback, obs_dim),
            dtype=np.float32
        )
        
        # Trading state
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        # Start at random point in data (with enough lookback)
        min_start = self.lookback
        max_start = len(self.data) - 100  # Leave room for episode
        self.current_step = np.random.randint(min_start, max_start) if max_start > min_start else min_start
        
        # Reset trading state
        self.capital = self.initial_capital
        self.position = None  # {type: 'call'/'put', qty: int, entry_price: float, entry_premium: float, strike: float, entry_step: int}
        self.trades = []
        self.daily_pnl = 0.0
        self.peak_capital = self.initial_capital
        
        return self._get_obs(), {}
    
    def _get_obs(self) -> np.ndarray:
        """
        Get current observation
        
        Returns:
            Observation array with shape (lookback, n_features)
        """
        start_idx = max(0, self.current_step - self.lookback)
        end_idx = self.current_step
        
        window = self.data.iloc[start_idx:end_idx].copy()
        
        # Pad if needed
        if len(window) < self.lookback:
            padding_needed = self.lookback - len(window)
            first_row = window.iloc[[0]].copy() if len(window) > 0 else pd.DataFrame()
            padding = pd.concat([first_row] * padding_needed, ignore_index=True)
            window = pd.concat([padding, window], ignore_index=True)
            window = window.tail(self.lookback)
        
        # Extract features
        features = []
        
        # OHLCV (5 features)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in window.columns:
                # Normalize by close price (except volume)
                if col == 'volume':
                    # Normalize volume by max in window
                    max_vol = window[col].max() if window[col].max() > 0 else 1.0
                    features.append((window[col].values / max_vol).reshape(-1, 1))
                else:
                    # Normalize OHLC by close
                    close = window['close'].values
                    close[close == 0] = 1.0  # Avoid division by zero
                    features.append((window[col].values / close).reshape(-1, 1))
            else:
                features.append(np.zeros((self.lookback, 1)))
        
        # VIX (1 feature) - normalized
        if 'vix' in window.columns:
            vix_normalized = (window['vix'].values / 20.0).reshape(-1, 1)  # Normalize around VIX 20
            features.append(vix_normalized)
        else:
            features.append(np.ones((self.lookback, 1)))
        
        # Regime (4 features - one-hot encoded)
        regime_map = {'calm': [1, 0, 0, 0], 'normal': [0, 1, 0, 0], 'storm': [0, 0, 1, 0], 'crash': [0, 0, 0, 1]}
        if 'regime' in window.columns:
            # Handle string or encoded regime values
            regime_values = window['regime'].values
            regime_encoded = []
            for r in regime_values:
                if isinstance(r, str):
                    regime_encoded.append(regime_map.get(r, [0, 1, 0, 0]))
                elif isinstance(r, (list, np.ndarray)):
                    regime_encoded.append(r)
                else:
                    regime_encoded.append([0, 1, 0, 0])
            regime_encoded = np.array(regime_encoded)
            features.append(regime_encoded)
        else:
            features.append(np.tile([0, 1, 0, 0], (self.lookback, 1)))
        
        # Position state (3 features: has_position, position_type, position_pnl)
        has_position = 1.0 if self.position is not None else 0.0
        position_type = 1.0 if (self.position and self.position['type'] == 'call') else -1.0 if (self.position and self.position['type'] == 'put') else 0.0
        position_pnl = self._calculate_position_pnl() if self.position else 0.0
        
        features.append(np.full((self.lookback, 1), has_position))
        features.append(np.full((self.lookback, 1), position_type))
        features.append(np.full((self.lookback, 1), position_pnl))
        
        # Greeks (4 features: Delta, Gamma, Theta, Vega) - if enabled
        if self.include_greeks and self.position:
            greeks = self._calculate_greeks()
            for greek in ['delta', 'gamma', 'theta', 'vega']:
                features.append(np.full((self.lookback, 1), greeks.get(greek, 0.0)))
        elif self.include_greeks:
            # No position - zeros for Greeks
            for _ in range(4):
                features.append(np.zeros((self.lookback, 1)))
        
        # Concatenate all features
        obs = np.concatenate(features, axis=1).astype(np.float32)
        
        return obs
    
    def _calculate_greeks(self) -> Dict[str, float]:
        """Calculate Greeks for current position"""
        if not self.position:
            return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
        
        try:
            from greeks_calculator import calculate_greeks
            
            current_price = float(self.data.iloc[self.current_step]['close'])
            strike = self.position['strike']
            option_type = self.position['type']
            entry_premium = self.position['entry_premium']
            
            # Estimate time to expiration (0DTE = ~6.5 hours = 390 minutes)
            time_to_exp = 390 / (252 * 390 * 24)  # Convert minutes to years
            
            # Estimate IV from entry premium
            if entry_premium > 0:
                # Rough IV estimation
                sigma = min(0.5, max(0.10, entry_premium / (current_price * np.sqrt(time_to_exp))))
            else:
                sigma = 0.20
            
            greeks = calculate_greeks(
                S=current_price,
                K=strike,
                T=time_to_exp,
                r=0.04,
                sigma=sigma,
                option_type=option_type
            )
            
            return greeks
            
        except Exception:
            return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
    
    def _calculate_position_pnl(self) -> float:
        """Calculate current position P&L as percentage"""
        if not self.position:
            return 0.0
        
        try:
            current_price = float(self.data.iloc[self.current_step]['close'])
            strike = self.position['strike']
            option_type = self.position['type']
            entry_premium = self.position['entry_premium']
            
            # Estimate current premium using Black-Scholes
            current_premium = self._estimate_premium(current_price, strike, option_type)
            
            # Calculate P&L percentage
            if entry_premium > 0:
                pnl_pct = (current_premium - entry_premium) / entry_premium
                return float(pnl_pct)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _estimate_premium(self, S: float, K: float, option_type: str) -> float:
        """Estimate option premium using Black-Scholes"""
        T = 390 / (252 * 390 * 24)  # 0DTE time to expiration
        r = 0.04
        sigma = 0.20
        
        if T <= 0:
            # Intrinsic value only
            if option_type == 'call':
                return max(0.01, S - K)
            else:
                return max(0.01, K - S)
        
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == 'call':
                premium = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                premium = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return max(0.01, premium)
        except Exception:
            return 0.01
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Continuous action value in [-1, 1]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        action_value = float(action[0])
        
        # Map continuous action to discrete trading actions
        trading_action = self._map_action(action_value)
        
        # Get current market state
        current_price = float(self.data.iloc[self.current_step]['close'])
        current_vix = float(self.data.iloc[self.current_step].get('vix', 20.0))
        current_regime = self.data.iloc[self.current_step].get('regime', 'normal')
        
        # Execute trading action
        reward = self._execute_action(trading_action, current_price, current_vix, current_regime)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        # Calculate reward
        reward = self._calculate_reward(reward, current_price)
        
        # Get next observation
        obs = self._get_obs()
        
        # Info dictionary
        info = {
            'capital': self.capital,
            'position': self.position,
            'daily_pnl': self.daily_pnl,
            'trades_count': len(self.trades)
        }
        
        return obs, reward, done, False, info
    
    def _map_action(self, action_value: float) -> int:
        """
        Map continuous action to discrete trading action
        
        Actions:
        0: HOLD
        1: BUY CALL
        2: BUY PUT
        3: TRIM 50%
        4: TRIM 70%
        5: FULL EXIT
        """
        if self.position is None:
            # No position - can only buy or hold
            if abs(action_value) < 0.35:
                return 0  # HOLD
            elif action_value > 0:
                return 1  # BUY CALL
            else:
                return 2  # BUY PUT
        else:
            # Have position - can trim, exit, or hold
            if abs(action_value) < 0.35:
                return 0  # HOLD
            elif action_value < 0.6:
                return 3  # TRIM 50%
            elif action_value < 0.85:
                return 4  # TRIM 70%
            else:
                return 5  # FULL EXIT
    
    def _execute_action(
        self,
        action: int,
        current_price: float,
        current_vix: float,
        current_regime: str
    ) -> float:
        """
        Execute trading action and return immediate reward
        
        Returns:
            Immediate reward from action
        """
        reward = 0.0
        
        if action == 0:  # HOLD
            # Small negative reward for holding (encourages action)
            reward = -0.001
        
        elif action == 1:  # BUY CALL
            if self.position is None:
                reward = self._buy_call(current_price, current_vix, current_regime)
            else:
                reward = -0.01  # Can't buy when already in position
        
        elif action == 2:  # BUY PUT
            if self.position is None:
                reward = self._buy_put(current_price, current_vix, current_regime)
            else:
                reward = -0.01  # Can't buy when already in position
        
        elif action == 3:  # TRIM 50%
            if self.position:
                reward = self._trim_position(0.5, current_price)
            else:
                reward = -0.01  # Can't trim without position
        
        elif action == 4:  # TRIM 70%
            if self.position:
                reward = self._trim_position(0.7, current_price)
            else:
                reward = -0.01  # Can't trim without position
        
        elif action == 5:  # FULL EXIT
            if self.position:
                reward = self._exit_position(current_price)
            else:
                reward = -0.01  # Can't exit without position
        
        return reward
    
    def _buy_call(self, price: float, vix: float, regime: str) -> float:
        """Buy call option"""
        # Calculate position size based on risk
        risk_pct = 0.10 if regime == 'calm' else 0.07 if regime == 'normal' else 0.05 if regime == 'storm' else 0.03
        risk_dollar = self.capital * risk_pct
        
        # Find ATM strike
        strike = round(price)
        premium = self._estimate_premium(price, strike, 'call')
        
        if premium <= 0:
            return -0.01
        
        # Calculate quantity
        qty = max(1, int(risk_dollar / (premium * 100)))
        
        # Check capital
        cost = qty * premium * 100
        if cost > self.capital:
            return -0.01
        
        # Execute buy
        self.position = {
            'type': 'call',
            'qty': qty,
            'entry_price': price,
            'entry_premium': premium,
            'strike': strike,
            'entry_step': self.current_step,
            'entry_vix': vix,
            'entry_regime': regime
        }
        
        self.capital -= cost
        
        return 0.0  # No immediate reward, wait for P&L
    
    def _buy_put(self, price: float, vix: float, regime: str) -> float:
        """Buy put option"""
        # Calculate position size
        risk_pct = 0.10 if regime == 'calm' else 0.07 if regime == 'normal' else 0.05 if regime == 'storm' else 0.03
        risk_dollar = self.capital * risk_pct
        
        # Find ATM strike
        strike = round(price)
        premium = self._estimate_premium(price, strike, 'put')
        
        if premium <= 0:
            return -0.01
        
        # Calculate quantity
        qty = max(1, int(risk_dollar / (premium * 100)))
        
        # Check capital
        cost = qty * premium * 100
        if cost > self.capital:
            return -0.01
        
        # Execute buy
        self.position = {
            'type': 'put',
            'qty': qty,
            'entry_price': price,
            'entry_premium': premium,
            'strike': strike,
            'entry_step': self.current_step,
            'entry_vix': vix,
            'entry_regime': regime
        }
        
        self.capital -= cost
        
        return 0.0
    
    def _trim_position(self, trim_pct: float, current_price: float) -> float:
        """Trim position (partial exit)"""
        if not self.position:
            return -0.01
        
        trim_qty = max(1, int(self.position['qty'] * trim_pct))
        current_premium = self._estimate_premium(
            current_price,
            self.position['strike'],
            self.position['type']
        )
        
        # Calculate P&L for trimmed portion
        pnl_per_contract = current_premium - self.position['entry_premium']
        pnl_dollar = pnl_per_contract * trim_qty * 100
        
        # Update position
        self.position['qty'] -= trim_qty
        self.capital += current_premium * trim_qty * 100
        
        # Close position if fully trimmed
        if self.position['qty'] <= 0:
            self.position = None
        
        # Reward based on P&L
        reward = pnl_dollar / self.initial_capital
        return float(reward)
    
    def _exit_position(self, current_price: float) -> float:
        """Full exit from position"""
        if not self.position:
            return -0.01
        
        current_premium = self._estimate_premium(
            current_price,
            self.position['strike'],
            self.position['type']
        )
        
        # Calculate total P&L
        pnl_per_contract = current_premium - self.position['entry_premium']
        pnl_dollar = pnl_per_contract * self.position['qty'] * 100
        
        # Update capital
        self.capital += current_premium * self.position['qty'] * 100
        
        # Record trade
        self.trades.append({
            'type': self.position['type'],
            'entry_premium': self.position['entry_premium'],
            'exit_premium': current_premium,
            'pnl_dollar': pnl_dollar,
            'pnl_pct': pnl_per_contract / self.position['entry_premium'] if self.position['entry_premium'] > 0 else 0.0,
            'qty': self.position['qty'],
            'regime': self.position.get('entry_regime', 'normal')
        })
        
        # Clear position
        self.position = None
        
        # Reward based on P&L
        reward = pnl_dollar / self.initial_capital
        self.daily_pnl += pnl_dollar
        return float(reward)
    
    def _calculate_reward(self, immediate_reward: float, current_price: float) -> float:
        """
        Calculate total reward including position P&L
        
        Args:
            immediate_reward: Reward from action execution
            current_price: Current market price
            
        Returns:
            Total reward
        """
        reward = immediate_reward
        
        # Add unrealized P&L if position exists
        if self.position:
            pnl_pct = self._calculate_position_pnl()
            reward += pnl_pct * 0.1  # Scale down unrealized P&L
        
        # Update peak capital
        self.peak_capital = max(self.peak_capital, self.capital)
        
        # Drawdown penalty
        drawdown = (self.capital - self.peak_capital) / self.peak_capital if self.peak_capital > 0 else 0.0
        if drawdown < 0:
            reward += drawdown * 0.5  # Penalize drawdowns
        
        return float(reward)

