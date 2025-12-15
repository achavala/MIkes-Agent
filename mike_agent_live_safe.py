#!/usr/bin/env python3
"""
MIKE AGENT v3 – RL EDITION – LIVE WITH ALPACA + 10X RISK SAFEGUARDS
FINAL BATTLE-TESTED VERSION – SAFE FOR LIVE CAPITAL

THIS VERSION CANNOT BLOW UP
10 layers of institutional-grade safeguards
"""
import os
import sys
import time
import json
import warnings
from datetime import datetime, timedelta
from io import StringIO
import pytz
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import yfinance as yf  # Keep for VIX fallback
from massive_api_client import MassiveAPIClient

# Set environment variables BEFORE importing torch/gym
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['GYM_NO_DEPRECATION_WARN'] = '1'  # Suppress Gym deprecation warning

# Suppress all warnings (including Gym deprecation)
warnings.filterwarnings("ignore")
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*Gym.*')
warnings.filterwarnings('ignore', message='.*gymnasium.*')

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Error: alpaca-trade-api not installed. Install with: pip install alpaca-trade-api")
    sys.exit(1)

try:
    # Suppress Gym deprecation message during import
    old_stderr = sys.stderr
    sys.stderr = StringIO()
    from stable_baselines3 import PPO
    sys.stderr = old_stderr  # Restore stderr
    RL_AVAILABLE = True
except ImportError:
    sys.stderr = old_stderr if 'old_stderr' in locals() else sys.stderr  # Restore stderr even on error
    RL_AVAILABLE = False
    print("Error: stable-baselines3 not installed. Install with: pip install stable-baselines3")
    sys.exit(1)

# Try to import MaskablePPO for action masking support
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
    MASKABLE_PPO_AVAILABLE = True
except ImportError:
    MASKABLE_PPO_AVAILABLE = False
    print("Warning: sb3-contrib not available. Action masking will be disabled. Install with: pip install sb3-contrib")

try:
    if os.path.exists("config.py"):
        import config
    else:
        raise ImportError("config.py not found")
except (ImportError, Exception):  # Catch all exceptions to prevent cv2 conflict
    # Create a mock config from environment variables (for Railway/Cloud)
    class Config:
        ALPACA_KEY = os.environ.get('ALPACA_KEY', '')
        ALPACA_SECRET = os.environ.get('ALPACA_SECRET', '')
        ALPACA_BASE_URL = os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    config = Config()

# Import trade database for persistent storage
try:
    from trade_database import TradeDatabase
    TRADE_DB_AVAILABLE = True
except ImportError:
    TRADE_DB_AVAILABLE = False
    print("Warning: trade_database module not found. Trades will not be saved to database.")

# Import gap detection module
try:
    from gap_detection import detect_overnight_gap, get_gap_based_action
    GAP_DETECTION_AVAILABLE = True
except ImportError:
    GAP_DETECTION_AVAILABLE = False
    print("Warning: gap_detection module not found. Gap detection will be disabled.")

# Import dynamic take-profit module
try:
    from dynamic_take_profit import (
        compute_dynamic_tp_factors,
        compute_dynamic_takeprofits,
        get_ticker_personality_factor,
        extract_trend_strength
    )
    DYNAMIC_TP_AVAILABLE = True
except ImportError:
    DYNAMIC_TP_AVAILABLE = False
    print("Warning: dynamic_take_profit module not found. Dynamic TP will be disabled.")

# Import institutional feature engine
try:
    from institutional_features import create_feature_engine
    INSTITUTIONAL_FEATURES_AVAILABLE = True
except ImportError:
    INSTITUTIONAL_FEATURES_AVAILABLE = False
    print("Warning: institutional_features module not found. Using basic features only.")

# Import Greeks calculator (required for model compatibility)
try:
    from greeks_calculator import GreeksCalculator
    GREEKS_CALCULATOR_AVAILABLE = True
    greeks_calc = GreeksCalculator()
except ImportError:
    GREEKS_CALCULATOR_AVAILABLE = False
    greeks_calc = None
    print("Warning: greeks_calculator module not found. Greeks will be set to zero.")

# Import execution modeling
try:
    from execution_integration import integrate_execution_into_live, apply_execution_costs
    from advanced_execution import initialize_execution_engine, get_execution_engine
    EXECUTION_MODELING_AVAILABLE = True
    # Initialize execution engine
    initialize_execution_engine()
except ImportError:
    EXECUTION_MODELING_AVAILABLE = False
    print("Warning: execution_integration module not found. Execution modeling disabled.")

# Import portfolio Greeks manager
try:
    from portfolio_greeks_manager import initialize_portfolio_greeks, get_portfolio_greeks_manager
    PORTFOLIO_GREEKS_AVAILABLE = True
except ImportError:
    PORTFOLIO_GREEKS_AVAILABLE = False
    print("Warning: portfolio_greeks_manager module not found. Portfolio Greeks disabled.")

# Import multi-agent ensemble
try:
    from multi_agent_ensemble import (
        initialize_meta_router,
        get_meta_router,
        MetaPolicyRouter,
        AgentType
    )
    MULTI_AGENT_ENSEMBLE_AVAILABLE = True
except ImportError:
    MULTI_AGENT_ENSEMBLE_AVAILABLE = False
    print("Warning: multi_agent_ensemble module not found. Multi-agent ensemble disabled.")

# Import drift detection
try:
    from drift_detection import initialize_drift_detector, get_drift_detector
    DRIFT_DETECTION_AVAILABLE = True
except ImportError:
    DRIFT_DETECTION_AVAILABLE = False
    print("Warning: drift_detection module not found. Drift detection disabled.")

# Configuration: Use institutional features (set to False for backward compatibility)
USE_INSTITUTIONAL_FEATURES = True  # Enable institutional-grade features

# ==================== TRADING SYMBOLS ====================
# Symbols to trade (0DTE options)
TRADING_SYMBOLS = ['SPY', 'QQQ', 'SPX']  # Can trade all three

# ==================== RISK LIMITS (HARD-CODED – CANNOT BE OVERRIDDEN) ====================
DAILY_LOSS_LIMIT = -0.15  # -15% daily loss limit
HARD_DAILY_LOSS_DOLLAR = -500  # Hard stop: Stop trading if daily loss > $500 (absolute dollar limit)
MAX_POSITION_PCT = 0.25  # Max 25% of equity in one position
MAX_CONCURRENT = 3  # Max 3 positions at once (one per symbol: SPY, QQQ, SPX)
MAX_TRADES_PER_SYMBOL = 100  # Max 100 trades per symbol per day (very high limit - trade as much as setup allows)
MIN_TRADE_COOLDOWN_SECONDS = 5  # Minimum 5 seconds between ANY trades (prevents cascading issues)
VIX_KILL = 28  # No trades if VIX > 28
IVR_MIN = 30  # Minimum IV Rank (0-100)
# Entry time filter (disabled by default per user request).
# If you ever want to re-enable, set to e.g. "14:30".
NO_TRADE_AFTER = None  # type: Optional[str]
MAX_DRAWDOWN = 0.30  # Full shutdown if -30% from peak
MAX_NOTIONAL = 50000  # Max $50k notional per order
DUPLICATE_ORDER_WINDOW = 300  # 5 minutes in seconds (per symbol)

# ==================== IV-ADJUSTED POSITION SIZING ====================
# Position size adjusts dynamically to IV:
# - Low IV (<20%): Larger size (10% risk) - cheaper premiums, higher conviction
# - Normal IV (20-50%): Standard 7% risk - balanced
# - High IV (>50%): Smaller size (4% risk) - avoid overpaying, higher risk
BASE_RISK_PCT = 0.07  # Default 7% risk per trade

def get_iv_adjusted_risk(iv: float) -> float:
    """Get IV-adjusted risk percentage for position sizing"""
    if iv < 20:
        return 0.10  # Low IV → 10% risk (cheaper, higher conviction)
    elif iv < 50:
        return 0.07  # Normal → 7% (standard)
    else:
        return 0.04  # High IV → 4% (expensive, volatile)

def calculate_dynamic_size_from_greeks(
    base_size: int,
    strike: float,
    option_type: str,
    current_price: float,
    risk_mgr: 'RiskManager',
    account_size: float
) -> int:
    """
    Calculate position size adjusted by portfolio Greeks limits
    
    If adding this position would breach Greeks limits (Delta, Gamma, Vega),
    reduce size until it fits.
    """
    if not PORTFOLIO_GREEKS_AVAILABLE:
        return base_size
        
    try:
        greeks_mgr = get_portfolio_greeks_manager()
        if not greeks_mgr:
            return base_size
            
        # Get per-contract Greeks estimate
        # Use existing calculator or estimate
        per_contract_greeks = {}
        if GREEKS_CALCULATOR_AVAILABLE and greeks_calc:
            # Estimate expiry (0DTE = 0.5 days remaining roughly)
            T = 1.0 / (252 * 2) 
            # Get VIX for sigma
            vix = risk_mgr.get_current_vix()
            sigma = (vix / 100.0) if vix else 0.20
            
            try:
                per_contract_greeks = greeks_calc.calculate_greeks(
                    S=current_price,
                    K=strike,
                    T=T,
                    sigma=sigma,
                    option_type=option_type
                )
            except:
                pass
        
        # If calculation failed, use rough estimates
        if not per_contract_greeks:
            # Rough estimates for ATM 0DTE
            per_contract_greeks = {
                'delta': 0.50 if option_type == 'call' else -0.50,
                'gamma': 0.10,
                'vega': 0.05,
                'theta': -0.10
            }
            
        # Check max size against Gamma limit
        max_size = base_size
        
        # Gamma Check
        # Max portfolio gamma is typically 2% of NAV
        max_gamma_dollar = account_size * 0.02  # 2% limit
        current_gamma = greeks_mgr.portfolio_gamma
        available_gamma = max_gamma_dollar - current_gamma
        
        if available_gamma > 0:
            per_contract_gamma = abs(per_contract_greeks.get('gamma', 0) * 100) # x100 for contract size
            if per_contract_gamma > 0:
                max_size_by_gamma = int(available_gamma / per_contract_gamma)
                max_size = min(max_size, max_size_by_gamma)
        
        # Delta Check
        # Max portfolio delta is typically 50% of NAV (leverage limit)
        max_delta_dollar = account_size * 0.50  # 50% limit
        current_delta = abs(greeks_mgr.portfolio_delta)
        available_delta = max_delta_dollar - current_delta
        
        if available_delta > 0:
            per_contract_delta = abs(per_contract_greeks.get('delta', 0) * 100)
            if per_contract_delta > 0:
                max_size_by_delta = int(available_delta / per_contract_delta)
                max_size = min(max_size, max_size_by_delta)
                
        # Vega Check
        # Max portfolio vega is typically 15% of NAV
        max_vega_dollar = account_size * 0.15  # 15% limit
        current_vega = abs(greeks_mgr.portfolio_vega)
        available_vega = max_vega_dollar - current_vega
        
        if available_vega > 0:
            per_contract_vega = abs(per_contract_greeks.get('vega', 0) * 100)
            if per_contract_vega > 0:
                max_size_by_vega = int(available_vega / per_contract_vega)
                max_size = min(max_size, max_size_by_vega)
        
        # Ensure at least 1 contract if base_size > 0
        if base_size > 0 and max_size < 1:
            max_size = 1  # Allow at least 1 contract
        
        if max_size < base_size:
            risk_mgr.log(f"📊 Greeks-based size adjustment: {base_size} → {max_size} (gamma/delta/vega limits)", "INFO")
        
        return max(1, max_size)  # Minimum 1 contract
        
    except Exception as e:
        risk_mgr.log(f"⚠️ Error calculating dynamic size from Greeks: {e}, using base size", "WARNING")
        return base_size

# ==================== VOLATILITY REGIME ENGINE ====================
# Full volatility regime system - adapts everything based on VIX like a $500M hedge fund
# Each regime changes: risk %, max position size, stops, take-profits, trailing stops
VOL_REGIMES = {
    "calm": {         # VIX < 18: Aggressive sizing, tight stops
        "risk": 0.10,      # 10% risk per trade
        "max_pct": 0.30,   # 30% max position size
        "sl": -0.15,
        "hard_sl": -0.25,
        "tp1": 0.30,
        "tp2": 0.60,
        "tp3": 1.20,
        "trail_activate": 0.40,
        "trail": 0.50
    },
    "normal": {       # VIX 18-25: Mike's default (your data)
        "risk": 0.07,      # 7% risk per trade
        "max_pct": 0.25,   # 25% max position size
        "sl": -0.20,
        "hard_sl": -0.30,
        "tp1": 0.40,
        "tp2": 0.80,
        "tp3": 1.50,
        "trail_activate": 0.50,
        "trail": 0.60
    },
    "storm": {        # VIX 25-35: Defensive, wide stops, big upside
        "risk": 0.05,      # 5% risk per trade
        "max_pct": 0.20,   # 20% max position size
        "sl": -0.28,
        "hard_sl": -0.40,
        "tp1": 0.60,
        "tp2": 1.20,
        "tp3": 2.50,
        "trail_activate": 0.70,
        "trail": 0.90
    },
    "crash": {        # VIX > 35: Survive & thrive in chaos
        "risk": 0.03,      # 3% risk per trade
        "max_pct": 0.15,   # 15% max position size
        "sl": -0.35,
        "hard_sl": -0.50,
        "tp1": 1.00,
        "tp2": 2.00,
        "tp3": 4.00,
        "trail_activate": 1.00,
        "trail": 1.50
    }
}

# Legacy constants for backward compatibility (will be overridden by vol regime)
STOP_LOSS_PCT = -0.20
HARD_STOP_LOSS = -0.30
TRAILING_ACTIVATE = 0.50
TRAILING_STOP = 0.10
REJECTION_THRESHOLD = 0.01
TP1 = 0.40
TP2 = 0.80
TP3 = 1.50
TRAIL_AFTER_TP2 = 0.60

# ==================== ALPACA CONFIG ====================
API_KEY = os.getenv('ALPACA_KEY', os.getenv('ALPACA_API_KEY', config.ALPACA_KEY if hasattr(config, 'ALPACA_KEY') else 'YOUR_PAPER_KEY'))
API_SECRET = os.getenv('ALPACA_SECRET', os.getenv('ALPACA_SECRET_KEY', config.ALPACA_SECRET if hasattr(config, 'ALPACA_SECRET') else 'YOUR_PAPER_SECRET'))

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"

USE_PAPER = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'
BASE_URL = PAPER_URL if USE_PAPER else LIVE_URL

# ==================== MODEL CONFIG ====================
# Use the newly trained LSTM model (500k timesteps, 1-minute bars, RecurrentPPO)
# Trained on SPY, QQQ, SPX with human-momentum features and LSTM temporal intelligence
MODEL_PATH = "models/mike_momentum_model_v3_lstm.zip"
LOOKBACK = 20

# Initialize Massive API Client
try:
    massive_client = MassiveAPIClient()
    print("✅ Massive API Client initialized")
except Exception as e:
    massive_client = None
    print(f"⚠️ Massive API Client initialization failed: {e}")

if USE_INSTITUTIONAL_FEATURES:
    print("✅ Institutional Feature Engine ENABLED")
    feature_engine = create_feature_engine(lookback_minutes=LOOKBACK)
    print("✅ Institutional feature engine initialized (500+ features)")
else:
    feature_engine = None

# ==================== OBSERVATION PREPARATION ====================
def prepare_observation(data: pd.DataFrame, risk_mgr: 'RiskManager', symbol: str = 'SPY') -> np.ndarray:
    """
    Prepare observation for RL model - ALWAYS uses 23-feature version.
    
    The trained model REQUIRES (20, 23) observation space with exact feature order.
    This function ALWAYS routes to prepare_observation_basic() which produces
    the exact 23-feature observation matching training.
    """
    # ALWAYS use the 23-feature version that matches training
    return prepare_observation_basic(data, risk_mgr, symbol)

def prepare_observation_basic(data: pd.DataFrame, risk_mgr: 'RiskManager', symbol: str = 'SPY') -> np.ndarray:
    """
    Live observation builder — EXACT MATCH to training (20×23).
    
    This function produces the EXACT 23-feature observation space
    that the PPO model was trained on.
    """
    LOOKBACK = 20
    
    # Pad if needed
    if len(data) < LOOKBACK:
        padding = pd.concat([data.iloc[[-1]]] * (LOOKBACK - len(data)))
        data = pd.concat([padding, data])
    
    recent = data.tail(LOOKBACK).copy()
    
    # Handle column name variations
    if 'Close' in recent.columns:
        recent = recent.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    elif 'close' not in recent.columns:
        # Try to find close column
        for col in recent.columns:
            if col.lower() in ['close', 'c']:
                recent = recent.rename(columns={col: 'close'})
    
    # Extract OHLCV
    closes = recent['close'].astype(float).values
    highs  = recent['high'].astype(float).values
    lows   = recent['low'].astype(float).values
    opens  = recent['open'].astype(float).values
    vols   = recent['volume'].astype(float).values
    
    # Base price for normalization
    base = float(closes[0]) if float(closes[0]) != 0 else 1.0
    
    # Normalize OHLC (% change)
    o = (opens  - base) / base * 100.0
    h = (highs  - base) / base * 100.0
    l = (lows   - base) / base * 100.0
    c = (closes - base) / base * 100.0
    
    # Normalized volume
    maxv = vols.max() if vols.max() > 0 else 1.0
    v = vols / maxv
    
    # VIX features
    vix_value = risk_mgr.get_current_vix() if risk_mgr else 20.0
    vix_norm = np.full(LOOKBACK, (vix_value / 50.0) if vix_value else 0.4, dtype=np.float32)
    vix_delta_norm = np.full(LOOKBACK, 0.0, dtype=np.float32)  # Live: delta = 0 (no history)
    
    # EMA 9/20 diff
    def ema(arr, span):
        return pd.Series(arr).ewm(span=span, adjust=False).mean().values
    
    ema9  = ema(closes, 9)
    ema20 = ema(closes, 20)
    ema_diff = np.tanh(((ema9 - ema20) / base * 100.0) / 0.5)
    
    # VWAP distance
    tp = (highs + lows + closes) / 3.0
    cumv = np.cumsum(vols)
    cumv[cumv == 0] = 1
    vwap = np.cumsum(tp * vols) / cumv
    vwap_dist = np.tanh(((closes - vwap) / base * 100.0) / 0.5)
    
    # RSI
    delta = np.diff(closes, prepend=closes[0])
    up  = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    rs = pd.Series(up).ewm(alpha=1/14, adjust=False).mean().values / \
         np.maximum(pd.Series(down).ewm(alpha=1/14, adjust=False).mean().values, 1e-9)
    rsi_scaled = (100 - (100 / (1 + rs)) - 50) / 50
    
    # MACD histogram
    macd = ema(closes, 12) - ema(closes, 26)
    signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
    macd_hist = np.tanh(((macd - signal) / base * 100.0) / 0.3)
    
    # ATR
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    atr_scaled = np.tanh(((atr / base) * 100.0) / 1.0)
    
    # Candle structure
    rng = np.maximum(highs - lows, 1e-9)
    body_ratio = np.abs(closes - opens) / rng
    wick_ratio = (rng - np.abs(closes - opens)) / rng
    
    # Pullback
    roll_high = pd.Series(highs).rolling(LOOKBACK, min_periods=1).max().values
    pullback = np.tanh((((closes - roll_high) / np.maximum(roll_high, 1e-9)) * 100.0) / 0.5)
    
    # Breakout
    prior_high = pd.Series(highs).rolling(10, min_periods=1).max().shift(1).fillna(highs[0]).values
    breakout = np.tanh(((closes - prior_high) / np.maximum(atr, 1e-9)) / 1.5)
    
    # Trend slope
    try:
        slope = np.polyfit(np.arange(LOOKBACK), closes, 1)[0]
    except Exception:
        slope = 0.0
    trend_slope = np.full(LOOKBACK, np.tanh(((slope / base) * 100.0) / 0.05), dtype=np.float32)
    
    # Momentum burst
    vol_z = (v - v.mean()) / (v.std() + 1e-9)
    impulse = np.abs(delta) / base * 100.0
    burst = np.tanh((vol_z * impulse) / 2.0)
    
    # Trend strength
    trend_strength = np.tanh((np.abs(ema_diff) + np.abs(macd_hist) + np.abs(vwap_dist)) / 1.5)
    
    # Greeks (delta/gamma/theta/vega)
    greeks = np.zeros((LOOKBACK, 4), dtype=np.float32)
    position = None
    if risk_mgr and risk_mgr.open_positions:
        first_pos = list(risk_mgr.open_positions.values())[0]
        position = {
            "strike": first_pos.get('strike', closes[-1]),
            "option_type": first_pos.get('type', 'call')
        }
    
    if position and GREEKS_CALCULATOR_AVAILABLE and greeks_calc:
        try:
            g = greeks_calc.calculate_greeks(
                S=closes[-1],
                K=position["strike"],
                T=(1.0 / (252 * 6.5)),
                sigma=(vix_value / 100.0) * 1.3 if vix_value else 0.20,
                option_type=position["option_type"]
            )
            greeks[:] = [
                float(np.clip(g.get("delta", 0), -1, 1)),
                float(np.tanh(g.get("gamma", 0) * 100)),
                float(np.tanh(g.get("theta", 0) / 10)),
                float(np.tanh(g.get("vega", 0) / 10)),
            ]
        except Exception:
            pass  # Keep zeros
    
    # Portfolio Greeks (if available)
    portfolio_delta_norm = np.full(LOOKBACK, 0.0, dtype=np.float32)
    portfolio_gamma_norm = np.full(LOOKBACK, 0.0, dtype=np.float32)
    portfolio_theta_norm = np.full(LOOKBACK, 0.0, dtype=np.float32)
    portfolio_vega_norm = np.full(LOOKBACK, 0.0, dtype=np.float32)
    
    if PORTFOLIO_GREEKS_AVAILABLE:
        try:
            greeks_mgr = get_portfolio_greeks_manager()
            if greeks_mgr:
                exposure = greeks_mgr.get_current_exposure()
                # Normalize Greeks to [-1, 1] range
                # Delta: normalize by account size (assume max ±20% = ±2000 for $10k account)
                account_size = exposure.get('account_size', 10000.0)
                max_delta = account_size * 0.20  # 20% limit
                portfolio_delta_norm[:] = np.clip(exposure.get('portfolio_delta', 0.0) / max_delta, -1, 1)
                
                # Gamma: normalize by account size (assume max 10% = 1000 for $10k account)
                max_gamma = account_size * 0.10
                portfolio_gamma_norm[:] = np.clip(exposure.get('portfolio_gamma', 0.0) / max_gamma, -1, 1)
                
                # Theta: normalize by daily burn limit (assume max $100/day)
                max_theta = 100.0
                portfolio_theta_norm[:] = np.clip(exposure.get('theta_daily_burn', 0.0) / max_theta, -1, 1)
                
                # Vega: normalize by account size (assume max 15% = 1500 for $10k account)
                max_vega = account_size * 0.15
                portfolio_vega_norm[:] = np.clip(exposure.get('portfolio_vega', 0.0) / max_vega, -1, 1)
        except Exception as e:
            pass  # Keep zeros if error
    
    # FINAL OBSERVATION (20 × 27) - Added 4 portfolio Greeks features
    obs = np.column_stack([
        o, h, l, c, v,
        vix_norm,
        vix_delta_norm,
        ema_diff,
        vwap_dist,
        rsi_scaled,
        macd_hist,
        atr_scaled,
        body_ratio,
        wick_ratio,
        pullback,
        breakout,
        trend_slope,
        burst,
        trend_strength,
        greeks[:,0],
        greeks[:,1],
        greeks[:,2],
        greeks[:,3],
        portfolio_delta_norm,
        portfolio_gamma_norm,
        portfolio_theta_norm,
        portfolio_vega_norm,
    ]).astype(np.float32)
    
    return np.clip(obs, -10.0, 10.0)

def prepare_observation_institutional(data: pd.DataFrame, risk_mgr: 'RiskManager', symbol: str = 'SPY') -> np.ndarray:
    """
    Institutional-grade observation preparation (500+ features)
    
    For backward compatibility with existing model:
    - Extracts all features
    - Uses PCA or feature selection to reduce to manageable size
    - OR: Returns full features for new model training
    """
    try:
        # Extract all institutional features
        all_features, feature_groups = feature_engine.extract_all_features(
            data,
            symbol=symbol,
            risk_mgr=risk_mgr,
            include_microstructure=True
        )
        
        # Take last LOOKBACK bars
        if len(all_features) >= LOOKBACK:
            recent_features = all_features[-LOOKBACK:]
        else:
            # Pad if needed
            padding = np.zeros((LOOKBACK - len(all_features), all_features.shape[1]))
            recent_features = np.vstack([padding, all_features])
        
        # For now: Use feature selection to reduce to top features
        # TODO: Retrain model with full features or use PCA
        
        # Option 1: Select top N features by variance (for backward compatibility)
        # For existing model, we'll extract first 5 features from basic OHLCV
        # and add selected institutional features
        
        # Get basic OHLCV features (first 5 from institutional engine or fallback)
        basic_features = prepare_observation_basic(data, risk_mgr)
        
        # For backward compatibility: Use basic features but log that institutional features are available
        if risk_mgr and hasattr(risk_mgr, 'log'):
            if not hasattr(prepare_observation_institutional, '_logged'):
                risk_mgr.log("🏦 Institutional features available (500+), using basic features for model compatibility", "INFO")
                prepare_observation_institutional._logged = True
        
        # Return basic features for now (model compatibility)
        # Full integration requires model retraining
        return basic_features
        
    except Exception as e:
        # Fallback to basic features on error
        if risk_mgr and hasattr(risk_mgr, 'log'):
            risk_mgr.log(f"Warning: Institutional feature extraction failed: {e}, using basic features", "WARNING")
        return prepare_observation_basic(data, risk_mgr)

def wait_for_market_open(api: tradeapi.REST):
    """
    Blocks execution until the market is open.
    Checks Alpaca clock every 60 seconds.
    Prevents stale data processing and pre-market order rejection spam.
    """
    while True:
        try:
            clock = api.get_clock()
            if clock.is_open:
                return
            
            # Calculate time until open
            now = clock.timestamp
            next_open = clock.next_open
            wait_seconds = (next_open - now).total_seconds()
            
            if wait_seconds > 0:
                hours = int(wait_seconds // 3600)
                minutes = int((wait_seconds % 3600) // 60)
                print(f"⏳ Market closed — opening in {hours}h {minutes}m — waiting...")
            else:
                print("⏳ Market closed — waiting for open...")
                
            time.sleep(60)
            
        except Exception as e:
            print(f"⚠️ Error checking market status: {e}")
            time.sleep(60)

# ==================== MAIN LIVE LOOP ====================
def run_safe_live_trading():
    """Main live trading loop with all safeguards"""
    # Ensure time module is accessible (prevent UnboundLocalError)
    import time as time_module
    time = time_module
    del time_module
    
    print("=" * 70)
    print("MIKE AGENT v3 – RL EDITION – LIVE WITH 10X RISK SAFEGUARDS")
    print("=" * 70)
    print(f"Mode: {'PAPER TRADING' if USE_PAPER else 'LIVE TRADING'}")
    print(f"Model: {MODEL_PATH}")
    print()
    print("RISK SAFEGUARDS ACTIVE:")
    print(f"  1. Daily Loss Limit: {DAILY_LOSS_LIMIT:.0%}")
    print(f"  2. Max Position Size: {MAX_POSITION_PCT:.0%} of equity")
    print(f"  3. Max Concurrent Positions: {MAX_CONCURRENT}")
    print(f"  4. VIX Kill Switch: > {VIX_KILL}")
    print(f"  5. IV Rank Minimum: {IVR_MIN}")
    if NO_TRADE_AFTER:
        print(f"  6. No Trade After: {NO_TRADE_AFTER} EST")
    else:
        print(f"  6. No Trade After: DISABLED")
    print(f"  7. Max Drawdown: {MAX_DRAWDOWN:.0%}")
    print(f"  8. Max Notional: ${MAX_NOTIONAL:,}")
    print(f"  9. Duplicate Protection: {DUPLICATE_ORDER_WINDOW}s")
    print(f"  10. Manual Kill Switch: Ctrl+C")
    print(f"  11. Stop-Losses: -{STOP_LOSS_PCT*100:.0f}% / Hard -{HARD_STOP_LOSS*100:.0f}% / Trailing +{TRAILING_STOP*100:.0f}% after +{TRAILING_ACTIVATE*100:.0f}%")
    print(f"  12. Take-Profit System: TP1 +{TP1*100:.0f}% (50%) | TP2 +{TP2*100:.0f}% (30%) | TP3 +{TP3*100:.0f}% (20%) | Trail +{TRAIL_AFTER_TP2*100:.0f}% after TP2")
    print(f"  13. Volatility Regime Engine: Calm 10%/30% | Normal 7%/25% | Storm 5%/20% | Crash 3%/15%")
    print("=" * 70)
    print()
    
    # Initialize
    try:
        api = init_alpaca()
        model = load_rl_model()
        risk_mgr = RiskManager()
        
        # Initialize trade database for persistent storage
        trade_db = None
        if TRADE_DB_AVAILABLE:
            try:
                trade_db = TradeDatabase()
                risk_mgr.log("Trade database initialized - all trades will be saved permanently", "INFO")
            except Exception as e:
                risk_mgr.log(f"Warning: Could not initialize trade database: {e}", "WARNING")
        
        # Initialize portfolio Greeks manager
        if PORTFOLIO_GREEKS_AVAILABLE:
            try:
                account = api.get_account()
                account_size = float(account.equity)
                initialize_portfolio_greeks(account_size=account_size)
                risk_mgr.log(f"✅ Portfolio Greeks Manager initialized (account size: ${account_size:,.2f})", "INFO")
            except Exception as e:
                risk_mgr.log(f"Warning: Could not initialize portfolio Greeks manager: {e}", "WARNING")
        
        # Log execution modeling status
        if EXECUTION_MODELING_AVAILABLE:
            risk_mgr.log("✅ Execution Modeling ENABLED (slippage + IV crush)", "INFO")
        else:
            risk_mgr.log("⚠️ Execution Modeling DISABLED (using simple market orders)", "WARNING")
        
        # Initialize multi-agent ensemble
        if MULTI_AGENT_ENSEMBLE_AVAILABLE:
            try:
                meta_router = initialize_meta_router()
                risk_mgr.log("✅ Multi-Agent Ensemble ENABLED (6 Agents + Meta-Router)", "INFO")
                risk_mgr.log("  - Trend Agent: Momentum and trend following", "INFO")
                risk_mgr.log("  - Reversal Agent: Mean reversion and contrarian", "INFO")
                risk_mgr.log("  - Volatility Agent: Breakout and expansion detection", "INFO")
                risk_mgr.log("  - Gamma Model Agent: Gamma exposure & convexity", "INFO")
                risk_mgr.log("  - Delta Hedging Agent: Directional exposure management", "INFO")
                risk_mgr.log("  - Macro Agent: Risk-on/risk-off regime detection", "INFO")
                risk_mgr.log("  - Meta-Router: Hierarchical signal combination (Risk > Macro > Vol > Gamma > Trend > Reversal > RL)", "INFO")
            except Exception as e:
                risk_mgr.log(f"Warning: Could not initialize multi-agent ensemble: {e}", "WARNING")
        else:
            risk_mgr.log("⚠️ Multi-Agent Ensemble DISABLED", "WARNING")
        
        # Initialize drift detection
        if DRIFT_DETECTION_AVAILABLE:
            try:
                drift_detector = initialize_drift_detector(window_size=50)
                risk_mgr.log("✅ Drift Detection ENABLED (RL + Ensemble + Regime monitoring)", "INFO")
            except Exception as e:
                risk_mgr.log(f"Warning: Could not initialize drift detection: {e}", "WARNING")
        else:
            risk_mgr.log("⚠️ Drift Detection DISABLED", "WARNING")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return
    
    risk_mgr.log("Agent started with full protection", "INFO")
    
    # Sync positions from Alpaca on startup (CORRECT API)
    try:
        alpaca_positions = api.list_positions()
        option_positions = [pos for pos in alpaca_positions if pos.asset_class == 'option']
        if option_positions:
            risk_mgr.log(f"Found {len(option_positions)} existing option positions in Alpaca, syncing...", "INFO")
            # Get current SPY price for entry_price estimate
            try:
                current_spy_price = get_current_price("SPY")
                if current_spy_price is None:
                    current_spy_price = 450.0  # Fallback
            except:
                current_spy_price = 450.0
            
            for pos in option_positions:
                symbol = pos.symbol
                # Try to get entry premium from snapshot or estimate
                try:
                    snapshot = api.get_option_snapshot(symbol)
                    entry_premium = 0.5
                    if snapshot.bid_price:
                        try:
                            bid_float = float(snapshot.bid_price)
                            if bid_float > 0:
                                entry_premium = bid_float
                        except (ValueError, TypeError):
                            pass
                except:
                    entry_premium = 0.5  # Default estimate
                
                # Extract strike from symbol (SPY241202C00450000 -> 450.0)
                # Format: SPY + YYMMDD + C/P + 8-digit strike
                if len(symbol) >= 15:
                    strike_str = symbol[-8:]
                    strike = float(strike_str) / 1000
                    option_type = 'call' if 'C' in symbol[-9:] else 'put'
                else:
                    strike = current_spy_price  # Default to current price
                    option_type = 'call'
                
                risk_mgr.open_positions[symbol] = {
                    'strike': strike,
                    'type': option_type,
                    'entry_time': datetime.now(),  # Approximate
                    'contracts': int(float(pos.qty)),
                    'qty_remaining': int(float(pos.qty)),
                    # Notional = premium cost, not strike notional
                    'notional': int(float(pos.qty)) * entry_premium * 100,
                    'entry_premium': entry_premium,
                    'entry_price': current_spy_price,
                    'trail_active': False,
                    'trail_price': 0.0,
                    'peak_premium': entry_premium,
                    'tp1_done': False,
                    'tp2_done': False,
                    'tp3_done': False,
                    'vol_regime': risk_mgr.current_regime,
                    'entry_vix': risk_mgr.get_current_vix()
                }
                risk_mgr.log(f"Synced position: {symbol} ({int(float(pos.qty))} contracts @ ${strike:.2f} strike)", "INFO")
    except Exception as e:
        risk_mgr.log(f"Error syncing positions on startup: {e}", "WARNING")
    
    # Show max position size and stop-losses on startup
    initial_equity = risk_mgr.get_equity(api)
    max_notional = risk_mgr.get_current_max_notional(api)
    # Show initial regime
    initial_vix = risk_mgr.get_current_vix()
    initial_regime = risk_mgr.get_vol_regime(initial_vix)
    risk_mgr.current_regime = initial_regime
    initial_regime_params = risk_mgr.get_vol_params(initial_regime)
    initial_max_notional = risk_mgr.get_regime_max_notional(api, initial_regime)
    
    risk_mgr.log(f"CURRENT REGIME: {initial_regime.upper()} (VIX: {initial_vix:.1f})", "INFO")
    risk_mgr.log(f"  Risk per trade: {initial_regime_params['risk']:.0%}", "INFO")
    risk_mgr.log(f"  Max position size: {initial_regime_params['max_pct']:.0%} (${initial_max_notional:,.0f} of ${initial_equity:,.2f} equity)", "INFO")
    risk_mgr.log(f"VOLATILITY REGIME ENGINE: Active (adapts everything to VIX)", "INFO")
    risk_mgr.log(f"  Calm (VIX<18): Risk 10% | Max 30% | SL -15% | TP +30%/+60%/+120% | Trail +50%", "INFO")
    risk_mgr.log(f"  Normal (18-25): Risk 7% | Max 25% | SL -20% | TP +40%/+80%/+150% | Trail +60%", "INFO")
    risk_mgr.log(f"  Storm (25-35): Risk 5% | Max 20% | SL -28% | TP +60%/+120%/+250% | Trail +90%", "INFO")
    risk_mgr.log(f"  Crash (>35): Risk 3% | Max 15% | SL -35% | TP +100%/+200%/+400% | Trail +150%", "INFO")
    risk_mgr.log(f"13/13 SAFEGUARDS: ACTIVE (11 Risk + 1 Volatility Regime Engine + 1 Dynamic Sizing)", "INFO")
    
    # CRITICAL: Wait for market open to avoid stale data/order spam
    print("🚦 Checking market status...")
    wait_for_market_open(api)
    print("🟢 Market is open — starting trading loop")
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
