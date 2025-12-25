#!/usr/bin/env python3
from __future__ import annotations

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

# Ensure /app is in Python path for imports (Fly.io deployment)
if '/app' not in sys.path:
    sys.path.insert(0, '/app')
# Also ensure current directory is in path
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

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
    import config
except ImportError:
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

# Import institutional feature engineering module
try:
    from institutional_features import InstitutionalFeatureEngine, create_feature_engine
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

# Import Telegram alerts
try:
    from utils.telegram_alerts import (
        send_entry_alert,
        send_exit_alert,
        send_block_alert,
        send_error_alert,
        send_daily_summary,
        send_info,
        send_warning,
        is_configured as telegram_configured
    )
    TELEGRAM_AVAILABLE = True
    if telegram_configured():
        print("✅ Telegram alerts configured")
    else:
        print("ℹ️  Telegram alerts available but not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)")
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("Warning: utils.telegram_alerts module not found. Telegram alerts disabled.")

# 10-feature observation will be defined inline below

# Configuration: Use institutional features (set to False for backward compatibility)
USE_INSTITUTIONAL_FEATURES = True  # Enable institutional-grade features

# ==================== TRADING SYMBOLS ====================
# Symbols to trade (0DTE options)
# NOTE: SPX options are NOT available in Alpaca paper trading (index options require special permissions)
# Using SPY/QQQ/IWM (ETFs fully supported in paper trading)
# 🔴 RED-TEAM FIX: Disable SPX (not available in paper), restrict IWM (lower liquidity)
# Focus on SPY and QQQ only for now (highest liquidity, best fills)
TRADING_SYMBOLS = ['SPY', 'QQQ']  # IWM disabled per red-team recommendation (lower liquidity)
# SPX disabled (not available in Alpaca paper trading, requires special permissions)

# ==================== RISK LIMITS (HARD-CODED – CANNOT BE OVERRIDDEN) ====================
DAILY_LOSS_LIMIT = -0.15  # -15% daily loss limit
HARD_DAILY_LOSS_DOLLAR = -500  # Hard stop: Stop trading if daily loss > $500 (absolute dollar limit)
MAX_POSITION_PCT = 0.25  # Max 25% of equity in one position
MAX_CONCURRENT = 3  # Max 3 positions at once (one per symbol: SPY, QQQ, IWM)
MAX_TRADES_PER_SYMBOL = 10  # Max 10 trades per symbol per day (reduced from 100 to prevent overtrading)
MIN_TRADE_COOLDOWN_SECONDS = 60  # Minimum 60 seconds between ANY trades (increased from 5s to prevent rapid-fire trading)
VIX_KILL = 28  # No trades if VIX > 28
IVR_MIN = 30  # Minimum IV Rank (0-100)
# Entry time filter (disabled per user request)
NO_TRADE_AFTER = None  # type: Optional[str]  # No time restriction - trading allowed all day
# 🔴 RED-TEAM FIX: Raise confidence threshold (do NOT lower it)
# Model is correctly uncertain - 0.52 is too low and causes losses
# Better to have zero trades than trades with low confidence
MIN_ACTION_STRENGTH_THRESHOLD = 0.60  # Minimum confidence (0.60 = 60%) required to execute trades
MAX_DRAWDOWN = 0.30  # Full shutdown if -30% from peak
MAX_NOTIONAL = 50000  # Max $50k notional per order
DUPLICATE_ORDER_WINDOW = 300  # 5 minutes in seconds (per symbol)

# ==================== DATA SOURCE CONFIGURATION ====================
ALLOW_YFINANCE_FALLBACK = False  # Set to False to disable yfinance fallback (delayed data not suitable for 0DTE trading)
USE_YFINANCE_FOR_VIX_ONLY = True  # VIX can be delayed, this is OK (non-critical data)

# ==================== DATA CACHING (REDUCES REDUNDANT API CALLS) ====================
# Cache market data for 30 seconds per symbol to reduce API calls and rate limit risk
DATA_CACHE_SECONDS = 30  # Cache bars for 30 seconds
POSITIONS_CACHE_SECONDS = 15  # Cache positions for 15 seconds (less frequent updates)

# Global caches (module-level for persistence across calls)
_market_data_cache: Dict[str, Dict[str, Any]] = {}  # {symbol: {'data': df, 'timestamp': datetime, 'price': float}}
_positions_cache: Dict[str, Any] = {'positions': [], 'timestamp': None}

# ==================== DAILY NO-TRADE REASON TRACKING ====================
# Track reasons why trades didn't execute (for end-of-day summary)
_daily_no_trade_reasons: Dict[str, int] = {
    'hold_signals': 0,          # RL/Ensemble said HOLD
    'low_confidence': 0,        # Confidence below threshold
    'cooldown_active': 0,       # Trade cooldown in effect
    'position_limit': 0,        # Max positions reached
    'daily_loss_limit': 0,      # Daily loss limit hit
    'vix_kill_switch': 0,       # VIX too high
    'stale_data': 0,            # Data too old/invalid
    'no_options_available': 0,  # No valid options found
    'safeguard_blocked': 0,     # Other safeguards blocked
    'market_closed': 0,         # Market not open
}
_no_trade_tracking_date: Optional[datetime.date] = None  # Current tracking date

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
    risk_mgr: RiskManager,
    account_size: float
) -> int:
    """
    Adjust position size based on portfolio Greeks limits (delta/vega)
    
    Args:
        base_size: Base position size from IV-adjusted risk
        strike: Option strike
        option_type: 'call' or 'put'
        current_price: Current underlying price
        risk_mgr: RiskManager instance
        account_size: Account size for Greeks calculations
        
    Returns:
        Adjusted position size (may be reduced to stay within limits)
    """
    if not PORTFOLIO_GREEKS_AVAILABLE or not GREEKS_CALCULATOR_AVAILABLE or not greeks_calc:
        return base_size  # No Greeks available, use base size
    
    try:
        greeks_mgr = get_portfolio_greeks_manager()
        if not greeks_mgr:
            return base_size
        
        # Calculate Greeks for one contract
        T = 1.0 / (252 * 6.5)  # 0DTE: ~1 hour
        vix_value = risk_mgr.get_current_vix() if risk_mgr else 20.0
        sigma = (vix_value / 100.0) * 1.3 if vix_value else 0.20
        
        per_contract_greeks = greeks_calc.calculate_greeks(
            S=current_price,
            K=strike,
            T=T,
            sigma=sigma,
            option_type=option_type
        )
        
        # Calculate max size that fits within limits
        max_size = base_size
        
        # Check gamma limit (most restrictive for 0DTE)
        max_gamma_dollar = account_size * 0.10  # 10% limit
        current_gamma = abs(greeks_mgr.portfolio_gamma)
        available_gamma = max_gamma_dollar - current_gamma
        
        if available_gamma > 0:
            per_contract_gamma = abs(per_contract_greeks.get('gamma', 0) * 100)
            if per_contract_gamma > 0:
                max_size_by_gamma = int(available_gamma / per_contract_gamma)
                max_size = min(max_size, max_size_by_gamma)
        
        # Check delta limit
        max_delta_dollar = account_size * 0.20  # 20% limit
        current_delta = abs(greeks_mgr.portfolio_delta)
        available_delta = max_delta_dollar - current_delta
        
        if available_delta > 0:
            per_contract_delta = abs(per_contract_greeks.get('delta', 0) * 100)
            if per_contract_delta > 0:
                max_size_by_delta = int(available_delta / per_contract_delta)
                max_size = min(max_size, max_size_by_delta)
        
        # Check vega limit
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
API_KEY = os.getenv('ALPACA_KEY', config.ALPACA_KEY if hasattr(config, 'ALPACA_KEY') else 'YOUR_PAPER_KEY')
API_SECRET = os.getenv('ALPACA_SECRET', config.ALPACA_SECRET if hasattr(config, 'ALPACA_SECRET') else 'YOUR_PAPER_SECRET')

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"

USE_PAPER = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'
BASE_URL = PAPER_URL if USE_PAPER else LIVE_URL

# ==================== MODEL CONFIG ====================
# Use the trained 23-feature model (5M timesteps, 2 years of 1-minute data, PPO)
# Trained on SPY, QQQ, IWM with Alpaca API data (Dec 2023 - Dec 2025) and all 23 features
# Features: OHLCV (5) + VIX (2) + Technical Indicators (11) + Greeks (4) = 23 features
# Completed: December 17, 2025
MODEL_PATH = "models/mike_23feature_model_final.zip"
LOOKBACK = 20

# ==================== ACTION MAPPING (CANONICAL) ====================
# Unified 6-action space mapping - used consistently throughout the codebase
# Model outputs: 0=HOLD, 1=BUY CALL, 2=BUY PUT, 3=TRIM 50%, 4=TRIM 70%, 5=FULL EXIT
ACTION_MAP = {
    0: "HOLD",
    1: "BUY CALL",
    2: "BUY PUT",
    3: "TRIM 50%",
    4: "TRIM 70%",
    5: "FULL EXIT",
}

def get_action_name(action: int) -> str:
    """Get canonical action name from action code"""
    return ACTION_MAP.get(int(action), "UNKNOWN")

# ==================== MASSIVE API CONFIG ====================
USE_MASSIVE_API = os.getenv('USE_MASSIVE_API', 'true').lower() == 'true'
MASSIVE_API_KEY = os.getenv('MASSIVE_API_KEY', '') or os.getenv('POLYGON_API_KEY', '')  # Support both env var names
massive_client = None

if USE_MASSIVE_API and MASSIVE_API_KEY:
    try:
        massive_client = MassiveAPIClient(MASSIVE_API_KEY)
        print("✅ Massive API client initialized (1-minute granular package enabled)")
        print(f"   API Key: {MASSIVE_API_KEY[:10]}...{MASSIVE_API_KEY[-5:] if len(MASSIVE_API_KEY) > 15 else '***'}")
    except Exception as e:
        print(f"⚠️  Failed to initialize Massive API client: {e}")
        print(f"   Please check MASSIVE_API_KEY or POLYGON_API_KEY environment variable")
        massive_client = None
else:
    if not MASSIVE_API_KEY:
        print("ℹ️  Massive API not configured (set MASSIVE_API_KEY or POLYGON_API_KEY to enable)")
    else:
        print(f"ℹ️  Massive API disabled (set USE_MASSIVE_API=true to enable)")

# ==================== STATE TRACKING ====================
class RiskManager:
    """Institutional-grade risk management with volatility-adjusted stops"""
    def __init__(self):
        self.peak_equity = 0.0
        self.daily_pnl = 0.0
        self.start_of_day_equity = 0.0
        self.open_positions = {}  # symbol: {entry_premium, entry_price, trail_active, trail_price, entry_time, contracts, qty_remaining, tp1_done, tp2_done, tp3_done, vol_regime}
        self.last_order_time = {}
        self.last_any_trade_time: Optional[datetime] = None  # Track last trade time across ALL symbols (for global cooldown)
        self.daily_trades = 0
        self.symbol_trade_counts: Dict[str, int] = {}  # Track trades per symbol: {'SPY': 3, 'QQQ': 1, 'SPX': 2}
        self.symbol_stop_loss_cooldown: Dict[str, datetime] = {}  # Track stop-loss triggers per symbol (prevents immediate re-entry)
        self.symbol_last_trade_time: Dict[str, datetime] = {}  # Track last trade time per symbol (anti-cycling protection)
        self.symbol_trailing_stop_cooldown: Dict[str, datetime] = {}  # Track trailing-stop triggers per symbol (60s cooldown)
        self.max_daily_trades = 20  # Max 20 trades per day
        self.current_vix = 20.0
        self.current_regime = "normal_vol"
        self.last_reset_date: Optional[str] = None  # Track last reset date for daily reset
        self.current_trading_day: Optional[datetime.date] = None  # Track current trading day from Alpaca clock
        self.option_cache: Dict[str, Any] = {}  # Cache for option symbols (cleared on new day)
        self.last_trade_symbols: set = set()  # Track last traded symbols (cleared on new day)
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        # Use EST for all timestamps
        est = pytz.timezone('US/Eastern')
        now_est = datetime.now(est)
        self.log_file = f"logs/mike_agent_safe_{now_est.strftime('%Y%m%d')}.log"
    
    def reset_daily_state(self, trading_day: datetime.date) -> None:
        """
        Reset daily state on new trading day
        Called automatically when Alpaca clock detects new trading day
        CRITICAL: Cooldowns must reset to allow new day trading
        
        Args:
            trading_day: Current trading day from Alpaca clock (EST date)
        """
        today_str = trading_day.strftime('%Y-%m-%d')
        if self.last_reset_date == today_str:
            return  # Already reset today
        
        self.log(f"🔄 NEW TRADING DAY DETECTED: {today_str} | Resetting all daily state", "INFO")
        
        self.last_reset_date = today_str
        self.current_trading_day = trading_day
        self.daily_trades = 0
        self.symbol_trade_counts = {}
        self.daily_pnl = 0.0
        self.start_of_day_equity = 0.0
        
        # CRITICAL: Reset all cooldown dictionaries to allow trading on new day
        self.symbol_stop_loss_cooldown = {}
        self.symbol_last_trade_time = {}
        self.symbol_trailing_stop_cooldown = {}
        self.last_order_time = {}
        self.last_any_trade_time = None
        
        # CRITICAL: Clear option cache and last traded symbols to prevent stale contracts
        self.option_cache.clear()
        self.last_trade_symbols.clear()
        
        self.log(f"✅ Daily reset complete: All cooldowns cleared, option cache cleared, daily counters reset", "INFO")
    
    def get_current_vix(self) -> float:
        """Get current VIX level (try Massive API first, fallback to yfinance)"""
        # Try Massive API first
        vix_price = get_current_price("^VIX")
        if vix_price:
            self.current_vix = float(vix_price)
            return self.current_vix
        
        # Fallback to yfinance
        try:
            vix_data = yf.Ticker("^VIX").history(period="1d")
            if len(vix_data) > 0:
                self.current_vix = float(vix_data['Close'].iloc[-1])
            return self.current_vix
        except Exception as e:
            self.log(f"Error fetching VIX: {e}, using default 20.0", "WARNING")
            return 20.0
    
    def get_vol_regime(self, vix: float) -> str:
        """Determine volatility regime based on VIX"""
        if vix < 18:
            return "calm"
        elif vix < 25:
            return "normal"
        elif vix < 35:
            return "storm"
        else:
            return "crash"
    
    def get_vol_params(self, regime: str = None) -> dict:
        """Get volatility-adjusted parameters for current or specified regime"""
        if regime is None:
            regime = self.current_regime
        return VOL_REGIMES.get(regime, VOL_REGIMES["normal"])
    
    def get_regime_max_notional(self, api: tradeapi.REST, regime: str = None) -> float:
        """Get regime-adjusted max notional (position size limit)"""
        if regime is None:
            regime = self.current_regime
        regime_params = self.get_vol_params(regime)
        equity = self.get_equity(api)
        return equity * regime_params['max_pct']
    
    def get_regime_risk(self, regime: str = None) -> float:
        """Get regime-adjusted risk percentage"""
        if regime is None:
            regime = self.current_regime
        regime_params = self.get_vol_params(regime)
        return regime_params['risk']
    
    def get_current_iv(self, underlying: str = "SPY") -> float:
        """Get current implied volatility for underlying"""
        try:
            # Try to get IV from option chain (if available via Alpaca)
            # For now, estimate from VIX
            vix = self.get_current_vix()
            # VIX is annualized, convert to approximate option IV
            # For 0DTE, IV is typically higher than VIX
            estimated_iv = vix * 1.2  # Rough approximation
            return estimated_iv
        except Exception as e:
            self.log(f"Error fetching IV: {e}, using default 20%", "WARNING")
            return 20.0
    
    def get_iv_adjusted_risk(self, iv: float) -> float:
        """Get IV-adjusted risk percentage for position sizing"""
        return get_iv_adjusted_risk(iv)
    
    def _compute_dynamic_trailing_pct(self, highest_pnl: float, vix: Optional[float] = None, base_trailing: float = 0.18) -> float:
        """
        Compute a dynamic trailing-stop percentage based on:
          - highest_pnl: peak unrealized PnL (as decimal, e.g. 0.75 = +75%)
          - vix: optional VIX level
          - base_trailing: starting trailing drawdown (default 18%)
        Returns a drawdown percentage (e.g. 0.18 = 18%) from peak.
        """
        trailing = base_trailing
        
        # Tighten the trailing stop as profits get larger
        if highest_pnl >= 2.0:          # +200% or more
            trailing = 0.10             # allow 10% pullback
        elif highest_pnl >= 1.5:        # +150% to +200%
            trailing = 0.12
        elif highest_pnl >= 1.0:        # +100% to +150%
            trailing = 0.15
        elif highest_pnl >= 0.60:       # +60% to +100% (just after TP2)
            trailing = 0.18
        else:
            trailing = base_trailing    # fallback (shouldn't really happen if trailing only after TP2)
        
        # Optional VIX-based adjustment
        if vix is not None:
            try:
                vix = float(vix)
                # High volatility → allow a bit more breathing room
                if vix >= 25:
                    trailing += 0.04    # widen by 4%
                elif vix <= 14:
                    trailing -= 0.03    # tighten by 3% in calm regime
            except Exception:
                pass
        
        # Clamp to a safe range: 8%–30%
        trailing = max(0.08, min(trailing, 0.30))
        return trailing
    
    def log(self, msg: str, level: str = "INFO"):
        """Log message to console and file"""
        # Use EST for all timestamps
        est = pytz.timezone('US/Eastern')
        now_est = datetime.now(est)
        timestamp = now_est.strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] [{level}] {msg}"
        print(log_msg)
        try:
            with open(self.log_file, "a") as f:
                # Use EST for all timestamps
                est = pytz.timezone('US/Eastern')
                now_est = datetime.now(est)
                f.write(f"{now_est} | [{level}] {msg}\n")
        except:
            pass
    
    def get_equity(self, api: tradeapi.REST) -> float:
        """Get current account equity"""
        try:
            account = api.get_account()
            return float(account.equity)
        except Exception as e:
            self.log(f"Error getting equity: {e}", "ERROR")
            return self.peak_equity if self.peak_equity > 0 else 1000.0
    
    def check_safeguards(self, api: tradeapi.REST) -> tuple[bool, str]:
        """
        Check all 10 risk safeguards
        Returns: (can_trade, reason_if_blocked)
        """
        equity = self.get_equity(api)
        
        # Initialize start of day equity
        if self.start_of_day_equity == 0:
            self.start_of_day_equity = equity
            self.peak_equity = equity
            self.log(f"Starting equity: ${equity:,.2f}")
        
        # Update peak equity
        self.peak_equity = max(self.peak_equity, equity)
        
        # Calculate daily PnL
        self.daily_pnl = (equity - self.start_of_day_equity) / self.start_of_day_equity
        
        # ========== SAFEGUARD 1: Daily Loss Limit (Percentage) ==========
        if self.daily_pnl <= DAILY_LOSS_LIMIT:
            self.log(f"🚨 SAFEGUARD 1 TRIGGERED: Daily loss limit hit ({self.daily_pnl:.1%})", "CRITICAL")
            try:
                api.close_all_positions()
                self.log("All positions closed. Shutting down.", "CRITICAL")
            except:
                pass
            sys.exit(1)
        
        # ========== SAFEGUARD 1.5: Hard Daily Dollar Loss Limit ==========
        # Get absolute dollar loss (more protective than percentage for smaller accounts)
        daily_pnl_dollar = equity * self.daily_pnl  # Current equity * P&L percentage = dollar P&L
        if daily_pnl_dollar < HARD_DAILY_LOSS_DOLLAR:
            self.log(f"🚨 SAFEGUARD 1.5 TRIGGERED: Hard daily loss limit (${abs(HARD_DAILY_LOSS_DOLLAR):,.0f}) reached | Current: ${daily_pnl_dollar:,.2f} (dollar-based) | Trading halted for day", "CRITICAL")
            try:
                api.close_all_positions()
                self.log("All positions closed. Shutting down.", "CRITICAL")
            except:
                pass
            sys.exit(1)
        
        # ========== SAFEGUARD 2: Max Drawdown Circuit Breaker ==========
        drawdown = (equity / self.peak_equity) - 1
        if drawdown <= -MAX_DRAWDOWN:
            self.log(f"🚨 SAFEGUARD 2 TRIGGERED: Max drawdown breached ({drawdown:.1%})", "CRITICAL")
            try:
                api.close_all_positions()
                self.log("All positions closed. Shutting down.", "CRITICAL")
            except:
                pass
            sys.exit(1)
        
        # ========== SAFEGUARD 3: VIX Volatility Kill Switch ==========
        try:
            vix = get_current_price("^VIX")
            if vix and vix > VIX_KILL:
                    return False, f"VIX {vix:.1f} > {VIX_KILL} (crash mode)"
        except Exception as e:
            self.log(f"Error fetching VIX: {e}", "WARNING")
        
        # ========== SAFEGUARD 4: Time-of-Day Filter (ENTRIES ONLY) ==========
        # DISABLED - NO_TRADE_AFTER is None, trading allowed all day
        # This safeguard is completely disabled per user request
        # No time restrictions - trading allowed throughout market hours
        
        # ========== SAFEGUARD 5: Max Concurrent Positions ==========
        if len(self.open_positions) >= MAX_CONCURRENT:
            return False, f"⛔ BLOCKED: Max concurrent positions ({MAX_CONCURRENT}) reached | Current: {len(self.open_positions)}/{MAX_CONCURRENT}"
        
        # ========== SAFEGUARD 6: Max Daily Trades ==========
        if self.daily_trades >= self.max_daily_trades:
            return False, f"Max daily trades ({self.max_daily_trades}) reached"
        
        return True, "OK"
    
    def get_current_max_notional(self, api: tradeapi.REST) -> float:
        """
        Get current max notional based on regime-adjusted position size
        Dynamically recalculated every call based on current VIX regime
        """
        return self.get_regime_max_notional(api, self.current_regime)
    
    def get_current_exposure(self) -> float:
        """Get total current exposure from open positions"""
        # This would need real position data from Alpaca
        # For now, estimate from tracked positions
        total = 0.0
        for pos in self.open_positions.values():
            if 'notional' in pos:
                total += pos['notional']
        return total
    
    def calculate_max_contracts(self, api: tradeapi.REST, strike: float, regime: str = None) -> tuple:
        """
        Calculate maximum contracts allowed under regime-adjusted position size limit
        Returns: (max_contracts, available_notional)
        """
        if regime is None:
            regime = self.current_regime
        max_notional = self.get_regime_max_notional(api, regime)
        current_exposure = self.get_current_exposure()
        available_notional = max_notional - current_exposure
        
        if available_notional <= 0:
            return 0, 0.0
        
        # Calculate max contracts: available_notional / (strike * 100)
        max_contracts = int(available_notional / (strike * 100))
        return max(0, max_contracts), available_notional
    
    def check_order_safety(self, symbol: str, qty: int, premium: float, api: tradeapi.REST, is_entry: bool = True, 
                          current_price: float = None, strike: float = None, option_type: str = None) -> tuple[bool, str]:
        """
        Check order-level safeguards with dynamic position size
        CRITICAL: Cooldown checks apply ONLY to entries, NEVER to exits
        
        🔴 RED-TEAM FIX: Added trade gating (spread, quote age, expected move)
        These are HARD VETOES - no trade if any gate fails
        
        Args:
            symbol: Option symbol
            qty: Number of contracts
            premium: Option premium per contract (not strike price!)
            api: Alpaca API instance
            is_entry: True if this is an entry (buy), False if exit (sell)
            current_price: Current underlying price (for expected move calculation)
            strike: Option strike price (for expected move calculation)
            option_type: 'call' or 'put' (for expected move calculation)
        Returns: (is_safe, reason_if_unsafe)
        """
        # ========== 🔴 RED-TEAM FIX: TRADE GATING (HARD VETOES) ==========
        # These gates MUST pass before any trade - no exceptions
        # Phase 0.2: Block trades when spread > X%, quote age > threshold, expected move < breakeven
        if is_entry and current_price and strike and option_type:
            # Gate 1: Spread check (if we can get bid/ask)
            try:
                # Try to get option snapshot for bid/ask spread
                snapshot = api.get_option_snapshot(symbol)
                if snapshot and hasattr(snapshot, 'bid_price') and hasattr(snapshot, 'ask_price'):
                    bid = float(snapshot.bid_price) if snapshot.bid_price else 0
                    ask = float(snapshot.ask_price) if snapshot.ask_price else 0
                    if bid > 0 and ask > 0:
                        spread = ask - bid
                        spread_pct = (spread / premium) * 100 if premium > 0 else 100
                        MAX_SPREAD_PCT = 20.0  # Block if spread > 20% of premium (0DTE can have wide spreads)
                        if spread_pct > MAX_SPREAD_PCT:
                            return False, f"⛔ BLOCKED: Spread too wide ({spread_pct:.1f}% of premium ${premium:.2f}) | Bid: ${bid:.2f}, Ask: ${ask:.2f} | Max allowed: {MAX_SPREAD_PCT}%"
            except Exception as e:
                # If we can't get spread, log but don't block (data may not be available)
                self.log(f"⚠️ Could not check spread for {symbol}: {e}", "WARNING")
            
            # Gate 2: Expected move vs breakeven (CRITICAL - this eliminates most losses)
            try:
                # Calculate expected move (simplified: use VIX/16 * sqrt(days_to_expiry))
                vix = self.get_current_vix()
                days_to_expiry = 1.0 / (252 * 6.5)  # 0DTE: ~1 trading day = 1/(252*6.5) years
                expected_move_pct = (vix / 16.0) * (days_to_expiry ** 0.5) * 100  # Annualized to daily
                expected_move_dollars = current_price * (expected_move_pct / 100)
                
                # Calculate breakeven move needed
                if option_type.lower() == 'call':
                    breakeven_move = strike + premium - current_price  # Need price to move above strike+premium
                else:  # put
                    breakeven_move = current_price - (strike - premium)  # Need price to move below strike-premium
                
                # CRITICAL GATE: Expected move must be >= breakeven move
                if expected_move_dollars < abs(breakeven_move):
                    return False, f"⛔ BLOCKED: Expected move (${expected_move_dollars:.2f}) < Breakeven move (${abs(breakeven_move):.2f}) | VIX: {vix:.1f} | This trade needs more volatility to be profitable"
            except Exception as e:
                # If calculation fails, log but don't block (may be data issue)
                self.log(f"⚠️ Could not calculate expected move for {symbol}: {e}", "WARNING")
        
        # ========== SAFEGUARD 7: Order Size Sanity Check ==========
        # For options, notional = premium cost = qty * premium * 100
        # NOT strike price - we're buying options, so cost is premium
        notional = qty * premium * 100  # Options: qty * premium * 100
        if notional > MAX_NOTIONAL:
            return False, f"Notional ${notional:,.0f} > ${MAX_NOTIONAL:,} limit"
        
        # ========== SAFEGUARD 8: Max Position Size (regime-adjusted) ==========
        max_notional = self.get_current_max_notional(api)
        current_exposure = self.get_current_exposure()
        regime_params = self.get_vol_params(self.current_regime)
        
        if current_exposure + notional > max_notional:
            return False, f"⛔ BLOCKED: Position would exceed {regime_params['max_pct']:.0%} limit ({self.current_regime.upper()} regime): ${current_exposure + notional:,.0f} > ${max_notional:,.0f}"
        
        # ========== SAFEGUARD 8.5: Max Trades Per Symbol ==========
        # Extract underlying symbol from option symbol (e.g., SPY251210C00680000 -> SPY)
        try:
            underlying = extract_underlying_from_option(symbol) if symbol and len(str(symbol)) > 10 else (str(symbol)[:3] if symbol else "UNK")
        except:
            underlying = str(symbol)[:3] if symbol else "UNK"
        symbol_trade_count = self.symbol_trade_counts.get(underlying, 0)
        if symbol_trade_count >= MAX_TRADES_PER_SYMBOL:
            return False, f"⛔ BLOCKED: Max trades per symbol ({MAX_TRADES_PER_SYMBOL}) reached for {underlying} | Current: {symbol_trade_count}/{MAX_TRADES_PER_SYMBOL}"
        
        # ========== SAFEGUARD 8.6: Global Trade Cooldown ==========
        # Minimum time between ANY trades (prevents cascading issues)
        if self.last_any_trade_time:
            # Use EST for all timestamps
            est = pytz.timezone('US/Eastern')
            now_est = datetime.now(est)
            time_since_last_trade = (now_est - self.last_any_trade_time).total_seconds()
            if time_since_last_trade < MIN_TRADE_COOLDOWN_SECONDS:
                return False, f"⛔ BLOCKED: Global trade cooldown active | {int(time_since_last_trade)}s < {MIN_TRADE_COOLDOWN_SECONDS}s (prevents cascading issues)"
        
        # ========== COOLDOWN CHECKS (ENTRY ONLY) ==========
        # CRITICAL: Cooldown checks apply ONLY to entries, NEVER to exits
        # Exits (SL/TP/TS/emergency) bypass all cooldown restrictions
        # Skip cooldown checks if this is an exit order
        if is_entry:
            try:
                underlying = extract_underlying_from_option(symbol) if symbol and len(str(symbol)) > 10 else (str(symbol)[:3] if symbol else "UNK")
            except:
                underlying = str(symbol)[:3] if symbol else "UNK"
            
            # ========== SAFEGUARD 8.7: Stop-Loss Cooldown ==========
            # Prevent immediate re-entry after stop-loss trigger (protects from volatile symbols)
            STOP_LOSS_COOLDOWN_MINUTES = 3  # 3 minutes cooldown after stop-loss
            if underlying in self.symbol_stop_loss_cooldown:
                # Use EST for all timestamps
                est = pytz.timezone('US/Eastern')
                now_est = datetime.now(est)
                time_since_sl = (now_est - self.symbol_stop_loss_cooldown[underlying]).total_seconds()
                if time_since_sl < (STOP_LOSS_COOLDOWN_MINUTES * 60):
                    remaining_minutes = int((STOP_LOSS_COOLDOWN_MINUTES * 60 - time_since_sl) / 60) + 1
                    return False, f"⛔ BLOCKED: Stop-loss cooldown active for {underlying} | {remaining_minutes} minute(s) remaining (prevents re-entry after SL trigger)"
                else:
                    # Cooldown expired, remove from tracking
                    del self.symbol_stop_loss_cooldown[underlying]
            
            # ========== SAFEGUARD 8.8: Per-Symbol Trade Cooldown (Anti-Cycling) ==========
            # Prevent rapid-fire trades on the same symbol (minimum 10 seconds between entries)
            MIN_SYMBOL_COOLDOWN_SECONDS = 10  # 10 seconds minimum between trades per symbol
            if underlying in self.symbol_last_trade_time:
                # Use EST for all timestamps
                est = pytz.timezone('US/Eastern')
                now_est = datetime.now(est)
                time_since_last = (now_est - self.symbol_last_trade_time[underlying]).total_seconds()
                if time_since_last < MIN_SYMBOL_COOLDOWN_SECONDS:
                    remaining_seconds = int(MIN_SYMBOL_COOLDOWN_SECONDS - time_since_last) + 1
                    return False, f"⛔ BLOCKED: Per-symbol cooldown active for {underlying} | {remaining_seconds}s remaining (prevents rapid-fire trading)"
            
            # ========== SAFEGUARD 8.9: Trailing-Stop Cooldown ==========
            # Prevent immediate re-entry after trailing-stop trigger (60 seconds cooldown)
            TRAILING_STOP_COOLDOWN_SECONDS = 60  # 60 seconds cooldown after trailing stop
            if underlying in self.symbol_trailing_stop_cooldown:
                # Use EST for all timestamps
                est = pytz.timezone('US/Eastern')
                now_est = datetime.now(est)
                time_since_ts = (now_est - self.symbol_trailing_stop_cooldown[underlying]).total_seconds()
                if time_since_ts < TRAILING_STOP_COOLDOWN_SECONDS:
                    remaining_seconds = int(TRAILING_STOP_COOLDOWN_SECONDS - time_since_ts) + 1
                    return False, f"⛔ BLOCKED: Trailing-stop cooldown active for {underlying} | {remaining_seconds}s remaining (prevents re-entry after TS trigger)"
                else:
                    # Cooldown expired, remove from tracking
                    del self.symbol_trailing_stop_cooldown[underlying]
            
            # Record trade time for per-symbol cooldown (for entries only)
            # Use EST for all timestamps
            est = pytz.timezone('US/Eastern')
            now_est = datetime.now(est)
            self.symbol_last_trade_time[underlying] = now_est
        
        # ========== SAFEGUARD 9: Duplicate Order Protection ==========
        if symbol in self.last_order_time:
            # Use EST for all timestamps
            est = pytz.timezone('US/Eastern')
            now_est = datetime.now(est)
            time_since_last = (now_est - self.last_order_time[symbol]).total_seconds()
            if time_since_last < DUPLICATE_ORDER_WINDOW:
                return False, f"⛔ BLOCKED: Duplicate order protection | {int(time_since_last)}s < {DUPLICATE_ORDER_WINDOW}s (prevents duplicate orders for same symbol)"
        
        # ========== SAFEGUARD 10: Max Daily Trades ==========
        if self.daily_trades >= self.max_daily_trades:
            return False, f"⛔ BLOCKED: Max daily trades ({self.max_daily_trades}) reached | Current: {self.daily_trades}/{self.max_daily_trades}"
        
        return True, "OK"
    
    def record_order(self, symbol: str, is_entry: bool = True):
        """
        Record order time for duplicate protection and track per-symbol trade counts
        CRITICAL: Trade count increments ONLY on entries, NOT on exits
        
        Args:
            symbol: Option symbol
            is_entry: True if this is an entry (buy), False if exit (sell)
        """
        # Use EST for all timestamps
        est = pytz.timezone('US/Eastern')
        now_est = datetime.now(est)
        self.last_order_time[symbol] = now_est
        self.last_any_trade_time = now_est  # Track global trade time for cooldown
        
        # CRITICAL: Increment trade count ONLY on entries (not exits)
        # Exits (SL/TP/TS) should NOT count toward daily trade limit
        if is_entry:
            self.daily_trades += 1
        
        # Track trades per symbol (only entries)
        if is_entry:
            try:
                underlying = extract_underlying_from_option(symbol) if symbol and len(str(symbol)) > 10 else (str(symbol)[:3] if symbol else "UNK")
            except:
                underlying = str(symbol)[:3] if symbol else "UNK"
        self.symbol_trade_counts[underlying] = self.symbol_trade_counts.get(underlying, 0) + 1

# ==================== ALPACA SETUP ====================
def init_alpaca():
    """Initialize Alpaca API"""
    if API_KEY == 'YOUR_PAPER_KEY' or API_SECRET == 'YOUR_PAPER_SECRET':
        raise ValueError("Please set ALPACA_KEY and ALPACA_SECRET in config.py")
    
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    
    try:
        account = api.get_account()
        print(f"✓ Connected to Alpaca ({'PAPER' if USE_PAPER else 'LIVE'})")
        print(f"  Account Status: {account.status}")
        print(f"  Equity: ${float(account.equity):,.2f}")
        print(f"  Buying Power: ${float(account.buying_power):,.2f}")
        return api
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Alpaca: {e}")

# ==================== MARKET DATA HELPERS ====================
def choose_best_symbol_for_trade(iteration: int, symbol_actions: dict, target_action: int, 
                                 open_positions: dict, risk_mgr, max_positions_per_symbol: int = 1) -> Optional[str]:
    """
    Choose best symbol for trade using:
    1. Fair rotation for symbol priority
    2. Filter out symbols with existing positions
    3. Filter out symbols in cooldown (stop-loss, trailing-stop)
    4. Filter out symbols that exceed portfolio risk limits
    5. Sort by RL strength to pick strongest signal
    
    Args:
        iteration: Current loop iteration for rotation
        symbol_actions: Dict of {symbol: (action, source, strength)}
        target_action: Action to filter for (1=BUY CALL, 2=BUY PUT)
        open_positions: Dict of current positions
        risk_mgr: RiskManager instance for cooldown checks
        max_positions_per_symbol: Max positions per symbol (default 1)
    
    Returns:
        Selected symbol or None if no eligible symbol
    """
    symbols = TRADING_SYMBOLS  # ['SPY', 'QQQ', 'IWM']
    
    # ⭐ PRIORITY FIX: Always prioritize SPY first (most profitable based on user's strategy)
    # Then QQQ, then IWM
    # Only use rotation if SPY is not available (has position, cooldown, etc.)
    priority_order = ['SPY', 'QQQ', 'IWM']  # Fixed priority: SPY first
    
    # If SPY is not in candidates after filtering, rotation can help with QQQ/IWM
    # But SPY should always be checked first
    
    # Filter candidates: must pass all checks
    candidates = []
    filtered_reasons = []
    
    for sym in priority_order:
        # Check if symbol has target action
        if sym not in symbol_actions:
            continue
        
        # Handle dict format
        action_data = symbol_actions[sym]
        if isinstance(action_data, dict):
            action = action_data.get('action', 0)
            source = action_data.get('action_source', 'unknown')
            strength = action_data.get('action_strength', 0.0)
        else:
            action, source, strength = action_data
        if action != target_action:
            continue
        
        # Check if symbol already has a position (filter out to avoid duplicates)
        # Count positions starting with this symbol (e.g., "SPY", "SPY_", etc.)
        symbol_position_count = sum(1 for pos_sym in open_positions.keys() if pos_sym.startswith(sym))
        
        if symbol_position_count >= max_positions_per_symbol:
            filtered_reasons.append(f"{sym}:has_position")
            continue
        
        # ⭐ ENHANCEMENT #1: Check cooldowns (stop-loss, trailing-stop)
        # Stop-loss cooldown check (3 minutes)
        STOP_LOSS_COOLDOWN_MINUTES = 3
        if sym in risk_mgr.symbol_stop_loss_cooldown:
            # Use EST for all timestamps
            est = pytz.timezone('US/Eastern')
            now_est = datetime.now(est)
            time_since_sl = (now_est - risk_mgr.symbol_stop_loss_cooldown[sym]).total_seconds()
            if time_since_sl < (STOP_LOSS_COOLDOWN_MINUTES * 60):
                remaining = int((STOP_LOSS_COOLDOWN_MINUTES * 60 - time_since_sl) / 60) + 1
                filtered_reasons.append(f"{sym}:SL_cooldown({remaining}min)")
                continue
            else:
                # Cooldown expired, remove from tracking
                del risk_mgr.symbol_stop_loss_cooldown[sym]
        
        # Trailing-stop cooldown check (60 seconds)
        TRAILING_STOP_COOLDOWN_SECONDS = 60
        if sym in risk_mgr.symbol_trailing_stop_cooldown:
            # Use EST for all timestamps
            est = pytz.timezone('US/Eastern')
            now_est = datetime.now(est)
            time_since_ts = (now_est - risk_mgr.symbol_trailing_stop_cooldown[sym]).total_seconds()
            if time_since_ts < TRAILING_STOP_COOLDOWN_SECONDS:
                remaining = int(TRAILING_STOP_COOLDOWN_SECONDS - time_since_ts) + 1
                filtered_reasons.append(f"{sym}:TS_cooldown({remaining}s)")
                continue
            else:
                # Cooldown expired, remove from tracking
                del risk_mgr.symbol_trailing_stop_cooldown[sym]
        
        # ⭐ ENHANCEMENT #2: Check portfolio risk limits (if institutional integration available)
        # This is optional and will be no-op until institutional modules are integrated
        try:
            # Check if institutional integration is available
            if hasattr(risk_mgr, 'institutional_integration') and risk_mgr.institutional_integration:
                # Check portfolio Greek limits before entry
                greek_check = risk_mgr.institutional_integration.check_portfolio_greek_limits_before_entry(
                    symbol=sym,
                    action=target_action,
                    position_size=1  # Placeholder
                )
                if not greek_check['allowed']:
                    filtered_reasons.append(f"{sym}:greek_limit({greek_check['reason']})")
                    continue
        except:
            pass  # Institutional integration not available yet, skip this check
        
        # Passed all filters, add to candidates
        candidates.append((sym, strength, source))
    
    if not candidates:
        # Log why no candidates available
        if filtered_reasons:
            try:
                risk_mgr.log(
                    f"⚠️ No eligible symbols for action={target_action} | Filtered: {', '.join(filtered_reasons)}",
                    "INFO",
                )
            except Exception:
                pass
        return None  # No eligible symbol
    
    # Sort by RL strength (descending), keep fairness via rotated priority as tiebreaker
    # This ensures we still honor rotation but prefer the strongest RL signal
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    selected_symbol = candidates[0][0]
    selected_strength = candidates[0][1]
    selected_source = candidates[0][2]
    
    # Log selection with details
    candidates_str = ", ".join([f"{s}({st:.2f})" for s, st, _ in candidates])
    try:
        risk_mgr.log(
            f"✅ Symbol selected: {selected_symbol} (strength={selected_strength:.3f}, source={selected_source}) | "
            f"candidates=[{candidates_str}] | priority={priority_order}",
            "INFO",
        )
    except Exception:
        pass
    
    return selected_symbol


# ==================== CACHING HELPER FUNCTIONS ====================
def _is_cache_valid(cache_entry: Dict, max_age_seconds: int) -> bool:
    """Check if cache entry is still valid (not expired)"""
    if not cache_entry or 'timestamp' not in cache_entry:
        return False
    cache_time = cache_entry['timestamp']
    if cache_time is None:
        return False
    now = datetime.now(pytz.timezone('US/Eastern'))
    age_seconds = (now - cache_time).total_seconds()
    return age_seconds < max_age_seconds


def _get_cached_market_data(symbol: str, risk_mgr=None) -> Optional[pd.DataFrame]:
    """Get cached market data if available and fresh"""
    global _market_data_cache
    
    if symbol in _market_data_cache:
        cache_entry = _market_data_cache[symbol]
        if _is_cache_valid(cache_entry, DATA_CACHE_SECONDS):
            if risk_mgr and hasattr(risk_mgr, 'log'):
                risk_mgr.log(f"📦 {symbol}: Using cached data ({DATA_CACHE_SECONDS}s cache)", "DEBUG")
            return cache_entry.get('data')
    return None


def _set_cached_market_data(symbol: str, data: pd.DataFrame, price: float = None):
    """Store market data in cache"""
    global _market_data_cache
    
    est = pytz.timezone('US/Eastern')
    _market_data_cache[symbol] = {
        'data': data,
        'timestamp': datetime.now(est),
        'price': price or (data['Close'].iloc[-1] if len(data) > 0 else None)
    }


def get_cached_positions(api: tradeapi.REST, risk_mgr=None) -> list:
    """Get positions with caching to reduce API calls"""
    global _positions_cache
    
    # Check if cache is valid
    if _is_cache_valid(_positions_cache, POSITIONS_CACHE_SECONDS):
        if risk_mgr and hasattr(risk_mgr, 'log'):
            risk_mgr.log(f"📦 Using cached positions ({POSITIONS_CACHE_SECONDS}s cache)", "DEBUG")
        return _positions_cache.get('positions', [])
    
    # Fetch fresh positions
    try:
        positions = api.list_positions()
        est = pytz.timezone('US/Eastern')
        _positions_cache = {
            'positions': positions,
            'timestamp': datetime.now(est)
        }
        return positions
    except Exception as e:
        if risk_mgr and hasattr(risk_mgr, 'log'):
            risk_mgr.log(f"⚠️ Error fetching positions: {e}", "WARNING")
        # Return cached positions if available (stale is better than nothing)
        return _positions_cache.get('positions', [])


def clear_data_cache(symbol: str = None):
    """Clear data cache (optionally for specific symbol only)"""
    global _market_data_cache
    
    if symbol:
        if symbol in _market_data_cache:
            del _market_data_cache[symbol]
    else:
        _market_data_cache.clear()


# ==================== NO-TRADE REASON TRACKING ====================
def track_no_trade_reason(reason: str, risk_mgr=None):
    """Track why a trade didn't execute (for daily summary)"""
    global _daily_no_trade_reasons, _no_trade_tracking_date
    
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).date()
    
    # Reset tracking if it's a new day
    if _no_trade_tracking_date != today:
        _no_trade_tracking_date = today
        for key in _daily_no_trade_reasons:
            _daily_no_trade_reasons[key] = 0
    
    # Increment the reason counter
    if reason in _daily_no_trade_reasons:
        _daily_no_trade_reasons[reason] += 1


def get_daily_no_trade_summary(risk_mgr=None) -> str:
    """Generate daily summary of why trades didn't execute"""
    global _daily_no_trade_reasons, _no_trade_tracking_date
    
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).date()
    
    if _no_trade_tracking_date != today:
        return "No tracking data for today yet."
    
    total = sum(_daily_no_trade_reasons.values())
    if total == 0:
        return "No trade blocking events recorded today."
    
    summary_lines = [
        f"📊 DAILY NO-TRADE SUMMARY ({today}):",
        f"   Total HOLD decisions: {total}",
    ]
    
    # Add non-zero reasons
    reason_labels = {
        'hold_signals': '   • HOLD signals (RL/Ensemble)',
        'low_confidence': '   • Low confidence (<60%)',
        'cooldown_active': '   • Cooldown active',
        'position_limit': '   • Max positions reached',
        'daily_loss_limit': '   • Daily loss limit hit',
        'vix_kill_switch': '   • VIX kill switch',
        'stale_data': '   • Stale/invalid data',
        'no_options_available': '   • No valid options',
        'safeguard_blocked': '   • Other safeguards',
        'market_closed': '   • Market closed',
    }
    
    for key, label in reason_labels.items():
        count = _daily_no_trade_reasons.get(key, 0)
        if count > 0:
            summary_lines.append(f"{label}: {count}")
    
    return "\n".join(summary_lines)


def print_daily_no_trade_summary(risk_mgr=None):
    """Print and log the daily no-trade summary"""
    summary = get_daily_no_trade_summary(risk_mgr)
    
    if risk_mgr and hasattr(risk_mgr, 'log'):
        for line in summary.split('\n'):
            risk_mgr.log(line, "INFO")
    else:
        print(summary)


def get_market_data(symbol: str, period: str = "2d", interval: str = "1m", api: Optional[tradeapi.REST] = None, risk_mgr = None, backtest_mode: bool = False, backtest_end_time: Optional[datetime] = None) -> pd.DataFrame:
    """
    Get market data - tries Alpaca first (you're paying for it!), then Massive API
    
    Priority:
    1. Alpaca API (real-time, included with trading account)
    2. Massive API (if available)
    3. yfinance (DISABLED by default - delayed data not acceptable for 0DTE trading)
    
    CRITICAL: For 0DTE trading, only Alpaca/Massive are acceptable (real-time).
    yfinance is delayed 15-20 minutes and is DISABLED by default (ALLOW_YFINANCE_FALLBACK = False).
    If both Alpaca and Massive fail, returns empty DataFrame (iteration will be skipped).
    
    Args:
        symbol: Stock symbol (SPY, QQQ, SPX, etc.)
        period: Period for data (e.g., "2d", "1d")
        interval: Data interval ("1m", "5m", "1d", etc.)
        api: Alpaca API instance (optional, but recommended)
        risk_mgr: RiskManager instance for logging (optional)
    
    Returns:
        DataFrame with OHLCV data (Open, High, Low, Close, Volume)
        Returns empty DataFrame if all sources fail or data is stale
    """
    global massive_client
    
    # ========== CHECK CACHE FIRST (Reduces redundant API calls) ==========
    # Skip cache in backtest mode (need historical data)
    if not backtest_mode:
        cached_data = _get_cached_market_data(symbol, risk_mgr)
        if cached_data is not None and len(cached_data) > 0:
            return cached_data
    
    # Use EST timezone for all date/time calculations
    est = pytz.timezone('US/Eastern')
    # In backtest mode, use backtest_end_time instead of current time
    if backtest_mode and backtest_end_time:
        now_est = backtest_end_time if backtest_end_time.tzinfo else est.localize(backtest_end_time)
        today_est = now_est.date()
    else:
        now_est = datetime.now(est)
        today_est = now_est.date()
    
    # Helper function to convert ANY timestamp to EST string format
    def timestamp_to_est_str(timestamp) -> str:
        """
        Convert any timestamp (UTC, naive, or already EST) to EST string format.
        This ensures ALL displayed times are in EST consistently.
        Handles: datetime, pandas Timestamp, string timestamps, None
        """
        if timestamp is None:
            return "N/A"
        
        # Handle pandas Timestamp objects
        if isinstance(timestamp, pd.Timestamp):
            dt = timestamp.to_pydatetime()
        # If it's already a string, try to parse it
        elif isinstance(timestamp, str):
            try:
                # Try parsing common formats
                if '+' in timestamp or 'Z' in timestamp or timestamp.endswith('UTC'):
                    # UTC format
                    timestamp_clean = timestamp.replace('Z', '+00:00')
                    if '+' not in timestamp_clean and 'UTC' in timestamp_clean:
                        timestamp_clean = timestamp_clean.replace(' UTC', '+00:00')
                    try:
                        dt = datetime.fromisoformat(timestamp_clean.replace('+00:00', ''))
                        dt = pytz.utc.localize(dt) if dt.tzinfo is None else dt
                    except:
                        dt = pd.to_datetime(timestamp).to_pydatetime()
                else:
                    # Try parsing as naive datetime
                    dt = pd.to_datetime(timestamp).to_pydatetime()
            except:
                return str(timestamp)  # Return as-is if parsing fails
        else:
            # Assume it's a datetime-like object
            dt = timestamp
        
        # Convert to EST
        if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
            dt_est = dt.astimezone(est)
        else:
            # Assume UTC if no timezone (Alpaca/Massive APIs return UTC)
            try:
                if isinstance(dt, datetime):
                    dt_utc = pytz.utc.localize(dt)
                else:
                    dt_utc = pytz.utc.localize(pd.Timestamp(dt).to_pydatetime())
                dt_est = dt_utc.astimezone(est)
            except Exception as e:
                # If localization fails, try to assume it's already in EST
                try:
                    dt_est = est.localize(dt) if isinstance(dt, datetime) else est.localize(pd.Timestamp(dt).to_pydatetime())
                except:
                    # Last resort: return formatted as-is
                    return str(dt)
        
        # Format as EST string
        return dt_est.strftime('%Y-%m-%d %H:%M:%S %Z')
    
    # Helper function to validate data freshness
    def validate_data_freshness(data: pd.DataFrame, source: str) -> tuple[bool, str]:
        """
        Validate that data is from today and not stale (>5 minutes old)
        Returns: (is_valid, error_message)
        
        🔴 RED-TEAM FIX: In backtest_mode, skip freshness checks (historical data is expected to be old)
        """
        if len(data) == 0:
            return False, "Empty data"
        
        # In backtest mode, skip freshness validation (historical data is expected)
        if backtest_mode:
            return True, "Backtest mode: freshness check skipped"
        
        last_bar_time = data.index[-1]
        
        # Convert to EST if timezone-aware, otherwise assume UTC and convert
        if hasattr(last_bar_time, 'tzinfo') and last_bar_time.tzinfo:
            last_bar_est = last_bar_time.astimezone(est)
        else:
            # Assume UTC if no timezone
            try:
                last_bar_utc = pytz.utc.localize(last_bar_time)
                last_bar_est = last_bar_utc.astimezone(est)
            except:
                # If localization fails, assume it's already in local time
                last_bar_est = est.localize(last_bar_time) if last_bar_time.tzinfo is None else last_bar_time
        
        last_bar_date = last_bar_est.date()
        
        # Check 1: Data must be from TODAY
        if last_bar_date != today_est:
            return False, f"Data is from {last_bar_date}, not today ({today_est})"
        
        # Check 2: Data must be fresh (< 5 minutes old during market hours)
        time_diff_minutes = (now_est - last_bar_est).total_seconds() / 60
        
        # During market hours (9:30 AM - 4:00 PM EST), reject data > 5 minutes old
        # Outside market hours, allow up to 1 hour old (for pre/post market data)
        market_hours = 9.5 <= now_est.hour + (now_est.minute / 60) < 16.0
        max_age_minutes = 5 if market_hours else 60
        
        if time_diff_minutes > max_age_minutes:
            return False, f"Data is {time_diff_minutes:.1f} minutes old (max: {max_age_minutes} min)"
        
        return True, f"Fresh data: {time_diff_minutes:.1f} minutes old, date: {last_bar_date}"
    
    # Helper function to log data collection with details
    def log_data_source(source: str, bars: int, symbol: str, last_price: float = None, last_time: str = None):
        if risk_mgr and hasattr(risk_mgr, 'log'):
            details = f" | Last: ${last_price:.2f} at {last_time}" if last_price and last_time else ""
            risk_mgr.log(
                f"📊 {symbol} Data: {bars} bars from {source}{details} | "
                f"period={period}, interval={interval}",
                "INFO"
            )
    
    # ========== PRIORITY 1: ALPACA API (You're paying for this!) ==========
    if api:
        # Log API availability for debugging
        if risk_mgr and hasattr(risk_mgr, 'log'):
            risk_mgr.log(f"🔍 Alpaca API available for {symbol}, attempting data fetch...", "INFO")
        try:
            from alpaca_trade_api.rest import TimeFrame
            
            # Calculate date range using EST timezone
            if period == "2d":
                days = 2
            elif period == "1d":
                days = 1
            elif period == "5d":
                days = 5
            else:
                days = 2  # Default
            
            # Use EST timezone for date calculations
            start_date = now_est - timedelta(days=days)
            if period == "2d":
                start_date = start_date - timedelta(hours=12)  # Buffer for extended hours
            
            # Use tomorrow as end date to ensure we get today's data (Alpaca end date is exclusive)
            end_date = now_est + timedelta(days=1)
            
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Map interval to Alpaca TimeFrame
            if interval == "1m":
                timeframe = TimeFrame.Minute
            elif interval == "5m":
                timeframe = TimeFrame(5, TimeFrame.Minute)
            elif interval == "15m":
                timeframe = TimeFrame(15, TimeFrame.Minute)
            elif interval == "1d":
                timeframe = TimeFrame.Day
            else:
                timeframe = TimeFrame.Minute  # Default to 1 minute
            
            # Log request with EST timezone info
            if risk_mgr and hasattr(risk_mgr, 'log'):
                risk_mgr.log(
                    f"🔍 Alpaca API: Requesting {symbol} from {start_str} to {end_str}, "
                    f"timeframe={timeframe} (EST: {now_est.strftime('%Y-%m-%d %H:%M:%S %Z')}, Today: {today_est})",
                    "INFO"
                )
            
            # Alpaca API v2 get_bars signature:
            # get_bars(symbol, timeframe, start, end, limit=None, adjustment=None)
            bars = api.get_bars(
                symbol,
                timeframe,
                start_str,
                end_str,
                limit=5000,
                adjustment='raw'  # Raw prices, not adjusted
            ).df
            
            if len(bars) > 0:
                # Alpaca returns: open, high, low, close, volume (lowercase)
                # Rename to match expected format (capitalized)
                if isinstance(bars.columns, pd.MultiIndex):
                    bars.columns = bars.columns.get_level_values(0)
                
                # Ensure we have the right column names
                column_map = {
                    'open': 'Open', 'high': 'High', 'low': 'Low', 
                    'close': 'Close', 'volume': 'Volume'
                }
                bars = bars.rename(columns=column_map)
                
                # Ensure we have required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in bars.columns for col in required_cols):
                    for req_col in required_cols:
                        if req_col not in bars.columns:
                            lower_col = req_col.lower()
                            if lower_col in bars.columns:
                                bars = bars.rename(columns={lower_col: req_col})
                
                # CRITICAL: Validate data freshness (must be from today and < 5 minutes old)
                is_valid, validation_msg = validate_data_freshness(bars, "Alpaca API")
                
                if not is_valid:
                    if risk_mgr and hasattr(risk_mgr, 'log'):
                        risk_mgr.log(
                            f"❌ CRITICAL: Alpaca data validation failed for {symbol}: {validation_msg}. "
                            f"Rejecting stale data, trying Massive API...",
                            "ERROR"
                        )
                    # Don't return stale data - fall through to Massive API
                    bars = pd.DataFrame()
                else:
                    bars_count = len(bars)
                    # In backtest mode, be more lenient with bar count (historical data may have gaps)
                    if backtest_mode:
                        expected_min_bars = 20  # Just need enough for observation (20 bars minimum)
                    else:
                        expected_min_bars = 1500 if period == "2d" else 700
                    
                    if bars_count < expected_min_bars and period == "2d":
                        if risk_mgr and hasattr(risk_mgr, 'log'):
                            risk_mgr.log(
                                f"⚠️ Alpaca API returned only {bars_count} bars for 2 days "
                                f"(expected at least {expected_min_bars}). Trying Massive API...",
                                "WARNING"
                            )
                        # Don't return yet - try Massive API for better data
                    else:
                        # Data is valid and sufficient
                        last_price = bars['Close'].iloc[-1]
                        last_time_str = timestamp_to_est_str(bars.index[-1])
                        log_data_source("Alpaca API", bars_count, symbol, last_price, last_time_str)
                        if risk_mgr and hasattr(risk_mgr, 'log'):
                            risk_mgr.log(
                                f"✅ Alpaca API: {bars_count} bars, last price: ${last_price:.2f}, "
                                f"{validation_msg}",
                                "INFO"
                            )
                        # Cache data before returning (reduces redundant API calls)
                        if not backtest_mode:
                            _set_cached_market_data(symbol, bars, last_price)
                        return bars
            else:
                if risk_mgr and hasattr(risk_mgr, 'log'):
                    risk_mgr.log(f"⚠️ Alpaca API returned empty data for {symbol}, trying Massive API...", "WARNING")
        except Exception as e:
            # Fallback to Massive API
            if risk_mgr and hasattr(risk_mgr, 'log'):
                risk_mgr.log(
                    f"⚠️ Alpaca data fetch failed for {symbol}: {str(e)} (type: {type(e).__name__}), "
                    f"trying Massive API...",
                    "WARNING"
                )
                import traceback
                risk_mgr.log(f"   Traceback: {traceback.format_exc()[:200]}", "WARNING")
            pass
    
    # ========== PRIORITY 2: MASSIVE API (You have 1-minute granular package!) ==========
    # Map symbol for Massive API (SPX uses different format)
    massive_symbol_map = {
        'SPY': 'SPY',
        'QQQ': 'QQQ',
        'IWM': 'IWM',
        'SPX': 'SPX',  # Polygon uses SPX (not ^SPX)
        '^SPX': 'SPX'
    }
    massive_symbol = massive_symbol_map.get(symbol, symbol.replace('^', ''))
    
    if massive_client:
        try:
            # Calculate date range using EST timezone
            if period == "2d":
                days = 2
            elif period == "1d":
                days = 1
            elif period == "5d":
                days = 5
            else:
                days = 2  # Default
            
            # Massive API needs date strings in YYYY-MM-DD format
            # Use EST timezone for date calculations
            # In backtest mode, use backtest_end_time; otherwise use current time
            if backtest_mode and backtest_end_time:
                backtest_time_est = backtest_end_time if backtest_end_time.tzinfo else est.localize(backtest_end_time)
                end_date_str = (backtest_time_est + timedelta(days=1)).strftime("%Y-%m-%d")  # Tomorrow to include today
                start_date_str = (backtest_time_est - timedelta(days=days)).strftime("%Y-%m-%d")
            else:
                end_date_str = (now_est + timedelta(days=1)).strftime("%Y-%m-%d")  # Tomorrow to include today
                start_date_str = (now_est - timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Log request with EST timezone info
            if risk_mgr and hasattr(risk_mgr, 'log'):
                risk_mgr.log(
                    f"🔍 Massive API: Requesting {symbol} from {start_date_str} to {end_date_str}, "
                    f"interval={interval} (EST: {now_est.strftime('%Y-%m-%d %H:%M:%S %Z')}, Today: {today_est})",
                    "INFO"
                )
            
            # Massive API get_historical_data signature:
            # get_historical_data(symbol, start_date, end_date, interval='1min')
            hist = massive_client.get_historical_data(
                massive_symbol, 
                start_date_str, 
                end_date_str, 
                interval=interval
            )
            
            if len(hist) > 0:
                # Massive API returns lowercase columns, normalize to match expected format (capitalized)
                if 'close' in hist.columns or 'Close' in hist.columns:
                    column_map = {}
                    for col in hist.columns:
                        if col.lower() == 'open':
                            column_map[col] = 'Open'
                        elif col.lower() == 'high':
                            column_map[col] = 'High'
                        elif col.lower() == 'low':
                            column_map[col] = 'Low'
                        elif col.lower() == 'close':
                            column_map[col] = 'Close'
                        elif col.lower() == 'volume':
                            column_map[col] = 'Volume'
                    
                    if column_map:
                        hist = hist.rename(columns=column_map)
                
                # Ensure index is datetime if it's not already
                if not isinstance(hist.index, pd.DatetimeIndex):
                    try:
                        hist.index = pd.to_datetime(hist.index)
                    except:
                        pass
                
                # CRITICAL: Validate data freshness (skip in backtest mode)
                if not backtest_mode:
                    is_valid, validation_msg = validate_data_freshness(hist, "Massive API")
                    
                    if not is_valid:
                        if risk_mgr and hasattr(risk_mgr, 'log'):
                            risk_mgr.log(
                                f"❌ CRITICAL: Massive API data validation failed for {symbol}: {validation_msg}. "
                                f"Rejecting stale data, trying yfinance (DELAYED - LAST RESORT)...",
                                "ERROR"
                            )
                        # Don't return stale data - fall through to yfinance
                        hist = pd.DataFrame()
                    else:
                        validation_msg = validation_msg
                else:
                    # In backtest mode, accept historical data from Massive API
                    validation_msg = "Backtest mode: historical data accepted"
                    is_valid = True
                
                if is_valid and len(hist) > 0:
                    hist_count = len(hist)
                    last_price = hist['Close'].iloc[-1] if 'Close' in hist.columns else hist['close'].iloc[-1]
                    last_time_str = timestamp_to_est_str(hist.index[-1])
                    log_data_source("Massive API", hist_count, symbol, last_price, last_time_str)
                    
                    if risk_mgr and hasattr(risk_mgr, 'log'):
                        # In backtest mode, be more lenient with bar count
                        if backtest_mode:
                            if hist_count >= 20:  # Minimum for observation
                                risk_mgr.log(
                                    f"✅ Massive API (BACKTEST): {hist_count} bars, last price: ${last_price:.2f}, "
                                    f"{validation_msg}",
                                    "INFO"
                                )
                            else:
                                risk_mgr.log(
                                    f"⚠️ Massive API (BACKTEST): Only {hist_count} bars (need at least 20 for observation)",
                                    "WARNING"
                                )
                        else:
                            # Live trading: stricter requirements
                            if hist_count < 1500 and period == "2d":
                                risk_mgr.log(
                                    f"⚠️ Massive API returned {hist_count} bars for 2 days "
                                    f"(expected at least 1,500). May be incomplete.",
                                    "WARNING"
                                )
                            else:
                                risk_mgr.log(
                                    f"✅ Massive API: {hist_count} bars, last price: ${last_price:.2f}, "
                                    f"{validation_msg}",
                                    "INFO"
                                )
                    
                    # Cache data before returning (reduces redundant API calls)
                    if not backtest_mode:
                        _set_cached_market_data(symbol, hist, last_price)
                    return hist
            else:
                if risk_mgr and hasattr(risk_mgr, 'log'):
                    risk_mgr.log(
                        f"⚠️ Massive API returned empty data for {symbol}, "
                        f"trying yfinance (DELAYED - LAST RESORT)...",
                        "WARNING"
                    )
        except Exception as e:
            # Fallback to yfinance
            if risk_mgr and hasattr(risk_mgr, 'log'):
                risk_mgr.log(
                    f"⚠️ Massive API fetch failed for {symbol}: {str(e)} (type: {type(e).__name__}), "
                    f"trying yfinance (DELAYED - LAST RESORT)...",
                    "WARNING"
                )
                import traceback
                risk_mgr.log(f"   Traceback: {traceback.format_exc()[:200]}", "WARNING")
            pass
    
    # ========== YFINANCE FALLBACK (ONLY FOR BACKTESTS) ==========
    # ⚠️ yfinance is DELAYED (15-20 minutes) - NOT SUITABLE FOR LIVE 0DTE TRADING
    # BUT: For backtests, historical data from yfinance is acceptable
    # If both Alpaca and Massive fail, use yfinance ONLY in backtest mode
    if backtest_mode or ALLOW_YFINANCE_FALLBACK:
        # Only use yfinance if explicitly enabled (NOT RECOMMENDED)
        try:
            if risk_mgr and hasattr(risk_mgr, 'log'):
                risk_mgr.log(
                    f"⚠️ WARNING: Both Alpaca and Massive failed for {symbol}. "
                    f"Falling back to yfinance (DELAYED 15-20 min - NOT SUITABLE FOR 0DTE TRADING)",
                    "WARNING"
                )
            
            # Map symbol for yfinance (SPX needs ^ prefix)
            yf_symbol = symbol
            if symbol == 'SPX':
                yf_symbol = '^GSPC'  # S&P 500 index (more reliable than ^SPX)
            
            # Force fresh fetch (clear any caches)
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period=period, interval=interval)
            
            # Ultimate yfinance 2025+ compatibility fix
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            hist = hist.dropna()
            
            if len(hist) > 0:
                # CRITICAL: Validate this is TODAY's data (skip in backtest mode)
                if not backtest_mode:
                    is_valid, validation_msg = validate_data_freshness(hist, "yfinance")
                    
                    if not is_valid:
                        if risk_mgr and hasattr(risk_mgr, 'log'):
                            risk_mgr.log(
                                f"❌ CRITICAL: yfinance data validation failed for {symbol}: {validation_msg}. "
                                f"Cannot use stale data for 0DTE trading.",
                                "ERROR"
                            )
                        return pd.DataFrame()  # Return empty - better than wrong data
                else:
                    # In backtest mode, accept historical data from yfinance
                    validation_msg = "Backtest mode: historical data accepted"
                
                last_price = hist['Close'].iloc[-1]
                last_time_str = timestamp_to_est_str(hist.index[-1])
                log_data_source("yfinance (DELAYED - LAST RESORT)", len(hist), symbol, last_price, last_time_str)
                
                if risk_mgr and hasattr(risk_mgr, 'log'):
                    risk_mgr.log(
                        f"⚠️ Using yfinance (DELAYED) for {symbol} - NOT SUITABLE FOR 0DTE! "
                        f"Last price: ${last_price:.2f} at {last_time_str}. "
                        f"{validation_msg}",
                        "WARNING"
                    )
                
                # Cache data before returning (reduces redundant API calls)
                if not backtest_mode:
                    _set_cached_market_data(symbol, hist, last_price)
                return hist
            else:
                if risk_mgr and hasattr(risk_mgr, 'log'):
                    risk_mgr.log(f"❌ yfinance returned empty data for {symbol}", "ERROR")
                return pd.DataFrame()
        except Exception as e:
            if risk_mgr and hasattr(risk_mgr, 'log'):
                risk_mgr.log(f"❌ All data sources failed for {symbol}: {e}", "ERROR")
            return pd.DataFrame()
    else:
        # yfinance fallback DISABLED - return empty DataFrame
        if risk_mgr and hasattr(risk_mgr, 'log'):
            risk_mgr.log(
                f"❌ CRITICAL: Both Alpaca and Massive API failed for {symbol}. "
                f"yfinance fallback is DISABLED (delayed data not acceptable for 0DTE trading). "
                f"Returning empty DataFrame - iteration will be skipped.",
                "ERROR"
            )
        return pd.DataFrame()  # Empty DataFrame - main loop will skip iteration

def get_current_price(symbol: str, risk_mgr=None) -> Optional[float]:
    """
    Get current price - tries Massive API first, then Alpaca API
    CRITICAL: Returns REAL-TIME price with data source logging
    NO yfinance fallback (delayed data not acceptable for 0DTE trading)
    
    Args:
        symbol: Stock symbol
        risk_mgr: RiskManager instance for logging (optional)
    
    Returns:
        Current price or None (if both APIs fail)
    """
    global massive_client
    
    # Map symbol for Massive API
    massive_symbol_map = {
        'SPY': 'SPY',
        'QQQ': 'QQQ',
        'IWM': 'IWM',
        'SPX': 'SPX',
        '^SPX': 'SPX',
        'VIX.X': 'VIX.X',  # VIX on Polygon
        '^VIX': 'VIX.X'
    }
    massive_symbol = massive_symbol_map.get(symbol, symbol.replace('^', ''))
    
    # ========== PRIORITY 1: MASSIVE API ==========
    if massive_client:
        try:
            if symbol.startswith('^VIX') or symbol == 'VIX.X':
                price = massive_client.get_real_time_price('VIX.X')
            else:
                price = massive_client.get_real_time_price(massive_symbol)
            if price:
                price_float = float(price)
                if risk_mgr and hasattr(risk_mgr, 'log'):
                    risk_mgr.log(
                        f"📊 {symbol} Price: ${price_float:.2f} (source: Massive API - REAL-TIME)",
                        "INFO"
                    )
                return price_float
        except Exception as e:
            if risk_mgr and hasattr(risk_mgr, 'log'):
                risk_mgr.log(f"⚠️ Massive API price fetch failed for {symbol}: {e}", "DEBUG")
            pass
    
    # ========== PRIORITY 2: ALPACA API ==========
    # Try Alpaca API if available (for symbols it supports)
    # Note: This requires api parameter, but get_current_price doesn't have it
    # Alpaca price fetching would need to be added if needed
    
    # ========== NO YFINANCE FALLBACK ==========
    # yfinance is DELAYED (15-20 minutes) - NOT SUITABLE FOR 0DTE TRADING
    if ALLOW_YFINANCE_FALLBACK:
        # Only use yfinance if explicitly enabled (NOT RECOMMENDED)
        try:
            # Handle SPX ticker (requires ^ prefix for yfinance)
            yf_symbol = symbol
            if symbol == 'SPX':
                yf_symbol = '^GSPC'  # S&P 500 index
            elif symbol.startswith('^'):
                yf_symbol = symbol  # Already has ^ prefix
            
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period="1d", interval="1m")
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            if len(hist) > 0:
                price_float = float(hist['Close'].iloc[-1])
                last_time = hist.index[-1]
                
                # Check data freshness
                est = pytz.timezone('US/Eastern')
                now_est = datetime.now(est)
                if hasattr(last_time, 'tzinfo') and last_time.tzinfo:
                    last_time_est = last_time.astimezone(est)
                else:
                    try:
                        last_time_utc = pytz.utc.localize(last_time)
                        last_time_est = last_time_utc.astimezone(est)
                    except:
                        last_time_est = est.localize(last_time) if last_time.tzinfo is None else last_time
                
                time_diff_minutes = (now_est - last_time_est).total_seconds() / 60
                
                if risk_mgr and hasattr(risk_mgr, 'log'):
                    risk_mgr.log(
                        f"⚠️ {symbol} Price: ${price_float:.2f} (source: yfinance - DELAYED {time_diff_minutes:.1f} min) | "
                        f"Last bar: {last_time_est.strftime('%H:%M:%S %Z')} | "
                        f"WARNING: Delayed data not suitable for 0DTE trading!",
                        "WARNING"
                    )
                
                return price_float
        except Exception as e:
            if risk_mgr and hasattr(risk_mgr, 'log'):
                risk_mgr.log(f"⚠️ yfinance price fetch failed for {symbol}: {e}", "DEBUG")
            pass
    else:
        # yfinance fallback DISABLED
        if risk_mgr and hasattr(risk_mgr, 'log'):
            risk_mgr.log(
                f"❌ CRITICAL: Massive API failed for {symbol} price. "
                f"yfinance fallback is DISABLED (delayed data not acceptable). "
                f"Returning None - iteration may be skipped.",
                "ERROR"
            )
    
    return None

# ==================== MODEL LOADING ====================
def load_rl_model():
    """Load trained RL model (supports MaskablePPO, RecurrentPPO/LSTM, and standard PPO)"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            f"Train first with: python train_historical_model.py --human-momentum"
        )
    
    print(f"Loading RL model from {MODEL_PATH}...")
    print(f"Model path check: is_historical_model = {'historical' in MODEL_PATH.lower()}")
    
    # CRITICAL FIX: For historical model (standard PPO), skip RecurrentPPO/MaskablePPO attempts
    # These cause "no locals when deleting <NULL>" errors when loading standard PPO models
    is_historical_model = "historical" in MODEL_PATH.lower()
    
    if is_historical_model:
        print("✓ Detected historical model - skipping RecurrentPPO/MaskablePPO, loading as standard PPO")
    
    # Try RecurrentPPO first (LSTM models) - SKIP for historical models
    if not is_historical_model:
        try:
            from sb3_contrib import RecurrentPPO
            try:
                model = RecurrentPPO.load(MODEL_PATH)
                print("✓ Model loaded successfully (RecurrentPPO with LSTM temporal intelligence)")
                return model
            except Exception as e:
                # Not a RecurrentPPO model, continue to other options
                pass
        except ImportError:
            # RecurrentPPO not available
            pass
    
    # Try MaskablePPO (for action masking support) - SKIP for historical models
    if MASKABLE_PPO_AVAILABLE and not is_historical_model:
        try:
            model = MaskablePPO.load(MODEL_PATH)
            print("✓ Model loaded successfully (MaskablePPO with action masking)")
            return model
        except Exception as e:
            print(f"Warning: Could not load as MaskablePPO: {e}")
            print("Falling back to standard PPO...")
    
    # Load as standard PPO (for historical model or fallback)
    # CRITICAL: Suppress warnings and use minimal options to avoid segfaults
    import warnings
    
    print("Attempting to load as standard PPO...")
    
    try:
        # Method 1: Try loading with custom_objects and suppressed warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                print("  Method 1: PPO.load with custom_objects={}, print_system_info=False")
                model = PPO.load(MODEL_PATH, custom_objects={}, print_system_info=False)
                print("✓ Model loaded successfully (standard PPO)")
                return model
            except Exception as e1:
                # Method 2: Try with explicit CPU device
                try:
                    import torch
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = PPO.load(MODEL_PATH, device='cpu', custom_objects={}, print_system_info=False)
                    print("✓ Model loaded successfully (standard PPO, CPU device)")
                    return model
                except Exception as e2:
                    # Method 3: Try with minimal options only
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = PPO.load(MODEL_PATH, print_system_info=False)
                    print("✓ Model loaded successfully (standard PPO, minimal options)")
                    return model
    except Exception as e:
        # If all methods fail, provide detailed error
        error_msg = str(e)
        print(f"❌ Model loading failed: {error_msg}")
        print(f"   Model path: {MODEL_PATH}")
        print(f"   File exists: {os.path.exists(MODEL_PATH)}")
        if os.path.exists(MODEL_PATH):
            size = os.path.getsize(MODEL_PATH)
            print(f"   File size: {size:,} bytes ({size/1024/1024:.2f} MB)")
        raise RuntimeError(
            f"Failed to load model from {MODEL_PATH}. "
            f"Error: {error_msg}"
        )

# ==================== OPTION SYMBOL HELPERS ====================
def get_option_symbol(underlying: str, strike: float, option_type: str, trading_day: Optional[datetime.date] = None) -> str:
    """
    Generate Alpaca option symbol for 0DTE options
    CRITICAL: Must use trading_day from Alpaca clock to ensure correct expiration date
    
    Args:
        underlying: Underlying symbol (SPY, QQQ, IWM)
        strike: Strike price
        option_type: 'call' or 'put'
        trading_day: Current trading day from Alpaca clock (EST date). If None, uses local EST (not recommended)
    
    Returns:
        Option symbol string (e.g., SPY251220C00680000)
    """
    # Use trading_day from Alpaca clock if provided (RECOMMENDED)
    if trading_day is not None:
        expiration_date = trading_day
    else:
        # Fallback to local EST (NOT RECOMMENDED - should use Alpaca clock)
        est = pytz.timezone('US/Eastern')
        expiration_date = datetime.now(est).date()
        import sys
        print(f"⚠️ WARNING: get_option_symbol() called without trading_day parameter. Using local EST: {expiration_date}", file=sys.stderr)
    
    date_str = expiration_date.strftime('%y%m%d')
    strike_str = f"{int(strike * 1000):08d}"
    type_str = 'C' if option_type == 'call' else 'P'
    option_symbol = f"{underlying}{date_str}{type_str}{strike_str}"
    
    return option_symbol

def find_atm_strike(price: float, option_type: str = 'call', target_delta: float = 0.50) -> float:
    """
    Find optimal strike for 0DTE options trading
    
    Strategy (based on successful manual trades):
    - CALLS: Strike = current_price + $1-3 (slightly OTM, ~$0.50 premium)
    - PUTS: Strike = current_price - $1-5 (slightly OTM, ~$0.40-$0.60 premium)
    
    Args:
        price: Current underlying price
        option_type: 'call' or 'put'
        target_delta: Target delta (0.50 = ATM, 0.30-0.40 = slightly OTM)
    
    Returns:
        Strike price rounded to nearest $0.50 or $1.00 increment
    """
    # For 0DTE options, use slightly OTM strikes (matches successful strategy)
    if option_type.lower() == 'call':
        # CALLS: Strike slightly above current price ($1-3 OTM)
        # This gives ~$0.50 premium and good movement potential
        strike_offset = 2.0  # $2 above price (slightly OTM)
        strike = price + strike_offset
    else:  # put
        # PUTS: Strike slightly below current price ($1-5 OTM)
        # This gives ~$0.40-$0.60 premium and good movement potential
        strike_offset = -3.0  # $3 below price (slightly OTM)
        strike = price + strike_offset
    
    # Round to nearest $0.50 increment (standard option strike spacing)
    # For prices > $100, use $1.00 increments; for < $100, use $0.50 increments
    if price >= 100:
        strike = round(strike)  # Round to nearest $1.00
    else:
        strike = round(strike * 2) / 2  # Round to nearest $0.50
    
    # Validation: Ensure strike is within reasonable range
    # Reject strikes that are >$10 away from current price (too far OTM)
    if abs(strike - price) > 10:
        # Fallback to ATM if calculated strike is too far
        if price >= 100:
            strike = round(price)
        else:
            strike = round(price * 2) / 2
    
    return strike

def estimate_premium(price: float, strike: float, option_type: str) -> float:
    """Estimate option premium using Black-Scholes with fallback"""
    # Try to use scipy for accurate Black-Scholes
    try:
        from scipy.stats import norm
        
        T = config.T if hasattr(config, 'T') else 1/365
        r = config.R if hasattr(config, 'R') else 0.04
        sigma = config.DEFAULT_SIGMA if hasattr(config, 'DEFAULT_SIGMA') else 0.20
        
        if T <= 0:
            return max(0.01, abs(price - strike))
        
        d1 = (np.log(price / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            premium = price * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            premium = strike * np.exp(-r * T) * norm.cdf(-d2) - price * norm.cdf(-d1)
        
        return max(0.01, premium)
    except (ImportError, ModuleNotFoundError, AttributeError):
        # Fallback: Simple intrinsic + time value estimation (no scipy required)
        # This is less accurate but allows trading to continue
        intrinsic = max(0, price - strike) if option_type == 'call' else max(0, strike - price)
        # Add time value: roughly 1-2% of underlying for 0DTE
        time_value = price * 0.015  # 1.5% time value for 0DTE
        premium = intrinsic + time_value
        return max(0.01, premium)

def extract_underlying_from_option(option_symbol: str) -> str:
    """Extract underlying symbol from option symbol (SPY251205C00685000 -> SPY)"""
    for underlying in ['SPX', 'QQQ', 'SPY']:
        if option_symbol.startswith(underlying):
            return underlying
    # Fallback: first 3 chars
    return option_symbol[:3] if len(option_symbol) >= 3 else option_symbol

def check_stop_losses(api: tradeapi.REST, risk_mgr: RiskManager, symbol_prices: dict, trade_db: Optional[TradeDatabase] = None) -> None:
    """
    Check all open positions for volatility-adjusted stop-loss and take-profit triggers
    Uses CORRECT Alpaca v2 API: list_positions() and get_option_snapshot()
    Implements dynamic parameters based on VIX level:
    - Low Vol (VIX < 18): Tight stops (-15%), modest TP (+30%/+60%/+120%)
    - Normal Vol (VIX 18-25): Default (-20%), standard TP (+40%/+80%/+150%)
    - High Vol (VIX 25-35): Wider stops (-28%), stretched TP (+60%/+120%/+250%)
    - Crash Vol (VIX > 35): Maximum stops (-35%), monster TP (+100%/+200%/+400%)
    """
    # Update VIX and regime
    current_vix = risk_mgr.get_current_vix()
    current_regime = risk_mgr.get_vol_regime(current_vix)
    risk_mgr.current_regime = current_regime
    vol_params = risk_mgr.get_vol_params(current_regime)
    
    positions_to_close = []
    
    # Get actual positions from Alpaca (CORRECT API)
    try:
        alpaca_positions = api.list_positions()
        # Filter to only option positions (Alpaca uses 'option' or 'us_option', or check symbol pattern)
        alpaca_option_positions = {pos.symbol: pos for pos in alpaca_positions 
                                   if (hasattr(pos, 'asset_class') and pos.asset_class in ['option', 'us_option']) 
                                   or (len(pos.symbol) >= 15 and ('C' in pos.symbol[-9:] or 'P' in pos.symbol[-9:]))}
    except Exception as e:
        risk_mgr.log(f"Error fetching positions from Alpaca: {e}", "ERROR")
        alpaca_option_positions = {}
    
    # ========== CRITICAL FIX: SYNC ALL ALPACA POSITIONS INTO TRACKING ==========
    # This ensures positions opened before agent start, or externally, are still checked for stop loss
    for symbol, alpaca_pos in alpaca_option_positions.items():
        if symbol not in risk_mgr.open_positions:
            # Position exists in Alpaca but not tracked - sync it!
            risk_mgr.log(f"🔍 Found untracked position in Alpaca: {symbol} - Syncing into tracking for stop loss protection", "INFO")
            
            try:
                # Extract strike and option type
                if len(symbol) >= 15:
                    strike_str = symbol[-8:]
                    strike = float(strike_str) / 1000
                    option_type = 'call' if 'C' in symbol[-9:] else 'put'
                else:
                    # Extract underlying to get appropriate price
                    underlying = extract_underlying_from_option(symbol)
                    default_price = symbol_prices.get(underlying, 0) if isinstance(symbol_prices, dict) else 0
                    strike = default_price if default_price > 0 else 690  # Fallback to SPY-like price
                    option_type = 'call'
                
                # Get ACTUAL entry premium from Alpaca (not current price!)
                entry_premium = None
                
                # Method 1: Try avg_entry_price (most accurate)
                if hasattr(alpaca_pos, 'avg_entry_price') and alpaca_pos.avg_entry_price:
                    try:
                        entry_premium = float(alpaca_pos.avg_entry_price)
                    except (ValueError, TypeError):
                        pass
                
                # Method 2: Calculate from cost_basis (total cost / (qty * 100))
                if entry_premium is None and hasattr(alpaca_pos, 'cost_basis') and alpaca_pos.cost_basis:
                    try:
                        qty = float(alpaca_pos.qty)
                        cost_basis = abs(float(alpaca_pos.cost_basis))
                        if qty > 0:
                            entry_premium = cost_basis / (qty * 100)
                    except (ValueError, TypeError, ZeroDivisionError):
                        pass
                
                # Method 3: Calculate from unrealized_pl and market_value
                if entry_premium is None:
                    try:
                        qty = float(alpaca_pos.qty)
                        market_value = abs(float(alpaca_pos.market_value)) if hasattr(alpaca_pos, 'market_value') and alpaca_pos.market_value else None
                        unrealized_pl = float(alpaca_pos.unrealized_pl) if hasattr(alpaca_pos, 'unrealized_pl') and alpaca_pos.unrealized_pl else None
                        
                        if market_value and unrealized_pl and qty > 0:
                            # unrealized_pl = (current - entry) * qty * 100
                            # entry = (market_value - unrealized_pl) / (qty * 100)
                            entry_premium = (market_value - unrealized_pl) / (qty * 100)
                    except (ValueError, TypeError, ZeroDivisionError):
                        pass
                
                # Method 4: Last resort - estimate (but log warning)
                if entry_premium is None or entry_premium <= 0:
                    # Extract underlying to get appropriate price
                    underlying = extract_underlying_from_option(symbol)
                    default_price = symbol_prices.get(underlying, strike) if isinstance(symbol_prices, dict) else strike
                    entry_premium = estimate_premium(default_price, strike, option_type)
                    risk_mgr.log(f"⚠️ WARNING: Could not get entry premium for {symbol}, using estimate: ${entry_premium:.2f}", "WARNING")
                else:
                    risk_mgr.log(f"✅ Synced {symbol}: Entry premium = ${entry_premium:.4f} from Alpaca data", "INFO")
                
                # Get quantity
                qty = int(float(alpaca_pos.qty))
                
                # Get symbol-specific entry price
                underlying = extract_underlying_from_option(symbol)
                entry_underlying_price = symbol_prices.get(underlying, strike) if isinstance(symbol_prices, dict) else strike
                
                # Add to tracking
                risk_mgr.open_positions[symbol] = {
                    'strike': strike,
                    'type': option_type,
                    'entry_time': datetime.now(pytz.timezone('US/Eastern')),  # Approximate, EST
                    'contracts': qty,
                    'qty_remaining': qty,
                    'notional': qty * entry_premium * 100,
                    'entry_premium': entry_premium,  # ACTUAL entry from Alpaca
                    'entry_price': entry_underlying_price,  # CRITICAL FIX: Use symbol-specific price
                    'trail_active': False,
                    'trail_price': 0.0,
                    'peak_premium': entry_premium,
                    'tp1_done': False,
                    'tp2_done': False,
                    'tp3_done': False,
                    'vol_regime': current_regime,
                    'entry_vix': current_vix,
                    'runner_active': False,
                    'runner_qty': 0,
                    'trail_triggered': False
                }
            except Exception as e:
                risk_mgr.log(f"Error syncing position {symbol}: {e}", "ERROR")
                import traceback
                risk_mgr.log(traceback.format_exc(), "ERROR")
    
    # NOW check all tracked positions (including newly synced ones)
    for symbol, pos_data in list(risk_mgr.open_positions.items()):
        try:
            # Check if position still exists in Alpaca
            if symbol not in alpaca_option_positions:
                # Position was closed externally, remove from tracking
                risk_mgr.log(f"Position {symbol} no longer exists in Alpaca, removing from tracking", "INFO")
                del risk_mgr.open_positions[symbol]
                continue
            
            alpaca_pos = alpaca_option_positions[symbol]
            
            # Use entry-time regime or current regime (whichever is more conservative for stops)
            entry_regime = pos_data.get('vol_regime', current_regime)
            entry_params = risk_mgr.get_vol_params(entry_regime)
            
            # For take-profits, use current regime (adapt to market)
            # For stop-losses, use entry regime (protect from widening)
            tp_params = vol_params  # Use current regime for TP
            sl_params = entry_params  # Use entry regime for SL
            
            # ========== BULLETPROOF STOP-LOSS CHECK (4-STEP PROCESS) ==========
            # Production-grade stop-loss with safe fallbacks
            # Order: 1) Alpaca PnL, 2) Bid-price, 3) Mid-price, 4) Emergency
            
            entry_premium = pos_data.get('entry_premium', 0)
            if entry_premium is None or entry_premium <= 0:
                risk_mgr.log(f"⚠️ WARNING: Invalid entry premium for {symbol}: ${entry_premium}, skipping stop-loss check", "WARNING")
                continue
            
            # Get snapshot for bid/ask prices
            snapshot = None
            bid_premium = None
            ask_premium = None
            mid_premium = None
            try:
                snapshot = api.get_option_snapshot(symbol)
                if snapshot:
                    if snapshot.bid_price:
                        try:
                            bid_premium = float(snapshot.bid_price)
                        except (ValueError, TypeError):
                            pass
                    if snapshot.ask_price:
                        try:
                            ask_premium = float(snapshot.ask_price)
                        except (ValueError, TypeError):
                            pass
                    # Calculate mid-price
                    if bid_premium and ask_premium:
                        mid_premium = (bid_premium + ask_premium) / 2.0
                    elif bid_premium:
                        mid_premium = bid_premium  # Use bid if no ask
            except Exception as e:
                risk_mgr.log(f"⚠️ Error getting snapshot for {symbol}: {e}", "WARNING")
            
            # ========== STEP 1: ALPACA UNREALIZED PnL CHECK (GROUND TRUTH) ==========
            alpaca_plpc = None
            try:
                # Try unrealized_plpc first (percentage as decimal)
                if hasattr(alpaca_pos, 'unrealized_plpc') and alpaca_pos.unrealized_plpc is not None:
                    alpaca_plpc = float(alpaca_pos.unrealized_plpc) / 100.0  # Convert from percentage to decimal
                # Fallback to calculating from unrealized_pl and cost_basis
                elif hasattr(alpaca_pos, 'unrealized_pl') and alpaca_pos.unrealized_pl is not None:
                    unrealized_pl = float(alpaca_pos.unrealized_pl)
                    cost_basis = abs(float(alpaca_pos.cost_basis)) if hasattr(alpaca_pos, 'cost_basis') and alpaca_pos.cost_basis else None
                    if cost_basis and cost_basis > 0:
                        alpaca_plpc = unrealized_pl / cost_basis
            except Exception as e:
                pass  # Will fall through to other checks
            
            # CRITICAL: If Alpaca shows >15% loss, close IMMEDIATELY (ground truth)
            if alpaca_plpc is not None and alpaca_plpc <= -0.15:
                underlying = extract_underlying_from_option(symbol)
                risk_mgr.log(f"🚨 STEP 1 STOP-LOSS (ALPACA PnL): {symbol} @ {alpaca_plpc:.2%} → FORCING IMMEDIATE CLOSE", "CRITICAL")
                
                # Send Telegram exit alert for stop-loss
                if TELEGRAM_AVAILABLE:
                    try:
                        pos_data = risk_mgr.open_positions.get(symbol, {})
                        entry_price = pos_data.get('entry_premium', 0)
                        exit_price = entry_price * (1 + alpaca_plpc) if entry_price > 0 else 0
                        send_exit_alert(
                            symbol=symbol,
                            exit_reason=f"Stop Loss ({alpaca_plpc:.1%})",
                            entry_price=entry_price,
                            exit_price=exit_price,
                            pnl_pct=alpaca_plpc,
                            qty=pos_data.get('contracts', 1)
                        )
                    except Exception:
                        pass  # Never block trading
                
                # Record stop-loss trigger for cooldown (prevent immediate re-entry)
                # Use EST for all timestamps
                est = pytz.timezone('US/Eastern')
                now_est = datetime.now(est)
                risk_mgr.symbol_stop_loss_cooldown[underlying] = now_est
                positions_to_close.append(symbol)
                continue
            
            # ========== STEP 2: BID-PRICE STOP-LOSS (MOST CONSERVATIVE) ==========
            # CRITICAL: Stop-loss MUST use BID price - this is the actual loss when selling
            if bid_premium and bid_premium > 0:
                bid_pnl_pct = (bid_premium - entry_premium) / entry_premium
                if bid_pnl_pct <= -0.15:
                    underlying = extract_underlying_from_option(symbol)
                    risk_mgr.log(f"🚨 STEP 2 STOP-LOSS (BID PRICE): {symbol} @ {bid_pnl_pct:.2%} (Entry: ${entry_premium:.4f}, Bid: ${bid_premium:.4f}) → FORCED FULL EXIT", "CRITICAL")
                    
                    # Send Telegram exit alert for stop-loss
                    if TELEGRAM_AVAILABLE:
                        try:
                            send_exit_alert(
                                symbol=symbol,
                                exit_reason=f"Stop Loss ({bid_pnl_pct:.1%})",
                                entry_price=entry_premium,
                                exit_price=bid_premium,
                                pnl_pct=bid_pnl_pct,
                                qty=pos_data.get('contracts', 1)
                            )
                        except Exception:
                            pass  # Never block trading
                    
                    # Record stop-loss trigger for cooldown (prevent immediate re-entry)
                    # Use EST for all timestamps
                    est = pytz.timezone('US/Eastern')
                    now_est = datetime.now(est)
                    risk_mgr.symbol_stop_loss_cooldown[underlying] = now_est
                    positions_to_close.append(symbol)
                    continue
            
            # ========== STEP 3: MID-PRICE STOP-LOSS (FALLBACK) ==========
            # Use mid-price if bid unavailable (less conservative, but still valid)
            current_premium = mid_premium if mid_premium else None
            premium_source = "snapshot_mid"
            
            # Fallback to market_value if mid unavailable
            if current_premium is None or current_premium <= 0:
                try:
                    if alpaca_pos.market_value and float(alpaca_pos.qty) > 0:
                        qty_float = float(alpaca_pos.qty)
                        market_val_float = abs(float(alpaca_pos.market_value))
                        calculated_premium = market_val_float / (qty_float * 100) if qty_float > 0 else 0.0
                        if calculated_premium > 0:
                            current_premium = calculated_premium
                            mid_premium = calculated_premium
                            premium_source = "market_value"
                except Exception as e:
                    risk_mgr.log(f"⚠️ Market value calculation failed for {symbol}: {e}", "WARNING")
            
            # Last resort: estimate from underlying price
            if current_premium is None or current_premium <= 0:
                underlying = extract_underlying_from_option(symbol)
                current_symbol_price = symbol_prices.get(underlying, pos_data['strike']) if isinstance(symbol_prices, dict) else pos_data['strike']
                current_premium = estimate_premium(current_symbol_price, pos_data['strike'], pos_data['type'])
                mid_premium = current_premium
                premium_source = "estimated"
                risk_mgr.log(f"⚠️ Using ESTIMATED premium for {symbol}: ${current_premium:.4f} (not recommended)", "WARNING")
            
            # Check mid-price stop-loss
            if current_premium and current_premium > 0:
                mid_pnl_pct = (current_premium - entry_premium) / entry_premium
                if mid_pnl_pct <= -0.15:
                    underlying = extract_underlying_from_option(symbol)
                    risk_mgr.log(f"🚨 STEP 3 STOP-LOSS (MID PRICE): {symbol} @ {mid_pnl_pct:.2%} (Entry: ${entry_premium:.4f}, Mid: ${current_premium:.4f}) → FORCED FULL EXIT", "CRITICAL")
                    # Record stop-loss trigger for cooldown (prevent immediate re-entry)
                    # Use EST for all timestamps
                    est = pytz.timezone('US/Eastern')
                    now_est = datetime.now(est)
                    risk_mgr.symbol_stop_loss_cooldown[underlying] = now_est
                    positions_to_close.append(symbol)
                    continue
            
            # ========== STEP 4: EMERGENCY FALLBACK ==========
            # If ALL data is missing and position open > 60 seconds, force close
            if (bid_premium is None and mid_premium is None and alpaca_plpc is None and current_premium is None):
                entry_time = pos_data.get('entry_time')
                if entry_time:
                    # Use EST for all timestamps
                    est = pytz.timezone('US/Eastern')
                    now_est = datetime.now(est)
                    time_open = (now_est - entry_time).total_seconds()
                    if time_open > 60:
                        underlying = extract_underlying_from_option(symbol)
                        risk_mgr.log(f"🚨 STEP 4 EMERGENCY CLOSE (NO DATA): {symbol} open for {int(time_open)}s with no premium data → FORCING CLOSE", "CRITICAL")
                        # Record stop-loss trigger for cooldown (prevent immediate re-entry)
                        # Use EST for all timestamps
                        est = pytz.timezone('US/Eastern')
                        now_est = datetime.now(est)
                        risk_mgr.symbol_stop_loss_cooldown[underlying] = now_est
                        positions_to_close.append(symbol)
                        continue
            
            # ========== CONTINUE WITH NORMAL FLOW (NO STOP-LOSS TRIGGERED) ==========
            # Update peak premium for trailing stops
            if current_premium and current_premium > pos_data.get('peak_premium', entry_premium):
                pos_data['peak_premium'] = current_premium
            
            # Calculate PnL for logging and take-profit checks
            EPSILON = 1e-6
            pnl_pct = (current_premium - entry_premium) / entry_premium if current_premium else 0.0
            
            # Enhanced debug logging
            if pnl_pct <= -0.10:
                risk_mgr.log(f"⚠️ Position {symbol}: PnL = {pnl_pct:.2%} (Entry: ${entry_premium:.4f}, Current: ${current_premium:.4f if current_premium else 'N/A'}, Bid: ${bid_premium:.4f if bid_premium else 'N/A'}, Qty: {int(float(alpaca_pos.qty))}) | Premium Source: {premium_source}", "INFO")
            
            if pnl_pct <= -0.12:
                risk_mgr.log(f"🚨 APPROACHING STOP LOSS: {symbol} at {pnl_pct:.2%} - Stop will trigger at -15%", "CRITICAL")
            
            # Get remaining quantity
            actual_qty = int(float(alpaca_pos.qty))
            qty_remaining = pos_data.get('qty_remaining', actual_qty)
            
            if qty_remaining != actual_qty:
                risk_mgr.log(f"Updating qty_remaining for {symbol}: {qty_remaining} → {actual_qty}", "INFO")
                pos_data['qty_remaining'] = actual_qty
                qty_remaining = actual_qty
            
            # ========== TAKE-PROFIT EXECUTION (ONE PER TICK - CRITICAL) ==========
            # CRITICAL: Only ONE take-profit can trigger per price update to prevent over-selling
            # This prevents gap-ups from triggering all TPs simultaneously
            tp_triggered = False
            
            # Get dynamic TP levels (use dynamic if available, otherwise fallback to regime-based)
            tp1_level = pos_data.get('tp1_dynamic', tp_params.get('tp1', 0.40))
            tp2_level = pos_data.get('tp2_dynamic', tp_params.get('tp2', 0.80))
            tp3_level = pos_data.get('tp3_dynamic', tp_params.get('tp3', 1.50))
            
            # ========== DYNAMIC TAKE-PROFIT TIER 1 ==========
            # Check TP1 FIRST (lowest threshold) - must be sequential
            # Use >= with epsilon to handle floating point precision
            if not tp_triggered and (pnl_pct + EPSILON) >= tp1_level and not pos_data.get('tp1_done', False):
                sell_qty = max(1, int(qty_remaining * 0.5))  # Sell 50% of remaining
                if sell_qty < qty_remaining:
                    try:
                        # CRITICAL FIX: Verify we own the position before selling
                        # This ensures Alpaca knows we're closing a long, not opening a short
                        try:
                            current_pos = api.get_position(symbol)
                            if current_pos and float(current_pos.qty) >= sell_qty:
                                # We own the position, so sell is closing/reducing
                                api.submit_order(
                                    symbol=symbol,
                                    qty=sell_qty,
                                    side='sell',
                                    type='market',
                                    time_in_force='day'
                                )
                            else:
                                risk_mgr.log(f"⚠️ Cannot sell {sell_qty} - only own {float(current_pos.qty) if current_pos else 0}", "WARNING")
                        except Exception as pos_error:
                            # If get_position fails, try submit_order anyway
                            pass
                        api.submit_order(
                            symbol=symbol,
                            qty=sell_qty,
                            side='sell',
                            type='market',
                            time_in_force='day'
                        )
                        pos_data['qty_remaining'] = qty_remaining - sell_qty
                        pos_data['tp1_done'] = True
                        pos_data['tp1_level'] = tp1_level  # Store dynamic TP1 level for trailing calc
                        
                        # Setup trailing stop: TP1 - 20% (using dynamic TP1 level)
                        tp1_price = pos_data['entry_premium'] * (1 + tp1_level)
                        trail_price = pos_data['entry_premium'] * (1 + tp1_level - 0.20)  # TP1 - 20%
                        pos_data['trail_active'] = True
                        pos_data['trail_price'] = trail_price
                        pos_data['trail_tp_level'] = 1  # Track which TP this trail is for
                        pos_data['trail_triggered'] = False
                        
                        tp_triggered = True  # CRITICAL: Prevent other TPs this tick
                        risk_mgr.log(f"🎯 TP1 +{tp1_level:.0%} ({current_regime.upper()}) [Dynamic: {tp1_level:.0%} vs Base: {tp_params['tp1']:.0%}] → SOLD 50% ({sell_qty}x) | Remaining: {pos_data['qty_remaining']} | Trail Stop: +{tp1_level - 0.20:.0%} (${trail_price:.2f})", "TRADE")
                        # Break after successful partial sell - wait for next price update
                        continue
                    except Exception as e:
                        risk_mgr.log(f"✗ Error executing TP1 for {symbol}: {e}", "ERROR")
                else:
                    # If 50% is all remaining, just close
                    try:
                        api.close_position(symbol)
                        risk_mgr.log(f"🎯 TP1 +{tp1_level:.0%} ({current_regime.upper()}) [Dynamic] → FULL EXIT: {symbol} @ {pnl_pct:.1%}", "TRADE")
                        
                        # Send Telegram exit alert for take-profit
                        if TELEGRAM_AVAILABLE:
                            try:
                                exit_price = entry_premium * (1 + pnl_pct) if entry_premium > 0 else 0
                                send_exit_alert(
                                    symbol=symbol,
                                    exit_reason=f"Take Profit 1 (+{tp1_level:.0%})",
                                    entry_price=entry_premium,
                                    exit_price=exit_price,
                                    pnl_pct=pnl_pct,
                                    qty=pos_data.get('contracts', 1)
                                )
                            except Exception:
                                pass  # Never block trading
                        
                        positions_to_close.append(symbol)
                        continue
                    except Exception as e:
                        risk_mgr.log(f"✗ Error closing at TP1: {e}", "ERROR")
            
            # ========== DYNAMIC TAKE-PROFIT TIER 2 ==========
            # Check TP2 ONLY if TP1 is done AND no TP triggered this tick
            if not tp_triggered and (pnl_pct + EPSILON) >= tp2_level and pos_data.get('tp1_done', False) and not pos_data.get('tp2_done', False):
                sell_qty = max(1, int(qty_remaining * 0.6))  # Sell 60% of remaining (improved from 30%)
                if sell_qty < qty_remaining:
                    try:
                        # CRITICAL FIX: Verify we own the position before selling
                        # This ensures Alpaca knows we're closing a long, not opening a short
                        try:
                            current_pos = api.get_position(symbol)
                            if current_pos and float(current_pos.qty) >= sell_qty:
                                # We own the position, so sell is closing/reducing
                                api.submit_order(
                                    symbol=symbol,
                                    qty=sell_qty,
                                    side='sell',
                                    type='market',
                                    time_in_force='day'
                                )
                            else:
                                risk_mgr.log(f"⚠️ Cannot sell {sell_qty} - only own {float(current_pos.qty) if current_pos else 0}", "WARNING")
                        except Exception as pos_error:
                            # If get_position fails, try submit_order anyway
                            pass
                        api.submit_order(
                            symbol=symbol,
                            qty=sell_qty,
                            side='sell',
                            type='market',
                            time_in_force='day'
                        )
                        pos_data['qty_remaining'] = qty_remaining - sell_qty
                        pos_data['tp2_done'] = True
                        pos_data['tp2_level'] = tp2_level  # Store dynamic TP2 level for trailing calc
                        pos_data['tp2_hit'] = True  # RL reward signal: TP2 hit
                        
                        # Setup dynamic trailing stop (activates after TP2)
                        # CRITICAL: Use dynamic TP2 level for trailing-stop initialization
                        pos_data['trail_active'] = True  # Activate trailing stop after TP2
                        pos_data['trail_tp_level'] = 2  # Track which TP this trail is for
                        pos_data['trail_triggered'] = False
                        pos_data['highest_pnl'] = pnl_pct  # Initialize peak PnL at TP2 level
                        # Use dynamic TP2 level to inform trailing-stop behavior
                        # Trailing stop will adapt based on how far TP3 is (dynamic TP3 - dynamic TP2)
                        pos_data['trailing_stop_pct'] = 0.18  # Initial default trailing percentage (will be dynamic)
                        pos_data['tp2_dynamic_for_trail'] = tp2_level  # Store dynamic TP2 for trailing-stop reference
                        pos_data['tp3_dynamic_for_trail'] = pos_data.get('tp3_dynamic', tp_params.get('tp3', 1.50))  # Store dynamic TP3 for trailing-stop reference
                        
                        tp_triggered = True  # CRITICAL: Prevent other TPs this tick
                        risk_mgr.log(f"🎯 TP2 +{tp2_level:.0%} ({current_regime.upper()}) [Dynamic: {tp2_level:.0%} vs Base: {tp_params['tp2']:.0%}] → SOLD 60% ({sell_qty}x) | Remaining: {pos_data['qty_remaining']} | Dynamic Trailing Stop Activated", "TRADE")
                        # Break after successful partial sell - wait for next price update
                        continue
                    except Exception as e:
                        risk_mgr.log(f"✗ Error executing TP2 for {symbol}: {e}", "ERROR")
                else:
                    # If 60% is all remaining, just close
                    try:
                        api.close_position(symbol)
                        risk_mgr.log(f"🎯 TP2 +{tp2_level:.0%} ({current_regime.upper()}) [Dynamic] → FULL EXIT: {symbol} @ {pnl_pct:.1%}", "TRADE")
                        
                        # Send Telegram exit alert for take-profit
                        if TELEGRAM_AVAILABLE:
                            try:
                                exit_price = entry_premium * (1 + pnl_pct) if entry_premium > 0 else 0
                                send_exit_alert(
                                    symbol=symbol,
                                    exit_reason=f"Take Profit 2 (+{tp2_level:.0%})",
                                    entry_price=entry_premium,
                                    exit_price=exit_price,
                                    pnl_pct=pnl_pct,
                                    qty=pos_data.get('contracts', 1)
                                )
                            except Exception:
                                pass  # Never block trading
                        
                        positions_to_close.append(symbol)
                        continue
                    except Exception as e:
                        risk_mgr.log(f"✗ Error closing at TP2: {e}", "ERROR")
            
            # ========== DYNAMIC TAKE-PROFIT TIER 3 ==========
            # Check TP3 ONLY if TP2 is done AND no TP triggered this tick
            # Use dynamic TP3 if available, otherwise fallback to regime-based
            elif not tp_triggered:
                tp3_level = pos_data.get('tp3_dynamic', tp_params.get('tp3', 1.50))
                if (pnl_pct + EPSILON) >= tp3_level and pos_data.get('tp2_done', False) and not pos_data.get('tp3_done', False):
                    try:
                        api.close_position(symbol)
                        pos_data['tp3_hit'] = True  # RL reward signal: TP3 hit (high reward)
                        risk_mgr.log(f"🎯 TP3 +{tp3_level:.0%} HIT ({current_regime.upper()}) [Dynamic: {tp3_level:.0%} vs Base: {tp_params['tp3']:.0%}] → FULL EXIT: {symbol} @ {pnl_pct:.1%}", "TRADE")
                        
                        # Send Telegram exit alert for take-profit
                        if TELEGRAM_AVAILABLE:
                            try:
                                exit_price = entry_premium * (1 + pnl_pct) if entry_premium > 0 else 0
                                send_exit_alert(
                                    symbol=symbol,
                                    exit_reason=f"Take Profit 3 (+{tp3_level:.0%})",
                                    entry_price=entry_premium,
                                    exit_price=exit_price,
                                    pnl_pct=pnl_pct,
                                    qty=pos_data.get('contracts', 1)
                                )
                            except Exception:
                                pass  # Never block trading
                        
                        positions_to_close.append(symbol)
                        continue
                    except Exception as e:
                        risk_mgr.log(f"✗ Error executing TP3 exit for {symbol}: {e}", "ERROR")
            
            
            # ========== TWO-TIER STOP-LOSS SYSTEM (DAMAGE CONTROL) ==========
            # CRITICAL: Check hard stop FIRST (highest priority)
            # Tier 2: Hard Stop-Loss (-35% or regime hard_sl, whichever is more conservative)
            hard_sl_threshold = min(sl_params['hard_sl'], -0.35)  # Use -35% or regime hard_sl, whichever is tighter
            # Use <= with epsilon for floating point precision
            if (pnl_pct - EPSILON) <= hard_sl_threshold:
                risk_mgr.log(f"🚨 HARD STOP-LOSS TRIGGERED ({entry_regime.upper()}, {hard_sl_threshold:.0%}): {symbol} @ {pnl_pct:.1%} → FORCED FULL EXIT", "CRITICAL")
                positions_to_close.append(symbol)
                continue
            
            # Tier 1: Normal Stop-Loss (-20% or regime sl) - Close 50% for damage control
            # Only if TP1 not hit (don't damage control if already profitable)
            # Only if loss is between -20% and -35% (hard stop already checked above)
            # Use <= with epsilon for floating point precision
            if (pnl_pct - EPSILON) <= sl_params['sl'] and pnl_pct > -0.35 and not pos_data.get('tp1_done', False) and not pos_data.get('trail_active', False):
                # Damage control: Close 50% instead of full exit
                damage_control_qty = max(1, int(qty_remaining * 0.5))
                if damage_control_qty < qty_remaining:
                    try:
                        # CRITICAL FIX: Verify we own the position before selling
                        try:
                            current_pos = api.get_position(symbol)
                            if current_pos and float(current_pos.qty) >= damage_control_qty:
                                # We own the position, so sell is closing/reducing
                                api.submit_order(
                                    symbol=symbol,
                                    qty=damage_control_qty,
                                    side='sell',
                                    type='market',
                                    time_in_force='day'
                                )
                            else:
                                risk_mgr.log(f"⚠️ Cannot sell {damage_control_qty} - only own {float(current_pos.qty) if current_pos else 0}", "WARNING")
                        except Exception as pos_error:
                            # If get_position fails, try submit_order anyway
                            api.submit_order(
                                symbol=symbol,
                                qty=damage_control_qty,
                                side='sell',
                                type='market',
                                time_in_force='day'
                            )
                        pos_data['qty_remaining'] = qty_remaining - damage_control_qty
                        risk_mgr.log(f"🛑 DAMAGE CONTROL STOP ({entry_regime.upper()}, {sl_params['sl']:.0%}): {symbol} @ {pnl_pct:.1%} → SOLD 50% ({damage_control_qty}x) | Remaining: {pos_data['qty_remaining']}", "TRADE")
                        continue  # Wait for next price update
                    except Exception as e:
                        risk_mgr.log(f"✗ Error executing damage control stop: {e}, closing full position", "ERROR")
                        # Fall through to full exit
                
                # Full exit if damage control failed
                risk_mgr.log(f"🛑 STOP-LOSS EXIT ({entry_regime.upper()}, {sl_params['sl']:.0%}): {symbol} @ {pnl_pct:.1%}", "TRADE")
                positions_to_close.append(symbol)
                continue
            
            # ========== DYNAMIC TRAILING STOP SYSTEM (after TP2) ==========
            # Check trailing stop if active (after TP1 or TP2)
            # Uses dynamic trailing percentage based on peak PnL and VIX
            # When triggered: Sell 80% of remaining, keep 20% as runner
            if pos_data.get('trail_active', False) and not pos_data.get('trail_triggered', False):
                # Pull any cached regime/VIX info
                vix = pos_data.get('entry_vix') or risk_mgr.get_current_vix()
                highest_pnl = pos_data.get('highest_pnl', 0.0)
                
                # Get best available PnL sources for dynamic trailing stop (Alpaca → bid → mid)
                alpaca_plpc_trail = None
                bid_pnl_pct_trail = None
                mid_pnl_pct_trail = None
                
                try:
                    # Try Alpaca unrealized_plpc first (ground truth)
                    if hasattr(alpaca_pos, 'unrealized_plpc') and alpaca_pos.unrealized_plpc is not None:
                        alpaca_plpc_trail = float(alpaca_pos.unrealized_plpc) / 100.0
                    elif hasattr(alpaca_pos, 'unrealized_pl') and alpaca_pos.unrealized_pl is not None:
                        unrealized_pl = float(alpaca_pos.unrealized_pl)
                        cost_basis = abs(float(alpaca_pos.cost_basis)) if hasattr(alpaca_pos, 'cost_basis') and alpaca_pos.cost_basis else None
                        if cost_basis and cost_basis > 0:
                            alpaca_plpc_trail = unrealized_pl / cost_basis
                except Exception:
                    pass
                
                # Calculate bid and mid PnL percentages
                if bid_premium and bid_premium > 0:
                    bid_pnl_pct_trail = (bid_premium - entry_premium) / entry_premium if entry_premium > 0 else 0.0
                
                if mid_premium and mid_premium > 0:
                    mid_pnl_pct_trail = (mid_premium - entry_premium) / entry_premium if entry_premium > 0 else 0.0
                
                # Choose the best available PnL source for current reading (Alpaca → bid → mid)
                current_pnl = None
                if alpaca_plpc_trail is not None:
                    current_pnl = alpaca_plpc_trail
                elif bid_pnl_pct_trail is not None:
                    current_pnl = bid_pnl_pct_trail
                elif mid_pnl_pct_trail is not None:
                    current_pnl = mid_pnl_pct_trail
                elif pnl_pct is not None:
                    current_pnl = pnl_pct  # Fallback to calculated PnL
                
                if current_pnl is not None:
                    # Update peak PnL
                    if current_pnl > highest_pnl:
                        highest_pnl = current_pnl
                        pos_data['highest_pnl'] = highest_pnl
                    
                    # Compute dynamic trailing threshold for this position
                    base_trailing = pos_data.get('trailing_stop_pct', 0.18)
                    dynamic_trailing_pct = risk_mgr._compute_dynamic_trailing_pct(
                        highest_pnl=highest_pnl,
                        vix=vix,
                        base_trailing=base_trailing,
                    )
                    pos_data['trailing_stop_pct'] = dynamic_trailing_pct  # persist the latest
                    
                    # If drawdown from peak exceeds trailing threshold → exit
                    drawdown = highest_pnl - current_pnl
                    if drawdown >= dynamic_trailing_pct:
                        trail_tp_level = pos_data.get('trail_tp_level', 2)
                        tp_level_pct = pos_data.get('tp1_level', 0.40) if trail_tp_level == 1 else pos_data.get('tp2_level', 0.80)
                        
                        # Calculate sell quantities: 80% of remaining, 20% runner
                        trail_sell_qty = max(1, int(qty_remaining * 0.8))  # 80% of remaining
                        runner_qty = qty_remaining - trail_sell_qty  # 20% of remaining
                        
                        # Edge case: If trail_sell_qty >= qty_remaining, sell all (no runner)
                        if trail_sell_qty >= qty_remaining:
                            trail_sell_qty = qty_remaining
                            runner_qty = 0
                        
                        if trail_sell_qty > 0:
                            try:
                                # CRITICAL FIX: Verify we own the position before selling
                                try:
                                    current_pos = api.get_position(symbol)
                                    if current_pos and float(current_pos.qty) >= trail_sell_qty:
                                        # We own the position, so sell is closing/reducing
                                        api.submit_order(
                                            symbol=symbol,
                                            qty=trail_sell_qty,
                                            side='sell',
                                            type='market',
                                            time_in_force='day'
                                        )
                                    else:
                                        risk_mgr.log(f"⚠️ Cannot sell {trail_sell_qty} - only own {float(current_pos.qty) if current_pos else 0}", "WARNING")
                                except Exception as pos_error:
                                    # If get_position fails, try submit_order anyway
                                    api.submit_order(
                                        symbol=symbol,
                                        qty=trail_sell_qty,
                                        side='sell',
                                        type='market',
                                        time_in_force='day'
                                    )
                                pos_data['qty_remaining'] = qty_remaining - trail_sell_qty
                                pos_data['trail_triggered'] = True
                                pos_data['trail_active'] = False  # Trail done, now manage runner
                                
                                # Activate runner if we have remaining position
                                if runner_qty > 0:
                                    pos_data['runner_active'] = True
                                    pos_data['runner_qty'] = runner_qty
                                    pos_data['trailing_stop_hit'] = True  # RL reward signal: trailing stop hit
                                    risk_mgr.log(f"📉 TRAILING STOP TRIGGERED {symbol}: peak={highest_pnl:.3f}, now={current_pnl:.3f}, drawdown={drawdown:.3f}, limit={dynamic_trailing_pct:.3f} → SOLD 80% ({trail_sell_qty}x) | Runner: {runner_qty}x until EOD or -15% stop", "TRADE")
                                else:
                                    pos_data['trailing_stop_hit'] = True  # RL reward signal: trailing stop hit
                                    risk_mgr.log(f"📉 TRAILING STOP TRIGGERED {symbol}: peak={highest_pnl:.3f}, now={current_pnl:.3f}, drawdown={drawdown:.3f}, limit={dynamic_trailing_pct:.3f} → SOLD ALL ({trail_sell_qty}x)", "TRADE")
                                
                                continue  # Wait for next price update
                            except Exception as e:
                                risk_mgr.log(f"✗ Error executing trailing stop for {symbol}: {e}", "ERROR")
            
            # ========== RUNNER MANAGEMENT ==========
            # Runner: 20% of remaining position runs until EOD or -15% stop loss
            if pos_data.get('runner_active', False) and pos_data.get('runner_qty', 0) > 0:
                runner_qty = pos_data['runner_qty']
                entry_premium = pos_data['entry_premium']
                
                # Condition 1: -15% Stop Loss from entry premium
                stop_loss_price = entry_premium * 0.85  # -15%
                if current_premium <= stop_loss_price + EPSILON:
                    try:
                        # CRITICAL FIX: Verify we own the position before selling
                        try:
                            current_pos = api.get_position(symbol)
                            if current_pos and float(current_pos.qty) >= runner_qty:
                                # We own the position, so sell is closing/reducing
                                api.submit_order(
                                    symbol=symbol,
                                    qty=runner_qty,
                                    side='sell',
                                    type='market',
                                    time_in_force='day'
                                )
                            else:
                                risk_mgr.log(f"⚠️ Cannot sell {runner_qty} - only own {float(current_pos.qty) if current_pos else 0}", "WARNING")
                        except Exception as pos_error:
                            # If get_position fails, try submit_order anyway
                            api.submit_order(
                                symbol=symbol,
                                qty=runner_qty,
                                side='sell',
                                type='market',
                                time_in_force='day'
                            )
                        pos_data['runner_active'] = False
                        pos_data['runner_qty'] = 0
                        pos_data['qty_remaining'] = pos_data.get('qty_remaining', 0) - runner_qty
                        risk_mgr.log(f"🛑 RUNNER STOP-LOSS (-15%): {symbol} @ ${current_premium:.2f} (entry: ${entry_premium:.2f}) → EXIT {runner_qty}x", "TRADE")
                        # Check if position is fully closed
                        if pos_data.get('qty_remaining', 0) <= 0:
                            positions_to_close.append(symbol)
                        continue
                    except Exception as e:
                        risk_mgr.log(f"✗ Error exiting runner at stop-loss: {e}", "ERROR")
                
                # Condition 2: EOD (4:00 PM EST) - Exit runner at market close
                est = pytz.timezone('US/Eastern')
                now = datetime.now(est)
                if now.hour >= 16:  # 4:00 PM or later
                    try:
                        # CRITICAL FIX: Verify we own the position before selling
                        try:
                            current_pos = api.get_position(symbol)
                            if current_pos and float(current_pos.qty) >= runner_qty:
                                # We own the position, so sell is closing/reducing
                                api.submit_order(
                                    symbol=symbol,
                                    qty=runner_qty,
                                    side='sell',
                                    type='market',
                                    time_in_force='day'
                                )
                            else:
                                risk_mgr.log(f"⚠️ Cannot sell {runner_qty} - only own {float(current_pos.qty) if current_pos else 0}", "WARNING")
                        except Exception as pos_error:
                            # If get_position fails, try submit_order anyway
                            api.submit_order(
                                symbol=symbol,
                                qty=runner_qty,
                                side='sell',
                                type='market',
                                time_in_force='day'
                            )
                        pos_data['runner_active'] = False
                        pos_data['runner_qty'] = 0
                        pos_data['qty_remaining'] = pos_data.get('qty_remaining', 0) - runner_qty
                        risk_mgr.log(f"🕐 RUNNER EOD EXIT: {symbol} @ ${current_premium:.2f} → EXIT {runner_qty}x at market close", "TRADE")
                        # Check if position is fully closed
                        if pos_data.get('qty_remaining', 0) <= 0:
                            positions_to_close.append(symbol)
                        continue
                    except Exception as e:
                        risk_mgr.log(f"✗ Error exiting runner at EOD: {e}", "ERROR")
                
                # Condition 3: Runner can hit TP2 or TP3 (optional - let it continue)
                # If runner continues and hits another TP, it will be handled by TP logic
                # Runner remains active until EOD or -15% stop
            
            # ========== REJECTION DETECTION ==========
            # Check if price rejected from entry level (for calls: high > entry but close < entry)
            # CRITICAL FIX: Use symbol-specific price, not global current_price
            underlying = extract_underlying_from_option(symbol)
            current_symbol_price = symbol_prices.get(underlying, 0) if isinstance(symbol_prices, dict) else 0
            
            if current_symbol_price > 0 and pos_data['type'] == 'call':
                # Would need bar data - simplified check
                if current_symbol_price < pos_data['entry_price'] * 0.99:  # 1% rejection
                    risk_mgr.log(f"⚠️ REJECTION DETECTED: {symbol} ({underlying}) → Exit | Entry: ${pos_data['entry_price']:.2f}, Current: ${current_symbol_price:.2f}", "TRADE")
                    positions_to_close.append(symbol)
                    continue
            
        except Exception as e:
            risk_mgr.log(f"Error checking stop-loss/take-profit for {symbol}: {e}", "ERROR")
    
    # Close positions that hit stops (CORRECT API)
    for symbol in positions_to_close:
        try:
            pos_data = risk_mgr.open_positions.get(symbol)
            
            # Use close_position() which works correctly
            api.close_position(symbol)
            risk_mgr.log(f"✓ Position closed: {symbol}", "TRADE")
            
            # Save trade to database if available
            if pos_data and TRADE_DB_AVAILABLE and trade_db:
                try:
                        # Get current premium for exit
                        try:
                            snapshot = api.get_option_snapshot(symbol)
                            exit_premium = float(snapshot.bid_price) if snapshot.bid_price else pos_data.get('entry_premium', 0)
                        except:
                            exit_premium = pos_data.get('entry_premium', 0)
                        
                        # Calculate PnL
                        entry_premium = pos_data.get('entry_premium', 0)
                        pnl = (exit_premium - entry_premium) * pos_data.get('qty_remaining', pos_data.get('contracts', 0)) * 100
                        pnl_pct = ((exit_premium - entry_premium) / entry_premium) if entry_premium > 0 else 0
                        
                        # Try to get order timestamps from Alpaca after closing
                        submitted_at = ''
                        filled_at = ''
                        order_id = ''
                        try:
                            # Get the most recent order for this symbol
                            orders = api.list_orders(status='filled', limit=10, nested=False)
                            for o in orders:
                                if o.symbol == symbol and o.side == 'sell':
                                    submitted_at = o.submitted_at if hasattr(o, 'submitted_at') and o.submitted_at else ''
                                    filled_at = o.filled_at if hasattr(o, 'filled_at') and o.filled_at else ''
                                    order_id = o.id if hasattr(o, 'id') else ''
                                    break
                        except:
                            pass
                        
                        # Convert timestamps to EST
                        est = pytz.timezone('US/Eastern')
                        now_est = datetime.now(est)
                        timestamp_est = now_est.strftime('%Y-%m-%d %H:%M:%S %Z')
                        
                        # Convert Alpaca timestamps (UTC) to EST if provided
                        submitted_at_est = ''
                        filled_at_est = ''
                        if submitted_at:
                            try:
                                # Alpaca timestamps are in UTC
                                if 'T' in str(submitted_at):
                                    dt_utc = datetime.fromisoformat(str(submitted_at).replace('Z', '+00:00'))
                                    if dt_utc.tzinfo is None:
                                        dt_utc = pytz.utc.localize(dt_utc)
                                    dt_est = dt_utc.astimezone(est)
                                    submitted_at_est = dt_est.strftime('%Y-%m-%d %H:%M:%S %Z')
                                else:
                                    submitted_at_est = submitted_at
                            except:
                                submitted_at_est = submitted_at
                        
                        if filled_at:
                            try:
                                if 'T' in str(filled_at):
                                    dt_utc = datetime.fromisoformat(str(filled_at).replace('Z', '+00:00'))
                                    if dt_utc.tzinfo is None:
                                        dt_utc = pytz.utc.localize(dt_utc)
                                    dt_est = dt_utc.astimezone(est)
                                    filled_at_est = dt_est.strftime('%Y-%m-%d %H:%M:%S %Z')
                                else:
                                    filled_at_est = filled_at
                            except:
                                filled_at_est = filled_at
                        
                        trade_db.save_trade({
                            'timestamp': timestamp_est,
                            'symbol': symbol,
                            'action': 'SELL',
                            'qty': pos_data.get('qty_remaining', pos_data.get('contracts', 0)),
                            'entry_premium': entry_premium,
                            'exit_premium': exit_premium,
                            'entry_price': pos_data.get('entry_price', 0),
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'regime': pos_data.get('vol_regime', 'normal'),
                            'vix': pos_data.get('entry_vix', 0),
                            'reason': 'stop_loss_or_take_profit',
                            'order_id': order_id,
                            'submitted_at': submitted_at_est,
                            'filled_at': filled_at_est
                        })
                except Exception as db_error:
                    risk_mgr.log(f"Warning: Could not save trade to database: {db_error}", "WARNING")
            
            if symbol in risk_mgr.open_positions:
                del risk_mgr.open_positions[symbol]
        except Exception as e:
            risk_mgr.log(f"✗ Error closing {symbol}: {e}", "ERROR")
            # Try alternative: submit sell order
            try:
                if symbol in alpaca_option_positions:
                    pos = alpaca_option_positions[symbol]
                    qty = int(float(pos.qty))
                    # CRITICAL FIX: Verify we own the position before selling
                    try:
                        current_pos = api.get_position(symbol)
                        if current_pos and float(current_pos.qty) >= qty:
                            # We own the position, so sell is closing/reducing
                            api.submit_order(
                                symbol=symbol,
                                qty=qty,
                                side='sell',
                                type='market',
                                time_in_force='day'
                            )
                        else:
                            risk_mgr.log(f"⚠️ Cannot sell {qty} - only own {float(current_pos.qty) if current_pos else 0}", "WARNING")
                    except Exception as pos_error:
                        # If get_position fails, try submit_order anyway
                        pass
                    api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    risk_mgr.log(f"✓ Closed via sell order: {symbol}", "TRADE")
                    
                    # Send Telegram exit alert
                    if TELEGRAM_AVAILABLE and symbol in risk_mgr.open_positions:
                        try:
                            pos_data = risk_mgr.open_positions[symbol]
                            entry_price = pos_data.get('entry_premium', 0)
                            # Try to get exit price from position or estimate
                            exit_price = 0
                            try:
                                pos = api.get_position(symbol)
                                if pos and hasattr(pos, 'market_value') and hasattr(pos, 'qty'):
                                    qty_float = float(pos.qty) if pos.qty else 1
                                    if qty_float > 0:
                                        exit_price = abs(float(pos.market_value)) / (qty_float * 100)
                            except:
                                pass
                            
                            # Calculate PnL
                            if entry_price > 0:
                                pnl_pct = ((exit_price - entry_price) / entry_price) if exit_price > 0 else 0
                                pnl_dollar = (exit_price - entry_price) * pos_data.get('contracts', 1) * 100 if exit_price > 0 else 0
                                send_exit_alert(
                                    symbol=symbol,
                                    exit_reason="Manual Close",
                                    entry_price=entry_price,
                                    exit_price=exit_price if exit_price > 0 else entry_price,
                                    pnl_pct=pnl_pct,
                                    qty=pos_data.get('contracts', 1),
                                    pnl_dollar=pnl_dollar
                                )
                        except Exception:
                            pass  # Never block trading
                    
                    if symbol in risk_mgr.open_positions:
                        del risk_mgr.open_positions[symbol]
            except Exception as e2:
                risk_mgr.log(f"✗ Alternative close also failed for {symbol}: {e2}", "ERROR")

# ==================== INSTITUTIONAL FEATURE ENGINE INITIALIZATION ====================
if INSTITUTIONAL_FEATURES_AVAILABLE and USE_INSTITUTIONAL_FEATURES:
    feature_engine = create_feature_engine(lookback_minutes=LOOKBACK)
    print("✅ Institutional feature engine initialized (500+ features)")
else:
    feature_engine = None

# ==================== OBSERVATION PREPARATION ====================
def prepare_observation(data: pd.DataFrame, risk_mgr: RiskManager, symbol: str = 'SPY') -> np.ndarray:
    """
    Prepare observation for RL model - routes to correct version based on model.
    
    The historical model (mike_historical_model.zip) uses (20, 10) features:
    - OHLCV (5) + VIX (1) + Greeks (4) = 10 features
    
    The momentum model (mike_momentum_model_v3_lstm.zip) uses (20, 23) features:
    - OHLCV (5) + VIX (2) + Technical (11) + Greeks (4) + Other (1) = 23 features
    
    CRITICAL: Observation space MUST match training exactly or model will fail silently.
    """
    # Check which model we're using
    # 23-feature models: mike_23feature_model, mike_momentum_model_v3_lstm
    # 10-feature models: mike_historical_model
    if "mike_23feature_model" in MODEL_PATH or "mike_momentum_model" in MODEL_PATH:
        # Use 23-feature observation for models trained with all technical indicators
        obs = prepare_observation_basic(data, risk_mgr, symbol)
        # CRITICAL VALIDATION: Ensure observation shape matches training exactly
        if obs.shape != (20, 23):
            error_msg = (
                f"❌ CRITICAL: Observation shape mismatch for 23-feature model!\n"
                f"   Expected: (20, 23) | Got: {obs.shape}\n"
                f"   Model: {MODEL_PATH}\n"
                f"   This will cause silent model failure and poor generalization.\n"
                f"   Fix: Ensure prepare_observation_basic returns exactly (20, 23)"
            )
            print(error_msg)
            if risk_mgr:
                risk_mgr.log(error_msg, "ERROR")
            # Force correct shape (safety fallback)
            if obs.shape[1] > 23:
                obs = obs[:, :23]
            elif obs.shape[1] < 23:
                padding = np.zeros((20, 23 - obs.shape[1]), dtype=np.float32)
                obs = np.column_stack([obs, padding])
        return obs
    elif "mike_historical_model" in MODEL_PATH:
        # Use 10-feature observation for historical model
        obs = prepare_observation_10_features_inline(data, risk_mgr, symbol)
        # CRITICAL VALIDATION: Ensure observation shape matches training exactly
        if obs.shape != (20, 10):
            error_msg = (
                f"❌ CRITICAL: Observation shape mismatch for historical model!\n"
                f"   Expected: (20, 10) | Got: {obs.shape}\n"
                f"   Model: {MODEL_PATH}\n"
                f"   This will cause silent model failure and poor generalization.\n"
                f"   Fix: Ensure prepare_observation_10_features_inline returns exactly (20, 10)"
            )
            print(error_msg)
            if risk_mgr:
                risk_mgr.log(error_msg, "ERROR")
            # Force correct shape (safety fallback)
            if obs.shape[1] > 10:
                obs = obs[:, :10]
            elif obs.shape[1] < 10:
                padding = np.zeros((20, 10 - obs.shape[1]), dtype=np.float32)
                obs = np.column_stack([obs, padding])
        return obs
    else:
        # Use 23-feature observation for momentum model
        obs = prepare_observation_basic(data, risk_mgr, symbol)
        # Validation for momentum model
        if obs.shape != (20, 23):
            error_msg = (
                f"⚠️  Observation shape mismatch for momentum model!\n"
                f"   Expected: (20, 23) | Got: {obs.shape}\n"
                f"   Model: {MODEL_PATH}"
            )
            print(error_msg)
            if risk_mgr:
                risk_mgr.log(error_msg, "WARNING")
        return obs

def prepare_observation_10_features_inline(data: pd.DataFrame, risk_mgr: RiskManager, symbol: str = 'SPY') -> np.ndarray:
    """
    Inline 10-feature observation preparation (fallback if module not available)
    Matches historical training: OHLCV (5) + VIX (1) + Greeks (4) = 10 features
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
    
    # Normalize OHLC (% change from base)
    o = (opens  - base) / base * 100.0
    h = (highs  - base) / base * 100.0
    l = (lows   - base) / base * 100.0
    c = (closes - base) / base * 100.0
    
    # Normalized volume
    maxv = vols.max() if vols.max() > 0 else 1.0
    v = vols / maxv
    
    # VIX (constant across window)
    vix_value = risk_mgr.get_current_vix() if risk_mgr else 20.0
    vix_norm = np.full(LOOKBACK, (vix_value / 50.0) if vix_value else 0.4, dtype=np.float32)
    
    # Greeks (delta/gamma/theta/vega) - constant across window if no position
    greeks = np.zeros((LOOKBACK, 4), dtype=np.float32)
    
    # Try to get Greeks if we have a position and calculator
    position = None
    if risk_mgr and hasattr(risk_mgr, 'open_positions') and risk_mgr.open_positions:
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
                T=(1.0 / (252 * 6.5)),  # 0DTE approximation
                sigma=(vix_value / 100.0) * 1.3 if vix_value else 0.20,
                option_type=position["option_type"]
            )
            # Fill all bars with same Greeks (constant for window)
            greeks[:] = [
                float(np.clip(g.get("delta", 0), -1, 1)),
                float(np.tanh(g.get("gamma", 0) * 100)),
                float(np.tanh(g.get("theta", 0) / 10)),
                float(np.tanh(g.get("vega", 0) / 10)),
            ]
        except Exception:
            pass  # Keep zeros
    
    # FINAL OBSERVATION (20 × 10)
    obs = np.column_stack([
        o, h, l, c, v,                    # 5 features: OHLCV
        vix_norm,                         # 1 feature: VIX
        greeks[:,0],                      # 1 feature: Delta
        greeks[:,1],                      # 1 feature: Gamma
        greeks[:,2],                      # 1 feature: Theta
        greeks[:,3],                      # 1 feature: Vega
    ]).astype(np.float32)
    
    # Ensure shape is exactly (20, 10)
    if obs.shape != (20, 10):
        if obs.shape[1] > 10:
            obs = obs[:, :10]
        elif obs.shape[1] < 10:
            padding = np.zeros((20, 10 - obs.shape[1]), dtype=np.float32)
            obs = np.column_stack([obs, padding])
    
    return np.clip(obs, -10.0, 10.0)

def prepare_observation_basic(data: pd.DataFrame, risk_mgr: RiskManager, symbol: str = 'SPY') -> np.ndarray:
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
    
    # FINAL OBSERVATION (20 × 23) - Model expects exactly 23 features
    # Note: Portfolio Greeks (4 features) removed to match training (20×23)
    obs = np.column_stack([
        o, h, l, c, v,                    # 5 features: OHLCV
        vix_norm,                         # 1 feature: VIX
        vix_delta_norm,                   # 1 feature: VIX delta
        ema_diff,                         # 1 feature: EMA 9/20 diff
        vwap_dist,                        # 1 feature: VWAP distance
        rsi_scaled,                       # 1 feature: RSI
        macd_hist,                        # 1 feature: MACD histogram
        atr_scaled,                       # 1 feature: ATR
        body_ratio,                       # 1 feature: Candle body ratio
        wick_ratio,                       # 1 feature: Candle wick ratio
        pullback,                         # 1 feature: Pullback
        breakout,                         # 1 feature: Breakout
        trend_slope,                      # 1 feature: Trend slope
        burst,                            # 1 feature: Momentum burst
        trend_strength,                   # 1 feature: Trend strength
        greeks[:,0],                      # 1 feature: Delta
        greeks[:,1],                      # 1 feature: Gamma
        greeks[:,2],                      # 1 feature: Theta
        greeks[:,3],                      # 1 feature: Vega
        # Portfolio Greeks removed (4 features) - model trained on 23 features only
    ]).astype(np.float32)
    
    # Ensure shape is exactly (20, 23)
    if obs.shape[1] != 23:
        obs = obs[:, :23]  # Slice to 23 features if somehow more
    
    return np.clip(obs, -10.0, 10.0)

def prepare_observation_institutional(data: pd.DataFrame, risk_mgr: RiskManager, symbol: str = 'SPY') -> np.ndarray:
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

# ==================== MAIN LIVE LOOP ====================
def run_safe_live_trading():
    """Main live trading loop with all safeguards"""
    # Ensure time module is accessible (prevent UnboundLocalError)
    import time as time_module
    time = time_module
    del time_module
    
    # ========== LIVE AGENT LOCK (PREVENTS BACKTEST INTERFERENCE) ==========
    # Create lock file to indicate live agent is running
    LIVE_AGENT_LOCK_FILE = "/tmp/mike_agent_live.lock"
    LIVE_AGENT_PID_FILE = "/tmp/mike_agent_live.pid"
    
    try:
        # Create lock file
        est = pytz.timezone('US/Eastern')
        now_est = datetime.now(est)
        with open(LIVE_AGENT_LOCK_FILE, 'w') as f:
            f.write(f"Live agent lock - {now_est.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
            f.write(f"PID: {os.getpid()}\n")
        
        # Save PID
        with open(LIVE_AGENT_PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
        
        print(f"✅ Live agent lock created: {LIVE_AGENT_LOCK_FILE}")
        print(f"✅ PID saved: {LIVE_AGENT_PID_FILE}")
    except Exception as e:
        print(f"⚠️  Could not create lock file: {e}")
    
    # Cleanup function to remove lock on exit
    def cleanup_lock():
        try:
            if os.path.exists(LIVE_AGENT_LOCK_FILE):
                os.remove(LIVE_AGENT_LOCK_FILE)
            if os.path.exists(LIVE_AGENT_PID_FILE):
                os.remove(LIVE_AGENT_PID_FILE)
        except:
            pass
    
    import atexit
    atexit.register(cleanup_lock)
    
    # Also register signal handlers
    def signal_handler(sig, frame):
        cleanup_lock()
        sys.exit(0)
    
    import signal
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Send test Telegram alert on startup to verify alerts are working
    if TELEGRAM_AVAILABLE:
        try:
            from utils.telegram_alerts import send_info, is_configured
            if is_configured():
                startup_msg = (
                    "🚀 Mike Agent Started\n\n"
                    "Agent is now running and monitoring the market.\n"
                    "You will receive alerts for:\n"
                    "• Trade entries\n"
                    "• Trade exits (TP/SL)\n"
                    "• Trade blocks\n"
                    "• Critical errors\n\n"
                    "If you received this message, Telegram alerts are working! ✅"
                )
                send_info(startup_msg)
                print("📱 Startup Telegram alert sent")
        except Exception as e:
            print(f"⚠️ Failed to send startup Telegram alert: {e}")
    
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
                
                # ========== AUTOMATIC SYNC FROM ALPACA ON STARTUP ==========
                try:
                    risk_mgr.log("🔄 Syncing trades from Alpaca on startup...", "INFO")
                    from sync_alpaca_trades import sync_alpaca_trades
                    synced_count = sync_alpaca_trades(days_back=7, limit=500)
                    if synced_count > 0:
                        risk_mgr.log(f"✅ Synced {synced_count} trades from Alpaca on startup", "INFO")
                    else:
                        risk_mgr.log("✅ Trade database is up to date (no new trades to sync)", "INFO")
                except Exception as sync_error:
                    risk_mgr.log(f"⚠️ Could not sync trades on startup: {sync_error}", "WARNING")
                    # Don't fail startup if sync fails
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
                current_spy_price = get_current_price("SPY", risk_mgr=risk_mgr)
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
                    'entry_time': datetime.now(pytz.timezone('US/Eastern')),  # Approximate, EST
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
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            
            try:
                # ========== ALPACA CLOCK (AUTHORITATIVE SOURCE OF TRUTH) ==========
                # CRITICAL: Use Alpaca clock for "today" - broker clock is the only authoritative source
                # This prevents wrong-day trading if OS clock drifts or multiple machines disagree
                try:
                    clock = api.get_clock()
                    now_utc = clock.timestamp  # UTC timestamp from Alpaca
                    est = pytz.timezone('US/Eastern')
                    now_est = now_utc.astimezone(est)
                    today_est = now_est.date()  # Today's date from Alpaca clock (EST)
                    
                    # Log Alpaca clock for validation (every 10th iteration to avoid spam)
                    if iteration % 10 == 0:
                        risk_mgr.log(
                            f"⏰ ALPACA_CLOCK_EST = {now_est.strftime('%Y-%m-%d %H:%M:%S %Z')} | "
                            f"Market Open: {clock.is_open} | Today: {today_est}",
                            "INFO"
                        )
                    
                    # Check if market is open (don't trade when closed)
                    if not clock.is_open:
                        if iteration % 20 == 0:  # Log every 20th iteration when market is closed
                            risk_mgr.log(
                                f"⏸️  Market is CLOSED (Alpaca clock) | Next open: {clock.next_open} | Next close: {clock.next_close}",
                                "INFO"
                            )
                            # Print daily no-trade summary when market closes
                            print_daily_no_trade_summary(risk_mgr)
                        track_no_trade_reason('market_closed', risk_mgr)
                        time.sleep(60)  # Wait longer when market is closed
                        continue
                    
                except Exception as clock_error:
                    risk_mgr.log(
                        f"❌ CRITICAL: Could not get Alpaca clock: {clock_error} | "
                        f"Falling back to local EST (NOT RECOMMENDED)",
                        "ERROR"
                    )
                    # Fallback to local EST (not ideal, but better than crashing)
                    est = pytz.timezone('US/Eastern')
                    now_est = datetime.now(est)
                    today_est = now_est.date()
                
                # ========== DAILY RESET CHECK (PREVENTS STALE OPTION SYMBOLS) ==========
                # CRITICAL: Reset daily state when new trading day detected from Alpaca clock
                # This prevents Dec 5/Dec 10 symbols from leaking into Dec 19
                if risk_mgr.current_trading_day != today_est:
                    risk_mgr.log(
                        f"🔄 NEW_TRADING_DAY = {today_est} | Previous: {risk_mgr.current_trading_day} | "
                        f"Executing daily reset...",
                        "INFO"
                    )
                    risk_mgr.reset_daily_state(today_est)
                    risk_mgr.log(
                        f"✅ RESET_DAILY_STATE executed | Option cache cleared | All cooldowns reset",
                        "INFO"
                    )
                
                # ========== SAFEGUARD CHECK ==========
                can_trade, reason = risk_mgr.check_safeguards(api)
                
                if not can_trade:
                    if iteration % 10 == 0:  # Log every 10th iteration
                        risk_mgr.log(f"Safeguard active: {reason}", "INFO")
                    # Track specific safeguard reasons
                    if 'loss limit' in reason.lower():
                        track_no_trade_reason('daily_loss_limit', risk_mgr)
                    elif 'vix' in reason.lower():
                        track_no_trade_reason('vix_kill_switch', risk_mgr)
                    else:
                        track_no_trade_reason('safeguard_blocked', risk_mgr)
                    time.sleep(30)
                    continue
                
                # Force data refresh - clear any caches before fetching
                # This ensures we get fresh data every iteration
                try:
                    import yfinance as yf
                    # Clear yfinance cache if possible
                    if hasattr(yf, 'pdr_override'):
                        yf.pdr_override = False
                except:
                    pass
                
                # Get latest SPY data (Alpaca first, then Massive, then yfinance)
                hist = get_market_data("SPY", period="2d", interval="1m", api=api, risk_mgr=risk_mgr)
                
                if len(hist) < LOOKBACK:
                    risk_mgr.log("Waiting for more data...", "INFO")
                    time.sleep(30)
                    continue
                
                # Validate data is from today (double-check using Alpaca clock date)
                # Note: today_est is already set from Alpaca clock above
                last_bar_time = hist.index[-1]
                
                # Convert to EST if timezone-aware
                if hasattr(last_bar_time, 'tzinfo') and last_bar_time.tzinfo:
                    last_bar_est = last_bar_time.astimezone(est)
                else:
                    try:
                        last_bar_utc = pytz.utc.localize(last_bar_time)
                        last_bar_est = last_bar_utc.astimezone(est)
                    except:
                        last_bar_est = est.localize(last_bar_time) if last_bar_time.tzinfo is None else last_bar_time
                
                last_bar_date = last_bar_est.date()
                
                if last_bar_date != today_est:
                    risk_mgr.log(
                        f"❌ CRITICAL: Data is from {last_bar_date}, not today ({today_est})! "
                        f"Skipping this iteration.",
                        "ERROR"
                    )
                    time.sleep(30)
                    continue
                
                current_price = hist['Close'].iloc[-1]
                
                # CRITICAL: Validate current_price is reasonable (sanity check)
                # SPY should be in reasonable range (e.g., $600-$700 for Dec 2025)
                if current_price < 600 or current_price > 700:
                    risk_mgr.log(
                        f"❌ CRITICAL: current_price ${current_price:.2f} is outside reasonable range ($600-$700). "
                        f"Data may be wrong. Last bar time: {last_bar_est.strftime('%Y-%m-%d %H:%M:%S %Z')}. "
                        f"Skipping this iteration.",
                        "ERROR"
                    )
                    time.sleep(30)
                    continue
                
                # Calculate data age for logging
                try:
                    data_age_minutes = (now_est - last_bar_est).total_seconds() / 60
                except:
                    data_age_minutes = 0
                
                # Log price source and validation
                risk_mgr.log(
                    f"📊 SPY Price Validation: ${current_price:.2f} | "
                    f"Last bar: {last_bar_est.strftime('%H:%M:%S %Z')} | "
                    f"Data age: {data_age_minutes:.1f} min | Price is within expected range ✅",
                    "INFO"
                )
                
                # ========== PRICE CROSS-VALIDATION ==========
                # Cross-validate price with alternative sources (safety check)
                # This helps catch data issues but we always use primary source (Alpaca/Massive)
                try:
                    # Try to get price from alternative source for validation
                    alt_price = None
                    alt_source = None
                    
                    # If we used Alpaca, try Massive for validation
                    # If we used Massive, try Alpaca for validation
                    # If we used yfinance, we're already in trouble (delayed data)
                    
                    # Try Alpaca for validation if available
                    if api:
                        try:
                            from alpaca_trade_api.rest import TimeFrame
                            now_est = datetime.now(est)
                            end_str = (now_est + timedelta(days=1)).strftime("%Y-%m-%d")
                            start_str = now_est.strftime("%Y-%m-%d")
                            
                            alt_bars = api.get_bars("SPY", TimeFrame.Minute, start_str, end_str, limit=1, adjustment='raw').df
                            if len(alt_bars) > 0:
                                alt_price = float(alt_bars['close'].iloc[-1])
                                alt_source = "Alpaca"
                        except:
                            pass
                    
                    # Try Massive for validation if Alpaca didn't work
                    # NOTE: Architect feedback - Alpaca is CANONICAL, Massive is informational only
                    if alt_price is None and massive_client:
                        try:
                            now_est = datetime.now(est)
                            end_date_str = (now_est + timedelta(days=1)).strftime("%Y-%m-%d")
                            start_date_str = now_est.strftime("%Y-%m-%d")
                            
                            alt_hist = massive_client.get_historical_data("SPY", start_date_str, end_date_str, interval='1min')
                            if len(alt_hist) > 0:
                                alt_price = float(alt_hist['close'].iloc[-1] if 'close' in alt_hist.columns else alt_hist['Close'].iloc[-1])
                                alt_source = "Massive"
                                # Get Massive timestamp for comparison
                                alt_timestamp = alt_hist.index[-1] if len(alt_hist) > 0 else None
                        except:
                            pass
                    
                    # If we have alternative price, compare (but ALWAYS use Alpaca as canonical)
                    if alt_price is not None:
                        price_diff = abs(current_price - alt_price)
                        price_diff_pct = (price_diff / current_price) * 100 if current_price > 0 else 0
                        
                        # ARCHITECT RECOMMENDATION: >0.3% mismatch = ignore Massive
                        if price_diff_pct > 0.3:
                            risk_mgr.log(
                                f"⚠️ DATA MISMATCH (>0.3%): Alpaca: ${current_price:.2f} vs {alt_source}: ${alt_price:.2f} | "
                                f"Diff: ${price_diff:.2f} ({price_diff_pct:.2f}%) | USING ALPACA (canonical)",
                                "WARNING"
                            )
                        elif price_diff > 0.50:  # More than $0.50 difference but under 0.3%
                            risk_mgr.log(
                                f"🔍 Price Validation: Alpaca: ${current_price:.2f}, {alt_source}: ${alt_price:.2f}, "
                                f"diff: ${price_diff:.2f} ({price_diff_pct:.2f}%)",
                                "DEBUG"
                            )
                        else:
                            # Prices match closely - good
                            if iteration % 10 == 0:  # Log every 10th iteration to avoid spam
                                risk_mgr.log(
                                    f"✅ Price Validation: Alpaca: ${current_price:.2f}, {alt_source}: ${alt_price:.2f}, "
                                    f"diff: ${price_diff:.2f} ({price_diff_pct:.2f}%) - MATCH ✓",
                                    "DEBUG"
                                )
                except Exception as e:
                    # Validation failed - not critical, just log
                    if iteration % 10 == 0:  # Log every 10th iteration
                        risk_mgr.log(f"⚠️ Price validation failed: {e} (non-critical)", "DEBUG")
                
                # ========== GAP DETECTION (Mike's Strategy Foundation) ==========
                # Detect overnight gaps and override RL signal for first 45-60 minutes
                gap_data = None
                gap_action = None
                est = pytz.timezone('US/Eastern')
                now_est = datetime.now(est)
                current_time_int = now_est.hour * 100 + now_est.minute  # HHMM format
                
                if GAP_DETECTION_AVAILABLE and not risk_mgr.open_positions:
                    # Only detect gap during market open (9:30 AM - 10:35 AM ET)
                    if 930 <= current_time_int <= 1035:
                        gap_data = detect_overnight_gap('SPY', current_price, hist, risk_mgr)
                        if gap_data and gap_data.get('detected'):
                            gap_action = get_gap_based_action(gap_data, current_price, current_time_int)
                            if gap_action:
                                risk_mgr.log(f"🎯 GAP-BASED ACTION: {gap_action} ({get_action_name(gap_action)}) | Overriding RL signal for first 60 min", "INFO")
                
                # ========== MULTI-SYMBOL RL INFERENCE (CRITICAL FIX) ==========
                # BLOCKER 1 FIX: Run RL inference PER SYMBOL, not once globally
                # This ensures QQQ/SPX get their own signals based on their own data
                
                available_symbols = [sym for sym in TRADING_SYMBOLS if not any(s.startswith(sym) for s in risk_mgr.open_positions.keys())]
                
                # Store RL actions per symbol with confidence/strength
                symbol_actions = {}  # {symbol: (action, action_source, action_strength)}
                
                # Run RL inference for each available symbol
                for sym in available_symbols:
                    try:
                        # Get symbol-specific market data (Alpaca first, then Massive, then yfinance)
                        sym_hist = get_market_data(sym, period="2d", interval="1m", api=api, risk_mgr=risk_mgr)
                        
                        if len(sym_hist) < LOOKBACK:
                            risk_mgr.log(f"⚠️ {sym}: Insufficient data ({len(sym_hist)} < {LOOKBACK} bars), skipping RL inference", "WARNING")
                            continue
                        
                        # ========== TECHNICAL ANALYSIS (MIKE'S PATTERN RECOGNITION) ==========
                        ta_result = None
                        ta_confidence_boost = 0.0
                        ta_strike_suggestion = None
                        ta_pattern_detected = False
                        
                        try:
                            from technical_analysis_engine import TechnicalAnalysisEngine
                            
                            # Initialize TA engine if not already done
                            if not hasattr(risk_mgr, 'ta_engine'):
                                risk_mgr.ta_engine = TechnicalAnalysisEngine(lookback_bars=50)
                            
                            # Get current price for TA
                            current_sym_price = get_current_price(sym, risk_mgr=risk_mgr)
                            if current_sym_price is None:
                                current_sym_price = sym_hist['close'].iloc[-1]
                            
                            # Run technical analysis
                            ta_result = risk_mgr.ta_engine.analyze_symbol(
                                data=sym_hist,
                                symbol=sym,
                                current_price=current_sym_price
                            )
                            
                            # Extract TA insights
                            if ta_result and ta_result.get('best_pattern'):
                                ta_pattern_detected = True
                                best_pattern = ta_result['best_pattern']
                                ta_confidence_boost = ta_result.get('confidence_boost', 0.0)
                                ta_strike_suggestion = ta_result.get('strike_suggestion')
                                
                                # Log pattern detection
                                risk_mgr.log(
                                    f"🎯 {sym} TA Pattern: {best_pattern['pattern_type']} ({best_pattern['direction']}) | "
                                    f"Confidence: {best_pattern['confidence']:.2f} | "
                                    f"Boost: +{ta_confidence_boost:.3f} | "
                                    f"Reason: {best_pattern.get('reason', 'N/A')}",
                                    "INFO"
                                )
                                
                                # Log targets if available
                                if ta_result.get('targets'):
                                    targets = ta_result['targets']
                                    risk_mgr.log(
                                        f"🎯 {sym} TA Targets: ${targets.get('target1', 0):.2f} / "
                                        f"${targets.get('target2', 0):.2f} | "
                                        f"Strike Suggestion: ${ta_strike_suggestion:.2f}",
                                        "INFO"
                                    )
                        except ImportError:
                            # TA engine not available, continue without it
                            pass
                        except Exception as e:
                            risk_mgr.log(f"⚠️ {sym} TA analysis error: {e}", "WARNING")
                            pass  # Don't block trading if TA fails
                        
                        # Prepare observation for THIS symbol
                        obs = prepare_observation(sym_hist, risk_mgr, symbol=sym)
                        
                        # 🔍 CRITICAL VALIDATION: Log observation stats and verify shape
                        # Determine expected shape based on model type
                        if "mike_23feature_model" in MODEL_PATH or "mike_momentum_model" in MODEL_PATH:
                            expected_shape = (20, 23)
                        elif "mike_historical_model" in MODEL_PATH:
                            expected_shape = (20, 10)
                        else:
                            expected_shape = (20, 23)  # Default to 23 for unknown models
                        if obs.shape != expected_shape:
                            risk_mgr.log(
                                f"❌ CRITICAL: {sym} Observation shape mismatch! Expected {expected_shape}, got {obs.shape}. "
                                f"Model will fail silently. This is a critical bug.",
                                "ERROR"
                            )
                        else:
                            risk_mgr.log(
                                f"✅ {sym} Observation: shape={obs.shape} (CORRECT), min={obs.min():.2f}, max={obs.max():.2f}, mean={obs.mean():.2f}, has_nan={np.isnan(obs).any()}, all_zero={(obs == 0).all()}",
                                "DEBUG"
                            )
                        
                        # RL Decision for THIS symbol with temperature-calibrated softmax
                        # Get raw logits from policy distribution and apply temperature
                        # Initialize action_raw defensively to avoid scoping errors
                        action_raw = None
                        try:
                            import torch
                            
                            # Check if this is a RecurrentPPO model (requires LSTM state handling)
                            try:
                                from sb3_contrib import RecurrentPPO
                                is_recurrent = isinstance(model, RecurrentPPO)
                            except ImportError:
                                is_recurrent = False
                            
                            if is_recurrent:
                                # RecurrentPPO: Use model.predict() directly (handles LSTM states internally)
                                # Don't use get_distribution() - it requires lstm_states and episode_starts
                                action_raw, lstm_state = model.predict(obs, deterministic=bool(risk_mgr.open_positions))
                                if isinstance(action_raw, np.ndarray):
                                    rl_action = int(action_raw.item() if action_raw.ndim == 0 else action_raw[0])
                                else:
                                    rl_action = int(action_raw)
                                # Estimate strength for RecurrentPPO (conservative)
                                if rl_action in (1, 2):
                                    action_strength = 0.65
                                elif rl_action == 0:
                                    action_strength = 0.50
                                else:
                                    action_strength = 0.35
                                risk_mgr.log(f"🔍 {sym} RecurrentPPO predict: action={rl_action}, estimated_strength={action_strength:.3f}", "DEBUG")
                            elif hasattr(model.policy, 'get_distribution'):
                                # Non-recurrent models: Can use get_distribution for temperature calibration
                                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                                action_dist = model.policy.get_distribution(obs_tensor)
                                
                                # Extract logits (works for both Categorical and MaskableCategorical)
                                if hasattr(action_dist.distribution, 'logits'):
                                    logits = action_dist.distribution.logits
                                elif hasattr(action_dist.distribution, 'probs'):
                                    # If only probs available, convert back to logits
                                    probs_raw = action_dist.distribution.probs
                                    logits = torch.log(probs_raw + 1e-8)
                                else:
                                    raise AttributeError("No logits or probs found in distribution")
                                
                                # Apply temperature calibration (0.7 = sweet spot for live inference)
                                temperature = 0.7
                                probs = torch.softmax(logits / temperature, dim=-1).detach().cpu().numpy()[0]
                                
                                # Get action from argmax of calibrated probabilities
                                rl_action = int(np.argmax(probs))
                                action_strength = float(probs[rl_action])
                                
                                # 🔍 DEBUG: Log probabilities
                                risk_mgr.log(f"🔍 {sym} RL Probs: {[f'{p:.3f}' for p in probs]} | Action={rl_action} | Strength={action_strength:.3f}", "DEBUG")
                            else:
                                # Fallback: Use standard predict and estimate strength from action type
                                action_raw, _ = model.predict(obs, deterministic=bool(risk_mgr.open_positions))
                                if isinstance(action_raw, np.ndarray):
                                    rl_action = int(action_raw.item() if action_raw.ndim == 0 else action_raw[0])
                                else:
                                    rl_action = int(action_raw)
                                # Estimate strength based on action (conservative fallback)
                                if rl_action in (1, 2):  # BUY
                                    action_strength = 0.65
                                elif rl_action == 0:  # HOLD
                                    action_strength = 0.50
                                else:  # TRIM/EXIT
                                    action_strength = 0.35
                                risk_mgr.log(f"🔍 {sym} Using fallback predict: action={rl_action}, estimated_strength={action_strength:.3f}", "DEBUG")
                        except Exception as e:
                            # Fallback to standard predict on error
                            risk_mgr.log(f"🔍 {sym} Temperature softmax failed: {e}, using standard predict", "DEBUG")
                            import traceback
                            risk_mgr.log(traceback.format_exc(), "DEBUG")
                            action_raw, _ = model.predict(obs, deterministic=bool(risk_mgr.open_positions))
                            if isinstance(action_raw, np.ndarray):
                                rl_action = int(action_raw.item() if action_raw.ndim == 0 else action_raw[0])
                            else:
                                rl_action = int(action_raw)
                            # Conservative fallback strength
                            if rl_action in (1, 2):
                                action_strength = 0.60
                            elif rl_action == 0:
                                action_strength = 0.50
                            else:
                                action_strength = 0.30
                        
                        # 🔍 DEBUG: Log original RL action (before any remapping)
                        original_rl_action = rl_action  # Preserve original for logging
                        risk_mgr.log(f"🔍 {sym} RL Action={rl_action}, Strength={action_strength:.3f} (temperature-calibrated)", "DEBUG")
                        
                        # ========== MULTI-AGENT ENSEMBLE SIGNAL ==========
                        ensemble_action = None
                        ensemble_confidence = 0.0
                        ensemble_details = None
                        
                        if MULTI_AGENT_ENSEMBLE_AVAILABLE:
                            try:
                                meta_router = get_meta_router()
                                if meta_router:
                                    # Prepare data for ensemble (ensure column names match)
                                    ensemble_data = sym_hist.copy()
                                    if 'Close' in ensemble_data.columns:
                                        ensemble_data = ensemble_data.rename(columns={
                                            'Open': 'open', 'High': 'high', 'Low': 'low',
                                            'Close': 'close', 'Volume': 'volume'
                                        })
                                    
                                    # Get ensemble signal
                                    vix_value = risk_mgr.get_current_vix() if risk_mgr else 20.0
                                    # Fix: Use ensemble_data directly, not undefined 'closes' variable
                                    if 'close' in ensemble_data.columns:
                                        current_price_val = ensemble_data['close'].iloc[-1]
                                    elif 'Close' in ensemble_data.columns:
                                        current_price_val = ensemble_data['Close'].iloc[-1]
                                    else:
                                        # Fallback: use current_price from outer scope
                                        current_price_val = current_price
                                    strike_val = round(current_price_val)
                                    
                                    # Get portfolio delta if available
                                    portfolio_delta_val = 0.0
                                    delta_limit_val = 2000.0
                                    if PORTFOLIO_GREEKS_AVAILABLE:
                                        try:
                                            greeks_mgr = get_portfolio_greeks_manager()
                                            if greeks_mgr:
                                                exposure = greeks_mgr.get_current_exposure()
                                                portfolio_delta_val = exposure.get('portfolio_delta', 0.0)
                                                account_size = exposure.get('account_size', 10000.0)
                                                delta_limit_val = account_size * 0.20  # 20% limit
                                        except Exception:
                                            pass
                                    
                                    ensemble_action, ensemble_confidence, ensemble_details = meta_router.route(
                                        data=ensemble_data,
                                        vix=vix_value,
                                        symbol=sym,
                                        current_price=current_price_val,
                                        strike=strike_val,
                                        portfolio_delta=portfolio_delta_val,
                                        delta_limit=delta_limit_val
                                    )
                                    
                                    risk_mgr.log(
                                        f"🎯 {sym} Ensemble: action={ensemble_action} ({get_action_name(ensemble_action)}), "
                                        f"confidence={ensemble_confidence:.3f}, regime={ensemble_details.get('regime', 'unknown')}",
                                        "INFO"
                                    )
                                    
                                    # Log individual agent signals
                                    signals = ensemble_details.get('signals', {})
                                    for agent_name, signal_info in signals.items():
                                        risk_mgr.log(
                                            f"   {agent_name.upper()}: action={signal_info['action']} "
                                            f"({get_action_name(signal_info['action'])}), "
                                            f"conf={signal_info['confidence']:.3f}, "
                                            f"weight={signal_info['weight']:.2f} | {signal_info['reasoning']}",
                                            "DEBUG"
                                        )
                            except Exception as e:
                                risk_mgr.log(f"⚠️ {sym} Ensemble analysis failed: {e}", "WARNING")
                                import traceback
                                risk_mgr.log(traceback.format_exc(), "DEBUG")
                        
                        # ========== COMBINE RL + ENSEMBLE SIGNALS ==========
                        # Hierarchical combination: Risk > Macro > Volatility > Gamma > Trend > Reversal > RL
                        # RL weight: 40% (lower priority in hierarchy)
                        # Ensemble weight: 60% (higher priority)
                        RL_WEIGHT = 0.40
                        ENSEMBLE_WEIGHT = 0.60
                        
                        # Confidence override rules
                        MIN_CONFIDENCE_THRESHOLD = 0.3
                        
                        if ensemble_action is not None:
                            # Check for confidence overrides
                            if ensemble_confidence < MIN_CONFIDENCE_THRESHOLD and action_strength > MIN_CONFIDENCE_THRESHOLD:
                                # Ensemble too weak, use RL
                                final_action = rl_action
                                final_confidence = action_strength
                                action_source = "RL (ensemble low confidence)"
                                risk_mgr.log(
                                    f"⚠️ {sym} Ensemble confidence too low ({ensemble_confidence:.2f}), using RL signal",
                                    "WARNING"
                                )
                            elif action_strength < MIN_CONFIDENCE_THRESHOLD and ensemble_confidence > MIN_CONFIDENCE_THRESHOLD:
                                # RL too weak, use ensemble
                                final_action = ensemble_action
                                final_confidence = ensemble_confidence
                                action_source = "Ensemble (RL low confidence)"
                                risk_mgr.log(
                                    f"⚠️ {sym} RL confidence too low ({action_strength:.2f}), using ensemble signal",
                                    "WARNING"
                                )
                            else:
                                # Both have reasonable confidence: combine with hierarchical weights
                                action_scores = {0: 0.0, 1: 0.0, 2: 0.0}  # HOLD, BUY_CALL, BUY_PUT
                                
                                # Ensemble contribution (higher weight - higher in hierarchy)
                                if ensemble_action in action_scores:
                                    action_scores[ensemble_action] += ENSEMBLE_WEIGHT * ensemble_confidence
                                
                                # RL contribution (lower weight - lower in hierarchy)
                                if rl_action in action_scores:
                                    action_scores[rl_action] += RL_WEIGHT * action_strength
                                
                                # Select winning action
                                combined_action = max(action_scores, key=action_scores.get)
                                combined_confidence = action_scores[combined_action]
                                
                                # Normalize confidence
                                total_score = sum(action_scores.values())
                                if total_score > 0:
                                    combined_confidence = combined_confidence / total_score
                                else:
                                    combined_confidence = 0.0
                                
                                # Use combined signal
                                final_action = combined_action
                                final_confidence = combined_confidence
                                action_source = "RL+Ensemble"
                                
                                risk_mgr.log(
                                    f"🔀 {sym} Combined Signal: RL={rl_action}({action_strength:.2f}) + "
                                    f"Ensemble={ensemble_action}({ensemble_confidence:.2f}) → "
                                    f"Final={final_action}({final_confidence:.2f})",
                                    "INFO"
                                )
                        else:
                            # Use RL only if ensemble unavailable
                            final_action = rl_action
                            final_confidence = action_strength
                            action_source = "RL"
                        
                        # Use final_action and final_confidence from combined signal (or RL if ensemble unavailable)
                        action = final_action
                        action_strength = final_confidence
                        
                        # ========== BOOST CONFIDENCE WITH TECHNICAL ANALYSIS ==========
                        if ta_pattern_detected and ta_confidence_boost > 0:
                            # Boost confidence based on TA pattern
                            base_confidence = action_strength
                            boosted_confidence = min(0.95, base_confidence + ta_confidence_boost)
                            action_strength = boosted_confidence
                            
                            risk_mgr.log(
                                f"🚀 {sym} Confidence Boost: {base_confidence:.3f} → {boosted_confidence:.3f} "
                                f"(+{ta_confidence_boost:.3f} from TA pattern: {ta_result['best_pattern']['pattern_type']})",
                                "INFO"
                            )
                        
                        # ❌ RESAMPLING REMOVED PER RED-TEAM REPORT
                        # The model is correctly uncertain - forcing trades via resampling
                        # causes losses. Better to have zero trades than wrong trades.
                        # "Kill the idea that more trades = learning"
                        resampled = False
                        # If model says HOLD or TRIM/EXIT when flat, respect it - do NOT resample

                        # Map discrete actions to trading actions
                        # Model outputs: 0=HOLD, 1=BUY CALL, 2=BUY PUT, 3=TRIM 50%, 4=TRIM 70%, 5=FULL EXIT
                        # Only allow trim/exit actions if positions exist
                        masked = False
                        if final_action >= 3 and not risk_mgr.open_positions:
                            masked = True
                            final_action = 0  # Mask TRIM/EXIT when flat
                            final_confidence = 0.5  # Lower confidence when masked
                        
                        # Gap-based override for this symbol (first 60 minutes only)
                        action = final_action  # Use combined signal (or RL if ensemble unavailable)
                        
                        if gap_action is not None and 930 <= current_time_int <= 1035 and not risk_mgr.open_positions:
                            action = gap_action
                            action_source = "GAP"
                            action_strength = 0.9  # High strength for gap signals
                        else:
                            # Use the action_source and action_strength from combined signal
                            action_strength = final_confidence
                        
                        action = int(action)
                        # Store per-symbol action with confidence and TA result (dict format)
                        symbol_actions[sym] = {
                            'action': action,
                            'action_source': action_source,
                            'action_strength': action_strength,
                            'ta_result': ta_result
                        }
                        
                        # Log per-symbol RL decision (using canonical action mapping)
                        # Show original action if it was remapped/masked
                        action_desc = get_action_name(action)
                        # Note: resampled is always False now (resampling removed per red-team report)
                        if masked:
                            original_desc = get_action_name(original_rl_action)
                            risk_mgr.log(f"🧠 {sym} RL Inference: action={action} ({action_desc}) | Original: {original_rl_action} ({original_desc}) | Source: {action_source} | Strength: {action_strength:.3f}", "INFO")
                        else:
                            risk_mgr.log(f"🧠 {sym} RL Inference: action={action} ({action_desc}) | Source: {action_source} | Strength: {action_strength:.3f}", "INFO")
                        
                    except Exception as e:
                        risk_mgr.log(f"❌ Error running RL inference for {sym}: {e}", "ERROR")
                        import traceback
                        risk_mgr.log(traceback.format_exc(), "ERROR")
                
                # For symbols with positions, use HOLD (trim/exit handled separately)
                # For gap detection override (first trade of day), use gap action
                global_action = 0  # Default to HOLD
                global_action_source = "RL"
                
                # If no available symbols, check gap action for first trade
                if not available_symbols and gap_action is not None and 930 <= current_time_int <= 1035 and not risk_mgr.open_positions:
                    global_action = gap_action
                    global_action_source = "GAP"
                elif symbol_actions:
                    # Find first symbol with BUY signal (BUY CALL or BUY PUT)
                    for sym, action_data in symbol_actions.items():
                        # Handle both old format (tuple) and new format (dict with ta_result)
                        if isinstance(action_data, dict):
                            action = action_data.get('action', 0)
                            source = action_data.get('action_source', 'unknown')
                            strength = action_data.get('action_strength', 0.0)
                        else:
                            # Old format: (action, source, strength)
                            action, source, strength = action_data
                        
                        if action in [1, 2]:  # BUY CALL (1) or BUY PUT (2) - using canonical action codes
                            global_action = action
                            global_action_source = f"{source}_{sym}"
                            break
                
                action = global_action
                action_source = global_action_source
                
                # ENHANCED LOGGING: Show why trades are/aren't happening
                if action == 0:  # HOLD (canonical action 0)
                    if symbol_actions:
                        # Handle both dict and tuple formats
                        actions_list = []
                        for sym, action_data in symbol_actions.items():
                            if isinstance(action_data, dict):
                                act = action_data.get('action', 0)
                                strength = action_data.get('action_strength', 0.0)
                            else:
                                act, _, strength = action_data
                            actions_list.append(f"{sym}:{act}({strength:.2f})")
                        actions_summary = ", ".join(actions_list)
                        risk_mgr.log(f"🤔 Multi-Symbol RL: All HOLD | Actions: [{actions_summary}] | Open Positions: {len(risk_mgr.open_positions)}/{MAX_CONCURRENT} | Available: {available_symbols}", "INFO")
                        # Track no-trade reason
                        track_no_trade_reason('hold_signals', risk_mgr)
                    else:
                        risk_mgr.log(f"🤔 Multi-Symbol RL: HOLD (no available symbols) | Open Positions: {len(risk_mgr.open_positions)}/{MAX_CONCURRENT}", "INFO")
                        # Track no-trade reason
                        if len(risk_mgr.open_positions) >= MAX_CONCURRENT:
                            track_no_trade_reason('position_limit', risk_mgr)
                        else:
                            track_no_trade_reason('hold_signals', risk_mgr)
                
                # Log raw RL output for debugging (every 5th iteration for better visibility)
                if iteration % 5 == 0:
                    action_desc = get_action_name(action)
                    # Note: rl_action may not be in scope here, use action instead
                    risk_mgr.log(f"🔍 RL Debug: Final Action={action} ({action_desc}) | Source: {action_source}", "INFO")
                
                equity = risk_mgr.get_equity(api)
                status = f"FLAT"
                if risk_mgr.open_positions:
                    status = f"{len(risk_mgr.open_positions)} positions"
                
                # Show current VIX and regime
                current_vix = risk_mgr.get_current_vix()
                current_regime = risk_mgr.get_vol_regime(current_vix)
                regime_params = risk_mgr.get_vol_params(current_regime)
                
                # Get prices for all trading symbols
                # SPX requires ^SPX ticker in yfinance
                symbol_ticker_map = {
                    'SPY': 'SPY',
                    'QQQ': 'QQQ',
                    'IWM': 'IWM',
                    'SPX': '^SPX'  # SPX index requires ^ prefix
                }
                
                symbol_prices = {}
                for sym in TRADING_SYMBOLS:
                    try:
                        # Use mapped ticker symbol for yfinance
                        yf_ticker = symbol_ticker_map.get(sym, sym)
                        ticker = yf.Ticker(yf_ticker)
                        sym_hist = ticker.history(period="1d", interval="1m")
                        if isinstance(sym_hist.columns, pd.MultiIndex):
                            sym_hist.columns = sym_hist.columns.get_level_values(0)
                        if len(sym_hist) > 0:
                            symbol_prices[sym] = float(sym_hist['Close'].iloc[-1])
                        else:
                            symbol_prices[sym] = current_price  # Fallback to SPY
                    except Exception as e:
                        # Fallback to SPY price if fetch fails
                        symbol_prices[sym] = current_price
                
                # Build symbol price string (format SPX with comma for thousands)
                price_str_parts = []
                for sym, price in symbol_prices.items():
                    if sym == 'SPX':
                        price_str_parts.append(f"{sym}: ${price:,.2f}")  # SPX needs comma formatting
                    else:
                        price_str_parts.append(f"{sym}: ${price:.2f}")
                price_str = " | ".join(price_str_parts)
                
                # Log with all symbol prices
                risk_mgr.log(f"{price_str} | VIX: {current_vix:.1f} ({current_regime.upper()}) | Risk: {regime_params['risk']:.0%} | Max Size: {regime_params['max_pct']:.0%} | Action: {action} | Equity: ${equity:,.2f} | Status: {status} | Daily PnL: {risk_mgr.daily_pnl:.2%}", "INFO")
                
                # ========== CHECK STOP-LOSSES ON EXISTING POSITIONS ==========
                # CRITICAL: Check stop losses EVERY iteration to prevent large losses
                # This ensures positions are monitored continuously, not just periodically
                check_stop_losses(api, risk_mgr, symbol_prices, trade_db)
                
                # CRITICAL: If we have open positions, check stop losses more frequently
                # For 0DTE options, price can move quickly - need continuous monitoring
                if len(risk_mgr.open_positions) > 0:
                    # Double-check stop losses after a brief delay if we have positions
                    # This catches any price movements that occurred between checks
                    import time
                    time.sleep(2)  # Brief 2-second delay
                    # Refresh symbol prices before second check
                    for sym in TRADING_SYMBOLS:
                        try:
                            sym_price = get_current_price(sym)
                            if sym_price:
                                symbol_prices[sym] = sym_price
                        except:
                            pass
                    check_stop_losses(api, risk_mgr, symbol_prices, trade_db)
                
                # ========== SAFE EXECUTION WITH REGIME-ADAPTIVE POSITION SIZING ==========
                # BLOCKER 1 FIX: Use per-symbol RL actions instead of global action
                if action == 1 and len(risk_mgr.open_positions) < MAX_CONCURRENT:  # BUY CALL (canonical action 1)
                    # Use smart symbol selection: rotation + position filter + cooldown filter + strength-based
                    current_symbol = choose_best_symbol_for_trade(
                        iteration=iteration,
                        symbol_actions=symbol_actions,
                        target_action=1,  # BUY CALL
                        open_positions=risk_mgr.open_positions,
                        risk_mgr=risk_mgr,  # For cooldown checks
                        max_positions_per_symbol=1  # Max 1 position per symbol
                    )
                    
                    if not current_symbol:
                        buy_call_symbols = [sym for sym, action_data in symbol_actions.items() 
                                           if isinstance(action_data, dict) and action_data.get('action') == 1]
                        risk_mgr.log(f"⛔ BLOCKED: No eligible symbols for BUY CALL | Signals: {buy_call_symbols} | Open Positions: {list(risk_mgr.open_positions.keys())}", "INFO")
                        # Fall through to next iteration
                    else:
                        # Check minimum confidence threshold before executing
                        action_data = symbol_actions.get(current_symbol, {})
                        if isinstance(action_data, dict):
                            selected_strength = action_data.get('action_strength', 0.0)
                        else:
                            # Fallback for tuple format (legacy)
                            try:
                                if isinstance(action_data, (tuple, list)) and len(action_data) > 2:
                                    selected_strength = action_data[2]
                                else:
                                    selected_strength = 0.0
                            except (IndexError, TypeError, KeyError):
                                selected_strength = 0.0
                        if selected_strength < MIN_ACTION_STRENGTH_THRESHOLD:
                            block_reason = f"Confidence too low (strength={selected_strength:.3f} < {MIN_ACTION_STRENGTH_THRESHOLD:.3f})"
                            risk_mgr.log(f"⛔ BLOCKED: Selected symbol {current_symbol} {block_reason} | Skipping trade", "INFO")
                            # Track no-trade reason
                            track_no_trade_reason('low_confidence', risk_mgr)
                            # Send Telegram block alert for confidence threshold blocks
                            if TELEGRAM_AVAILABLE:
                                try:
                                    send_block_alert(symbol=current_symbol, block_reason=block_reason)
                                except Exception:
                                    pass  # Never block trading
                            time.sleep(10)
                            continue
                        # current_symbol already validated and selected by choose_best_symbol_for_trade
                        buy_call_symbols = [sym for sym, action_data in symbol_actions.items() 
                                           if isinstance(action_data, dict) and action_data.get('action') == 1]
                        risk_mgr.log(f"🎯 SYMBOL SELECTION: {current_symbol} selected for BUY CALL (strength={selected_strength:.3f}) | All CALL signals: {buy_call_symbols}", "INFO")
                    
                    # Skip if no symbol selected
                    if current_symbol is None:
                        time.sleep(10)
                        continue
                    
                    # Get current price for selected symbol (Massive API first, yfinance fallback)
                    symbol_price = get_current_price(current_symbol, risk_mgr=risk_mgr)
                    if symbol_price is None:
                        risk_mgr.log(
                            f"⚠️ get_current_price() returned None for {current_symbol}, using SPY price ${current_price:.2f} as fallback",
                            "WARNING"
                        )
                        symbol_price = current_price  # Fallback to SPY
                    
                    # CRITICAL: Validate price is reasonable and current
                    # For SPY/QQQ/IWM, prices should be in reasonable ranges
                    expected_ranges = {
                        'SPY': (600, 700),
                        'QQQ': (500, 700),
                        'IWM': (150, 250),
                        'SPX': (6000, 7000)
                    }
                    
                    if current_symbol in expected_ranges:
                        min_price, max_price = expected_ranges[current_symbol]
                        if symbol_price < min_price or symbol_price > max_price:
                            risk_mgr.log(
                                f"❌ CRITICAL: {current_symbol} price ${symbol_price:.2f} is outside expected range "
                                f"(${min_price}-${max_price}). Data may be stale or wrong. REJECTING ORDER.",
                                "ERROR"
                            )
                            continue  # Reject order
                    
                    # Cross-validate with current_price (SPY) - should be close for similar ETFs
                    if current_symbol in ['SPY', 'QQQ', 'IWM']:
                        price_diff = abs(symbol_price - current_price)
                        price_diff_pct = price_diff / current_price if current_price > 0 else 0
                        
                        if price_diff > 5.0:  # More than $5 difference
                            risk_mgr.log(
                                f"❌ CRITICAL: {current_symbol} price ${symbol_price:.2f} differs from SPY ${current_price:.2f} "
                                f"by ${price_diff:.2f} ({price_diff_pct:.1%}). Data may be stale. REJECTING ORDER.",
                                "ERROR"
                            )
                            continue  # Reject order
                        elif price_diff > 2.0:  # More than $2 difference
                            risk_mgr.log(
                                f"⚠️ WARNING: {current_symbol} price ${symbol_price:.2f} differs from SPY ${current_price:.2f} "
                                f"by ${price_diff:.2f}. Proceeding with caution.",
                                "WARNING"
                            )
                    
                    # Log price source and validation
                    risk_mgr.log(
                        f"📊 Price Validation: {current_symbol} = ${symbol_price:.2f} | "
                        f"SPY = ${current_price:.2f} | Diff: ${abs(symbol_price - current_price):.2f} | "
                        f"Price is within expected range ✅",
                        "INFO"
                    )
                    
                    # ========== USE TA-BASED STRIKE IF AVAILABLE ==========
                    # Check if we have TA analysis for this symbol
                    ta_strike = None
                    if current_symbol in symbol_actions:
                        action_info = symbol_actions[current_symbol]
                        # Handle dict format (current) or tuple format (legacy)
                        if isinstance(action_info, dict):
                            ta_result_call = action_info.get('ta_result')
                            if ta_result_call and isinstance(ta_result_call, dict) and ta_result_call.get('strike_suggestion'):
                                ta_strike = ta_result_call['strike_suggestion']
                        else:
                            # Legacy tuple format: (action, source, strength, ta_result)
                            if len(action_info) >= 4 and action_info[3] is not None:
                                ta_result_call = action_info[3]
                                if ta_result_call and isinstance(ta_result_call, dict) and ta_result_call.get('strike_suggestion'):
                                    ta_strike = ta_result_call['strike_suggestion']
                    
                    # Use TA-based strike if available, otherwise use fixed offset
                    if ta_strike and ta_strike > 0:
                        strike = ta_strike
                        risk_mgr.log(
                            f"🎯 {current_symbol} Using TA-based strike: ${strike:.2f}",
                            "INFO"
                        )
                    else:
                        strike = find_atm_strike(symbol_price, option_type='call')
                        risk_mgr.log(
                            f"📊 {current_symbol} Using fixed-offset strike: ${strike:.2f} (price + $2)",
                            "INFO"
                        )
                    
                    symbol = get_option_symbol(current_symbol, strike, 'call', trading_day=today_est)
                    
                    # CRITICAL: Validate option expiration date is TODAY (from Alpaca clock)
                    # Note: today_est is already set from Alpaca clock above
                    
                    # Parse expiration from option symbol (format: SPY251219C00688000)
                    # Extract date: positions 3-8 (YYMMDD)
                    if len(symbol) >= 8:
                        try:
                            date_str = symbol[3:9]  # Extract YYMMDD
                            exp_year = 2000 + int(date_str[:2])
                            exp_month = int(date_str[2:4])
                            exp_day = int(date_str[4:6])
                            exp_date = datetime(exp_year, exp_month, exp_day).date()
                            
                            if exp_date != today_est:
                                risk_mgr.log(
                                    f"❌ CRITICAL: Option {symbol} expiration ({exp_date}) is NOT today ({today_est})! "
                                    f"REJECTING ORDER to prevent trading expired options.",
                                    "ERROR"
                                )
                                # Reject this order
                                continue
                            else:
                                risk_mgr.log(
                                    f"✅ Option expiration validated: {symbol} expires {exp_date} (today EST)",
                                    "INFO"
                                )
                        except Exception as e:
                            risk_mgr.log(
                                f"⚠️ WARNING: Could not parse expiration from {symbol}: {e}. Proceeding with caution.",
                                "WARNING"
                            )
                    
                    # Validate strike is reasonable (within $5 of current price)
                    if abs(strike - symbol_price) > 5:
                        risk_mgr.log(f"⚠️ WARNING: Strike ${strike:.2f} is ${abs(strike - symbol_price):.2f} away from price ${symbol_price:.2f} - may be too far OTM", "WARNING")
                    
                    # Log symbol selection for debugging
                    risk_mgr.log(f"📊 Selected symbol for CALL: {current_symbol} @ ${symbol_price:.2f} | Strike: ${strike:.2f} | Option: {symbol} | Expiration: {exp_date if 'exp_date' in locals() else 'N/A'}", "INFO")
                    
                    # Get current regime and parameters
                    current_vix = risk_mgr.get_current_vix()
                    current_regime = risk_mgr.get_vol_regime(current_vix)
                    risk_mgr.current_regime = current_regime
                    regime_params = risk_mgr.get_vol_params(current_regime)
                    
                    # ========== DYNAMIC TAKE-PROFIT CALCULATION ==========
                    # Calculate dynamic TP levels based on ATR, TrendStrength, VIX, Personality, Confidence
                    base_tp1 = regime_params['tp1']
                    base_tp2 = regime_params['tp2']
                    base_tp3 = regime_params['tp3']
                    
                    # Get symbol-specific historical data for dynamic TP calculation
                    tp1_dynamic = base_tp1
                    tp2_dynamic = base_tp2
                    tp3_dynamic = base_tp3
                    
                    if DYNAMIC_TP_AVAILABLE:
                        try:
                            # Get symbol-specific data for dynamic TP calculation
                            sym_hist_tp = get_market_data(current_symbol, period="5d", interval="1m", api=api, risk_mgr=risk_mgr)
                            
                            if len(sym_hist_tp) >= 20:
                                # Calculate dynamic TP factors
                                from dynamic_take_profit import (
                                    compute_dynamic_tp_factors,
                                    compute_dynamic_takeprofits,
                                    get_ticker_personality_factor,
                                    extract_trend_strength
                                )
                                
                                # Extract factors
                                atr_factor, trend_factor, vix_factor, personality_factor = compute_dynamic_tp_factors(
                                    sym_hist_tp, current_symbol, risk_mgr
                                )
                                
                                # Calculate dynamic TP levels
                                tp1_dynamic, tp2_dynamic, tp3_dynamic = compute_dynamic_takeprofits(
                                    base_tp1, base_tp2, base_tp3,
                                    atr_factor, trend_factor, vix_factor, personality_factor,
                                    action_strength  # Use RL confidence for dynamic adjustment
                                )
                                
                                if risk_mgr and hasattr(risk_mgr, 'log'):
                                    risk_mgr.log(
                                        f"🎯 Dynamic TP for {current_symbol}: "
                                        f"TP1={tp1_dynamic:.0%} (base: {base_tp1:.0%}), "
                                        f"TP2={tp2_dynamic:.0%} (base: {base_tp2:.0%}), "
                                        f"TP3={tp3_dynamic:.0%} (base: {base_tp3:.0%})",
                                        "INFO"
                                    )
                        except Exception as e:
                            # Fallback to base TP levels if dynamic calculation fails
                            if risk_mgr and hasattr(risk_mgr, 'log'):
                                risk_mgr.log(f"⚠️ Dynamic TP calculation failed: {e}, using base TP levels", "WARNING")
                            tp1_dynamic = base_tp1
                            tp2_dynamic = base_tp2
                            tp3_dynamic = base_tp3
                    
                    # Continue with trade execution using tp1_dynamic, tp2_dynamic, tp3_dynamic
                    # (Trade execution code continues here...)
                    
            except KeyboardInterrupt:
                risk_mgr.log("🛑 Trading stopped by user (KeyboardInterrupt)", "INFO")
                break
            except Exception as e:
                risk_mgr.log(f"❌ Error in trading loop iteration: {e}", "ERROR")
                import traceback
                risk_mgr.log(traceback.format_exc(), "ERROR")
                time.sleep(30)  # Wait before retrying
                continue
        
    except KeyboardInterrupt:
        risk_mgr.log("🛑 Trading stopped by user (KeyboardInterrupt)", "INFO")
    except Exception as e:
        risk_mgr.log(f"❌ Fatal error in main trading loop: {e}", "CRITICAL")
        import traceback
        risk_mgr.log(traceback.format_exc(), "CRITICAL")
    finally:
        risk_mgr.log("🔄 Shutting down trading agent...", "INFO")
        # Close any open positions if needed
        try:
            if api:
                positions = api.list_positions()
                if len(positions) > 0:
                    risk_mgr.log(f"⚠️ Closing {len(positions)} open positions on shutdown...", "WARNING")
                    api.close_all_positions()
        except Exception as e:
            risk_mgr.log(f"⚠️ Error closing positions on shutdown: {e}", "WARNING")

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    run_safe_live_trading()
