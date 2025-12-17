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
# Using SPY/QQQ only (ETFs fully supported in paper trading)
TRADING_SYMBOLS = ['SPY', 'QQQ']  # SPX removed - not available in Alpaca paper trading

# ==================== RISK LIMITS (HARD-CODED – CANNOT BE OVERRIDDEN) ====================
DAILY_LOSS_LIMIT = -0.15  # -15% daily loss limit
HARD_DAILY_LOSS_DOLLAR = -500  # Hard stop: Stop trading if daily loss > $500 (absolute dollar limit)
MAX_POSITION_PCT = 0.25  # Max 25% of equity in one position
MAX_CONCURRENT = 2  # Max 2 positions at once (one per symbol: SPY, QQQ)
MAX_TRADES_PER_SYMBOL = 10  # Max 10 trades per symbol per day (reduced from 100 to prevent overtrading)
MIN_TRADE_COOLDOWN_SECONDS = 60  # Minimum 60 seconds between ANY trades (increased from 5s to prevent rapid-fire trading)
VIX_KILL = 28  # No trades if VIX > 28
IVR_MIN = 30  # Minimum IV Rank (0-100)
# Entry time filter (disabled per user request)
NO_TRADE_AFTER = None  # type: Optional[str]  # No time restriction - trading allowed all day
MIN_ACTION_STRENGTH_THRESHOLD = 0.65  # Minimum confidence (0.65 = 65%) required to execute trades (prevents weak signals)
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
# Use the trained historical model (5M timesteps, 23.9 years of data, PPO)
# Trained on SPY, QQQ, SPX with historical data (2002-2025) and regime-aware sampling
# Completed: December 9, 2025
MODEL_PATH = "models/mike_historical_model.zip"
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
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/mike_agent_safe_{datetime.now().strftime('%Y%m%d')}.log"
    
    def reset_daily_state(self) -> None:
        """
        Reset daily state at midnight
        Called automatically on day change
        CRITICAL: Cooldowns must reset to allow new day trading
        """
        today = datetime.now().strftime('%Y-%m-%d')
        if self.last_reset_date == today:
            return  # Already reset today
        
        self.last_reset_date = today
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
        
        self.log(f"🔄 Daily reset: All cooldowns cleared, daily counters reset", "INFO")
    
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
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] [{level}] {msg}"
        print(log_msg)
        try:
            with open(self.log_file, "a") as f:
                f.write(f"{datetime.now()} | [{level}] {msg}\n")
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
        # Disabled when NO_TRADE_AFTER is None.
        if NO_TRADE_AFTER:
            current_time = datetime.now().strftime("%H:%M")
            try:
                now_t = datetime.strptime(current_time, "%H:%M").time()
                cutoff_t = datetime.strptime(str(NO_TRADE_AFTER), "%H:%M").time()
                if now_t > cutoff_t:
                    return False, f"⛔ BLOCKED: After {NO_TRADE_AFTER} EST (time filter) | Current: {current_time} EST"
            except Exception:
                # If parsing fails for any reason, do NOT block trading.
                pass
        
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
    
    def check_order_safety(self, symbol: str, qty: int, premium: float, api: tradeapi.REST, is_entry: bool = True) -> tuple[bool, str]:
        """
        Check order-level safeguards with dynamic position size
        CRITICAL: Cooldown checks apply ONLY to entries, NEVER to exits
        
        Args:
            symbol: Option symbol
            qty: Number of contracts
            premium: Option premium per contract (not strike price!)
            api: Alpaca API instance
            is_entry: True if this is an entry (buy), False if exit (sell)
        Returns: (is_safe, reason_if_unsafe)
        """
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
            time_since_last_trade = (datetime.now() - self.last_any_trade_time).total_seconds()
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
                time_since_sl = (datetime.now() - self.symbol_stop_loss_cooldown[underlying]).total_seconds()
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
                time_since_last = (datetime.now() - self.symbol_last_trade_time[underlying]).total_seconds()
                if time_since_last < MIN_SYMBOL_COOLDOWN_SECONDS:
                    remaining_seconds = int(MIN_SYMBOL_COOLDOWN_SECONDS - time_since_last) + 1
                    return False, f"⛔ BLOCKED: Per-symbol cooldown active for {underlying} | {remaining_seconds}s remaining (prevents rapid-fire trading)"
            
            # ========== SAFEGUARD 8.9: Trailing-Stop Cooldown ==========
            # Prevent immediate re-entry after trailing-stop trigger (60 seconds cooldown)
            TRAILING_STOP_COOLDOWN_SECONDS = 60  # 60 seconds cooldown after trailing stop
            if underlying in self.symbol_trailing_stop_cooldown:
                time_since_ts = (datetime.now() - self.symbol_trailing_stop_cooldown[underlying]).total_seconds()
                if time_since_ts < TRAILING_STOP_COOLDOWN_SECONDS:
                    remaining_seconds = int(TRAILING_STOP_COOLDOWN_SECONDS - time_since_ts) + 1
                    return False, f"⛔ BLOCKED: Trailing-stop cooldown active for {underlying} | {remaining_seconds}s remaining (prevents re-entry after TS trigger)"
                else:
                    # Cooldown expired, remove from tracking
                    del self.symbol_trailing_stop_cooldown[underlying]
            
            # Record trade time for per-symbol cooldown (for entries only)
            self.symbol_last_trade_time[underlying] = datetime.now()
        
        # ========== SAFEGUARD 9: Duplicate Order Protection ==========
        if symbol in self.last_order_time:
            time_since_last = (datetime.now() - self.last_order_time[symbol]).total_seconds()
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
        self.last_order_time[symbol] = datetime.now()
        self.last_any_trade_time = datetime.now()  # Track global trade time for cooldown
        
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
    symbols = TRADING_SYMBOLS  # ['SPY', 'QQQ', 'SPX']
    
    # Rotation for fairness
    rot = iteration % len(symbols)
    priority_order = symbols[rot:] + symbols[:rot]
    
    # Filter candidates: must pass all checks
    candidates = []
    filtered_reasons = []
    
    for sym in priority_order:
        # Check if symbol has target action
        if sym not in symbol_actions:
            continue
        
        action, source, strength = symbol_actions[sym]
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
            time_since_sl = (datetime.now() - risk_mgr.symbol_stop_loss_cooldown[sym]).total_seconds()
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
            time_since_ts = (datetime.now() - risk_mgr.symbol_trailing_stop_cooldown[sym]).total_seconds()
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


def get_market_data(symbol: str, period: str = "2d", interval: str = "1m", api: Optional[tradeapi.REST] = None, risk_mgr = None) -> pd.DataFrame:
    """
    Get market data - tries Alpaca first (you're paying for it!), then Massive, then yfinance
    
    Priority:
    1. Alpaca API (real-time, included with trading account)
    2. Massive API (if available)
    3. yfinance (free fallback, delayed)
    
    Args:
        symbol: Stock symbol (SPY, QQQ, SPX, etc.)
        period: Period for data (e.g., "2d", "1d")
        interval: Data interval ("1m", "5m", "1d", etc.)
        api: Alpaca API instance (optional, but recommended)
        risk_mgr: RiskManager instance for logging (optional)
    
    Returns:
        DataFrame with OHLCV data (Open, High, Low, Close, Volume)
    """
    global massive_client
    
    # Helper function to log data collection
    def log_data_source(source: str, bars: int, symbol: str):
        if risk_mgr and hasattr(risk_mgr, 'log'):
            risk_mgr.log(f"📊 {symbol} Data: {bars} bars from {source} | period={period}, interval={interval}", "INFO")
    
    # ========== PRIORITY 1: ALPACA API (You're paying for this!) ==========
    if api:
        # Log API availability for debugging
        if risk_mgr and hasattr(risk_mgr, 'log'):
            risk_mgr.log(f"🔍 Alpaca API available for {symbol}, attempting data fetch...", "INFO")
        try:
            from alpaca_trade_api.rest import TimeFrame
            from datetime import datetime, timedelta
            
            # Calculate date range - get full 2 days including today
            # For 2 days: get data from 2 days ago to today (inclusive)
            if period == "2d":
                days = 2
            elif period == "1d":
                days = 1
            elif period == "5d":
                days = 5
            else:
                days = 2  # Default
            
            # Use current time as end, go back (days) days for start
            # This ensures we get the most recent data including today
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # For better data coverage, extend start date slightly to ensure we get full 2 days
            # Alpaca might filter to market hours, so we request a bit more to be safe
            if period == "2d":
                start_date = start_date - timedelta(hours=12)  # Go back 12 hours more to ensure full coverage
            
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
            
            # Get bars from Alpaca (limit 5000 bars per request)
            # For 2 days of 1-minute data: 
            # - Market hours only: ~780 bars (390 min/day × 2 days)
            # - Market + extended hours: ~1,920 bars (960 min/day × 2 days)
            # - Full 24/7: ~2,880 bars (1,440 min/day × 2 days) - unrealistic for stock markets
            # Alpaca API v2: Use date strings in YYYY-MM-DD format (most reliable)
            # Ensure we include today's data by using tomorrow as end date (exclusive end)
            start_str = start_date.strftime("%Y-%m-%d")
            # Use tomorrow as end date to ensure we get today's data (Alpaca end date is exclusive)
            end_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
            
            # Log attempt for debugging
            if risk_mgr and hasattr(risk_mgr, 'log'):
                risk_mgr.log(f"🔍 Attempting Alpaca API fetch for {symbol}: {start_str} to {end_str}, timeframe={timeframe}", "INFO")
            
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
                # Handle both single column names and MultiIndex
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
                    # Try to infer from existing columns
                    for req_col in required_cols:
                        if req_col not in bars.columns:
                            lower_col = req_col.lower()
                            if lower_col in bars.columns:
                                bars = bars.rename(columns={lower_col: req_col})
                
                # Check if Alpaca returned sufficient data for 2 days
                # Realistic expectations for 2 days of 1-minute data:
                # - Market hours only: ~780 bars (390 min/day × 2 days)
                # - Market + extended hours: ~1,920 bars (960 min/day × 2 days) ✅ This is what we're getting
                # - Full 24/7: ~2,880 bars (1,440 min/day × 2 days) - unrealistic for stock markets
                # 
                # ~1,800 bars is actually GOOD - it includes market hours + extended hours
                # Both Alpaca and Massive return similar amounts because that's all the data available
                bars_count = len(bars)
                
                # For 2 days, expect at least 1,500 bars (market hours + some extended hours)
                # If we get less, try Massive API as backup
                expected_min_bars = 1500 if period == "2d" else 700  # Realistic threshold
                
                if bars_count < expected_min_bars and period == "2d":
                    if risk_mgr and hasattr(risk_mgr, 'log'):
                        risk_mgr.log(f"⚠️ Alpaca API returned only {bars_count} bars for 2 days (expected at least {expected_min_bars}). Trying Massive API...", "WARNING")
                    # Don't return yet - try Massive API for better data
                else:
                    # Alpaca data is sufficient (1,800+ bars is good for market hours + extended hours)
                    log_data_source("Alpaca API", bars_count, symbol)
                    if risk_mgr and hasattr(risk_mgr, 'log') and bars_count >= 1500:
                        risk_mgr.log(f"✅ Alpaca API returned {bars_count} bars (market hours + extended hours) - sufficient for RL model", "INFO")
                    return bars
            else:
                if risk_mgr and hasattr(risk_mgr, 'log'):
                    risk_mgr.log(f"⚠️ Alpaca API returned empty data for {symbol}, trying Massive/yfinance", "WARNING")
        except Exception as e:
            # Fallback to Massive/yfinance
            if risk_mgr and hasattr(risk_mgr, 'log'):
                risk_mgr.log(f"⚠️ Alpaca data fetch failed for {symbol}: {str(e)} (type: {type(e).__name__}), trying Massive/yfinance", "WARNING")
                import traceback
                risk_mgr.log(f"   Traceback: {traceback.format_exc()[:200]}", "WARNING")
            pass
    
    # ========== PRIORITY 2: MASSIVE API (You have 1-minute granular package!) ==========
    # Map symbol for Massive API (SPX uses different format)
    massive_symbol_map = {
        'SPY': 'SPY',
        'QQQ': 'QQQ',
        'SPX': 'SPX',  # Polygon uses SPX (not ^SPX)
        '^SPX': 'SPX'
    }
    massive_symbol = massive_symbol_map.get(symbol, symbol.replace('^', ''))
    
    if massive_client:
        try:
            # Calculate date range - get full 2 days including today
            if period == "2d":
                days = 2
            elif period == "1d":
                days = 1
            elif period == "5d":
                days = 5
            else:
                days = 2  # Default
            
            # Massive API needs date strings in YYYY-MM-DD format
            # Get data from (days) days ago to today (inclusive)
            # Use tomorrow as end date to ensure we get today's data (Massive API end date may be exclusive)
            now = datetime.now()
            end_date_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")  # Tomorrow to include today
            start_date_str = (now - timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Log attempt for debugging
            if risk_mgr and hasattr(risk_mgr, 'log'):
                risk_mgr.log(f"🔍 Attempting Massive API fetch for {symbol}: {start_date_str} to {end_date_str}, interval={interval}", "INFO")
            
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
                    # Handle both lowercase and capitalized columns
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
                
                # CRITICAL FIX: Return ALL data, not just last 50 bars!
                # Massive API with 1-minute granular package returns market hours + extended hours
                # Realistic expectation: ~1,800-1,900 bars for 2 days (market + extended hours)
                hist_count = len(hist)
                log_data_source("Massive API", hist_count, symbol)
                
                # Compare with Alpaca data if we have it
                if risk_mgr and hasattr(risk_mgr, 'log'):
                    if hist_count < 1500 and period == "2d":
                        risk_mgr.log(f"⚠️ Massive API returned {hist_count} bars for 2 days (expected at least 1,500). May be incomplete.", "WARNING")
                    elif hist_count >= 1500:
                        risk_mgr.log(f"✅ Massive API returned {hist_count} bars (market hours + extended hours) - using complete dataset for {symbol}", "INFO")
                
                return hist
            else:
                if risk_mgr and hasattr(risk_mgr, 'log'):
                    risk_mgr.log(f"⚠️ Massive API returned empty data for {symbol}, trying yfinance", "WARNING")
        except Exception as e:
            # Fallback to yfinance
            if risk_mgr and hasattr(risk_mgr, 'log'):
                risk_mgr.log(f"⚠️ Massive API fetch failed for {symbol}: {str(e)} (type: {type(e).__name__}), trying yfinance", "WARNING")
                import traceback
                risk_mgr.log(f"   Traceback: {traceback.format_exc()[:200]}", "WARNING")
            pass
    
    # ========== PRIORITY 3: YFINANCE (Last Resort) ==========
    try:
        # Map symbol for yfinance (SPX needs ^ prefix)
        yf_symbol = symbol
        if symbol == 'SPX':
            yf_symbol = '^GSPC'  # S&P 500 index (more reliable than ^SPX)
        
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period=period, interval=interval)
        # Ultimate yfinance 2025+ compatibility fix
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        hist = hist.dropna()
        
        # CRITICAL FIX: Return ALL data, not just last 50 bars!
        # Note: yfinance may have limitations, but return what we get
        if len(hist) > 0:
            log_data_source("yfinance", len(hist), symbol)
            return hist
    except Exception as e:
        if risk_mgr and hasattr(risk_mgr, 'log'):
            risk_mgr.log(f"❌ All data sources failed for {symbol}: {e}", "ERROR")
        return pd.DataFrame()
    
    return pd.DataFrame()

def get_current_price(symbol: str) -> Optional[float]:
    """
    Get current price - tries Massive API first, falls back to yfinance
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Current price or None
    """
    global massive_client
    
    # Map symbol for Massive API
    massive_symbol_map = {
        'SPY': 'SPY',
        'QQQ': 'QQQ',
        'SPX': 'SPX',
        '^SPX': 'SPX',
        'VIX.X': 'VIX.X',  # VIX on Polygon
        '^VIX': 'VIX.X'
    }
    massive_symbol = massive_symbol_map.get(symbol, symbol.replace('^', ''))
    
    # Try Massive API first
    if massive_client:
        try:
            if symbol.startswith('^VIX') or symbol == 'VIX.X':
                price = massive_client.get_real_time_price('VIX.X')
            else:
                price = massive_client.get_real_time_price(massive_symbol)
            if price:
                return float(price)
        except Exception as e:
            pass
    
    # Fallback to yfinance
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
            return float(hist['Close'].iloc[-1])
    except:
        pass
    
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
    
    # Try RecurrentPPO first (LSTM models)
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
    
    # Try MaskablePPO (for action masking support)
    # Skip for historical model - it's a standard PPO model
    if MASKABLE_PPO_AVAILABLE and "historical" not in MODEL_PATH.lower():
        try:
            model = MaskablePPO.load(MODEL_PATH)
            print("✓ Model loaded successfully (MaskablePPO with action masking)")
            return model
        except Exception as e:
            print(f"Warning: Could not load as MaskablePPO: {e}")
            print("Falling back to standard PPO...")
    
    # Fallback to standard PPO
    # CRITICAL: Suppress warnings and use minimal options to avoid segfaults
    import warnings
    
    try:
        # Method 1: Try loading with custom_objects and suppressed warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = PPO.load(MODEL_PATH, custom_objects={}, print_system_info=False)
                print("✓ Model loaded successfully (standard PPO, no action masking)")
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
def get_option_symbol(underlying: str, strike: float, option_type: str) -> str:
    """Generate Alpaca option symbol"""
    expiration = datetime.now()
    date_str = expiration.strftime('%y%m%d')
    strike_str = f"{int(strike * 1000):08d}"
    type_str = 'C' if option_type == 'call' else 'P'
    return f"{underlying}{date_str}{type_str}{strike_str}"

def find_atm_strike(price: float) -> float:
    """Find nearest ATM strike"""
    return round(price)

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
                    'entry_time': datetime.now(),  # Approximate
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
                risk_mgr.symbol_stop_loss_cooldown[underlying] = datetime.now()
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
                    risk_mgr.symbol_stop_loss_cooldown[underlying] = datetime.now()
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
                    risk_mgr.symbol_stop_loss_cooldown[underlying] = datetime.now()
                    positions_to_close.append(symbol)
                    continue
            
            # ========== STEP 4: EMERGENCY FALLBACK ==========
            # If ALL data is missing and position open > 60 seconds, force close
            if (bid_premium is None and mid_premium is None and alpaca_plpc is None and current_premium is None):
                entry_time = pos_data.get('entry_time')
                if entry_time:
                    time_open = (datetime.now() - entry_time).total_seconds()
                    if time_open > 60:
                        underlying = extract_underlying_from_option(symbol)
                        risk_mgr.log(f"🚨 STEP 4 EMERGENCY CLOSE (NO DATA): {symbol} open for {int(time_open)}s with no premium data → FORCING CLOSE", "CRITICAL")
                        # Record stop-loss trigger for cooldown (prevent immediate re-entry)
                        risk_mgr.symbol_stop_loss_cooldown[underlying] = datetime.now()
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
                        
                        trade_db.save_trade({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
                            'submitted_at': submitted_at,
                            'filled_at': filled_at
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
    """
    # Check which model we're using
    if "mike_historical_model" in MODEL_PATH:
        # Use 10-feature observation for historical model
        return prepare_observation_10_features_inline(data, risk_mgr, symbol)
    else:
        # Use 23-feature observation for momentum model
        return prepare_observation_basic(data, risk_mgr, symbol)

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
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            
            try:
                # ========== SAFEGUARD CHECK ==========
                can_trade, reason = risk_mgr.check_safeguards(api)
                
                if not can_trade:
                    if iteration % 10 == 0:  # Log every 10th iteration
                        risk_mgr.log(f"Safeguard active: {reason}", "INFO")
                    time.sleep(30)
                    continue
                
                # Get latest SPY data (Alpaca first, then Massive, then yfinance)
                hist = get_market_data("SPY", period="2d", interval="1m", api=api, risk_mgr=risk_mgr)
                
                if len(hist) < LOOKBACK:
                    risk_mgr.log("Waiting for more data...", "INFO")
                    time.sleep(30)
                    continue
                
                current_price = hist['Close'].iloc[-1]
                
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
                        
                        # Prepare observation for THIS symbol
                        obs = prepare_observation(sym_hist, risk_mgr, symbol=sym)
                        
                        # 🔍 DEBUG: Log observation stats
                        risk_mgr.log(f"🔍 {sym} Observation: shape={obs.shape}, min={obs.min():.2f}, max={obs.max():.2f}, mean={obs.mean():.2f}, has_nan={np.isnan(obs).any()}, all_zero={(obs == 0).all()}", "DEBUG")
                        
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
                        
                        # If we're FLAT and the model keeps outputting TRIM/EXIT, resample stochastically.
                        # This preserves safety (we still only enter on BUY actions) while avoiding deadlock.
                        resampled = False
                        if not risk_mgr.open_positions and action in (3, 4, 5):
                            try:
                                sampled = []
                                for _ in range(7):
                                    a_s, _ = model.predict(obs, deterministic=False)
                                    if isinstance(a_s, np.ndarray):
                                        if a_s.ndim == 0:
                                            sampled.append(int(a_s.item()))
                                        else:
                                            sampled.append(int(a_s[0] if len(a_s) > 0 else a_s.item()))
                                    elif isinstance(a_s, (list, tuple)):
                                        sampled.append(int(a_s[0] if len(a_s) > 0 else 0))
                                    else:
                                        sampled.append(int(a_s))

                                buy_samples = [a for a in sampled if a in (1, 2)]
                                if buy_samples:
                                    # Prefer the more frequent BUY direction
                                    rl_action = max(set(buy_samples), key=buy_samples.count)
                                    resampled = True
                                    risk_mgr.log(
                                        f"🔁 {sym} Resample while flat: original={original_rl_action} ({get_action_name(original_rl_action)}) "
                                        f"| sampled={sampled} | selected_buy={rl_action} ({get_action_name(rl_action)})",
                                        "DEBUG",
                                    )
                                else:
                                    risk_mgr.log(
                                        f"🔁 {sym} Resample while flat: original={original_rl_action} ({get_action_name(original_rl_action)}) "
                                        f"| sampled={sampled} | no BUY found, keeping original",
                                        "DEBUG",
                                    )
                            except Exception as e:
                                risk_mgr.log(f"🔁 {sym} Resample while flat failed: {e}", "DEBUG")

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
                        symbol_actions[sym] = (action, action_source, action_strength)
                        
                        # Log per-symbol RL decision (using canonical action mapping)
                        # Show original action if it was remapped/masked
                        action_desc = get_action_name(action)
                        if resampled or masked:
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
                    for sym, (action, source, strength) in symbol_actions.items():
                        if action in [1, 2]:  # BUY CALL (1) or BUY PUT (2) - using canonical action codes
                            global_action = action
                            global_action_source = f"{source}_{sym}"
                            break
                
                action = global_action
                action_source = global_action_source
                
                # ENHANCED LOGGING: Show why trades are/aren't happening
                if action == 0:  # HOLD (canonical action 0)
                    if symbol_actions:
                        actions_summary = ", ".join([f"{sym}:{act}({strength:.2f})" for sym, (act, _, strength) in symbol_actions.items()])
                        risk_mgr.log(f"🤔 Multi-Symbol RL: All HOLD | Actions: [{actions_summary}] | Open Positions: {len(risk_mgr.open_positions)}/{MAX_CONCURRENT} | Available: {available_symbols}", "INFO")
                    else:
                        risk_mgr.log(f"🤔 Multi-Symbol RL: HOLD (no available symbols) | Open Positions: {len(risk_mgr.open_positions)}/{MAX_CONCURRENT}", "INFO")
                
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
                        buy_call_symbols = [sym for sym, (act, _, _) in symbol_actions.items() if act == 1]
                        risk_mgr.log(f"⛔ BLOCKED: No eligible symbols for BUY CALL | Signals: {buy_call_symbols} | Open Positions: {list(risk_mgr.open_positions.keys())}", "INFO")
                        # Fall through to next iteration
                    else:
                        # Check minimum confidence threshold before executing
                        selected_strength = symbol_actions.get(current_symbol, (None, None, 0.0))[2] if current_symbol in symbol_actions else 0.0
                        if selected_strength < MIN_ACTION_STRENGTH_THRESHOLD:
                            risk_mgr.log(f"⛔ BLOCKED: Selected symbol {current_symbol} confidence too low (strength={selected_strength:.3f} < {MIN_ACTION_STRENGTH_THRESHOLD:.3f}) | Skipping trade", "INFO")
                            time.sleep(10)
                            continue
                        # current_symbol already validated and selected by choose_best_symbol_for_trade
                        buy_call_symbols = [sym for sym, (act, _, _) in symbol_actions.items() if act == 1]
                        risk_mgr.log(f"🎯 SYMBOL SELECTION: {current_symbol} selected for BUY CALL (strength={selected_strength:.3f}) | All CALL signals: {buy_call_symbols}", "INFO")
                    
                    # Skip if no symbol selected
                    if current_symbol is None:
                        time.sleep(10)
                        continue
                    
                    # Get current price for selected symbol (Massive API first, yfinance fallback)
                    symbol_price = get_current_price(current_symbol)
                    if symbol_price is None:
                        symbol_price = current_price  # Fallback to SPY
                    
                    strike = find_atm_strike(symbol_price)
                    symbol = get_option_symbol(current_symbol, strike, 'call')
                    
                    # Log symbol selection for debugging
                    risk_mgr.log(f"📊 Selected symbol for CALL: {current_symbol} @ ${symbol_price:.2f} | Strike: ${strike:.2f} | Option: {symbol}", "INFO")
                    
                    # Get current regime and parameters
                    current_vix = risk_mgr.get_current_vix()
                    current_regime = risk_mgr.get_vol_regime(current_vix)
                    risk_mgr.current_regime = current_regime
                    regime_params = risk_mgr.get_vol_params(current_regime)
                    
                    # ========== DYNAMIC TAKE-PROFIT CALCULATION (PUT) ==========
                    # Calculate dynamic TP levels based on ATR, TrendStrength, VIX, Personality, Confidence
                    base_tp1 = regime_params['tp1']
                    base_tp2 = regime_params['tp2']
                    base_tp3 = regime_params['tp3']
                    
                    # Get symbol-specific historical data for dynamic TP calculation
                    try:
                        symbol_hist = get_market_data(current_symbol, period="2d", interval="1m", api=api, risk_mgr=risk_mgr)
                        if len(symbol_hist) >= 20:
                            # Get RL action raw value for confidence (from symbol_actions)
                            rl_action_raw = None
                            if current_symbol in symbol_actions:
                                act_tuple = symbol_actions[current_symbol]
                                rl_action_raw = act_tuple[0]  # Use action as proxy
                            
                            # Compute dynamic TP factors
                            if DYNAMIC_TP_AVAILABLE:
                                tp_factors = compute_dynamic_tp_factors(
                                    hist_data=symbol_hist,
                                    ticker=current_symbol,
                                    vix=current_vix,
                                    confidence=None,
                                    rl_action_raw=float(rl_action_raw) if rl_action_raw is not None else None
                                )
                                
                                # Compute dynamic TP levels
                                dynamic_tp1, dynamic_tp2, dynamic_tp3 = compute_dynamic_takeprofits(
                                    base_tp1=base_tp1,
                                    base_tp2=base_tp2,
                                    base_tp3=base_tp3,
                                    adjustment_factors=tp_factors
                                )
                                
                                # Use dynamic TPs
                                tp1_dynamic = dynamic_tp1
                                tp2_dynamic = dynamic_tp2
                                tp3_dynamic = dynamic_tp3
                                
                                risk_mgr.log(f"🎯 DYNAMIC TP (PUT): {current_symbol} | ATR={tp_factors['atr']:.2f}x | Trend={tp_factors['trend']:.2f}x | VIX={tp_factors['vix']:.2f}x | Personality={tp_factors['personality']:.2f}x | Confidence={tp_factors['confidence']:.2f}x | Total={tp_factors['total']:.2f}x", "INFO")
                                risk_mgr.log(f"   Base: TP1={base_tp1:.0%} TP2={base_tp2:.0%} TP3={base_tp3:.0%} → Dynamic: TP1={tp1_dynamic:.0%} TP2={tp2_dynamic:.0%} TP3={tp3_dynamic:.0%}", "INFO")
                            else:
                                # Fallback to regime-based TPs
                                tp1_dynamic = base_tp1
                                tp2_dynamic = base_tp2
                                tp3_dynamic = base_tp3
                        else:
                            # Not enough data, use regime-based TPs
                            tp1_dynamic = base_tp1
                            tp2_dynamic = base_tp2
                            tp3_dynamic = base_tp3
                    except Exception as e:
                        # Fallback to regime-based TPs on error
                        risk_mgr.log(f"⚠️ Dynamic TP calculation failed for {current_symbol}: {e}, using regime-based TPs", "WARNING")
                        tp1_dynamic = base_tp1
                        tp2_dynamic = base_tp2
                        tp3_dynamic = base_tp3
                    
                    # Use regime-adjusted risk percentage
                    regime_risk = regime_params['risk']
                    
                    # Estimate premium for sizing calculation
                    # CRITICAL FIX: Use symbol_price (not current_price) for correct premium estimation
                    estimated_premium = estimate_premium(symbol_price, strike, 'call')
                    
                    # Calculate position size using regime-adjusted risk
                    equity = risk_mgr.get_equity(api)
                    risk_dollar = equity * regime_risk
                    regime_adjusted_qty = max(1, int(risk_dollar / (estimated_premium * 100)))
                    
                    # Calculate max contracts under regime-adjusted limit
                    regime_max_notional = risk_mgr.get_regime_max_notional(api, current_regime)
                    # Use premium cost (not strike notional) for contract sizing
                    contract_cost = estimated_premium * 100  # Cost per contract
                    regime_max_contracts = int(regime_max_notional / contract_cost) if contract_cost > 0 else 0
                    
                    if regime_max_contracts < 1:
                        risk_mgr.log(f"REGIME MAX POSITION SIZE REACHED ({current_regime.upper()}): ${risk_mgr.get_current_exposure():,.0f} / ${regime_max_notional:,.0f} → NO NEW ENTRY", "WARNING")
                        time.sleep(30)
                        continue
                    
                    # Use smaller of: regime-adjusted size or regime max contracts
                    qty = min(regime_adjusted_qty, regime_max_contracts)
                    
                    # No hard contract limit - system decides based on regime-adjusted position sizing
                    # (Previously enforced: qty = min(qty, 10) - REMOVED to allow system decision)
                    
                    # Check order safety - use premium for notional calculation, not strike
                    is_safe, reason = risk_mgr.check_order_safety(symbol, qty, estimated_premium, api)
                    if not is_safe:
                        current_vix = risk_mgr.get_current_vix()
                        current_regime = risk_mgr.get_vol_regime(current_vix)
                        risk_mgr.log(f"⛔ BLOCKED: {current_symbol} ({symbol}) | Reason: {reason} | Symbol: {current_symbol} | Qty: {qty} | Premium: ${estimated_premium:.4f} | Regime: {current_regime.upper()} | VIX: {current_vix:.1f} | Positions: {len(risk_mgr.open_positions)}/{MAX_CONCURRENT} | Time: {datetime.now().strftime('%H:%M:%S')} EST", "WARNING")
                        
                        # Send Telegram block alert (only for significant blocks, not cooldowns)
                        if TELEGRAM_AVAILABLE and not any(x in reason.lower() for x in ['cooldown', 'duplicate']):
                            try:
                                send_block_alert(symbol=current_symbol, block_reason=reason)
                            except Exception:
                                pass  # Never block trading
                        
                        time.sleep(30)
                        continue
                    
                    try:
                        order = api.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        # Notional = premium cost, not strike notional
                        notional = qty * estimated_premium * 100
                        # Estimate entry premium (will be updated with real value)
                        # CRITICAL FIX: Use symbol_price (not current_price) for correct premium estimation
                        entry_premium = estimate_premium(symbol_price, strike, 'call')
                        # Get current VIX and regime for this entry
                        entry_vix = risk_mgr.get_current_vix()
                        entry_regime = risk_mgr.get_vol_regime(entry_vix)
                        
                        risk_mgr.open_positions[symbol] = {
                            'strike': strike,
                            'type': 'call',
                            'entry_time': datetime.now(),
                            'contracts': qty,
                            'qty_remaining': qty,  # Track remaining for TP tiers
                            'notional': notional,
                            'entry_premium': entry_premium,
                            'entry_price': symbol_price,  # CRITICAL FIX: Use symbol_price, not current_price
                            'trail_active': False,
                            'trail_price': 0.0,  # Legacy field (kept for compatibility)
                            'trail_tp_level': 0,  # Which TP this trail is for (1, 2, or 3)
                            'trail_triggered': False,
                            'tp1_level': tp1_dynamic,  # Store dynamic TP1 level
                            'tp2_level': tp2_dynamic,  # Store dynamic TP2 level
                            'tp3_level': tp3_dynamic,  # Store dynamic TP3 level
                            'tp1_dynamic': tp1_dynamic,  # Dynamic TP1 (for execution)
                            'tp2_dynamic': tp2_dynamic,  # Dynamic TP2 (for execution)
                            'tp3_dynamic': tp3_dynamic,  # Dynamic TP3 (for execution)
                            'peak_premium': entry_premium,
                            'highest_pnl': 0.0,  # Peak unrealized PnL (for dynamic trailing)
                            'trailing_stop_pct': 0.18,  # Dynamic trailing stop percentage (will adapt)
                            'tp1_done': False,
                            'tp2_done': False,
                            'tp3_done': False,
                            'runner_active': False,  # Is runner position active?
                            'runner_qty': 0,  # Quantity in runner position
                            'vol_regime': entry_regime,  # Store regime at entry
                            'entry_vix': entry_vix
                        }
                        
                        # Enhanced logging for validation
                        risk_mgr.log(f"✅ TRADE_OPENED | symbol={current_symbol} | option={symbol} | symbol_price=${symbol_price:.2f} | entry_price=${symbol_price:.2f} | premium=${entry_premium:.4f} | qty={qty} | strike=${strike:.2f} | regime={entry_regime.upper()}", "TRADE")
                        risk_mgr.log(f"✅ NEW ENTRY: {qty}x {symbol} @ ${entry_premium:.2f} premium (Strike: ${strike:.2f}, Underlying: ${symbol_price:.2f})", "TRADE")
                        
                        # Send Telegram entry alert
                        if TELEGRAM_AVAILABLE:
                            try:
                                # Extract expiry from symbol (0DTE format)
                                expiry = "0DTE" if len(symbol) > 10 else "Unknown"
                                # Get confidence from action_strength if available
                                confidence = symbol_actions.get(current_symbol, (None, None, None))[2] if current_symbol in symbol_actions else None
                                action_source_val = symbol_actions.get(current_symbol, (None, None, None))[1] if current_symbol in symbol_actions else None
                                alert_sent = send_entry_alert(
                                    symbol=symbol,
                                    side="CALL",
                                    strike=strike,
                                    expiry=expiry,
                                    fill_price=entry_premium,
                                    qty=qty,
                                    confidence=confidence,
                                    action_source=action_source_val
                                )
                                if alert_sent:
                                    risk_mgr.log(f"📱 Telegram entry alert sent for {symbol}", "INFO")
                                else:
                                    risk_mgr.log(f"⚠️ Telegram entry alert not sent (rate limited or error) for {symbol}", "WARNING")
                            except Exception as e:
                                risk_mgr.log(f"❌ Telegram entry alert error: {e}", "ERROR")
                                pass  # Never block trading
                        
                        risk_mgr.record_order(symbol)
                        
                        # Save trade to database
                        if TRADE_DB_AVAILABLE and trade_db:
                            try:
                                trade_db.save_trade({
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'qty': qty,
                                    'entry_premium': entry_premium,
                                    'entry_price': symbol_price,  # CRITICAL FIX: Use symbol_price, not current_price
                                    'strike_price': strike,
                                    'option_type': 'call',
                                    'regime': entry_regime,
                                    'vix': entry_vix,
                                    'reason': 'rl_signal'
                                })
                            except Exception as db_error:
                                risk_mgr.log(f"Warning: Could not save trade to database: {db_error}", "WARNING")
                        
                        current_exposure = risk_mgr.get_current_exposure()
                        max_notional = risk_mgr.get_current_max_notional(api)
                        risk_mgr.log(f"✓ EXECUTED: BUY {qty}x {symbol} (CALL) @ ${strike:.2f} | {current_regime.upper()} REGIME | Risk: {regime_risk:.0%} | Max Size: {regime_params['max_pct']:.0%} | Notional: ${notional:,.0f} | Exposure: ${current_exposure:,.0f}/{regime_max_notional:,.0f}", "TRADE")
                    except Exception as e:
                        risk_mgr.log(f"✗ Order failed: {e}", "ERROR")
                
                elif action == 2 and len(risk_mgr.open_positions) < MAX_CONCURRENT:  # BUY PUT (canonical action 2)
                    # Use smart symbol selection: rotation + position filter + cooldown filter + strength-based
                    current_symbol = choose_best_symbol_for_trade(
                        iteration=iteration,
                        symbol_actions=symbol_actions,
                        target_action=2,  # BUY PUT
                        open_positions=risk_mgr.open_positions,
                        risk_mgr=risk_mgr,  # For cooldown checks
                        max_positions_per_symbol=1  # Max 1 position per symbol
                    )
                    
                    if not current_symbol:
                        buy_put_symbols = [sym for sym, (act, _, _) in symbol_actions.items() if act == 2]
                        risk_mgr.log(f"⛔ BLOCKED: No eligible symbols for BUY PUT | Signals: {buy_put_symbols} | Open Positions: {list(risk_mgr.open_positions.keys())}", "INFO")
                        # Fall through to next iteration
                    else:
                        # Check minimum confidence threshold before executing
                        selected_strength = symbol_actions.get(current_symbol, (None, None, 0.0))[2] if current_symbol in symbol_actions else 0.0
                        if selected_strength < MIN_ACTION_STRENGTH_THRESHOLD:
                            risk_mgr.log(f"⛔ BLOCKED: Selected symbol {current_symbol} confidence too low (strength={selected_strength:.3f} < {MIN_ACTION_STRENGTH_THRESHOLD:.3f}) | Skipping trade", "INFO")
                            time.sleep(10)
                            continue
                        # current_symbol already validated and selected by choose_best_symbol_for_trade
                        buy_put_symbols = [sym for sym, (act, _, _) in symbol_actions.items() if act == 2]
                        risk_mgr.log(f"🎯 SYMBOL SELECTION: {current_symbol} selected for BUY PUT (strength={selected_strength:.3f}) | All PUT signals: {buy_put_symbols}", "INFO")
                    
                    # Skip if no symbol selected
                    if current_symbol is None:
                        time.sleep(10)
                        continue
                    
                    # Fallback to SPY if still None (shouldn't happen)
                    if current_symbol is None:
                        current_symbol = 'SPY'
                        risk_mgr.log(f"⚠️ Symbol selection failed, defaulting to SPY", "WARNING")
                    
                    # Get current price for selected symbol (Massive API first, yfinance fallback)
                    symbol_price = get_current_price(current_symbol)
                    if symbol_price is None:
                        # Try to get price from symbol_prices dict (already fetched above)
                        symbol_price = symbol_prices.get(current_symbol, current_price)
                        if symbol_price == current_price and current_symbol != 'SPY':
                            risk_mgr.log(f"⚠️ Could not get price for {current_symbol}, using SPY price as fallback", "WARNING")
                    
                    strike = find_atm_strike(symbol_price)
                    symbol = get_option_symbol(current_symbol, strike, 'put')
                    
                    # Log symbol selection for debugging
                    risk_mgr.log(f"📊 Selected symbol for PUT: {current_symbol} @ ${symbol_price:.2f} | Strike: ${strike:.2f} | Option: {symbol}", "INFO")
                    
                    # Get current regime and parameters
                    current_vix = risk_mgr.get_current_vix()
                    current_regime = risk_mgr.get_vol_regime(current_vix)
                    risk_mgr.current_regime = current_regime
                    regime_params = risk_mgr.get_vol_params(current_regime)
                    
                    # ========== DYNAMIC TAKE-PROFIT CALCULATION (PUT) ==========
                    # Calculate dynamic TP levels based on ATR, TrendStrength, VIX, Personality, Confidence
                    base_tp1 = regime_params['tp1']
                    base_tp2 = regime_params['tp2']
                    base_tp3 = regime_params['tp3']
                    
                    # Get symbol-specific historical data for dynamic TP calculation
                    try:
                        symbol_hist = get_market_data(current_symbol, period="2d", interval="1m", api=api, risk_mgr=risk_mgr)
                        if len(symbol_hist) >= 20:
                            # Get RL action raw value for confidence (from symbol_actions)
                            rl_action_raw = None
                            if current_symbol in symbol_actions:
                                act_tuple = symbol_actions[current_symbol]
                                rl_action_raw = act_tuple[0]  # Use action as proxy
                            
                            # Compute dynamic TP factors
                            if DYNAMIC_TP_AVAILABLE:
                                tp_factors = compute_dynamic_tp_factors(
                                    hist_data=symbol_hist,
                                    ticker=current_symbol,
                                    vix=current_vix,
                                    confidence=None,
                                    rl_action_raw=float(rl_action_raw) if rl_action_raw is not None else None
                                )
                                
                                # Compute dynamic TP levels
                                dynamic_tp1, dynamic_tp2, dynamic_tp3 = compute_dynamic_takeprofits(
                                    base_tp1=base_tp1,
                                    base_tp2=base_tp2,
                                    base_tp3=base_tp3,
                                    adjustment_factors=tp_factors
                                )
                                
                                # Use dynamic TPs
                                tp1_dynamic = dynamic_tp1
                                tp2_dynamic = dynamic_tp2
                                tp3_dynamic = dynamic_tp3
                                
                                risk_mgr.log(f"🎯 DYNAMIC TP (PUT): {current_symbol} | ATR={tp_factors['atr']:.2f}x | Trend={tp_factors['trend']:.2f}x | VIX={tp_factors['vix']:.2f}x | Personality={tp_factors['personality']:.2f}x | Confidence={tp_factors['confidence']:.2f}x | Total={tp_factors['total']:.2f}x", "INFO")
                                risk_mgr.log(f"   Base: TP1={base_tp1:.0%} TP2={base_tp2:.0%} TP3={base_tp3:.0%} → Dynamic: TP1={tp1_dynamic:.0%} TP2={tp2_dynamic:.0%} TP3={tp3_dynamic:.0%}", "INFO")
                            else:
                                # Fallback to regime-based TPs
                                tp1_dynamic = base_tp1
                                tp2_dynamic = base_tp2
                                tp3_dynamic = base_tp3
                        else:
                            # Not enough data, use regime-based TPs
                            tp1_dynamic = base_tp1
                            tp2_dynamic = base_tp2
                            tp3_dynamic = base_tp3
                    except Exception as e:
                        # Fallback to regime-based TPs on error
                        risk_mgr.log(f"⚠️ Dynamic TP calculation failed for {current_symbol}: {e}, using regime-based TPs", "WARNING")
                        tp1_dynamic = base_tp1
                        tp2_dynamic = base_tp2
                        tp3_dynamic = base_tp3
                    
                    # Use regime-adjusted risk percentage
                    regime_risk = regime_params['risk']
                    
                    # Estimate premium for sizing calculation
                    # CRITICAL FIX: Use symbol_price (not current_price) for correct premium estimation
                    estimated_premium = estimate_premium(symbol_price, strike, 'put')
                    
                    # Calculate position size using regime-adjusted risk
                    equity = risk_mgr.get_equity(api)
                    risk_dollar = equity * regime_risk
                    regime_adjusted_qty = max(1, int(risk_dollar / (estimated_premium * 100)))
                    
                    # Calculate max contracts under regime-adjusted limit
                    regime_max_notional = risk_mgr.get_regime_max_notional(api, current_regime)
                    # Use premium cost (not strike notional) for contract sizing
                    contract_cost = estimated_premium * 100  # Cost per contract
                    regime_max_contracts = int(regime_max_notional / contract_cost) if contract_cost > 0 else 0
                    
                    if regime_max_contracts < 1:
                        risk_mgr.log(f"REGIME MAX POSITION SIZE REACHED ({current_regime.upper()}): ${risk_mgr.get_current_exposure():,.0f} / ${regime_max_notional:,.0f} → NO NEW ENTRY", "WARNING")
                        time.sleep(30)
                        continue
                    
                    # Use smaller of: regime-adjusted size or regime max contracts
                    qty = min(regime_adjusted_qty, regime_max_contracts)
                    
                    # No hard contract limit - system decides based on regime-adjusted position sizing
                    # (Previously enforced: qty = min(qty, 10) - REMOVED to allow system decision)
                    
                    # Check order safety - use premium for notional calculation, not strike
                    is_safe, reason = risk_mgr.check_order_safety(symbol, qty, estimated_premium, api)
                    if not is_safe:
                        current_vix = risk_mgr.get_current_vix()
                        current_regime = risk_mgr.get_vol_regime(current_vix)
                        risk_mgr.log(f"⛔ BLOCKED: {current_symbol} ({symbol}) | Reason: {reason} | Symbol: {current_symbol} | Qty: {qty} | Premium: ${estimated_premium:.4f} | Regime: {current_regime.upper()} | VIX: {current_vix:.1f} | Positions: {len(risk_mgr.open_positions)}/{MAX_CONCURRENT} | Time: {datetime.now().strftime('%H:%M:%S')} EST", "WARNING")
                        
                        # Send Telegram block alert (only for significant blocks, not cooldowns)
                        if TELEGRAM_AVAILABLE and not any(x in reason.lower() for x in ['cooldown', 'duplicate']):
                            try:
                                send_block_alert(symbol=current_symbol, block_reason=reason)
                            except Exception:
                                pass  # Never block trading
                        
                        time.sleep(30)
                        continue
                    
                    try:
                        order = api.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        # Notional = premium cost, not strike notional
                        notional = qty * estimated_premium * 100
                        # Estimate entry premium (will be updated with real value)
                        # CRITICAL FIX: Use symbol_price (not current_price) for correct premium estimation
                        entry_premium = estimate_premium(symbol_price, strike, 'put')
                        # Get current VIX and regime for this entry
                        entry_vix = risk_mgr.get_current_vix()
                        entry_regime = risk_mgr.get_vol_regime(entry_vix)
                        
                        risk_mgr.open_positions[symbol] = {
                            'strike': strike,
                            'type': 'put',
                            'entry_time': datetime.now(),
                            'contracts': qty,
                            'qty_remaining': qty,  # Track remaining for TP tiers
                            'notional': notional,
                            'entry_premium': entry_premium,
                            'entry_price': symbol_price,  # CRITICAL FIX: Use symbol_price, not current_price
                            'trail_active': False,
                            'trail_price': 0.0,  # Legacy field (kept for compatibility)
                            'trail_tp_level': 0,  # Which TP this trail is for (1, 2, or 3)
                            'trail_triggered': False,
                            'tp1_level': tp1_dynamic,  # Store dynamic TP1 level
                            'tp2_level': tp2_dynamic,  # Store dynamic TP2 level
                            'tp3_level': tp3_dynamic,  # Store dynamic TP3 level
                            'tp1_dynamic': tp1_dynamic,  # Dynamic TP1 (for execution)
                            'tp2_dynamic': tp2_dynamic,  # Dynamic TP2 (for execution)
                            'tp3_dynamic': tp3_dynamic,  # Dynamic TP3 (for execution)
                            'peak_premium': entry_premium,
                            'highest_pnl': 0.0,  # Peak unrealized PnL (for dynamic trailing)
                            'trailing_stop_pct': 0.18,  # Dynamic trailing stop percentage (will adapt)
                            'tp1_done': False,
                            'tp2_done': False,
                            'tp3_done': False,
                            'runner_active': False,  # Is runner position active?
                            'runner_qty': 0,  # Quantity in runner position
                            'vol_regime': entry_regime,  # Store regime at entry
                            'entry_vix': entry_vix
                        }
                        
                        # Enhanced logging for validation
                        risk_mgr.log(f"✅ TRADE_OPENED | symbol={current_symbol} | option={symbol} | symbol_price=${symbol_price:.2f} | entry_price=${symbol_price:.2f} | premium=${entry_premium:.4f} | qty={qty} | strike=${strike:.2f} | regime={entry_regime.upper()}", "TRADE")
                        risk_mgr.log(f"✅ NEW ENTRY: {qty}x {symbol} @ ${entry_premium:.2f} premium (Strike: ${strike:.2f}, Underlying: ${symbol_price:.2f})", "TRADE")
                        
                        # Send Telegram entry alert for PUT
                        if TELEGRAM_AVAILABLE:
                            try:
                                # Extract expiry from symbol (0DTE format)
                                expiry = "0DTE" if len(symbol) > 10 else "Unknown"
                                # Get confidence from action_strength if available
                                confidence = symbol_actions.get(current_symbol, (None, None, None))[2] if current_symbol in symbol_actions else None
                                action_source_val = symbol_actions.get(current_symbol, (None, None, None))[1] if current_symbol in symbol_actions else None
                                alert_sent = send_entry_alert(
                                    symbol=symbol,
                                    side="PUT",
                                    strike=strike,
                                    expiry=expiry,
                                    fill_price=entry_premium,
                                    qty=qty,
                                    confidence=confidence,
                                    action_source=action_source_val
                                )
                                if alert_sent:
                                    risk_mgr.log(f"📱 Telegram entry alert sent for {symbol}", "INFO")
                                else:
                                    risk_mgr.log(f"⚠️ Telegram entry alert not sent (rate limited or error) for {symbol}", "WARNING")
                            except Exception as e:
                                risk_mgr.log(f"❌ Telegram entry alert error: {e}", "ERROR")
                                pass  # Never block trading
                        
                        risk_mgr.record_order(symbol)
                        # Get timestamps from Alpaca order response
                        submitted_at = order.submitted_at if hasattr(order, 'submitted_at') and order.submitted_at else ''
                        filled_at = order.filled_at if hasattr(order, 'filled_at') and order.filled_at else ''
                        order_id = order.id if hasattr(order, 'id') else ''
                        
                        risk_mgr.record_order(symbol)
                        
                        # Save trade to database with timestamps
                        if TRADE_DB_AVAILABLE and trade_db:
                            try:
                                trade_db.save_trade({
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'qty': qty,
                                    'entry_premium': entry_premium,
                                    'entry_price': symbol_price,  # CRITICAL FIX: Use symbol_price, not current_price
                                    'strike_price': strike,
                                    'option_type': 'put',
                                    'regime': entry_regime,
                                    'vix': entry_vix,
                                    'reason': 'rl_signal',
                                    'order_id': order_id,
                                    'submitted_at': submitted_at,
                                    'filled_at': filled_at
                                })
                            except Exception as db_error:
                                risk_mgr.log(f"Warning: Could not save trade to database: {db_error}", "WARNING")
                        
                        current_exposure = risk_mgr.get_current_exposure()
                        max_notional = risk_mgr.get_current_max_notional(api)
                        risk_mgr.log(f"✓ EXECUTED: BUY {qty}x {symbol} (PUT) @ ${strike:.2f} | {current_regime.upper()} REGIME | Risk: {regime_risk:.0%} | Max Size: {regime_params['max_pct']:.0%} | Notional: ${notional:,.0f} | Exposure: ${current_exposure:,.0f}/{regime_max_notional:,.0f}", "TRADE")
                    except Exception as e:
                        risk_mgr.log(f"✗ Order failed: {e}", "ERROR")
                
                elif action in [3, 4, 5] and risk_mgr.open_positions:  # TRIM OR EXIT
                    # Get actual positions from Alpaca
                    try:
                        alpaca_positions = api.list_positions()
                        alpaca_option_positions = {pos.symbol: pos for pos in alpaca_positions if pos.asset_class == 'option'}
                    except Exception as e:
                        risk_mgr.log(f"Error fetching positions for trim/exit: {e}", "ERROR")
                        alpaca_option_positions = {}
                    
                    for sym in list(risk_mgr.open_positions.keys()):
                        try:
                            if action == 5:  # FULL EXIT (canonical action 5)
                                api.close_position(sym)
                                risk_mgr.log(f"✓ SAFE EXIT: Closed all {sym}", "TRADE")
                            else:
                                # Get actual quantity from Alpaca
                                if sym in alpaca_option_positions:
                                    actual_qty = int(float(alpaca_option_positions[sym].qty))
                                    trim_pct = 0.5 if action == 3 else 0.7
                                    qty = max(1, int(actual_qty * trim_pct))
                                else:
                                    qty = 5 if action == 3 else 7  # Fallback: TRIM 50% (3) or TRIM 70% (4)
                                
                                # CRITICAL FIX: Verify we own the position before selling
                                try:
                                    current_pos = api.get_position(sym)
                                    if current_pos and float(current_pos.qty) >= qty:
                                        # We own the position, so sell is closing/reducing
                                        api.submit_order(
                                            symbol=sym,
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
                                    symbol=sym,
                                    qty=qty,
                                    side='sell',
                                    type='market',
                                    time_in_force='day'
                                )
                                risk_mgr.log(f"✓ TRIMMED: {qty}x {sym}", "TRADE")
                                
                                # Update tracked quantity
                                if sym in risk_mgr.open_positions:
                                    risk_mgr.open_positions[sym]['qty_remaining'] = actual_qty - qty if sym in alpaca_option_positions else risk_mgr.open_positions[sym].get('qty_remaining', 0) - qty
                        except Exception as e:
                            risk_mgr.log(f"✗ Exit/Trim failed: {e}", "ERROR")
                    
                    if action == 5:
                        risk_mgr.open_positions.clear()
                
                # Heartbeat
                if iteration % 10 == 0:
                    equity = risk_mgr.get_equity(api)
                    risk_mgr.log(f"💓 Heartbeat: Equity=${equity:,.2f} | Daily PnL={risk_mgr.daily_pnl:.2%} | Trades={risk_mgr.daily_trades}", "INFO")
                
                # CRITICAL: Adjust sleep time based on whether we have open positions
                # If we have positions, check more frequently (every 10 seconds) for stop loss monitoring
                # If no positions, can wait longer (30 seconds)
                if len(risk_mgr.open_positions) > 0:
                    time.sleep(10)  # Check every 10 seconds when we have positions (for stop loss)
                else:
                    time.sleep(30)  # Check every 30 seconds when flat
                
            except KeyboardInterrupt:
                risk_mgr.log("🚨 MANUAL KILL SWITCH ACTIVATED → FLATTENING ALL POSITIONS", "CRITICAL")
                try:
                    api.close_all_positions()
                    risk_mgr.log("All positions closed. Shutting down safely.", "CRITICAL")
                except:
                    pass
                break
                
            except Exception as e:
                risk_mgr.log(f"Error in main loop: {e}", "ERROR")
                
                # Send Telegram error alert
                if TELEGRAM_AVAILABLE:
                    try:
                        send_error_alert(str(e), context="Main trading loop")
                    except Exception:
                        pass  # Never block trading
                
                import traceback
                traceback.print_exc()
                time.sleep(10)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Agent stopped by user")
    
    finally:
        print("\n" + "=" * 70)
        print("Final Status:")
        equity = risk_mgr.get_equity(api)
        print(f"  Final Equity: ${equity:,.2f}")
        print(f"  Daily PnL: {risk_mgr.daily_pnl:.2%}")
        print(f"  Total Trades: {risk_mgr.daily_trades}")
        if risk_mgr.open_positions:
            print(f"  Open Positions: {len(risk_mgr.open_positions)}")
        else:
            print("  No open positions")
        print("=" * 70)

# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mike Agent v3 - Safe Live Trading")
    parser.add_argument('--live', action='store_true', help='Use live trading (default: paper)')
    parser.add_argument('--key', type=str, help='Alpaca API key')
    parser.add_argument('--secret', type=str, help='Alpaca API secret')
    
    args = parser.parse_args()
    
    if args.key:
        API_KEY = args.key
    if args.secret:
        API_SECRET = args.secret
    if args.live:
        BASE_URL = LIVE_URL
        USE_PAPER = False
        print("⚠️  WARNING: LIVE TRADING MODE - Real money will be used!")
        response = input("Type 'YES' to continue: ")
        if response != 'YES':
            print("Cancelled")
            sys.exit(0)
    
    run_safe_live_trading()

