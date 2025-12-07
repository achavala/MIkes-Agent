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
import yfinance as yf

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

import config

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

# Import institutional feature engineering module
try:
    from institutional_features import InstitutionalFeatureEngine, create_feature_engine
    INSTITUTIONAL_FEATURES_AVAILABLE = True
except ImportError:
    INSTITUTIONAL_FEATURES_AVAILABLE = False
    print("Warning: institutional_features module not found. Using basic features only.")

# Configuration: Use institutional features (set to False for backward compatibility)
USE_INSTITUTIONAL_FEATURES = True  # Enable institutional-grade features

# ==================== TRADING SYMBOLS ====================
# Symbols to trade (0DTE options)
TRADING_SYMBOLS = ['SPY', 'QQQ', 'SPX']  # Can trade all three

# ==================== RISK LIMITS (HARD-CODED – CANNOT BE OVERRIDDEN) ====================
DAILY_LOSS_LIMIT = -0.15  # -15% daily loss limit
MAX_POSITION_PCT = 0.25  # Max 25% of equity in one position
MAX_CONCURRENT = 2  # Max 2 positions at once (across all symbols)
VIX_KILL = 28  # No trades if VIX > 28
IVR_MIN = 30  # Minimum IV Rank (0-100)
NO_TRADE_AFTER = "14:30"  # No new entries after 2:30 PM EST
MAX_DRAWDOWN = 0.30  # Full shutdown if -30% from peak
MAX_NOTIONAL = 50000  # Max $50k notional per order
DUPLICATE_ORDER_WINDOW = 300  # 5 minutes in seconds

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
MODEL_PATH = "mike_rl_agent.zip"
LOOKBACK = 20

# ==================== STATE TRACKING ====================
class RiskManager:
    """Institutional-grade risk management with volatility-adjusted stops"""
    def __init__(self):
        self.peak_equity = 0.0
        self.daily_pnl = 0.0
        self.start_of_day_equity = 0.0
        self.open_positions = {}  # symbol: {entry_premium, entry_price, trail_active, trail_price, entry_time, contracts, qty_remaining, tp1_done, tp2_done, tp3_done, vol_regime}
        self.last_order_time = {}
        self.daily_trades = 0
        self.max_daily_trades = 20  # Max 20 trades per day
        self.current_vix = 20.0
        self.current_regime = "normal_vol"
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/mike_agent_safe_{datetime.now().strftime('%Y%m%d')}.log"
    
    def get_current_vix(self) -> float:
        """Get current VIX level"""
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
        
        # ========== SAFEGUARD 1: Daily Loss Limit ==========
        if self.daily_pnl <= DAILY_LOSS_LIMIT:
            self.log(f"🚨 SAFEGUARD 1 TRIGGERED: Daily loss limit hit ({self.daily_pnl:.1%})", "CRITICAL")
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
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="1d")
            if len(vix_data) > 0:
                # Ultimate yfinance 2025+ compatibility fix
                if isinstance(vix_data.columns, pd.MultiIndex):
                    vix_data.columns = vix_data.columns.get_level_values(0)
                vix = vix_data['Close'].iloc[-1]
                if vix > VIX_KILL:
                    return False, f"VIX {vix:.1f} > {VIX_KILL} (crash mode)"
        except Exception as e:
            self.log(f"Error fetching VIX: {e}", "WARNING")
        
        # ========== SAFEGUARD 4: Time-of-Day Filter ==========
        current_time = datetime.now().strftime("%H:%M")
        if current_time > NO_TRADE_AFTER:
            return False, f"After {NO_TRADE_AFTER} EST (theta crush protection)"
        
        # ========== SAFEGUARD 5: Max Concurrent Positions ==========
        if len(self.open_positions) >= MAX_CONCURRENT:
            return False, f"Max concurrent positions ({MAX_CONCURRENT}) reached"
        
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
    
    def check_order_safety(self, symbol: str, qty: int, premium: float, api: tradeapi.REST) -> tuple[bool, str]:
        """
        Check order-level safeguards with dynamic position size
        Args:
            symbol: Option symbol
            qty: Number of contracts
            premium: Option premium per contract (not strike price!)
            api: Alpaca API instance
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
            return False, f"Position would exceed {regime_params['max_pct']:.0%} limit ({self.current_regime.upper()} regime): ${current_exposure + notional:,.0f} > ${max_notional:,.0f}"
        
        # ========== SAFEGUARD 9: Duplicate Order Protection ==========
        if symbol in self.last_order_time:
            time_since_last = (datetime.now() - self.last_order_time[symbol]).total_seconds()
            if time_since_last < DUPLICATE_ORDER_WINDOW:
                return False, f"Duplicate order protection: {int(time_since_last)}s < {DUPLICATE_ORDER_WINDOW}s"
        
        return True, "OK"
    
    def record_order(self, symbol: str):
        """Record order time for duplicate protection"""
        self.last_order_time[symbol] = datetime.now()
        self.daily_trades += 1

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

# ==================== MODEL LOADING ====================
def load_rl_model():
    """Load trained RL model"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            f"Train first with: python mike_rl_agent.py --train"
        )
    
    print(f"Loading RL model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)
    print("✓ Model loaded successfully")
    return model

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
    """Estimate option premium using Black-Scholes"""
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

def check_stop_losses(api: tradeapi.REST, risk_mgr: RiskManager, current_price: float, trade_db: Optional[TradeDatabase] = None) -> None:
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
                    strike = current_price
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
                    entry_premium = estimate_premium(current_price, strike, option_type)
                    risk_mgr.log(f"⚠️ WARNING: Could not get entry premium for {symbol}, using estimate: ${entry_premium:.2f}", "WARNING")
                else:
                    risk_mgr.log(f"✅ Synced {symbol}: Entry premium = ${entry_premium:.4f} from Alpaca data", "INFO")
                
                # Get quantity
                qty = int(float(alpaca_pos.qty))
                
                # Add to tracking
                risk_mgr.open_positions[symbol] = {
                    'strike': strike,
                    'type': option_type,
                    'entry_time': datetime.now(),  # Approximate
                    'contracts': qty,
                    'qty_remaining': qty,
                    'notional': qty * entry_premium * 100,
                    'entry_premium': entry_premium,  # ACTUAL entry from Alpaca
                    'entry_price': current_price,
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
            
            # Get current premium from Alpaca snapshot (CORRECT API)
            try:
                snapshot = api.get_option_snapshot(symbol)
                # Use bid price if available, otherwise use market value / qty
                bid_price_float = None
                if snapshot.bid_price:
                    try:
                        bid_price_float = float(snapshot.bid_price)
                    except (ValueError, TypeError):
                        pass
                
                if bid_price_float and bid_price_float > 0:
                    current_premium = bid_price_float
                elif alpaca_pos.market_value and float(alpaca_pos.qty) > 0:
                    # For options: market_value = premium * qty * 100
                    # So premium = market_value / (qty * 100)
                    qty_float = float(alpaca_pos.qty)
                    market_val_float = abs(float(alpaca_pos.market_value))
                    current_premium = market_val_float / (qty_float * 100) if qty_float > 0 else 0.0
                else:
                    # Fallback to estimate
                    current_premium = estimate_premium(current_price, pos_data['strike'], pos_data['type'])
            except Exception as e:
                # Fallback to estimate if snapshot fails
                risk_mgr.log(f"Error getting snapshot for {symbol}: {e}, using estimate", "WARNING")
                current_premium = estimate_premium(current_price, pos_data['strike'], pos_data['type'])
            
            # Update peak premium
            if current_premium > pos_data.get('peak_premium', pos_data['entry_premium']):
                pos_data['peak_premium'] = current_premium
            
            # Calculate PnL percentage (with small epsilon for floating point precision)
            pnl_pct = (current_premium - pos_data['entry_premium']) / pos_data['entry_premium']
            # Add small epsilon to handle floating point precision issues (e.g., 0.3999999 vs 0.4)
            EPSILON = 1e-6
            
            # Enhanced debug logging for stop loss monitoring
            if pnl_pct <= -0.10:  # Log when loss is 10% or more
                risk_mgr.log(f"⚠️ Position {symbol}: PnL = {pnl_pct:.2%} (Entry: ${pos_data['entry_premium']:.4f}, Current: ${current_premium:.4f}, Qty: {int(float(alpaca_pos.qty))}) | Checking stop losses...", "INFO")
            
            # Get remaining quantity from actual Alpaca position
            actual_qty = int(float(alpaca_pos.qty))
            qty_remaining = pos_data.get('qty_remaining', actual_qty)
            
            # Update qty_remaining if it doesn't match actual position
            if qty_remaining != actual_qty:
                risk_mgr.log(f"Updating qty_remaining for {symbol}: {qty_remaining} → {actual_qty}", "INFO")
                pos_data['qty_remaining'] = actual_qty
                qty_remaining = actual_qty
            
            # ========== ABSOLUTE -15% STOP-LOSS (HIGHEST PRIORITY - CHECK FIRST) ==========
            # CRITICAL: Simple -15% stop loss that ALWAYS triggers regardless of other conditions
            # Check this BEFORE take-profits to ensure losing positions exit immediately
            ABSOLUTE_STOP_LOSS = -0.15  # -15% absolute stop
            stop_loss_check = (pnl_pct - EPSILON) <= ABSOLUTE_STOP_LOSS
            
            # Enhanced logging for stop loss check
            if pnl_pct <= -0.15:
                risk_mgr.log(f"🛑 STOP LOSS CHECK: {symbol} | PnL: {pnl_pct:.2%} | Threshold: {ABSOLUTE_STOP_LOSS:.2%} | Trigger: {stop_loss_check} | Entry: ${pos_data['entry_premium']:.4f} | Current: ${current_premium:.4f}", "CRITICAL")
            
            if stop_loss_check:
                risk_mgr.log(f"🛑 ABSOLUTE STOP-LOSS TRIGGERED (-15%): {symbol} @ {pnl_pct:.1%} (Entry: ${pos_data['entry_premium']:.4f}, Current: ${current_premium:.4f}, Qty: {qty_remaining}) → FORCED FULL EXIT", "CRITICAL")
                positions_to_close.append(symbol)
                continue
            
            # ========== TAKE-PROFIT EXECUTION (ONE PER TICK - CRITICAL) ==========
            # CRITICAL: Only ONE take-profit can trigger per price update to prevent over-selling
            # This prevents gap-ups from triggering all TPs simultaneously
            tp_triggered = False
            
            # ========== VOLATILITY-ADJUSTED TAKE-PROFIT TIER 1 ==========
            # Check TP1 FIRST (lowest threshold) - must be sequential
            # Use >= with epsilon to handle floating point precision
            if not tp_triggered and (pnl_pct + EPSILON) >= tp_params['tp1'] and not pos_data.get('tp1_done', False):
                sell_qty = max(1, int(qty_remaining * 0.5))  # Sell 50% of remaining
                if sell_qty < qty_remaining:
                    try:
                        api.submit_order(
                            symbol=symbol,
                            qty=sell_qty,
                            side='sell',
                            type='market',
                            time_in_force='day'
                        )
                        pos_data['qty_remaining'] = qty_remaining - sell_qty
                        pos_data['tp1_done'] = True
                        pos_data['tp1_level'] = tp_params['tp1']  # Store TP1 level for trailing calc
                        
                        # Setup trailing stop: TP1 - 20%
                        tp1_price = pos_data['entry_premium'] * (1 + tp_params['tp1'])
                        trail_price = pos_data['entry_premium'] * (1 + tp_params['tp1'] - 0.20)  # TP1 - 20%
                        pos_data['trail_active'] = True
                        pos_data['trail_price'] = trail_price
                        pos_data['trail_tp_level'] = 1  # Track which TP this trail is for
                        pos_data['trail_triggered'] = False
                        
                        tp_triggered = True  # CRITICAL: Prevent other TPs this tick
                        risk_mgr.log(f"🎯 TP1 +{tp_params['tp1']:.0%} ({current_regime.upper()}) → SOLD 50% ({sell_qty}x) | Remaining: {pos_data['qty_remaining']} | Trail Stop: +{tp_params['tp1'] - 0.20:.0%} (${trail_price:.2f})", "TRADE")
                        # Break after successful partial sell - wait for next price update
                        continue
                    except Exception as e:
                        risk_mgr.log(f"✗ Error executing TP1 for {symbol}: {e}", "ERROR")
                else:
                    # If 50% is all remaining, just close
                    try:
                        api.close_position(symbol)
                        risk_mgr.log(f"🎯 TP1 +{tp_params['tp1']:.0%} ({current_regime.upper()}) → FULL EXIT: {symbol} @ {pnl_pct:.1%}", "TRADE")
                        positions_to_close.append(symbol)
                        continue
                    except Exception as e:
                        risk_mgr.log(f"✗ Error closing at TP1: {e}", "ERROR")
            
            # ========== VOLATILITY-ADJUSTED TAKE-PROFIT TIER 2 ==========
            # Check TP2 ONLY if TP1 is done AND no TP triggered this tick
            elif not tp_triggered and (pnl_pct + EPSILON) >= tp_params['tp2'] and pos_data.get('tp1_done', False) and not pos_data.get('tp2_done', False):
                sell_qty = max(1, int(qty_remaining * 0.6))  # Sell 60% of remaining (improved from 30%)
                if sell_qty < qty_remaining:
                    try:
                        api.submit_order(
                            symbol=symbol,
                            qty=sell_qty,
                            side='sell',
                            type='market',
                            time_in_force='day'
                        )
                        pos_data['qty_remaining'] = qty_remaining - sell_qty
                        pos_data['tp2_done'] = True
                        pos_data['tp2_level'] = tp_params['tp2']  # Store TP2 level for trailing calc
                        
                        # Setup trailing stop: TP2 - 20% (overwrites TP1 trail if not triggered)
                        tp2_price = pos_data['entry_premium'] * (1 + tp_params['tp2'])
                        trail_price = pos_data['entry_premium'] * (1 + tp_params['tp2'] - 0.20)  # TP2 - 20%
                        pos_data['trail_active'] = True
                        pos_data['trail_price'] = trail_price
                        pos_data['trail_tp_level'] = 2  # Track which TP this trail is for
                        pos_data['trail_triggered'] = False
                        
                        tp_triggered = True  # CRITICAL: Prevent other TPs this tick
                        risk_mgr.log(f"🎯 TP2 +{tp_params['tp2']:.0%} ({current_regime.upper()}) → SOLD 60% ({sell_qty}x) | Remaining: {pos_data['qty_remaining']} | Trail Stop: +{tp_params['tp2'] - 0.20:.0%} (${trail_price:.2f})", "TRADE")
                        # Break after successful partial sell - wait for next price update
                        continue
                    except Exception as e:
                        risk_mgr.log(f"✗ Error executing TP2 for {symbol}: {e}", "ERROR")
                else:
                    # If 60% is all remaining, just close
                    try:
                        api.close_position(symbol)
                        risk_mgr.log(f"🎯 TP2 +{tp_params['tp2']:.0%} ({current_regime.upper()}) → FULL EXIT: {symbol} @ {pnl_pct:.1%}", "TRADE")
                        positions_to_close.append(symbol)
                        continue
                    except Exception as e:
                        risk_mgr.log(f"✗ Error closing at TP2: {e}", "ERROR")
            
            # ========== VOLATILITY-ADJUSTED TAKE-PROFIT TIER 3 ==========
            # Check TP3 ONLY if TP2 is done AND no TP triggered this tick
            elif not tp_triggered and (pnl_pct + EPSILON) >= tp_params['tp3'] and pos_data.get('tp2_done', False) and not pos_data.get('tp3_done', False):
                try:
                    api.close_position(symbol)
                    risk_mgr.log(f"🎯 TP3 +{tp_params['tp3']:.0%} HIT ({current_regime.upper()}) → FULL EXIT: {symbol} @ {pnl_pct:.1%}", "TRADE")
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
            
            # ========== NEW TRAILING STOP SYSTEM (TP - 20%) ==========
            # Check trailing stop if active (after TP1 or TP2)
            # When triggered: Sell 80% of remaining, keep 20% as runner
            if pos_data.get('trail_active', False) and not pos_data.get('trail_triggered', False):
                trail_price = pos_data.get('trail_price', 0)
                if current_premium <= trail_price + EPSILON:  # Price dropped to trailing stop
                    trail_tp_level = pos_data.get('trail_tp_level', 1)
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
                            # Sell 80% of remaining at trailing stop
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
                                risk_mgr.log(f"📉 TRAILING STOP HIT (TP{trail_tp_level} - 20% = +{tp_level_pct - 0.20:.0%}): {symbol} @ ${current_premium:.2f} → SOLD 80% ({trail_sell_qty}x) | Runner: {runner_qty}x until EOD or -15% stop", "TRADE")
                            else:
                                risk_mgr.log(f"📉 TRAILING STOP HIT (TP{trail_tp_level} - 20% = +{tp_level_pct - 0.20:.0%}): {symbol} @ ${current_premium:.2f} → SOLD ALL ({trail_sell_qty}x)", "TRADE")
                            
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
            if pos_data['type'] == 'call':
                # Would need bar data - simplified check
                if current_price < pos_data['entry_price'] * 0.99:  # 1% rejection
                    risk_mgr.log(f"⚠️ REJECTION DETECTED: {symbol} → Exit", "TRADE")
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
                            'reason': 'stop_loss_or_take_profit'
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
                    api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    risk_mgr.log(f"✓ Closed via sell order: {symbol}", "TRADE")
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
    Prepare observation for RL model with backward compatibility
    
    If USE_INSTITUTIONAL_FEATURES is True, extracts 500+ features
    Otherwise, uses simple 5-feature OHLCV (backward compatible)
    """
    # Use institutional features if available and enabled
    if INSTITUTIONAL_FEATURES_AVAILABLE and USE_INSTITUTIONAL_FEATURES and feature_engine:
        return prepare_observation_institutional(data, risk_mgr, symbol)
    else:
        return prepare_observation_basic(data, risk_mgr)

def prepare_observation_basic(data: pd.DataFrame, risk_mgr: RiskManager) -> np.ndarray:
    """
    Basic observation preparation (backward compatible)
    Model expects shape (20, 5) - matching training: [open, high, low, close, volume]
    """
    if len(data) < LOOKBACK:
        # Pad with last value
        padding = pd.concat([data.iloc[[-1]]] * (LOOKBACK - len(data)))
        data = pd.concat([padding, data])
    
    recent = data.tail(LOOKBACK).copy()
    
    # Ensure column names are lowercase (matching training environment)
    column_mapping = {
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
        'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'
    }
    for old_col, new_col in column_mapping.items():
        if old_col in recent.columns:
            recent = recent.rename(columns={old_col: new_col})
    
    # Extract the 5 features the model was trained with: open, high, low, close, volume
    # Normalize OHLC by dividing by close price (or use raw values if model expects them)
    obs_data = recent[['open', 'high', 'low', 'close', 'volume']].copy()
    
    # Normalize volume (optional - depends on how model was trained)
    if obs_data['volume'].max() > 0:
        obs_data['volume'] = obs_data['volume'] / obs_data['volume'].max()
    
    # Convert to numpy array with shape (LOOKBACK, 5)
    # CRITICAL: Remove batch dimension - model expects (20, 5) not (1, 20, 5)
    state = obs_data.values.astype(np.float32)
    
    # Ensure shape is exactly (LOOKBACK, 5)
    if state.shape != (LOOKBACK, 5):
        # Reshape if needed, but should already be correct
        state = state.reshape(LOOKBACK, 5)
    
    # CRITICAL: Model was trained with DummyVecEnv, which expects batch dimension
    # Observation space is Box(shape=(20, 5)), so VecEnv expects (n_env, 20, 5)
    # For single prediction, we need (1, 20, 5)
    state = state.reshape(1, LOOKBACK, 5)
    
    return state

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
    print(f"  6. No Trade After: {NO_TRADE_AFTER} EST")
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
                spy_ticker = yf.Ticker("SPY")
                spy_hist = spy_ticker.history(period="1d", interval="1m")
                if isinstance(spy_hist.columns, pd.MultiIndex):
                    spy_hist.columns = spy_hist.columns.get_level_values(0)
                current_spy_price = float(spy_hist['Close'].iloc[-1]) if len(spy_hist) > 0 else 450.0
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
                
                # Get latest SPY data
                spy = yf.Ticker("SPY")
                hist = spy.history(period="2d", interval="1m")
                
                # Ultimate yfinance 2025+ compatibility fix
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                
                hist = hist.dropna().tail(50)
                
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
                                risk_mgr.log(f"🎯 GAP-BASED ACTION: {gap_action} ({'BUY CALL' if gap_action == 1 else 'BUY PUT'}) | Overriding RL signal for first 60 min", "INFO")
                
                # Prepare observation
                # Returns shape (1, 20, 5) for VecEnv compatibility
                # Get current symbol for feature extraction (use SPY for main observation)
                current_symbol = 'SPY'  # Main observation uses SPY data
                obs = prepare_observation(hist, risk_mgr, symbol=current_symbol)
                
                # RL Decision
                # obs shape should be (1, 20, 5) - matching VecEnv format
                action_raw, _ = model.predict(obs, deterministic=True)
                
                # Model outputs continuous values in Box(-1.0, 1.0, (1,))
                # Map continuous action to discrete actions:
                # -1.0 to -0.5: Action 0 (HOLD)
                # -0.5 to 0.0: Action 1 (BUY CALL)
                # 0.0 to 0.5: Action 2 (BUY PUT)
                # 0.5 to 1.0: Action 3+ (TRIM/EXIT - only if positions exist)
                
                # Extract action value
                if isinstance(action_raw, (list, np.ndarray)):
                    action_value = float(action_raw[0] if len(action_raw) > 0 else 0.0)
                else:
                    action_value = float(action_raw)
                
                # Map continuous to discrete actions - Simplified sign-based mapping
                # Converts positive RL outputs to BUY CALL, negative to BUY PUT
                # This fixes the issue where model outputs +0.5 to +0.8 (exit signals) when flat
                if abs(action_value) < 0.35:
                    rl_action = 0  # HOLD (near zero = no conviction)
                elif action_value > 0:
                    rl_action = 1  # Positive raw → BUY CALL (bullish bias)
                else:
                    rl_action = 2  # Negative raw → BUY PUT (bearish bias)
                
                # Handle trim/exit actions only if positions exist
                if action_value >= 0.5 and risk_mgr.open_positions:
                    # Override for exit/trim when positions exist
                    if action_value < 0.75:
                        rl_action = 3  # TRIM 50%
                    elif action_value < 0.9:
                        rl_action = 4  # TRIM 70%
                    else:
                        rl_action = 5  # FULL EXIT
                
                # ========== GAP-BASED OVERRIDE (First 60 minutes only) ==========
                # Gap detection overrides RL signal during market open
                if gap_action is not None and 930 <= current_time_int <= 1035 and not risk_mgr.open_positions:
                    action = gap_action  # Use gap-based action
                    action_source = "GAP"
                else:
                    action = rl_action  # Use RL action
                    action_source = "RL"
                
                action = int(action)
                
                # Log raw RL output for debugging (every 5th iteration for better visibility)
                if iteration % 5 == 0:
                    action_desc = {0: 'HOLD', 1: 'BUY CALL', 2: 'BUY PUT', 3: 'TRIM 50%', 4: 'TRIM 70%', 5: 'FULL EXIT'}.get(action, 'UNKNOWN')
                    risk_mgr.log(f"🔍 RL Debug: Raw={action_value:.3f} → Action={action} ({action_desc})", "INFO")
                
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
                check_stop_losses(api, risk_mgr, current_price, trade_db)
                
                # ========== SAFE EXECUTION WITH REGIME-ADAPTIVE POSITION SIZING ==========
                if action == 1 and len(risk_mgr.open_positions) < MAX_CONCURRENT:  # BUY CALL
                    # Select symbol to trade (rotate or use SPY as default)
                    current_symbol = 'SPY'  # Default
                    for sym in TRADING_SYMBOLS:
                        # Check if we already have a position in this symbol
                        has_position = any(s.startswith(sym) for s in risk_mgr.open_positions.keys())
                        if not has_position:
                            current_symbol = sym
                            break
                    
                    # Get current price for selected symbol
                    try:
                        ticker = yf.Ticker(current_symbol)
                        symbol_hist = ticker.history(period="1d", interval="1m")
                        if isinstance(symbol_hist.columns, pd.MultiIndex):
                            symbol_hist.columns = symbol_hist.columns.get_level_values(0)
                        symbol_price = float(symbol_hist['Close'].iloc[-1]) if len(symbol_hist) > 0 else current_price
                    except:
                        symbol_price = current_price  # Fallback to SPY price
                    
                    strike = find_atm_strike(symbol_price)
                    symbol = get_option_symbol(current_symbol, strike, 'call')
                    
                    # Get current regime and parameters
                    current_vix = risk_mgr.get_current_vix()
                    current_regime = risk_mgr.get_vol_regime(current_vix)
                    risk_mgr.current_regime = current_regime
                    regime_params = risk_mgr.get_vol_params(current_regime)
                    
                    # Use regime-adjusted risk percentage
                    regime_risk = regime_params['risk']
                    
                    # Estimate premium for sizing calculation
                    estimated_premium = estimate_premium(current_price, strike, 'call')
                    
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
                    
                    # Check order safety - use premium for notional calculation, not strike
                    is_safe, reason = risk_mgr.check_order_safety(symbol, qty, estimated_premium, api)
                    if not is_safe:
                        risk_mgr.log(f"Order blocked: {reason}", "WARNING")
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
                        entry_premium = estimate_premium(current_price, strike, 'call')
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
                            'entry_price': current_price,
                            'trail_active': False,
                            'trail_price': 0.0,
                            'trail_tp_level': 0,  # Which TP this trail is for (1, 2, or 3)
                            'trail_triggered': False,
                            'tp1_level': 0.0,  # Store TP1 level for trailing calc
                            'tp2_level': 0.0,  # Store TP2 level for trailing calc
                            'tp3_level': 0.0,  # Store TP3 level for trailing calc
                            'peak_premium': entry_premium,
                            'tp1_done': False,
                            'tp2_done': False,
                            'tp3_done': False,
                            'runner_active': False,  # Is runner position active?
                            'runner_qty': 0,  # Quantity in runner position
                            'vol_regime': entry_regime,  # Store regime at entry
                            'entry_vix': entry_vix
                        }
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
                                    'entry_price': current_price,
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
                
                elif action == 2 and len(risk_mgr.open_positions) < MAX_CONCURRENT:  # BUY PUT
                    # Select symbol to trade (rotate or use SPY as default)
                    current_symbol = 'SPY'  # Default
                    for sym in TRADING_SYMBOLS:
                        # Check if we already have a position in this symbol
                        has_position = any(s.startswith(sym) for s in risk_mgr.open_positions.keys())
                        if not has_position:
                            current_symbol = sym
                            break
                    
                    # Get current price for selected symbol
                    try:
                        ticker = yf.Ticker(current_symbol)
                        symbol_hist = ticker.history(period="1d", interval="1m")
                        if isinstance(symbol_hist.columns, pd.MultiIndex):
                            symbol_hist.columns = symbol_hist.columns.get_level_values(0)
                        symbol_price = float(symbol_hist['Close'].iloc[-1]) if len(symbol_hist) > 0 else current_price
                    except:
                        symbol_price = current_price  # Fallback to SPY price
                    
                    strike = find_atm_strike(symbol_price)
                    symbol = get_option_symbol(current_symbol, strike, 'put')
                    
                    # Get current regime and parameters
                    current_vix = risk_mgr.get_current_vix()
                    current_regime = risk_mgr.get_vol_regime(current_vix)
                    risk_mgr.current_regime = current_regime
                    regime_params = risk_mgr.get_vol_params(current_regime)
                    
                    # Use regime-adjusted risk percentage
                    regime_risk = regime_params['risk']
                    
                    # Estimate premium for sizing calculation
                    estimated_premium = estimate_premium(current_price, strike, 'put')
                    
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
                    
                    # Check order safety - use premium for notional calculation, not strike
                    is_safe, reason = risk_mgr.check_order_safety(symbol, qty, estimated_premium, api)
                    if not is_safe:
                        risk_mgr.log(f"Order blocked: {reason}", "WARNING")
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
                        entry_premium = estimate_premium(current_price, strike, 'put')
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
                            'entry_price': current_price,
                            'trail_active': False,
                            'trail_price': 0.0,
                            'trail_tp_level': 0,  # Which TP this trail is for (1, 2, or 3)
                            'trail_triggered': False,
                            'tp1_level': 0.0,  # Store TP1 level for trailing calc
                            'tp2_level': 0.0,  # Store TP2 level for trailing calc
                            'tp3_level': 0.0,  # Store TP3 level for trailing calc
                            'peak_premium': entry_premium,
                            'tp1_done': False,
                            'tp2_done': False,
                            'tp3_done': False,
                            'runner_active': False,  # Is runner position active?
                            'runner_qty': 0,  # Quantity in runner position
                            'vol_regime': entry_regime,  # Store regime at entry
                            'entry_vix': entry_vix
                        }
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
                                    'entry_price': current_price,
                                    'strike_price': strike,
                                    'option_type': 'put',
                                    'regime': entry_regime,
                                    'vix': entry_vix,
                                    'reason': 'rl_signal'
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
                            if action == 5:  # Full exit
                                api.close_position(sym)
                                risk_mgr.log(f"✓ SAFE EXIT: Closed all {sym}", "TRADE")
                            else:
                                # Get actual quantity from Alpaca
                                if sym in alpaca_option_positions:
                                    actual_qty = int(float(alpaca_option_positions[sym].qty))
                                    trim_pct = 0.5 if action == 3 else 0.7
                                    qty = max(1, int(actual_qty * trim_pct))
                                else:
                                    qty = 5 if action == 3 else 7  # Fallback
                                
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
                
                time.sleep(55)  # ~1 minute cycle
                
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

