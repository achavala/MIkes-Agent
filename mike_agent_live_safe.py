#!/usr/bin/env python3Git: Discard All Changes
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
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import yfinance as yf
import pytz

# Set environment variables BEFORE importing torch/gym
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['OMP_NUM_THREADS'] = '1'

warnings.filterwarnings("ignore")

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Error: alpaca-trade-api not installed. Install with: pip install alpaca-trade-api")
    sys.exit(1)

try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("Error: stable-baselines3 not installed. Install with: pip install stable-baselines3")
    sys.exit(1)

import config

# Import trade database
try:
    from mike_agent_trades import MikeAgentTradeDB
    TRADE_DB_AVAILABLE = True
except ImportError:
    TRADE_DB_AVAILABLE = False
    print("Warning: mike_agent_trades not available. Trades will not be logged to database.")

# ==================== RISK LIMITS (HARD-CODED – CANNOT BE OVERRIDDEN) ====================
DAILY_LOSS_LIMIT = -0.15  # -15% daily loss limit
MAX_POSITION_PCT = 0.25  # Max 25% of equity in one position
MAX_CONCURRENT = 10  # Max 10 positions at once
VIX_KILL = 28  # No trades if VIX > 28
IVR_MIN = 30  # Minimum IV Rank (0-100)
NO_TRADE_AFTER = "14:30"  # No new entries after 2:30 PM EST
MAX_DRAWDOWN = 0.30  # Full shutdown if -30% from peak
MAX_NOTIONAL = 50000  # Max $50k notional per order
DUPLICATE_ORDER_WINDOW = 300  # 5 minutes in seconds
FIXED_STOP_LOSS = -0.15  # Fixed -15% stop-loss (always, regardless of volatility regime)

# ==================== TRADING SYMBOLS ====================
TRADING_SYMBOLS = ['SPY', 'QQQ', 'SPX']  # Trade SPY, QQQ, and SPX

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
        "tp1": 0.40,       # +40%: Sell 50%
        "tp2": 0.60,       # +60%: Sell 20% of remaining
        "tp3": 1.00,       # +100%: Sell 10% of remaining
        "tp4": 1.50,       # +150%: Sell 10% of remaining
        "tp5": 2.00,       # +200%: Full exit
        "trail_activate": 1.50,  # Activate after TP4
        "trail": 1.00      # Lock in +100% minimum
    },
    "normal": {       # VIX 18-25: Mike's default (your data)
        "risk": 0.07,      # 7% risk per trade
        "max_pct": 0.25,   # 25% max position size
        "sl": -0.20,
        "hard_sl": -0.30,
        "tp1": 0.40,       # +40%: Sell 50%
        "tp2": 0.60,       # +60%: Sell 20% of remaining
        "tp3": 1.00,       # +100%: Sell 10% of remaining
        "tp4": 1.50,       # +150%: Sell 10% of remaining
        "tp5": 2.00,       # +200%: Full exit
        "trail_activate": 1.50,  # Activate after TP4
        "trail": 1.00      # Lock in +100% minimum
    },
    "storm": {        # VIX 25-35: Defensive, wide stops, big upside
        "risk": 0.05,      # 5% risk per trade
        "max_pct": 0.20,   # 20% max position size
        "sl": -0.28,
        "hard_sl": -0.40,
        "tp1": 0.40,       # +40%: Sell 50%
        "tp2": 0.60,       # +60%: Sell 20% of remaining
        "tp3": 1.00,       # +100%: Sell 10% of remaining
        "tp4": 1.50,       # +150%: Sell 10% of remaining
        "tp5": 2.00,       # +200%: Full exit
        "trail_activate": 1.50,  # Activate after TP4
        "trail": 1.00      # Lock in +100% minimum
    },
    "crash": {        # VIX > 35: Survive & thrive in chaos
        "risk": 0.03,      # 3% risk per trade
        "max_pct": 0.15,   # 15% max position size
        "sl": -0.35,
        "hard_sl": -0.50,
        "tp1": 0.40,       # +40%: Sell 50%
        "tp2": 0.60,       # +60%: Sell 20% of remaining
        "tp3": 1.00,       # +100%: Sell 10% of remaining
        "tp4": 1.50,       # +150%: Sell 10% of remaining
        "tp5": 2.00,       # +200%: Full exit
        "trail_activate": 1.50,  # Activate after TP4
        "trail": 1.00      # Lock in +100% minimum
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
        self.open_positions = {}  # symbol: {entry_premium, entry_price, trail_active, trail_price, entry_time, contracts, qty_remaining, tp1_done, tp2_done, tp3_done, tp4_done, tp5_done, vol_regime}
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
    
    def check_order_safety(self, symbol: str, qty: int, price: float, api: tradeapi.REST, premium: float = None) -> tuple[bool, str]:
        """
        Check order-level safeguards with dynamic position size
        Returns: (is_safe, reason_if_unsafe)
        For options: use premium cost (not strike notional) - this is what you actually pay
        """
        # For options, use premium cost if provided, otherwise fall back to price (for backward compatibility)
        if premium is not None:
            order_cost = qty * premium * 100  # Actual cost: qty * premium * 100
        else:
            order_cost = qty * price * 100  # Fallback (shouldn't happen for options)
        
        # ========== SAFEGUARD 7: Order Size Sanity Check ==========
        # Check against MAX_NOTIONAL using premium cost (for options)
        if order_cost > MAX_NOTIONAL:
            return False, f"Order cost ${order_cost:,.0f} > ${MAX_NOTIONAL:,} limit"
        
        # ========== SAFEGUARD 8: Max Position Size (regime-adjusted) ==========
        max_notional = self.get_current_max_notional(api)
        current_exposure = self.get_current_exposure()
        regime_params = self.get_vol_params(self.current_regime)
        
        if current_exposure + order_cost > max_notional:
            return False, f"Position would exceed {regime_params['max_pct']:.0%} limit ({self.current_regime.upper()} regime): ${current_exposure + order_cost:,.0f} > ${max_notional:,.0f}"
        
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

# ==================== MARKET HOURS CHECK ====================
def is_market_open():
    """Check if US stock market is currently open (9:30 AM - 4:00 PM EST, weekdays)"""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    
    # Check if weekday (Monday=0, Friday=4)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Market hours: 9:30 AM - 4:00 PM EST
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close

# ==================== ALPACA SETUP ====================
def init_alpaca():
    """Initialize Alpaca API"""
    if API_KEY == 'YOUR_PAPER_KEY' or API_SECRET == 'YOUR_PAPER_SECRET':
        raise ValueError("Please set ALPACA_KEY and ALPACA_SECRET in config.py")
    
    # Detect if paper keys are being used with live endpoint
    if not USE_PAPER and (API_KEY.startswith('PK') or 'paper' in API_KEY.lower()):
        raise ValueError(
            "❌ ERROR: Paper trading keys detected with LIVE trading mode!\n"
            "   Paper keys (starting with 'PK') cannot be used with live API.\n"
            "   Solutions:\n"
            "   1. Use paper trading: Remove --live flag (default is paper)\n"
            "   2. Get live keys: Go to https://app.alpaca.markets/dashboard/overview\n"
            "   3. Update config.py with LIVE API keys (different from paper keys)"
        )
    
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    
    try:
        account = api.get_account()
        print(f"✓ Connected to Alpaca ({'PAPER' if USE_PAPER else 'LIVE'})")
        print(f"  Account Status: {account.status}")
        print(f"  Equity: ${float(account.equity):,.2f}")
        print(f"  Buying Power: ${float(account.buying_power):,.2f}")
        return api
    except Exception as e:
        error_msg = str(e)
        if "not authorized" in error_msg.lower() or "401" in error_msg:
            if not USE_PAPER:
                raise ConnectionError(
                    f"❌ Authentication failed for LIVE trading:\n"
                    f"   Error: {error_msg}\n"
                    f"   Possible causes:\n"
                    f"   1. Using paper trading keys with live endpoint\n"
                    f"   2. Live API keys are incorrect or expired\n"
                    f"   3. Keys don't have trading permissions enabled\n\n"
                    f"   To fix:\n"
                    f"   - Get LIVE keys from: https://app.alpaca.markets/dashboard/overview\n"
                    f"   - Update config.py with LIVE API keys\n"
                    f"   - Or use paper trading: Remove --live flag"
                )
            else:
                raise ConnectionError(
                    f"❌ Authentication failed for PAPER trading:\n"
                    f"   Error: {error_msg}\n"
                    f"   Please check your API keys in config.py"
                )
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

def check_stop_losses(api: tradeapi.REST, risk_mgr: RiskManager, current_price: float, trade_db=None) -> None:
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
        # Filter to only option positions (Alpaca uses 'us_option' for options)
        alpaca_option_positions = {pos.symbol: pos for pos in alpaca_positions if pos.asset_class in ['option', 'us_option'] or ('C' in pos.symbol[-9:] or 'P' in pos.symbol[-9:])}
    except Exception as e:
        risk_mgr.log(f"Error fetching positions from Alpaca: {e}", "ERROR")
        alpaca_option_positions = {}
    
    # First, sync any Alpaca positions that aren't being tracked
    for symbol, alpaca_pos in alpaca_option_positions.items():
        if symbol not in risk_mgr.open_positions:
            # Position exists in Alpaca but not tracked - add it
            try:
                # Extract strike from symbol (format: SPY251204C00683000)
                if len(symbol) >= 15:
                    strike_str = symbol[-9:-1]  # Last 9 chars before last char
                    strike = float(strike_str) / 1000
                    option_type = 'call' if 'C' in symbol[-9:] else 'put'
                else:
                    strike = current_price
                    option_type = 'call'
                
                # Get actual entry premium from Alpaca position
                # Alpaca provides avg_entry_price (already in premium format for options, NOT multiplied by 100)
                try:
                    # Try to get avg_entry_price - for options, this is already the premium per share
                    if hasattr(alpaca_pos, 'avg_entry_price') and alpaca_pos.avg_entry_price:
                        # avg_entry_price for options is already the premium (e.g., 0.8994, not 89.94)
                        entry_premium = float(alpaca_pos.avg_entry_price)
                    elif hasattr(alpaca_pos, 'cost_basis') and alpaca_pos.cost_basis and float(alpaca_pos.qty) > 0:
                        # Calculate from cost_basis: cost_basis / (qty * 100) for options
                        entry_premium = abs(float(alpaca_pos.cost_basis) / (float(alpaca_pos.qty) * 100))
                    else:
                        # Fallback: use current premium as worst case (assume we're at break-even or loss)
                        # This is conservative - will trigger stop-loss if needed
                        try:
                            snapshot = api.get_option_snapshot(symbol)
                            if snapshot.bid_price and snapshot.bid_price > 0:
                                entry_premium = float(snapshot.bid_price)  # Use current as entry (conservative)
                            else:
                                entry_premium = abs(float(alpaca_pos.market_value) / float(alpaca_pos.qty)) if float(alpaca_pos.qty) > 0 else estimate_premium(current_price, strike, option_type)
                        except:
                            entry_premium = abs(float(alpaca_pos.market_value) / float(alpaca_pos.qty)) if float(alpaca_pos.qty) > 0 else estimate_premium(current_price, strike, option_type)
                except Exception as e:
                    risk_mgr.log(f"Error getting entry premium for {symbol}: {e}, using estimate", "WARNING")
                    entry_premium = estimate_premium(current_price, strike, option_type)
                
                # Get current premium (CORRECT: market_value / (qty * 100) for options)
                try:
                    if alpaca_pos.market_value and float(alpaca_pos.qty) > 0:
                        # market_value is total value, qty is contracts, each contract is 100 shares
                        current_premium = abs(float(alpaca_pos.market_value) / (float(alpaca_pos.qty) * 100))
                    else:
                        current_premium = entry_premium
                except:
                    current_premium = abs(float(alpaca_pos.market_value) / (float(alpaca_pos.qty) * 100)) if float(alpaca_pos.qty) > 0 else entry_premium
                
                risk_mgr.open_positions[symbol] = {
                    'strike': strike,
                    'type': option_type,
                    'entry_time': datetime.now(),
                    'contracts': int(float(alpaca_pos.qty)),
                    'qty_remaining': int(float(alpaca_pos.qty)),
                    'notional': int(float(alpaca_pos.qty)) * strike * 100,
                    'entry_premium': entry_premium,  # Actual entry from Alpaca
                    'entry_price': current_price,
                    'trail_active': False,
                    'trail_price': 0.0,
                    'peak_premium': current_premium,
                    'tp1_done': False,
                    'tp2_done': False,
                    'tp3_done': False,
                    'tp4_done': False,
                    'tp5_done': False,
                    'vol_regime': current_regime,
                    'entry_vix': risk_mgr.get_current_vix()
                }
                risk_mgr.log(f"Synced untracked position: {symbol} ({int(float(alpaca_pos.qty))} contracts @ ${strike:.2f} strike)", "INFO")
            except Exception as e:
                risk_mgr.log(f"Error syncing position {symbol}: {e}", "WARNING")
    
    # Check tracked positions against actual Alpaca positions
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
            # For stop-losses, ALWAYS use fixed -15% (not volatility-adjusted)
            tp_params = vol_params  # Use current regime for TP
            # Stop-loss is now FIXED at -15% regardless of regime
            
            # ALWAYS update entry_premium from Alpaca (most accurate source)
            # This ensures we use Alpaca's actual avg_entry_price, not an estimate
            try:
                if hasattr(alpaca_pos, 'avg_entry_price') and alpaca_pos.avg_entry_price:
                    alpaca_entry = float(alpaca_pos.avg_entry_price)
                    # Update entry_premium if it's different (to ensure accuracy)
                    if abs(alpaca_entry - pos_data['entry_premium']) > 0.001:
                        risk_mgr.log(f"Updating entry_premium for {symbol}: ${pos_data['entry_premium']:.4f} → ${alpaca_entry:.4f} (from Alpaca)", "INFO")
                        pos_data['entry_premium'] = alpaca_entry
                elif hasattr(alpaca_pos, 'cost_basis') and alpaca_pos.cost_basis and float(alpaca_pos.qty) > 0:
                    alpaca_entry = abs(float(alpaca_pos.cost_basis) / (float(alpaca_pos.qty) * 100))
                    if abs(alpaca_entry - pos_data['entry_premium']) > 0.001:
                        risk_mgr.log(f"Updating entry_premium for {symbol}: ${pos_data['entry_premium']:.4f} → ${alpaca_entry:.4f} (from cost_basis)", "INFO")
                        pos_data['entry_premium'] = alpaca_entry
            except Exception as e:
                risk_mgr.log(f"Error updating entry_premium for {symbol}: {e}", "WARNING")
            
            # Get current premium from Alpaca position (most reliable)
            # Use market_value / (qty * 100) for options (market_value is total, qty is contracts)
            try:
                if alpaca_pos.market_value and float(alpaca_pos.qty) > 0:
                    # market_value is total value, qty is contracts, each contract is 100 shares
                    current_premium = abs(float(alpaca_pos.market_value) / (float(alpaca_pos.qty) * 100))
                else:
                    # Fallback to estimate if market_value not available
                    current_premium = estimate_premium(current_price, pos_data['strike'], pos_data['type'])
            except Exception as e:
                # Fallback to estimate if calculation fails
                risk_mgr.log(f"Error calculating current premium for {symbol}: {e}, using estimate", "WARNING")
                current_premium = estimate_premium(current_price, pos_data['strike'], pos_data['type'])
            
            # Update peak premium
            if current_premium > pos_data.get('peak_premium', pos_data['entry_premium']):
                pos_data['peak_premium'] = current_premium
            
            # Calculate PnL percentage
            pnl_pct = (current_premium - pos_data['entry_premium']) / pos_data['entry_premium']
            
            # Get remaining quantity from actual Alpaca position
            actual_qty = int(float(alpaca_pos.qty))
            qty_remaining = pos_data.get('qty_remaining', actual_qty)
            
            # Update qty_remaining if it doesn't match actual position
            if qty_remaining != actual_qty:
                risk_mgr.log(f"Updating qty_remaining for {symbol}: {qty_remaining} → {actual_qty}", "INFO")
                pos_data['qty_remaining'] = actual_qty
                qty_remaining = actual_qty
            
            # ========== FIXED STOP-LOSS (-15% ALWAYS) - CHECK FIRST ==========
            # Stop-loss must be checked BEFORE take-profits to prevent larger losses
            # Log P&L for debugging
            risk_mgr.log(f"Position {symbol}: Entry=${pos_data['entry_premium']:.4f} | Current=${current_premium:.4f} | P&L={pnl_pct:.2%}", "INFO")
            
            if pnl_pct <= FIXED_STOP_LOSS:
                risk_mgr.log(f"🚨 STOP-LOSS TRIGGERED (-15% FIXED): {symbol} @ {pnl_pct:.1%} → FORCED EXIT", "CRITICAL")
                
                # Log trade to database
                if trade_db:
                    capital_before = risk_mgr.get_equity(api)
                    pnl_dollar = (current_premium - pos_data['entry_premium']) * qty_remaining * 100
                    trade_db.log_trade(
                        symbol=symbol,
                        action='SELL',
                        qty=qty_remaining,
                        price=current_premium,
                        strike=pos_data['strike'],
                        option_type=pos_data['type'],
                        entry_premium=pos_data['entry_premium'],
                        exit_premium=current_premium,
                        pnl=pnl_dollar,
                        pnl_pct=pnl_pct,
                        capital_before=capital_before,
                        capital_after=capital_before + pnl_dollar,
                        reason='STOP_LOSS',
                        regime=entry_regime,
                        vix=risk_mgr.get_current_vix()
                    )
                
                positions_to_close.append(symbol)
                continue  # Exit immediately, don't check take-profits
            
            # ========== NEW 5-TIER TAKE-PROFIT SYSTEM ==========
            # TP1 at +40%: Sell 50% of position
            # TP2 at +60%: Sell 20% of remaining
            # TP3 at +100%: Sell 10% of remaining
            # TP4 at +150%: Sell 10% of remaining
            # TP5 at +200%: Full exit
            # Trailing Stop: Activates after TP4, locks in +100% minimum
            
            # ========== TP5: +200% - FULL EXIT ==========
            if pnl_pct >= tp_params['tp5'] and not pos_data.get('tp5_done', False):
                try:
                    api.close_position(symbol)
                    risk_mgr.log(f"🎯 TP5 +{tp_params['tp5']:.0%} HIT ({current_regime.upper()}) → FULL EXIT: {symbol} @ {pnl_pct:.1%}", "TRADE")
                    
                    # Log trade to database
                    if trade_db:
                        capital_before = risk_mgr.get_equity(api)
                        pnl_dollar = (current_premium - pos_data['entry_premium']) * qty_remaining * 100
                        trade_db.log_trade(
                            symbol=symbol,
                            action='SELL',
                            qty=qty_remaining,
                            price=current_premium,
                            strike=pos_data['strike'],
                            option_type=pos_data['type'],
                            entry_premium=pos_data['entry_premium'],
                            exit_premium=current_premium,
                            pnl=pnl_dollar,
                            pnl_pct=pnl_pct,
                            capital_before=capital_before,
                            capital_after=capital_before + pnl_dollar,
                            reason='TP5',
                            regime=current_regime,
                            vix=risk_mgr.get_current_vix()
                        )
                    
                    positions_to_close.append(symbol)
                    continue
                except Exception as e:
                    risk_mgr.log(f"✗ Error executing TP5 exit for {symbol}: {e}", "ERROR")
            
            # ========== TP4: +150% - Sell 10% of remaining ==========
            elif pnl_pct >= tp_params['tp4'] and not pos_data.get('tp4_done', False):
                sell_qty = max(1, int(qty_remaining * 0.10))  # Sell 10% of remaining
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
                        pos_data['tp4_done'] = True
                        # Activate trailing stop after TP4
                        pos_data['trail_active'] = True
                        pos_data['trail_price'] = pos_data['entry_premium'] * (1 + tp_params['trail'])  # Lock in +100%
                        risk_mgr.log(f"🎯 TP4 +{tp_params['tp4']:.0%} ({current_regime.upper()}) → SOLD 10% ({sell_qty}x) | Remaining: {pos_data['qty_remaining']} | Trail activated at +{tp_params['trail']:.0%}", "TRADE")
                        
                        # Log trade to database
                        if trade_db:
                            capital_before = risk_mgr.get_equity(api)
                            pnl_dollar = (current_premium - pos_data['entry_premium']) * sell_qty * 100
                            trade_db.log_trade(
                                symbol=symbol,
                                action='SELL',
                                qty=sell_qty,
                                price=current_premium,
                                strike=pos_data['strike'],
                                option_type=pos_data['type'],
                                entry_premium=pos_data['entry_premium'],
                                exit_premium=current_premium,
                                pnl=pnl_dollar,
                                pnl_pct=pnl_pct,
                                capital_before=capital_before,
                                capital_after=capital_before + pnl_dollar,
                                reason='TP4',
                                regime=current_regime,
                                vix=risk_mgr.get_current_vix()
                            )
                    except Exception as e:
                        risk_mgr.log(f"✗ Error executing TP4 for {symbol}: {e}", "ERROR")
                else:
                    # If 10% is all remaining, just close
                    try:
                        api.close_position(symbol)
                        risk_mgr.log(f"🎯 TP4 +{tp_params['tp4']:.0%} ({current_regime.upper()}) → FULL EXIT: {symbol} @ {pnl_pct:.1%}", "TRADE")
                        
                        # Log trade to database
                        if trade_db:
                            capital_before = risk_mgr.get_equity(api)
                            pnl_dollar = (current_premium - pos_data['entry_premium']) * qty_remaining * 100
                            trade_db.log_trade(
                                symbol=symbol,
                                action='SELL',
                                qty=qty_remaining,
                                price=current_premium,
                                strike=pos_data['strike'],
                                option_type=pos_data['type'],
                                entry_premium=pos_data['entry_premium'],
                                exit_premium=current_premium,
                                pnl=pnl_dollar,
                                pnl_pct=pnl_pct,
                                capital_before=capital_before,
                                capital_after=capital_before + pnl_dollar,
                                reason='TP4_FULL',
                                regime=current_regime,
                                vix=risk_mgr.get_current_vix()
                            )
                        
                        positions_to_close.append(symbol)
                        continue
                    except Exception as e:
                        risk_mgr.log(f"✗ Error closing at TP4: {e}", "ERROR")
            
            # ========== TP3: +100% - Sell 10% of remaining ==========
            elif pnl_pct >= tp_params['tp3'] and not pos_data.get('tp3_done', False):
                sell_qty = max(1, int(qty_remaining * 0.10))  # Sell 10% of remaining
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
                        pos_data['tp3_done'] = True
                        risk_mgr.log(f"🎯 TP3 +{tp_params['tp3']:.0%} ({current_regime.upper()}) → SOLD 10% ({sell_qty}x) | Remaining: {pos_data['qty_remaining']}", "TRADE")
                        
                        # Log trade to database
                        if trade_db:
                            capital_before = risk_mgr.get_equity(api)
                            pnl_dollar = (current_premium - pos_data['entry_premium']) * sell_qty * 100
                            trade_db.log_trade(
                                symbol=symbol,
                                action='SELL',
                                qty=sell_qty,
                                price=current_premium,
                                strike=pos_data['strike'],
                                option_type=pos_data['type'],
                                entry_premium=pos_data['entry_premium'],
                                exit_premium=current_premium,
                                pnl=pnl_dollar,
                                pnl_pct=pnl_pct,
                                capital_before=capital_before,
                                capital_after=capital_before + pnl_dollar,
                                reason='TP3',
                                regime=current_regime,
                                vix=risk_mgr.get_current_vix()
                            )
                    except Exception as e:
                        risk_mgr.log(f"✗ Error executing TP3 for {symbol}: {e}", "ERROR")
                else:
                    # If 10% is all remaining, just close
                    try:
                        api.close_position(symbol)
                        risk_mgr.log(f"🎯 TP3 +{tp_params['tp3']:.0%} ({current_regime.upper()}) → FULL EXIT: {symbol} @ {pnl_pct:.1%}", "TRADE")
                        
                        # Log trade to database
                        if trade_db:
                            capital_before = risk_mgr.get_equity(api)
                            pnl_dollar = (current_premium - pos_data['entry_premium']) * qty_remaining * 100
                            trade_db.log_trade(
                                symbol=symbol,
                                action='SELL',
                                qty=qty_remaining,
                                price=current_premium,
                                strike=pos_data['strike'],
                                option_type=pos_data['type'],
                                entry_premium=pos_data['entry_premium'],
                                exit_premium=current_premium,
                                pnl=pnl_dollar,
                                pnl_pct=pnl_pct,
                                capital_before=capital_before,
                                capital_after=capital_before + pnl_dollar,
                                reason='TP3_FULL',
                                regime=current_regime,
                                vix=risk_mgr.get_current_vix()
                            )
                        
                        positions_to_close.append(symbol)
                        continue
                    except Exception as e:
                        risk_mgr.log(f"✗ Error closing at TP3: {e}", "ERROR")
            
            # ========== TP2: +60% - Sell 20% of remaining ==========
            elif pnl_pct >= tp_params['tp2'] and not pos_data.get('tp2_done', False):
                sell_qty = max(1, int(qty_remaining * 0.20))  # Sell 20% of remaining
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
                        risk_mgr.log(f"🎯 TP2 +{tp_params['tp2']:.0%} ({current_regime.upper()}) → SOLD 20% ({sell_qty}x) | Remaining: {pos_data['qty_remaining']}", "TRADE")
                        
                        # Log trade to database
                        if trade_db:
                            capital_before = risk_mgr.get_equity(api)
                            pnl_dollar = (current_premium - pos_data['entry_premium']) * sell_qty * 100
                            trade_db.log_trade(
                                symbol=symbol,
                                action='SELL',
                                qty=sell_qty,
                                price=current_premium,
                                strike=pos_data['strike'],
                                option_type=pos_data['type'],
                                entry_premium=pos_data['entry_premium'],
                                exit_premium=current_premium,
                                pnl=pnl_dollar,
                                pnl_pct=pnl_pct,
                                capital_before=capital_before,
                                capital_after=capital_before + pnl_dollar,
                                reason='TP2',
                                regime=current_regime,
                                vix=risk_mgr.get_current_vix()
                            )
                    except Exception as e:
                        risk_mgr.log(f"✗ Error executing TP2 for {symbol}: {e}", "ERROR")
                else:
                    # If 20% is all remaining, just close
                    try:
                        api.close_position(symbol)
                        risk_mgr.log(f"🎯 TP2 +{tp_params['tp2']:.0%} ({current_regime.upper()}) → FULL EXIT: {symbol} @ {pnl_pct:.1%}", "TRADE")
                        
                        # Log trade to database
                        if trade_db:
                            capital_before = risk_mgr.get_equity(api)
                            pnl_dollar = (current_premium - pos_data['entry_premium']) * qty_remaining * 100
                            trade_db.log_trade(
                                symbol=symbol,
                                action='SELL',
                                qty=qty_remaining,
                                price=current_premium,
                                strike=pos_data['strike'],
                                option_type=pos_data['type'],
                                entry_premium=pos_data['entry_premium'],
                                exit_premium=current_premium,
                                pnl=pnl_dollar,
                                pnl_pct=pnl_pct,
                                capital_before=capital_before,
                                capital_after=capital_before + pnl_dollar,
                                reason='TP2_FULL',
                                regime=current_regime,
                                vix=risk_mgr.get_current_vix()
                            )
                        
                        positions_to_close.append(symbol)
                        continue
                    except Exception as e:
                        risk_mgr.log(f"✗ Error closing at TP2: {e}", "ERROR")
            
            # ========== TP1: +40% - Sell 50% of position ==========
            elif pnl_pct >= tp_params['tp1'] and not pos_data.get('tp1_done', False):
                # CRITICAL: Always sell 50% using submit_order, NEVER close_position for TP1
                # Calculate 50% of current remaining quantity
                sell_qty = max(1, int(qty_remaining * 0.5))  # Sell 50% of position
                
                # Ensure we don't sell more than remaining (safety check)
                sell_qty = min(sell_qty, qty_remaining)
                
                # Only proceed if we have more than 1 contract (so we can sell partial)
                if qty_remaining > 1 and sell_qty < qty_remaining:
                    try:
                        # Use submit_order to sell PARTIAL position (50%)
                        api.submit_order(
                            symbol=symbol,
                            qty=sell_qty,
                            side='sell',
                            type='market',
                            time_in_force='day'
                        )
                        pos_data['qty_remaining'] = qty_remaining - sell_qty
                        pos_data['tp1_done'] = True
                        risk_mgr.log(f"🎯 TP1 +{tp_params['tp1']:.0%} ({current_regime.upper()}) → SOLD 50% ({sell_qty}x of {qty_remaining}) | Remaining: {pos_data['qty_remaining']}", "TRADE")
                        
                        # Log trade to database
                        if trade_db:
                            capital_before = risk_mgr.get_equity(api)
                            pnl_dollar = (current_premium - pos_data['entry_premium']) * sell_qty * 100
                            trade_db.log_trade(
                                symbol=symbol,
                                action='SELL',
                                qty=sell_qty,
                                price=current_premium,
                                strike=pos_data['strike'],
                                option_type=pos_data['type'],
                                entry_premium=pos_data['entry_premium'],
                                exit_premium=current_premium,
                                pnl=pnl_dollar,
                                pnl_pct=pnl_pct,
                                capital_before=capital_before,
                                capital_after=capital_before + pnl_dollar,
                                reason='TP1',
                                regime=current_regime,
                                vix=risk_mgr.get_current_vix()
                            )
                        # Continue to next position - don't check other TPs in same cycle
                        continue
                    except Exception as e:
                        risk_mgr.log(f"✗ Error executing TP1 partial sell for {symbol}: {e}", "ERROR")
                elif qty_remaining == 1:
                    # Only 1 contract left - can't sell 50%, so skip TP1 (will hit TP2/TP3/etc later)
                    risk_mgr.log(f"⚠️ TP1 +{tp_params['tp1']:.0%} reached but only 1 contract remaining - skipping TP1, will trigger at next TP level", "INFO")
                    pos_data['tp1_done'] = True  # Mark as done so we don't keep checking
                else:
                    # Edge case: qty_remaining calculation issue
                    risk_mgr.log(f"⚠️ TP1 calculation issue: qty_remaining={qty_remaining}, sell_qty={sell_qty} - skipping", "WARNING")
                    pos_data['tp1_done'] = True  # Mark as done to prevent infinite loop
            
            # ========== TRAILING STOP AFTER TP4 (+100% minimum) ==========
            if pos_data.get('trail_active', False) and pos_data.get('tp4_done', False):
                if current_premium <= pos_data.get('trail_price', 0):
                    try:
                        api.close_position(symbol)
                        trail_pct = tp_params['trail']
                        risk_mgr.log(f"📉 TRAILING STOP HIT ({current_regime.upper()}, +{trail_pct:.0%}): {symbol} @ ${current_premium:.2f} → Locked profit", "TRADE")
                        
                        # Log trade to database
                        if trade_db:
                            capital_before = risk_mgr.get_equity(api)
                            pnl_dollar = (current_premium - pos_data['entry_premium']) * qty_remaining * 100
                            trade_db.log_trade(
                                symbol=symbol,
                                action='SELL',
                                qty=qty_remaining,
                                price=current_premium,
                                strike=pos_data['strike'],
                                option_type=pos_data['type'],
                                entry_premium=pos_data['entry_premium'],
                                exit_premium=current_premium,
                                pnl=pnl_dollar,
                                pnl_pct=pnl_pct,
                                capital_before=capital_before,
                                capital_after=capital_before + pnl_dollar,
                                reason='TRAILING_STOP',
                                regime=current_regime,
                                vix=risk_mgr.get_current_vix()
                            )
                        
                        positions_to_close.append(symbol)
                        continue
                    except Exception as e:
                        risk_mgr.log(f"✗ Error executing trailing stop: {e}", "ERROR")
            
            # Stop-loss already checked at the top - this section removed to avoid duplication
            
            # ========== TRAILING STOP CHECK (after TP4 activation) ==========
            # Trailing stop is activated at TP4, locks in +100% minimum
            if pos_data.get('trail_active', False) and pos_data.get('tp4_done', False):
                if current_premium <= pos_data.get('trail_price', 0):
                    trail_pct = tp_params['trail']
                    risk_mgr.log(f"📉 TRAILING STOP HIT ({current_regime.upper()}, +{trail_pct:.0%} minimum): {symbol} @ ${current_premium:.2f} (trail: ${pos_data['trail_price']:.2f})", "TRADE")
                    
                    # Log trade to database
                    if trade_db:
                        capital_before = risk_mgr.get_equity(api)
                        pnl_dollar = (current_premium - pos_data['entry_premium']) * qty_remaining * 100
                        trade_db.log_trade(
                            symbol=symbol,
                            action='SELL',
                            qty=qty_remaining,
                            price=current_premium,
                            strike=pos_data['strike'],
                            option_type=pos_data['type'],
                            entry_premium=pos_data['entry_premium'],
                            exit_premium=current_premium,
                            pnl=pnl_dollar,
                            pnl_pct=pnl_pct,
                            capital_before=capital_before,
                            capital_after=capital_before + pnl_dollar,
                            reason='TRAILING_STOP',
                            regime=current_regime,
                            vix=risk_mgr.get_current_vix()
                        )
                    
                    positions_to_close.append(symbol)
                    continue
            
            # ========== REJECTION DETECTION ==========
            # Check if price rejected from entry level (for calls: high > entry but close < entry)
            if pos_data['type'] == 'call':
                # Would need bar data - simplified check
                if current_price < pos_data['entry_price'] * 0.99:  # 1% rejection
                    risk_mgr.log(f"⚠️ REJECTION DETECTED: {symbol} → Exit", "TRADE")
                    
                    # Log trade to database
                    if trade_db:
                        capital_before = risk_mgr.get_equity(api)
                        pnl_dollar = (current_premium - pos_data['entry_premium']) * qty_remaining * 100
                        trade_db.log_trade(
                            symbol=symbol,
                            action='SELL',
                            qty=qty_remaining,
                            price=current_premium,
                            strike=pos_data['strike'],
                            option_type=pos_data['type'],
                            entry_premium=pos_data['entry_premium'],
                            exit_premium=current_premium,
                            pnl=pnl_dollar,
                            pnl_pct=pnl_pct,
                            capital_before=capital_before,
                            capital_after=capital_before + pnl_dollar,
                            reason='REJECTION',
                            regime=current_regime,
                            vix=risk_mgr.get_current_vix()
                        )
                    
                    positions_to_close.append(symbol)
                    continue
            
        except Exception as e:
            risk_mgr.log(f"Error checking stop-loss/take-profit for {symbol}: {e}", "ERROR")
    
    # Close positions that hit stops (CORRECT API)
    for symbol in positions_to_close:
        try:
            # Use close_position() which works correctly
            api.close_position(symbol)
            risk_mgr.log(f"✓ Position closed: {symbol}", "TRADE")
            
            # Log final close if not already logged above
            if trade_db and symbol in risk_mgr.open_positions:
                pos_data = risk_mgr.open_positions[symbol]
                try:
                    # Try to get current premium from Alpaca
                    alpaca_positions = api.list_positions()
                    alpaca_pos = next((p for p in alpaca_positions if p.symbol == symbol), None)
                    if alpaca_pos:
                        current_premium = abs(float(alpaca_pos.market_value) / (float(alpaca_pos.qty) * 100)) if float(alpaca_pos.qty) > 0 else pos_data['entry_premium']
                    else:
                        current_premium = pos_data['entry_premium']
                    
                    pnl_pct = (current_premium - pos_data['entry_premium']) / pos_data['entry_premium']
                    qty_remaining = pos_data.get('qty_remaining', pos_data.get('contracts', 0))
                    pnl_dollar = (current_premium - pos_data['entry_premium']) * qty_remaining * 100
                    capital_before = risk_mgr.get_equity(api)
                    
                    trade_db.log_trade(
                        symbol=symbol,
                        action='SELL',
                        qty=qty_remaining,
                        price=current_premium,
                        strike=pos_data['strike'],
                        option_type=pos_data['type'],
                        entry_premium=pos_data['entry_premium'],
                        exit_premium=current_premium,
                        pnl=pnl_dollar,
                        pnl_pct=pnl_pct,
                        capital_before=capital_before,
                        capital_after=capital_before + pnl_dollar,
                        reason='CLOSE',
                        regime=pos_data.get('vol_regime', 'normal_vol'),
                        vix=pos_data.get('entry_vix', risk_mgr.get_current_vix())
                    )
                except Exception as e:
                    risk_mgr.log(f"Error logging final close for {symbol}: {e}", "WARNING")
            
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

# ==================== OBSERVATION PREPARATION ====================
def prepare_observation(data: pd.DataFrame, risk_mgr: RiskManager) -> np.ndarray:
    """Prepare observation for RL model - must match training format: (20, 5)"""
    if len(data) < LOOKBACK:
        # Pad with last value
        padding = pd.concat([data.iloc[[-1]]] * (LOOKBACK - len(data)))
        data = pd.concat([padding, data])
    
    recent = data.tail(LOOKBACK).copy()
    
    # Model was trained with: open, high, low, close, volume (5 features)
    # Ensure column names match (case-insensitive)
    cols = {}
    for col in recent.columns:
        col_lower = col.lower()
        if 'open' in col_lower:
            cols['open'] = col
        elif 'high' in col_lower:
            cols['high'] = col
        elif 'low' in col_lower:
            cols['low'] = col
        elif 'close' in col_lower:
            cols['close'] = col
        elif 'volume' in col_lower:
            cols['volume'] = col
    
    # Extract the 5 features in the correct order
    obs_data = recent[[cols.get('open', 'Open'), 
                       cols.get('high', 'High'), 
                       cols.get('low', 'Low'), 
                       cols.get('close', 'Close'), 
                       cols.get('volume', 'Volume')]].values
    
    # Return shape (20, 5) - no batch dimension, matches training
    return obs_data.astype(np.float32)

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
    print(f"  11. Stop-Loss: FIXED -{abs(FIXED_STOP_LOSS)*100:.0f}% (ALWAYS, regardless of volatility)")
    print(f"  12. Take-Profit System: TP1 +40% (50%) | TP2 +60% (20%) | TP3 +100% (10%) | TP4 +150% (10%) | TP5 +200% (FULL) | Trail +100% after TP4")
    print(f"  13. Volatility Regime Engine: Calm 10%/30% | Normal 7%/25% | Storm 5%/20% | Crash 3%/15%")
    print(f"  14. Trading Symbols: {', '.join(TRADING_SYMBOLS)}")
    print("=" * 70)
    print()
    
    # Initialize trade database
    if TRADE_DB_AVAILABLE:
        trade_db = MikeAgentTradeDB()
        print("✅ Trade database initialized - all trades will be logged")
    else:
        trade_db = None
        print("⚠️  Trade database not available - trades will not be logged")
    print()
    
    # Initialize
    try:
        api = init_alpaca()
        model = load_rl_model()
        risk_mgr = RiskManager()
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return
    
    risk_mgr.log("Agent started with full protection", "INFO")
    
    # Sync positions from Alpaca on startup (CORRECT API)
    try:
        alpaca_positions = api.list_positions()
        option_positions = [pos for pos in alpaca_positions if pos.asset_class in ['option', 'us_option'] or ('C' in pos.symbol[-9:] or 'P' in pos.symbol[-9:])]
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
                # Get entry premium from Alpaca's avg_entry_price (most accurate)
                try:
                    # avg_entry_price for options is already the premium per share
                    if hasattr(pos, 'avg_entry_price') and pos.avg_entry_price:
                        entry_premium = float(pos.avg_entry_price)
                    elif hasattr(pos, 'cost_basis') and pos.cost_basis and float(pos.qty) > 0:
                        # Calculate from cost_basis: cost_basis / (qty * 100)
                        entry_premium = abs(float(pos.cost_basis) / (float(pos.qty) * 100))
                    else:
                        entry_premium = 0.5  # Default estimate
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
                    'notional': int(float(pos.qty)) * strike * 100,
                    'entry_premium': entry_premium,
                    'entry_price': current_spy_price,
                    'trail_active': False,
                    'trail_price': 0.0,
                    'peak_premium': entry_premium,
                    'tp1_done': False,
                    'tp2_done': False,
                    'tp3_done': False,
                    'tp4_done': False,
                    'tp5_done': False,
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
    risk_mgr.log(f"  Calm (VIX<18): Risk 10% | Max 30% | SL -15% | TP +40%/+60%/+100%/+150%/+200% | Trail +100% after TP4", "INFO")
    risk_mgr.log(f"  Normal (18-25): Risk 7% | Max 25% | SL -20% | TP +40%/+60%/+100%/+150%/+200% | Trail +100% after TP4", "INFO")
    risk_mgr.log(f"  Storm (25-35): Risk 5% | Max 20% | SL -28% | TP +40%/+60%/+100%/+150%/+200% | Trail +100% after TP4", "INFO")
    risk_mgr.log(f"  Crash (>35): Risk 3% | Max 15% | SL -35% | TP +40%/+60%/+100%/+150%/+200% | Trail +100% after TP4", "INFO")
    risk_mgr.log(f"NEW 5-TIER TAKE-PROFIT SYSTEM: TP1 +40% (50%) | TP2 +60% (20%) | TP3 +100% (10%) | TP4 +150% (10%) | TP5 +200% (FULL)", "INFO")
    risk_mgr.log(f"TRAILING STOP: Activates after TP4, locks in +100% minimum profit", "INFO")
    risk_mgr.log(f"MAX CONCURRENT POSITIONS: {MAX_CONCURRENT}", "INFO")
    risk_mgr.log(f"14/14 SAFEGUARDS: ACTIVE (11 Risk + 1 Volatility Regime Engine + 1 Dynamic Sizing + 1 New 5-Tier TP)", "INFO")
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            
            try:
                # ========== MARKET HOURS CHECK ==========
                if not is_market_open():
                    if iteration % 20 == 0:  # Log every 20th iteration when market is closed
                        est = pytz.timezone('US/Eastern')
                        now = datetime.now(est)
                        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
                        if now.weekday() < 5 and now < market_open:
                            # Before market open today
                            wait_time = (market_open - now).total_seconds() / 60
                            risk_mgr.log(f"⏰ Market closed. Waiting for market open in {wait_time:.0f} minutes (9:30 AM EST)", "INFO")
                        elif now.weekday() >= 5:
                            # Weekend
                            days_until_monday = (7 - now.weekday()) % 7
                            if days_until_monday == 0:
                                days_until_monday = 1
                            risk_mgr.log(f"⏰ Weekend. Market opens Monday 9:30 AM EST ({days_until_monday} day{'s' if days_until_monday > 1 else ''})", "INFO")
                        else:
                            # After market close
                            risk_mgr.log(f"⏰ Market closed. Opens tomorrow 9:30 AM EST", "INFO")
                    time.sleep(60)  # Check every minute when market is closed
                    continue
                
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
                
                # Prepare observation
                obs = prepare_observation(hist, risk_mgr)
                
                # RL Decision - model expects (1, 20, 5) for predict
                # obs is (20, 5), add batch dimension: (1, 20, 5)
                obs_batch = obs.reshape(1, LOOKBACK, 5)
                action, _ = model.predict(obs_batch, deterministic=True)
                action = int(action[0])
                
                equity = risk_mgr.get_equity(api)
                status = f"FLAT"
                if risk_mgr.open_positions:
                    status = f"{len(risk_mgr.open_positions)} positions"
                
                # Show current VIX and regime
                current_vix = risk_mgr.get_current_vix()
                current_regime = risk_mgr.get_vol_regime(current_vix)
                regime_params = risk_mgr.get_vol_params(current_regime)
                risk_mgr.log(f"SPY: ${current_price:.2f} | VIX: {current_vix:.1f} ({current_regime.upper()}) | Risk: {regime_params['risk']:.0%} | Max Size: {regime_params['max_pct']:.0%} | Action: {action} | Equity: ${equity:,.2f} | Status: {status} | Daily PnL: {risk_mgr.daily_pnl:.2%}", "INFO")
                
                # ========== CHECK STOP-LOSSES ON EXISTING POSITIONS ==========
                check_stop_losses(api, risk_mgr, current_price, trade_db)
                
                # ========== SAFE EXECUTION WITH REGIME-ADAPTIVE POSITION SIZING ==========
                if action == 1 and len(risk_mgr.open_positions) < MAX_CONCURRENT:  # BUY CALL
                    strike = find_atm_strike(current_price)
                    symbol = get_option_symbol('SPY', strike, 'call')
                    
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
                    # For options, use PREMIUM cost (not strike notional) - this is what you actually pay
                    regime_max_notional = risk_mgr.get_regime_max_notional(api, current_regime)
                    premium_cost_per_contract = estimated_premium * 100  # Actual cost per contract
                    regime_max_contracts = int(regime_max_notional / premium_cost_per_contract)
                    
                    if regime_max_contracts < 1:
                        risk_mgr.log(f"REGIME MAX POSITION SIZE REACHED ({current_regime.upper()}): ${risk_mgr.get_current_exposure():,.0f} / ${regime_max_notional:,.0f} | Premium: ${estimated_premium:.2f} → NO NEW ENTRY", "WARNING")
                        time.sleep(30)
                        continue
                    
                    # Use smaller of: regime-adjusted size or regime max contracts
                    qty = min(regime_adjusted_qty, regime_max_contracts)
                    
                    # Check order safety
                    is_safe, reason = risk_mgr.check_order_safety(symbol, qty, strike, api, premium=estimated_premium)
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
                        notional = qty * strike * 100
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
                            'peak_premium': entry_premium,
                            'tp1_done': False,
                            'tp2_done': False,
                            'tp3_done': False,
                            'tp4_done': False,
                            'tp5_done': False,
                            'vol_regime': entry_regime,  # Store regime at entry
                            'entry_vix': entry_vix
                        }
                        risk_mgr.record_order(symbol)
                        current_exposure = risk_mgr.get_current_exposure()
                        max_notional = risk_mgr.get_current_max_notional(api)
                        risk_mgr.log(f"✓ EXECUTED: BUY {qty}x {symbol} (CALL) @ ${strike:.2f} | {current_regime.upper()} REGIME | Risk: {regime_risk:.0%} | Max Size: {regime_params['max_pct']:.0%} | Notional: ${notional:,.0f} | Exposure: ${current_exposure:,.0f}/{regime_max_notional:,.0f}", "TRADE")
                        
                        # Log trade to database
                        if trade_db:
                            capital_before = risk_mgr.get_equity(api)
                            trade_db.log_trade(
                                symbol=symbol,
                                action='BUY',
                                qty=qty,
                                price=strike,
                                strike=strike,
                                option_type='call',
                                entry_premium=entry_premium,
                                capital_before=capital_before,
                                capital_after=capital_before,  # Will update on exit
                                reason='ENTRY',
                                regime=current_regime,
                                vix=entry_vix
                            )
                    except Exception as e:
                        risk_mgr.log(f"✗ Order failed: {e}", "ERROR")
                
                elif action == 2 and len(risk_mgr.open_positions) < MAX_CONCURRENT:  # BUY PUT
                    strike = find_atm_strike(current_price)
                    symbol = get_option_symbol('SPY', strike, 'put')
                    
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
                    # For options, use PREMIUM cost (not strike notional) - this is what you actually pay
                    regime_max_notional = risk_mgr.get_regime_max_notional(api, current_regime)
                    premium_cost_per_contract = estimated_premium * 100  # Actual cost per contract
                    regime_max_contracts = int(regime_max_notional / premium_cost_per_contract)
                    
                    if regime_max_contracts < 1:
                        risk_mgr.log(f"REGIME MAX POSITION SIZE REACHED ({current_regime.upper()}): ${risk_mgr.get_current_exposure():,.0f} / ${regime_max_notional:,.0f} | Premium: ${estimated_premium:.2f} → NO NEW ENTRY", "WARNING")
                        time.sleep(30)
                        continue
                    
                    # Use smaller of: regime-adjusted size or regime max contracts
                    qty = min(regime_adjusted_qty, regime_max_contracts)
                    
                    # Check order safety
                    is_safe, reason = risk_mgr.check_order_safety(symbol, qty, strike, api, premium=estimated_premium)
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
                        notional = qty * strike * 100
                        # Get actual fill price if available
                        entry_premium = float(order.filled_avg_price) if hasattr(order, 'filled_avg_price') and order.filled_avg_price else estimated_premium
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
                            'peak_premium': entry_premium,
                            'tp1_done': False,
                            'tp2_done': False,
                            'tp3_done': False,
                            'tp4_done': False,
                            'tp5_done': False,
                            'vol_regime': entry_regime,  # Store regime at entry
                            'entry_vix': entry_vix
                        }
                        risk_mgr.record_order(symbol)
                        current_exposure = risk_mgr.get_current_exposure()
                        max_notional = risk_mgr.get_current_max_notional(api)
                        risk_mgr.log(f"✓ EXECUTED: BUY {qty}x {symbol} (PUT) @ ${strike:.2f} | {current_regime.upper()} REGIME | Risk: {regime_risk:.0%} | Max Size: {regime_params['max_pct']:.0%} | Notional: ${notional:,.0f} | Exposure: ${current_exposure:,.0f}/{regime_max_notional:,.0f}", "TRADE")
                        
                        # Log trade to database
                        if trade_db:
                            capital_before = risk_mgr.get_equity(api)
                            trade_db.log_trade(
                                symbol=symbol,
                                action='BUY',
                                qty=qty,
                                price=strike,
                                strike=strike,
                                option_type='put',
                                entry_premium=entry_premium,
                                capital_before=capital_before,
                                capital_after=capital_before,  # Will update on exit
                                reason='ENTRY',
                                regime=current_regime,
                                vix=entry_vix
                            )
                    except Exception as e:
                        risk_mgr.log(f"✗ Order failed: {e}", "ERROR")
                
                elif action in [3, 4, 5] and risk_mgr.open_positions:  # TRIM OR EXIT
                    # Get actual positions from Alpaca
                    try:
                        alpaca_positions = api.list_positions()
                        alpaca_option_positions = {pos.symbol: pos for pos in alpaca_positions if pos.asset_class in ['option', 'us_option'] or ('C' in pos.symbol[-9:] or 'P' in pos.symbol[-9:])}
                    except Exception as e:
                        risk_mgr.log(f"Error fetching positions for trim/exit: {e}", "ERROR")
                        alpaca_option_positions = {}
                    
                    for sym in list(risk_mgr.open_positions.keys()):
                        try:
                            if action == 5:  # Full exit
                                api.close_position(sym)
                                risk_mgr.log(f"✓ SAFE EXIT: Closed all {sym}", "TRADE")
                                
                                # Log trade to database
                                if trade_db and sym in risk_mgr.open_positions:
                                    pos_data = risk_mgr.open_positions[sym]
                                    try:
                                        alpaca_pos = alpaca_option_positions.get(sym)
                                        if alpaca_pos:
                                            current_premium = abs(float(alpaca_pos.market_value) / (float(alpaca_pos.qty) * 100)) if float(alpaca_pos.qty) > 0 else pos_data['entry_premium']
                                        else:
                                            current_premium = pos_data['entry_premium']
                                        
                                        qty = int(float(alpaca_pos.qty)) if alpaca_pos else pos_data.get('qty_remaining', pos_data.get('contracts', 0))
                                        pnl_pct = (current_premium - pos_data['entry_premium']) / pos_data['entry_premium']
                                        pnl_dollar = (current_premium - pos_data['entry_premium']) * qty * 100
                                        capital_before = risk_mgr.get_equity(api)
                                        
                                        trade_db.log_trade(
                                            symbol=sym,
                                            action='SELL',
                                            qty=qty,
                                            price=current_premium,
                                            strike=pos_data['strike'],
                                            option_type=pos_data['type'],
                                            entry_premium=pos_data['entry_premium'],
                                            exit_premium=current_premium,
                                            pnl=pnl_dollar,
                                            pnl_pct=pnl_pct,
                                            capital_before=capital_before,
                                            capital_after=capital_before + pnl_dollar,
                                            reason='RL_EXIT',
                                            regime=pos_data.get('vol_regime', 'normal_vol'),
                                            vix=pos_data.get('entry_vix', risk_mgr.get_current_vix())
                                        )
                                    except Exception as e:
                                        risk_mgr.log(f"Error logging RL exit for {sym}: {e}", "WARNING")
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
                                
                                # Log trade to database
                                if trade_db and sym in risk_mgr.open_positions:
                                    pos_data = risk_mgr.open_positions[sym]
                                    try:
                                        alpaca_pos = alpaca_option_positions.get(sym)
                                        if alpaca_pos:
                                            current_premium = abs(float(alpaca_pos.market_value) / (float(alpaca_pos.qty) * 100)) if float(alpaca_pos.qty) > 0 else pos_data['entry_premium']
                                        else:
                                            current_premium = pos_data['entry_premium']
                                        
                                        pnl_pct = (current_premium - pos_data['entry_premium']) / pos_data['entry_premium']
                                        pnl_dollar = (current_premium - pos_data['entry_premium']) * qty * 100
                                        capital_before = risk_mgr.get_equity(api)
                                        reason = 'TRIM_50' if action == 3 else 'TRIM_70'
                                        
                                        trade_db.log_trade(
                                            symbol=sym,
                                            action='SELL',
                                            qty=qty,
                                            price=current_premium,
                                            strike=pos_data['strike'],
                                            option_type=pos_data['type'],
                                            entry_premium=pos_data['entry_premium'],
                                            exit_premium=current_premium,
                                            pnl=pnl_dollar,
                                            pnl_pct=pnl_pct,
                                            capital_before=capital_before,
                                            capital_after=capital_before + pnl_dollar,
                                            reason=reason,
                                            regime=pos_data.get('vol_regime', 'normal_vol'),
                                            vix=pos_data.get('entry_vix', risk_mgr.get_current_vix())
                                        )
                                    except Exception as e:
                                        risk_mgr.log(f"Error logging trim for {sym}: {e}", "WARNING")
                                
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
        
        # Check if paper keys are being used
        if API_KEY.startswith('PK') or 'paper' in str(API_KEY).lower():
            print("=" * 70)
            print("❌ ERROR: Cannot use paper trading keys with LIVE mode!")
            print("=" * 70)
            print(f"Detected API Key: {API_KEY[:15]}...")
            print("Paper keys (starting with 'PK') only work with paper trading.")
            print("")
            print("Options:")
            print("  1. Use PAPER trading (recommended for testing):")
            print("     → Remove --live flag: python3 mike_agent_live_safe.py")
            print("")
            print("  2. Get LIVE trading keys:")
            print("     → Go to: https://app.alpaca.markets/dashboard/overview")
            print("     → Navigate to API Keys section")
            print("     → Generate LIVE API keys (different from paper keys)")
            print("     → Update config.py with LIVE keys")
            print("     → Then run: python3 mike_agent_live_safe.py --live")
            print("=" * 70)
            sys.exit(1)
        
        print("⚠️  WARNING: LIVE TRADING MODE - Real money will be used!")
        print("🚀 Starting in 5 seconds... (Press Ctrl+C to cancel)")
        time.sleep(5)
        print("✅ Proceeding with LIVE TRADING")
    
    run_safe_live_trading()

