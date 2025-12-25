"""
Phase 0 Replay Loop - Main Backtest Engine

Replays historical data with Phase 0 logic (frozen model, hard gates, no resampling).
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import pytz

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase0_backtest.engine.gatekeeper import Gatekeeper
from phase0_backtest.engine.risk_book import RiskBook
from phase0_backtest.engine.fill_model import FillModel
from phase0_backtest.engine.option_universe import OptionUniverseFilter

# Import from main agent (for observation preparation and model loading)
try:
    from mike_agent_live_safe import (
        prepare_observation_basic,
        load_rl_model,
        get_market_data,
        get_current_price,
        find_atm_strike,
        get_option_symbol,
        RiskManager
    )
except ImportError as e:
    print(f"Error importing from mike_agent_live_safe: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class Phase0ReplayLoop:
    """
    Phase 0 Replay Loop
    
    Purpose: Test Phase 0 logic on historical data.
    Philosophy: Read-only logic replay - no training, no tuning.
    """
    
    def __init__(self, model_path: str, starting_equity: float = 10000.0, api_key: str = None, api_secret: str = None):
        """
        Initialize replay loop
        
        Args:
            model_path: Path to frozen PPO model
            starting_equity: Starting equity
            api_key: Alpaca API key (for option universe filtering)
            api_secret: Alpaca API secret (for option universe filtering)
        """
        self.model_path = model_path
        self.starting_equity = starting_equity
        
        # Load frozen model (NO TRAINING)
        print(f"Loading frozen model from {model_path}...")
        self.model = load_rl_model()
        print("✅ Model loaded (FROZEN - no training)")
        
        # Initialize components
        self.gatekeeper = Gatekeeper()
        self.risk_book = RiskBook(daily_loss_limit=-250.0, max_trades_per_day=5)
        self.fill_model = FillModel()
        self.risk_mgr = RiskManager()
        
        # Initialize option universe filter (if credentials provided)
        if api_key and api_secret:
            self.option_universe = OptionUniverseFilter(api_key, api_secret)
            print("✅ Option universe filter initialized (real quote filtering)")
        else:
            self.option_universe = None
            print("⚠️  Option universe filter not initialized (no API credentials)")
        
        # Trading symbols (Phase 0: SPY, QQQ only)
        self.trading_symbols = ['SPY', 'QQQ']
        
        # Results tracking
        self.trade_log: List[dict] = []
        self.rejection_log: List[dict] = []
        self.daily_summaries: List[dict] = []
        
        self.est = pytz.timezone('US/Eastern')
    
    def get_trading_days(self, start_date: datetime, end_date: datetime) -> List[date]:
        """Get list of trading days between start and end"""
        trading_days = []
        current = start_date.date()
        end = end_date.date()
        
        while current <= end:
            # Skip weekends (Saturday=5, Sunday=6)
            if current.weekday() < 5:
                trading_days.append(current)
            current += timedelta(days=1)
        
        return trading_days
    
    def get_intraday_minutes(self, trading_date: date) -> List[int]:
        """Get list of intraday minutes for a trading day (9:30 AM - 4:00 PM EST)"""
        minutes = []
        # Market hours: 9:30 AM (930) to 4:00 PM (1600)
        for hour in range(9, 16):
            for minute in [0, 30]:
                if hour == 9 and minute < 30:
                    continue  # Skip before 9:30
                if hour == 16 and minute > 0:
                    break  # Stop at 4:00 PM
                time_int = hour * 100 + minute
                minutes.append(time_int)
        return minutes
    
    def estimate_premium(self, current_price: float, strike: float, option_type: str, vix: float) -> float:
        """
        Estimate option premium (simplified Black-Scholes approximation)
        
        Args:
            current_price: Current underlying price
            strike: Strike price
            option_type: 'call' or 'put'
            vix: Current VIX level
        
        Returns:
            Estimated premium
        """
        # Simplified: intrinsic value + time value estimate
        if option_type.lower() == 'call':
            intrinsic = max(0, current_price - strike)
        else:  # put
            intrinsic = max(0, strike - current_price)
        
        # Time value: rough estimate based on VIX and time to expiry
        days_to_expiry = 1.0 / (252 * 6.5)  # 0DTE: ~1 trading day
        time_value = (vix / 100.0) * current_price * (days_to_expiry ** 0.5) * 0.1
        
        premium = intrinsic + time_value
        return max(0.10, premium)  # Minimum $0.10
    
    def calculate_expected_move(self, current_price: float, vix: float) -> float:
        """Calculate expected move in dollars"""
        days_to_expiry = 1.0 / (252 * 6.5)  # 0DTE
        expected_move_pct = (vix / 16.0) * (days_to_expiry ** 0.5) * 100
        return current_price * (expected_move_pct / 100)
    
    def calculate_breakeven_move(self, current_price: float, strike: float, premium: float, option_type: str) -> float:
        """Calculate breakeven move needed in dollars"""
        if option_type.lower() == 'call':
            breakeven_price = strike + premium
            return breakeven_price - current_price
        else:  # put
            breakeven_price = strike - premium
            return current_price - breakeven_price
    
    def replay_day(self, trading_date: date, api=None) -> dict:
        """
        Replay a single trading day
        
        Args:
            trading_date: Date to replay
            api: Alpaca API instance (optional, for data fetching)
        
        Returns:
            Daily summary
        """
        print(f"\n{'='*80}")
        print(f"📅 REPLAYING DAY: {trading_date}")
        print(f"{'='*80}")
        
        # Reset risk book for new day
        self.risk_book.reset_day(trading_date, self.starting_equity)
        
        # Get intraday minutes
        minutes = self.get_intraday_minutes(trading_date)
        
        # Track decisions for this day
        day_decisions = []
        
        for minute_int in minutes:
            try:
                # Convert minute_int to datetime
                hour = minute_int // 100
                minute = minute_int % 100
                current_time = datetime.combine(trading_date, datetime.min.time().replace(hour=hour, minute=minute))
                current_time = self.est.localize(current_time)
                
                # Check if trading is halted
                can_trade, reason = self.risk_book.can_open_new_trade()
                if not can_trade:
                    if len(day_decisions) == 0 or day_decisions[-1].get('halted') != True:
                        print(f"  ⏸️  Trading halted: {reason}")
                        day_decisions.append({
                            'time': current_time,
                            'action': 'HALTED',
                            'reason': reason
                        })
                    continue
                
                # Process each symbol
                for symbol in self.trading_symbols:
                    # Get market data (last 20 bars)
                    # For backtest, we need historical data for the specific date/time
                    try:
                        # Calculate date range for this minute
                        start_dt = current_time - timedelta(days=2)
                        end_dt = current_time
                        
                        # Try to get data from Alpaca or yfinance (backtest_mode=True for historical data)
                        # Pass current_time so get_market_data uses correct date range
                        hist = get_market_data(
                            symbol,
                            period="2d",
                            interval="1m",
                            api=api,
                            risk_mgr=self.risk_mgr,
                            backtest_mode=True,  # Disable freshness checks for historical data
                            backtest_end_time=current_time  # Use backtest time instead of current time
                        )
                        
                        # Filter to data up to current_time (ensure we don't use future data)
                        if len(hist) > 0:
                            hist = hist[hist.index <= current_time]
                        
                        if len(hist) < 20:
                            continue  # Not enough data
                        
                        # Get current price
                        current_price = float(hist['Close'].iloc[-1])
                        
                        # Get VIX
                        vix = self.risk_mgr.get_current_vix()
                        
                        # Prepare observation
                        obs = prepare_observation_basic(hist, self.risk_mgr, symbol=symbol)
                        
                        # RL inference (FROZEN MODEL - NO TRAINING)
                        action_raw, _ = self.model.predict(obs, deterministic=True)
                        if isinstance(action_raw, np.ndarray):
                            rl_action = int(action_raw.item() if action_raw.ndim == 0 else action_raw[0])
                        else:
                            rl_action = int(action_raw)
                        
                        # Get confidence (temperature-calibrated softmax)
                        try:
                            import torch
                            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                            action_dist = self.model.policy.get_distribution(obs_tensor)
                            
                            if hasattr(action_dist.distribution, 'logits'):
                                logits = action_dist.distribution.logits
                            elif hasattr(action_dist.distribution, 'probs'):
                                probs_raw = action_dist.distribution.probs
                                logits = torch.log(probs_raw + 1e-8)
                            else:
                                raise AttributeError("No logits or probs found")
                            
                            temperature = 0.7
                            probs = torch.softmax(logits / temperature, dim=-1).detach().cpu().numpy()[0]
                            confidence = float(probs[rl_action])
                        except Exception:
                            # Fallback confidence
                            if rl_action in [1, 2]:
                                confidence = 0.65
                            elif rl_action == 0:
                                confidence = 0.50
                            else:
                                confidence = 0.35
                        
                        # Only process BUY actions (1=BUY_CALL, 2=BUY_PUT)
                        if rl_action not in [1, 2]:
                            continue
                        
                        # ========== 🔴 CORRECT PHASE 0 FLOW: OPTION UNIVERSE FILTER FIRST ==========
                        # Step 1: Get tradeable options BEFORE RL selection
                        # RL never fabricates symbols - it only chooses from pre-filtered universe
                        tradeable_options = {}
                        expected_move = self.calculate_expected_move(current_price, vix)
                        
                        if self.option_universe:
                            # Use real option universe filter (ground truth)
                            try:
                                tradeable_options = self.option_universe.get_tradeable_options(
                                    underlying=symbol,
                                    current_price=current_price,
                                    trading_day=trading_date,
                                    expected_move=expected_move
                                )
                            except Exception as e:
                                print(f"  ⚠️  Error fetching tradeable options: {e}")
                                continue
                        else:
                            # Fallback: Generate option symbol manually (no real quotes)
                            # This is less ideal but allows backtest to run without API
                            option_type = 'call' if rl_action == 1 else 'put'
                            strike = find_atm_strike(current_price, option_type=option_type)
                            option_symbol = get_option_symbol(symbol, strike, option_type, trading_day=trading_date)
                            premium = self.estimate_premium(current_price, strike, option_type, vix)
                            spread_estimate = self.fill_model.estimate_spread(premium)
                            
                            # Create synthetic tradeable option entry
                            tradeable_options[option_symbol] = {
                                'bid': premium * 0.90,  # Estimate
                                'ask': premium * 1.10,  # Estimate
                                'mid': premium,
                                'spread': spread_estimate,
                                'spread_pct': (spread_estimate / premium) * 100 if premium > 0 else 20.0,
                                'bid_size': 100,  # Estimate
                                'ask_size': 100,  # Estimate
                                'snapshot': None
                            }
                        
                        # Step 2: If no tradeable options, HOLD (correct behavior)
                        if not tradeable_options:
                            self.rejection_log.append({
                                'date': trading_date,
                                'time': current_time,
                                'symbol': symbol,
                                'reason': "No tradeable options available (filtered by liquidity/spread)",
                                'rl_action': rl_action,
                                'confidence': confidence
                            })
                            continue
                        
                        # Step 3: RL selects from tradeable options (direction only)
                        option_type = 'call' if rl_action == 1 else 'put'
                        best_option = self.option_universe.get_best_strike(
                            tradeable_options,
                            current_price,
                            option_type=option_type
                        ) if self.option_universe else None
                        
                        if not best_option:
                            # Fallback: pick first tradeable option of correct type
                            type_char = 'C' if option_type == 'call' else 'P'
                            matching = {sym: data for sym, data in tradeable_options.items() if sym[6] == type_char}
                            if matching:
                                best_symbol = list(matching.keys())[0]
                                best_option = (best_symbol, matching[best_symbol])
                            else:
                                self.rejection_log.append({
                                    'date': trading_date,
                                    'time': current_time,
                                    'symbol': symbol,
                                    'reason': f"No tradeable {option_type} options available",
                                    'rl_action': rl_action,
                                    'confidence': confidence
                                })
                                continue
                        
                        option_symbol, option_data = best_option
                        
                        # Extract real quote data
                        real_bid = option_data.get('bid')
                        real_ask = option_data.get('ask')
                        real_mid = option_data.get('mid')
                        spread_estimate = option_data.get('spread')
                        spread_pct = option_data.get('spread_pct', 0.0)
                        
                        # Extract strike from option symbol
                        try:
                            strike_str = option_symbol[7:15]
                            strike = float(strike_str) / 1000.0
                        except (ValueError, IndexError):
                            # Fallback: estimate from symbol
                            strike = find_atm_strike(current_price, option_type=option_type)
                        
                        # Use real mid price as premium, or estimate
                        premium = real_mid if real_mid else self.estimate_premium(current_price, strike, option_type, vix)
                        
                        # Calculate breakeven move
                        breakeven_move = self.calculate_breakeven_move(current_price, strike, premium, option_type)
                        
                        # GATEKEEPER CHECK (HARD VETOES) - Now with real quotes
                        is_allowed, gate_reason = self.gatekeeper.allow_trade(
                            action=rl_action,
                            confidence=confidence,
                            symbol=symbol,
                            current_price=current_price,
                            strike=strike,
                            option_type=option_type,
                            premium=premium,
                            spread_estimate=spread_estimate,
                            expected_move=expected_move,
                            breakeven_move=breakeven_move,
                            time_of_day=minute_int,
                            vix=vix
                        )
                        
                        # Log decision (with real quote data)
                        decision = {
                            'time': current_time,
                            'symbol': symbol,
                            'option_symbol': option_symbol,
                            'price': current_price,
                            'rl_action': rl_action,
                            'confidence': confidence,
                            'strike': strike,
                            'premium': premium,
                            'real_bid': real_bid,
                            'real_ask': real_ask,
                            'real_mid': real_mid,
                            'spread_pct': spread_pct,
                            'expected_move': expected_move,
                            'breakeven_move': breakeven_move,
                            'spread_estimate': spread_estimate,
                            'vix': vix,
                            'allowed': is_allowed,
                            'gate_reason': gate_reason,
                            'tradeable_options_count': len(tradeable_options)
                        }
                        day_decisions.append(decision)
                        
                        if not is_allowed:
                            # Trade rejected by gatekeeper
                            self.rejection_log.append({
                                'date': trading_date,
                                'time': current_time,
                                'symbol': symbol,
                                'reason': gate_reason,
                                **decision
                            })
                            continue
                        
                        # Calculate position size (simplified: 1 contract for Phase 0)
                        qty = 1
                        
                        # Check risk book
                        can_open, risk_reason = self.risk_book.can_open_new_trade()
                        if not can_open:
                            self.rejection_log.append({
                                'date': trading_date,
                                'time': current_time,
                                'symbol': symbol,
                                'reason': f"Risk book: {risk_reason}",
                                **decision
                            })
                            continue
                        
                        # Execute entry (using real bid/ask if available)
                        entry_fill = self.fill_model.execute_entry(
                            mid_price=real_mid if real_mid else premium,
                            premium=premium,
                            spread_estimate=spread_estimate,
                            qty=qty,
                            real_bid=real_bid,
                            real_ask=real_ask
                        )
                        
                        # Record trade
                        # option_symbol already determined from universe filter
                        self.risk_book.record_trade(option_symbol, current_price, qty, entry_fill, rl_action)
                        
                        # Log trade
                        trade_entry = {
                            'date': trading_date,
                            'time': current_time,
                            'symbol': symbol,
                            'option_symbol': option_symbol,
                            'action': 'BUY_CALL' if rl_action == 1 else 'BUY_PUT',
                            'strike': strike,
                            'entry_price': current_price,
                            'entry_premium': entry_fill,
                            'qty': qty,
                            'confidence': confidence,
                            'expected_move': expected_move,
                            'breakeven_move': breakeven_move,
                            'vix': vix
                        }
                        self.trade_log.append(trade_entry)
                        
                        # Log trade with real quote info
                        quote_info = ""
                        if real_bid and real_ask:
                            quote_info = f" | Bid: ${real_bid:.2f}, Ask: ${real_ask:.2f}, Spread: {spread_pct:.1f}%"
                        
                        print(f"  ✅ TRADE: {symbol} {option_type.upper()} @ ${strike:.2f} | "
                              f"Premium: ${entry_fill:.2f} | Confidence: {confidence:.3f}{quote_info} | "
                              f"Time: {current_time.strftime('%H:%M')} | "
                              f"Tradeable options: {len(tradeable_options)}")
                        
                        # Simulate position lifecycle (simplified: close at end of day or stop-loss)
                        # For Phase 0, we'll close at end of day or if stop-loss hit
                        # This is handled in the position monitoring loop
                        
                    except Exception as e:
                        print(f"  ⚠️  Error processing {symbol} at {current_time}: {e}")
                        continue
                
                # Update unrealized PnL for open positions
                for pos_symbol in list(self.risk_book.open_positions.keys()):
                    # Simplified: assume premium decays linearly (for Phase 0)
                    # In reality, this would use actual option pricing
                    pos = self.risk_book.open_positions[pos_symbol]
                    # Estimate current premium (simplified)
                    current_premium_est = pos['entry_premium'] * 0.95  # 5% decay estimate
                    self.risk_book.update_unrealized_pnl(pos_symbol, current_premium_est)
                
                # Check daily loss limit
                if self.risk_book.daily_loss_exceeded():
                    print(f"  🚨 DAILY LOSS LIMIT EXCEEDED - Trading halted")
                    break
                
            except Exception as e:
                print(f"  ⚠️  Error at minute {minute_int}: {e}")
                continue
        
        # Finalize day
        daily_summary = self.risk_book.finalize_day()
        daily_summary['decisions'] = day_decisions
        self.daily_summaries.append(daily_summary)
        
        print(f"\n  📊 Day Summary:")
        print(f"     Trades: {daily_summary['trades_taken']}")
        print(f"     P&L: ${daily_summary['total_pnl']:.2f}")
        print(f"     Halted: {daily_summary['trading_halted']}")
        
        return daily_summary
    
    def run_backtest(self, start_date: datetime, end_date: datetime, api=None) -> dict:
        """
        Run Phase 0 backtest
        
        Args:
            start_date: Start date
            end_date: End date
            api: Alpaca API instance (optional)
        
        Returns:
            Complete backtest results
        """
        print(f"\n{'='*80}")
        print(f"🚀 PHASE 0 BACKTEST")
        print(f"{'='*80}")
        print(f"Start: {start_date.date()}")
        print(f"End: {end_date.date()}")
        print(f"Model: {self.model_path} (FROZEN)")
        print(f"Starting Equity: ${self.starting_equity:,.2f}")
        print(f"{'='*80}\n")
        
        # Get trading days
        trading_days = self.get_trading_days(start_date, end_date)
        print(f"📅 Trading days: {len(trading_days)}")
        
        # Replay each day
        for trading_date in trading_days:
            try:
                self.replay_day(trading_date, api=api)
            except Exception as e:
                print(f"❌ Error replaying {trading_date}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Compile results
        results = {
            'start_date': start_date.date(),
            'end_date': end_date.date(),
            'trading_days': len(trading_days),
            'total_trades': len(self.trade_log),
            'total_rejections': len(self.rejection_log),
            'daily_summaries': self.daily_summaries,
            'trade_log': self.trade_log,
            'rejection_log': self.rejection_log,
            'gatekeeper_summary': self.gatekeeper.get_gate_summary()
        }
        
        return results

