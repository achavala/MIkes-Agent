#!/usr/bin/env python3
"""
30-DAY BACKTEST RUNNER
Institutional-grade backtest with comprehensive logging
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Try to import yfinance (optional - may have dependency conflicts)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError as e:
    YFINANCE_AVAILABLE = False
    print(f"⚠️ Warning: yfinance not available ({e})")
    print("   Will use alternative data source or existing backtest infrastructure")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from institutional_logging import initialize_logger, get_logger
from realistic_fill_modeling import calculate_realistic_fill
from online_learning_system import initialize_online_learning, get_online_learning_system
from multi_agent_ensemble import initialize_meta_router, get_meta_router
from log_compression import compress_daily_logs
from weekly_review_system import initialize_review_system, get_review_system
from end_of_run_verdict import initialize_verdict_system, get_verdict_system
from data_provider_router import initialize_data_router, get_data_router, DataProvider
from behavioral_profile import get_behavioral_profile, apply_behavioral_overrides
from trade_block_aggregator import initialize_block_aggregator, get_block_aggregator

# Import agent (will need to be adapted)
try:
    from mike_agent import MikeAgent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    print("Warning: mike_agent not available")


class InstitutionalBacktest:
    """
    30-day backtest with institutional-grade logging
    """
    
    def __init__(
        self,
        symbols: list = ['SPY', 'QQQ', 'SPX'],
        capital: float = 100000.0,
        mode: str = 'behavioral',  # 'behavioral', 'pnl', or 'paper'
        log_dir: str = "logs"
    ):
        self.symbols = symbols
        self.capital = capital
        self.mode = mode
        self.current_capital = capital
        
        # Store log_dir as instance attribute (Fix Option A)
        self.log_dir = log_dir
        
        # Initialize logging
        self.logger = initialize_logger(log_dir=log_dir)
        
        # Initialize online learning
        self.learning_system = initialize_online_learning(
            model_dir="models",
            rolling_window_days=30,
            min_retrain_interval_hours=20
        )
        
        # Initialize meta router
        self.meta_router = initialize_meta_router()
        
        # Initialize review systems
        self.review_system = initialize_review_system(log_dir=log_dir)
        self.verdict_system = initialize_verdict_system(log_dir=log_dir)
        
        # Initialize data provider router (institutional mode)
        self.data_router = initialize_data_router(institutional_mode=True)
        
        # Initialize trade block aggregator
        self.block_aggregator = initialize_block_aggregator(log_dir=log_dir)
        
        # Load profile based on mode
        self.behavioral_profile = None
        if mode == 'behavioral':
            from behavioral_profile import get_behavioral_profile
            self.behavioral_profile = get_behavioral_profile()
            print(f"📋 Behavioral mode: Relaxation profile loaded")
            print(f"   - Gamma limit: {self.behavioral_profile['risk_manager']['gamma_limit_multiplier']}x")
            print(f"   - Min agent agreement: {self.behavioral_profile['ensemble']['min_agent_agreement']}")
            print(f"   - IV crush penalty: {'OFF' if not self.behavioral_profile['execution']['apply_iv_crush'] else 'ON'}")
            signal_floor = self.behavioral_profile.get("signal_floor", {})
            if signal_floor.get("enabled", False):
                print(f"   - Signal floor: ENABLED (RL≥{signal_floor.get('rl_confidence_min', 0.52):.2f}, Ensemble≥{signal_floor.get('ensemble_confidence_min', 0.50):.2f})")
            action_nudge = self.behavioral_profile.get("action_nudge", {})
            if action_nudge.get("enabled", False):
                print(f"   - Action nudge: ENABLED (threshold={action_nudge.get('rl_action_raw_threshold', 0.15):.2f}, probe_trades={'ON' if action_nudge.get('force_probe_trade', False) else 'OFF'})")
        
        # Track backtest start date
        self.backtest_start_date = None
        
        # Track trades per day for minimum trade expectation
        self.trades_per_day: Dict[str, int] = {}
        
        # Store historical data for each symbol (for ensemble lookback)
        self.historical_data: Dict[str, pd.DataFrame] = {}
        
        # Track positions
        self.positions = {}
        self.trade_counter = {}
        for symbol in symbols:
            self.trade_counter[symbol] = 0
        
        # Statistics
        self.stats = {
            'total_decisions': 0,
            'total_trades': 0,
            'total_pnl': 0.0,
            'winning_trades': 0,
            'losing_trades': 0
        }
    
    def run_backtest(
        self,
        start_date: str,
        end_date: str
    ) -> Dict:
        """
        Run 30-day backtest
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Backtest results
        """
        print("="*70)
        print("  30-DAY INSTITUTIONAL BACKTEST")
        print("="*70)
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Mode: {self.mode.upper()}")
        print(f"Capital: ${self.capital:,.2f}")
        print(f"Date Range: {start_date} to {end_date}")
        print("="*70)
        
        # Load data for all symbols
        all_data = {}
        
        # Use institutional data provider router
        print("\n📊 Using Institutional Data Provider Router")
        print("   Priority: Massive > Alpaca > Polygon > yfinance")
        print("   Institutional Mode: ENABLED (yfinance blocked)")
        
        for symbol in self.symbols:
            print(f"\nLoading data for {symbol}...")
            try:
                # Use data router with priority-based fallback
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(end_date, "%Y-%m-%d")
                
                all_chunks = []
                current_start = start
                chunk_days = 7  # Use 7 days per chunk for providers with limits
                
                while current_start < end:
                    current_end = min(current_start + timedelta(days=chunk_days), end)
                    chunk_start_str = current_start.strftime("%Y-%m-%d")
                    chunk_end_str = current_end.strftime("%Y-%m-%d")
                    
                    print(f"  Loading chunk: {chunk_start_str} to {chunk_end_str}...")
                    
                    # Use data router
                    chunk_data, provider_used, fallback_count = self.data_router.fetch_data(
                        symbol=symbol,
                        data_type="minute_bars",
                        start_date=chunk_start_str,
                        end_date=chunk_end_str
                    )
                    
                    if chunk_data is not None and len(chunk_data) > 0:
                        # Clean and format
                        if isinstance(chunk_data, pd.DataFrame):
                            chunk_data.columns = [col.lower() if isinstance(col, str) else col for col in chunk_data.columns]
                            if isinstance(chunk_data.columns, pd.MultiIndex):
                                chunk_data = chunk_data.xs(symbol, axis=1, level=1)
                            
                            all_chunks.append(chunk_data)
                            print(f"    ✅ Loaded {len(chunk_data)} bars from {provider_used.value}")
                            if fallback_count > 0:
                                print(f"    ⚠️ Used {fallback_count} fallback(s)")
                        else:
                            print(f"    ⚠️ Invalid data format from {provider_used.value}")
                    else:
                        print(f"    ⚠️ No data for this chunk")
                    
                    current_start = current_end
                
                if all_chunks:
                    # Combine all chunks
                    data = pd.concat(all_chunks, axis=0)
                    data = data.sort_index()
                    data = data[~data.index.duplicated(keep='first')]  # Remove duplicates
                    all_data[symbol] = data
                    print(f"  ✅ Total loaded: {len(data)} bars")
                else:
                    print(f"  ⚠️ No data loaded for {symbol}")
                    continue
                    
            except Exception as e:
                print(f"  ❌ Error loading {symbol}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_data:
            print("\n❌ No data available for any symbol")
            return {}
        
        # Run backtest
        print("\n" + "="*70)
        print("  RUNNING BACKTEST...")
        print("="*70)
        
        # Store start date for reviews
        self.backtest_start_date = start_date
        
        # Process each trading day
        trading_days = pd.bdate_range(start=start_date, end=end_date)
        day_number = 0
        
        for day in trading_days:
            day_number += 1
            day_str = day.strftime("%Y-%m-%d")
            print(f"\n📅 Processing {day_str}...")
            
            # Process each symbol
            for symbol in self.symbols:
                if symbol not in all_data:
                    continue
                
                # Get day's data
                day_data = all_data[symbol][all_data[symbol].index.date == day.date()]
                if len(day_data) == 0:
                    continue
                
                # Initialize historical data for this symbol if not exists
                if symbol not in self.historical_data:
                    self.historical_data[symbol] = pd.DataFrame()
                
                # Process each bar
                for idx, (timestamp, row) in enumerate(day_data.iterrows()):
                    # Update historical data (keep last 50 bars for ensemble lookback)
                    bar_df = pd.DataFrame({
                        'close': [row.get('close', row.get('Close', 0))],
                        'open': [row.get('open', row.get('Open', row.get('close', row.get('Close', 0))))],
                        'high': [row.get('high', row.get('High', row.get('close', row.get('Close', 0))))],
                        'low': [row.get('low', row.get('Low', row.get('close', row.get('Close', 0))))],
                        'volume': [row.get('volume', row.get('Volume', 0))]
                    }, index=[timestamp])
                    
                    # Append to historical data
                    self.historical_data[symbol] = pd.concat([self.historical_data[symbol], bar_df])
                    # Keep only last 50 bars to avoid memory issues
                    if len(self.historical_data[symbol]) > 50:
                        self.historical_data[symbol] = self.historical_data[symbol].tail(50)
                    
                    self._process_bar(symbol, timestamp, row, day_str)
            
            # End of day: flush buffers, check for retraining
            self.logger.flush_buffers()
            self._check_daily_retraining(day_str)
            
            # Check minimum trade expectation
            trades_today = self.trades_per_day.get(day_str, 0)
            if trades_today == 0:
                # Check if market was active (had decisions)
                decisions_today = len([d for d in self.logger.decision_buffer if d.get('timestamp', '')[:10] == day_str])
                if decisions_today > 0:
                    print(f"   ⚠️ No trades today ({decisions_today} decisions) — possible over-constrained policy")
                    if self.logger:
                        self.logger.log_feedback(
                            date=day_str,
                            reviewer="SYSTEM",
                            comment=f"No trades today despite {decisions_today} decisions — possible over-constrained policy",
                            severity="MEDIUM",
                            category="BEHAVIOR"
                        )
            
            # Print daily block summary
            self.block_aggregator.print_daily_summary(day_str)
            
            # Weekly review checkpoints (days 5, 10, 20, 30)
            if day_number in [5, 10, 20, 30]:
                print(f"\n📊 Conducting Day {day_number} Review...")
                try:
                    review = self.review_system.conduct_review(day_number, start_date)
                    answers = review.get('answers', {})
                    if answers:
                        print(f"   ✅ Review complete: {len(answers)} metrics computed")
                    else:
                        print(f"   ⚠️ Review complete: No metrics computed (insufficient data)")
                except Exception as e:
                    print(f"   ⚠️ Review failed (non-critical): {e}")
                    # Continue - review failure shouldn't stop backtest
        
        # Compress logs (recommendation: handle log volume)
        # Wrapped defensively - post-run cleanup must never invalidate results
        print("\n📦 Compressing daily logs...")
        try:
            compress_daily_logs(log_dir=self.log_dir)
            print("   ✅ Log compression complete")
        except Exception as e:
            print(f"   ⚠️ Log compression failed (non-critical): {e}")
            # Log error but don't fail the backtest
            if self.logger:
                self.logger.log_feedback(
                    date=end_date,
                    reviewer="SYSTEM",
                    comment=f"Log compression failed: {e}",
                    severity="LOW",
                    category="EXECUTION"
                )
        
        # Generate final report
        results = self._generate_report()
        
        # Print period block summary
        self.block_aggregator.print_period_summary(start_date, end_date)
        self.block_aggregator.save_summary(start_date, end_date)
        
        # Check minimum trade expectation for verdict
        total_trades = sum(self.trades_per_day.values())
        trading_days = len([d for d in pd.bdate_range(start=start_date, end=end_date)])
        avg_trades_per_day = total_trades / trading_days if trading_days > 0 else 0
        
        print(f"\n📊 Trade Activity Summary:")
        print(f"   Total trades: {total_trades}")
        print(f"   Trading days: {trading_days}")
        print(f"   Avg trades/day: {avg_trades_per_day:.2f}")
        
        # Minimum trade expectation check
        if total_trades == 0:
            print(f"   ⚠️ ZERO TRADES — Automatic REJECT (over-constrained policy)")
            if self.logger:
                self.logger.log_feedback(
                    date=end_date,
                    reviewer="SYSTEM",
                    comment=f"Zero trades over {trading_days} days — over-constrained policy detected",
                    severity="HIGH",
                    category="BEHAVIOR"
                )
        elif avg_trades_per_day < 0.1:
            print(f"   ⚠️ Very low trade activity — possible over-constraint")
        elif 0.1 <= avg_trades_per_day <= 3.0:
            print(f"   ✅ Acceptable trade activity")
        elif 3.0 < avg_trades_per_day <= 10.0:
            print(f"   ✅ Ideal trade activity")
        else:
            print(f"   ⚠️ High trade activity — possible overtrading")
        
        # Generate end-of-run verdict
        print("\n📋 Generating End-of-Run Verdict...")
        # CRITICAL: Pass total_trades from backtest stats to verdict system
        # This ensures verdict uses actual trade count, not empty log lists
        verdict = self.verdict_system.generate_verdict(
            start_date, 
            end_date, 
            mode=self.mode,
            total_trades_override=total_trades  # Pass actual trade count from stats
        )
        
        # Override verdict if zero trades
        if total_trades == 0:
            verdict['recommendation'] = {
                'decision': 'REJECT',
                'reason': 'Zero trades over entire period — over-constrained policy',
                'next_steps': [
                    'Review trade block summary to identify dominant constraint',
                    'Apply behavioral profile relaxation',
                    'Re-run with relaxed constraints'
                ]
            }
            print(f"\n   ⚠️ Verdict overridden: REJECT (zero trades)")
        
        # Get data provider usage summary
        provider_stats = self.data_router.get_provider_stats()
        
        print("\n" + "="*70)
        print("  BACKTEST COMPLETE")
        print("="*70)
        print(f"\n🎯 FINAL VERDICT: {verdict.get('recommendation', {}).get('decision', 'UNKNOWN')}")
        print(f"   Reason: {verdict.get('recommendation', {}).get('reason', 'N/A')}")
        print(f"\n📊 Scorecards:")
        print(f"   Behavior: {verdict.get('overall_scores', {}).get('behavior', 0):.2f}")
        print(f"   Risk: {verdict.get('overall_scores', {}).get('risk', 0):.2f}")
        print(f"   Execution: {verdict.get('overall_scores', {}).get('execution', 0):.2f}")
        print(f"   Learning: {verdict.get('overall_scores', {}).get('learning', 0):.2f}")
        print(f"   Average: {verdict.get('overall_scores', {}).get('average', 0):.2f}")
        
        print(f"\n📡 Data Provider Usage:")
        percentages = provider_stats.get('percentages', {})
        print(f"   Massive: {percentages.get('massive', 0):.1f}%")
        print(f"   Alpaca: {percentages.get('alpaca', 0):.1f}%")
        print(f"   Polygon: {percentages.get('polygon', 0):.1f}%")
        yfinance_pct = percentages.get('yfinance', 0)
        yfinance_flag = provider_stats.get('yfinance_red_flag', False)
        flag_text = '🔴 RED FLAG' if yfinance_flag else '✅ OK'
        print(f"   yfinance: {yfinance_pct:.1f}% {flag_text}")
        
        results['verdict'] = verdict
        results['provider_stats'] = provider_stats
        results['trade_activity'] = {
            'total_trades': total_trades,
            'trading_days': trading_days,
            'avg_trades_per_day': avg_trades_per_day
        }
        
        return results
    
    def _process_bar(self, symbol: str, timestamp: datetime, bar: pd.Series, date_str: str):
        """Process a single bar"""
        price = bar.get('close', bar.get('Close', 0))
        if price <= 0:
            return
        
        # Calculate time to expiry (0DTE = ~6.5 hours from 9:30 AM)
        # Normalize to same timezone as timestamp
        if hasattr(timestamp, 'tz') and timestamp.tz is not None:
            # Timestamp is tz-aware (pandas Timestamp)
            market_open = timestamp.normalize().replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = timestamp.normalize().replace(hour=16, minute=0, second=0, microsecond=0)
        else:
            # Timestamp is tz-naive (datetime)
            market_open = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = timestamp.replace(hour=16, minute=0, second=0, microsecond=0)
        time_to_expiry_hours = (market_close - timestamp).total_seconds() / 3600
        time_to_expiry_min = int(time_to_expiry_hours * 60)
        
        # Detect regime (simplified - use VIX if available)
        vix = 20.0  # Default, would get from data
        regime = self._detect_regime(vix)
        
        # Get RL action (simulated for now)
        # In real implementation, this would come from the trained PPO model
        # For now, simulate with some randomness to test signal generation
        import numpy as np
        rl_action_raw = np.random.uniform(-0.2, 0.2)  # Simulated raw action (would come from PPO)
        rl_confidence = abs(rl_action_raw) * 2.5  # Convert to confidence (0-0.5 range)
        rl_confidence = min(rl_confidence, 0.95)  # Cap at 0.95
        
        # Map raw action to discrete action
        # Standard threshold would be ~0.35, but we'll apply behavioral nudge
        if self.behavioral_profile and self.behavioral_profile.get("action_nudge", {}).get("enabled", False):
            nudge_threshold = self.behavioral_profile["action_nudge"]["rl_action_raw_threshold"]
            if rl_action_raw >= nudge_threshold:
                rl_action = 1  # BUY_CALL
            elif rl_action_raw <= -nudge_threshold:
                rl_action = 2  # BUY_PUT
            else:
                rl_action = 0  # HOLD
        else:
            # Standard threshold (higher)
            if rl_action_raw >= 0.35:
                rl_action = 1  # BUY_CALL
            elif rl_action_raw <= -0.35:
                rl_action = 2  # BUY_PUT
            else:
                rl_action = 0  # HOLD
        
        # Get ensemble action
        try:
            # Prepare data for ensemble - use historical data (last 50 bars)
            # This ensures agents have enough lookback data (they need 20+ bars)
            if symbol in self.historical_data and len(self.historical_data[symbol]) > 0:
                ensemble_data = self.historical_data[symbol].copy()
                # Ensure column names are lowercase
                ensemble_data.columns = [col.lower() if isinstance(col, str) else col for col in ensemble_data.columns]
            else:
                # Fallback: create minimal DataFrame if no history yet
                ensemble_data = pd.DataFrame({
                    'close': [price],
                    'open': [bar.get('open', price)],
                    'high': [bar.get('high', price)],
                    'low': [bar.get('low', price)],
                    'volume': [bar.get('volume', 0)]
                })
            
            ensemble_action, ensemble_confidence, ensemble_details = self.meta_router.route(
                data=ensemble_data,
                vix=vix,
                symbol=symbol,
                current_price=price,
                strike=round(price),
                portfolio_delta=0.0,  # Would get from portfolio
                delta_limit=2000.0
            )
            
            # Convert action to string
            action_map = {0: "HOLD", 1: "BUY_CALL", 2: "BUY_PUT"}
            ensemble_action_str = action_map.get(ensemble_action, "HOLD")
            rl_action_str = action_map.get(0, "HOLD")  # Simulated
            
            # Combine RL + Ensemble (60% ensemble, 40% RL)
            # Apply behavioral signal floor if in behavioral mode
            if self.behavioral_profile and self.behavioral_profile.get("signal_floor", {}).get("enabled", False):
                signal_floor = self.behavioral_profile["signal_floor"]
                rl_min = signal_floor.get("rl_confidence_min", 0.52)
                ensemble_min = signal_floor.get("ensemble_confidence_min", 0.50)
                
                # Behavioral signal floor: allow weak-but-consistent signals
                if rl_confidence >= rl_min and ensemble_confidence >= ensemble_min:
                    # Both meet minimum thresholds - allow trade
                    if ensemble_action_str != "HOLD":
                        final_action = ensemble_action_str
                        final_confidence = ensemble_confidence * 0.6 + rl_confidence * 0.4
                    elif rl_action_str != "HOLD":
                        final_action = rl_action_str
                        final_confidence = rl_confidence
                    else:
                        final_action = "HOLD"
                        final_confidence = min(rl_confidence, ensemble_confidence)
                elif ensemble_confidence >= 0.3:
                    # Standard logic if signal floor not met
                    # Changed to >= 0.3 to allow signals at exactly 0.3 threshold
                    final_action = ensemble_action_str
                    final_confidence = ensemble_confidence * 0.6 + rl_confidence * 0.4
                else:
                    final_action = rl_action_str
                    final_confidence = rl_confidence
            else:
                # Standard logic (non-behavioral mode, including paper mode)
                # Changed to >= 0.3 to allow signals at exactly 0.3 threshold
                if ensemble_confidence >= 0.3:
                    final_action = ensemble_action_str
                    final_confidence = ensemble_confidence * 0.6 + rl_confidence * 0.4
                else:
                    final_action = rl_action_str
                    final_confidence = rl_confidence
            
            # Get agent votes
            agent_votes = {}
            signals = ensemble_details.get('signals', {})
            for agent_name, signal_info in signals.items():
                agent_action = signal_info.get('action', 0)
                agent_votes[agent_name] = action_map.get(agent_action, "HOLD")
            
        except Exception as e:
            print(f"  ⚠️ Ensemble error: {e}")
            final_action = "HOLD"
            final_confidence = 0.0
            agent_votes = {}
            ensemble_action_str = "HOLD"
            ensemble_confidence = 0.0
        
        # Check for probe trade if no signal proposed and in behavioral mode
        probe_trade_triggered = False
        if self.behavioral_profile and final_action == "HOLD":
            action_nudge = self.behavioral_profile.get("action_nudge", {})
            if action_nudge.get("force_probe_trade", False):
                # Check if we should force a probe trade (1 per day max)
                trades_today = self.trades_per_day.get(date_str, 0)
                if trades_today == 0:
                    # Force a small probe trade to observe behavior
                    # Use weak directional signal from RL
                    if abs(rl_action_raw) > 0.05:  # Even weaker threshold for probe
                        if rl_action_raw > 0:
                            final_action = "BUY_CALL"
                            final_confidence = 0.3  # Low confidence probe
                            probe_trade_triggered = True
                        else:
                            final_action = "BUY_PUT"
                            final_confidence = 0.3  # Low confidence probe
                            probe_trade_triggered = True
        
        # Store metadata for trade execution and logging
        # Always store RL metadata (for both behavioral and paper modes)
        decision_metadata = {
            "rl_action_raw": float(rl_action_raw),
            "rl_confidence": float(rl_confidence),
        }
        if self.behavioral_profile:
            decision_metadata.update({
                "probe_trade": probe_trade_triggered,
                "signal_floor_applied": self.behavioral_profile.get("signal_floor", {}).get("enabled", False)
            })
        
        # Log no-signal diagnostics if no trade proposed
        if final_action == "HOLD" and not probe_trade_triggered:
            no_trade_reason = "NO_SIGNAL_PROPOSED"
            decision_metadata["no_trade_reason"] = no_trade_reason
            decision_metadata["ensemble_votes"] = agent_votes
        
        # Store metadata for trade execution
        self._last_decision_metadata = decision_metadata
        
        # Log decision
        self.logger.log_decision(
            timestamp=timestamp,
            symbol=symbol,
            price=price,
            regime=regime,
            time_to_expiry_min=time_to_expiry_min,
            action_final=final_action,
            confidence_final=final_confidence,
            rl_action=rl_action_str,
            rl_confidence=rl_confidence,
            ensemble_action=ensemble_action_str,
            ensemble_confidence=ensemble_confidence,
            agent_votes=agent_votes,
            metadata=decision_metadata
        )
        
        self.stats['total_decisions'] += 1
        
        # Risk check with behavioral profile overrides
        portfolio_delta = 0.0  # Would get from portfolio
        portfolio_gamma = 0.0
        portfolio_theta = 0.0
        portfolio_vega = 0.0
        
        # Apply behavioral profile overrides if in behavioral mode
        gamma_limit = 0.025
        delta_limit = 2000.0
        
        if self.behavioral_profile:
            gamma_limit *= self.behavioral_profile["risk_manager"]["gamma_limit_multiplier"]
            delta_limit *= self.behavioral_profile["risk_manager"]["delta_limit_multiplier"]
        
        risk_action = "ALLOW"
        risk_reason = None
        
        if final_action != "HOLD":
            # Check risk limits (with behavioral overrides if applicable)
            if portfolio_gamma > gamma_limit:
                risk_action = "BLOCK"
                risk_reason = "GAMMA_LIMIT_EXCEEDED"
            elif portfolio_delta > delta_limit:
                risk_action = "BLOCK"
                risk_reason = "DELTA_LIMIT_EXCEEDED"
            # Check ensemble agreement (mode-aware)
            if self.behavioral_profile:
                # Use profile's min_agent_agreement (behavioral=1, paper=2)
                min_agreement = self.behavioral_profile["ensemble"]["min_agent_agreement"]
                buy_votes = sum(1 for v in agent_votes.values() if 'BUY' in str(v))
                # Block if agreement requirement not met
                # Behavioral: min_agreement=1 (only blocks if 0 agents agree)
                # Paper: min_agreement=2 (requires 2+ agents)
                if buy_votes < min_agreement:
                    risk_action = "BLOCK"
                    risk_reason = "ENSEMBLE_DISAGREEMENT"
            # If no profile (standard mode), use default ensemble agreement check
            elif len(agent_votes) > 0:
                buy_votes = sum(1 for v in agent_votes.values() if 'BUY' in str(v))
                if buy_votes < 2:  # Standard: require at least 2 agents
                    risk_action = "BLOCK"
                    risk_reason = "ENSEMBLE_DISAGREEMENT"
            # Check macro agent (RISK-OFF blocks)
            if "macro" in agent_votes and agent_votes.get("macro") == "RISK_OFF":
                risk_action = "BLOCK"
                risk_reason = "MACRO_RISK_OFF"
        
        self.logger.log_risk_check(
            timestamp=timestamp,
            symbol=symbol,
            portfolio_delta=portfolio_delta,
            portfolio_gamma=portfolio_gamma,
            portfolio_theta=portfolio_theta,
            portfolio_vega=portfolio_vega,
            gamma_limit=gamma_limit,
            delta_limit=delta_limit,
            risk_action=risk_action,
            risk_reason=risk_reason
        )
        
        # Log block reason if blocked
        if risk_action == "BLOCK" and risk_reason:
            self.block_aggregator.log_block(date_str, risk_reason, symbol)
        
        # Execute trade if allowed
        if final_action != "HOLD" and risk_action == "ALLOW":
            # Check if this is a probe trade (smaller size)
            is_probe_trade = False
            size_multiplier = 1.0
            if self.behavioral_profile:
                action_nudge = self.behavioral_profile.get("action_nudge", {})
                # Check metadata for probe trade flag
                if hasattr(self, '_last_decision_metadata') and self._last_decision_metadata.get("probe_trade", False):
                    is_probe_trade = True
                    size_multiplier = action_nudge.get("probe_trade_size", 0.1)
            
            self._execute_trade(symbol, timestamp, price, final_action, date_str, size_multiplier=size_multiplier, is_probe_trade=is_probe_trade)
            
            # Track trades per day
            if date_str not in self.trades_per_day:
                self.trades_per_day[date_str] = 0
            self.trades_per_day[date_str] += 1
    
    def _execute_trade(self, symbol: str, timestamp: datetime, price: float, action: str, date_str: str, size_multiplier: float = 1.0, is_probe_trade: bool = False):
        """Execute a trade with realistic fill modeling"""
        # Generate trade ID
        self.trade_counter[symbol] += 1
        trade_id = f"{symbol}_{date_str}_{self.trade_counter[symbol]:03d}"
        if is_probe_trade:
            trade_id += "_PROBE"
        
        # Calculate realistic fill
        mid = price * 0.05  # Estimate option premium (5% of underlying)
        bid = mid * 0.98
        ask = mid * 1.02
        spread = ask - bid
        
        # Calculate time to expiry (normalize to same timezone)
        # Option A: Make expiry tz-aware in same timezone as timestamp
        if hasattr(timestamp, 'tz') and timestamp.tz is not None:
            # Timestamp is tz-aware (pandas Timestamp)
            expiry = timestamp.normalize().replace(hour=16, minute=0, second=0, microsecond=0)
        else:
            # Timestamp is tz-naive (datetime)
            expiry = timestamp.replace(hour=16, minute=0, second=0, microsecond=0)
        time_to_expiry = (expiry - timestamp).total_seconds() / 3600
        
        # Apply behavioral profile execution overrides
        apply_iv_crush = True
        apply_theta_penalty = True
        slippage_multiplier = 1.0
        
        if self.behavioral_profile:
            apply_iv_crush = self.behavioral_profile["execution"]["apply_iv_crush"]
            apply_theta_penalty = self.behavioral_profile["execution"]["apply_theta_penalty"]
            slippage_multiplier = self.behavioral_profile["execution"]["slippage_multiplier"]
        
        # Apply size multiplier for probe trades
        qty = max(1, int(1 * size_multiplier))  # At least 1 contract, scaled by multiplier
        
        fill_price, fill_details = calculate_realistic_fill(
            mid=mid,
            bid=bid,
            ask=ask,
            qty=qty,
            side='buy' if 'BUY' in action else 'sell',
            time_to_expiry=time_to_expiry,
            vix=20.0,
            volume=1000000,
            has_news=False,
            gamma_exposure=0.0,
            hidden_liquidity_pct=0.1
        )
        
        # Apply slippage multiplier if in behavioral mode
        if self.behavioral_profile and slippage_multiplier != 1.0:
            original_slippage = fill_details.get('slippage_pct', 0)
            fill_details['slippage_pct'] = original_slippage * slippage_multiplier
            # Adjust fill price accordingly
            price_adjustment = (fill_price - mid) * (1 - slippage_multiplier)
            fill_price = mid + price_adjustment
        
        # Apply IV crush and theta penalties based on profile
        if not apply_iv_crush and 'iv_crush_impact' in fill_details:
            fill_details['iv_crush_impact'] = 0.0
        if not apply_theta_penalty and 'theta_impact' in fill_details:
            fill_details['theta_impact'] = 0.0
        
        # Log execution
        self.logger.log_execution(
            timestamp=timestamp,
            symbol=symbol,
            order_type=action,
            mid_price=mid,
            fill_price=fill_price,
            spread=spread,
            slippage_pct=fill_details.get('slippage_pct', 0),
            qty=qty,
            gamma_impact=fill_details.get('gamma_impact'),
            iv_crush_impact=fill_details.get('iv_crush_impact'),
            theta_impact=fill_details.get('theta_impact'),
            liquidity_factor=fill_details.get('liquidity_factor')
        )
        
        # Log position entry
        self.logger.log_position_entry(
            trade_id=trade_id,
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            qty=qty,
            entry_price=price,
            strike=round(price),
            premium=fill_price,
            is_probe_trade=is_probe_trade
        )
        
        # Store position
        self.positions[trade_id] = {
            'entry_time': timestamp,
            'entry_price': price,
            'entry_premium': fill_price,
            'symbol': symbol,
            'action': action
        }
        
        self.stats['total_trades'] += 1
    
    def _detect_regime(self, vix: float) -> str:
        """Detect market regime from VIX"""
        if vix < 18:
            return "mean_reverting"
        elif vix < 25:
            return "neutral"
        elif vix < 35:
            return "volatile"
        else:
            return "trending"
    
    def _check_daily_retraining(self, date_str: str):
        """Check if model should be retrained"""
        try:
            regime = "neutral"  # Would detect from data
            should_retrain, reason = self.learning_system.should_retrain(regime)
            
            if should_retrain:
                # Get rolling window data (simplified)
                training_data = pd.DataFrame({'close': np.random.randn(100) + 500})
                
                version_id = self.learning_system.retrain_model(
                    training_data=training_data,
                    current_regime=regime,
                    training_config={'epochs': 100}
                )
                
                self.logger.log_learning_event(
                    date=date_str,
                    regime=regime,
                    retrained=True,
                    model_candidate=version_id,
                    production_model=self.learning_system.current_production_version,
                    retrain_reason=reason
                )
            else:
                self.logger.log_learning_event(
                    date=date_str,
                    regime=regime,
                    retrained=False,
                    retrain_reason=reason
                )
        except Exception as e:
            print(f"  ⚠️ Retraining check error: {e}")
    
    def _generate_report(self) -> Dict:
        """Generate backtest report"""
        return {
            'stats': self.stats,
            'final_capital': self.current_capital,
            'total_return_pct': ((self.current_capital - self.capital) / self.capital * 100) if self.capital > 0 else 0.0
        }


def main():
    """Run 30-day backtest"""
    # Calculate date range (last 30 trading days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=45)  # 45 calendar days to get ~30 trading days
    
    backtest = InstitutionalBacktest(
        symbols=['SPY'],  # Start with SPY for testing
        capital=100000.0,
        mode='behavioral',
        log_dir="logs"
    )
    
    results = backtest.run_backtest(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    
    print("\n📊 Backtest Results:")
    print(f"  Total Decisions: {results.get('stats', {}).get('total_decisions', 0)}")
    print(f"  Total Trades: {results.get('stats', {}).get('total_trades', 0)}")
    print(f"  Final Capital: ${results.get('final_capital', 0):,.2f}")
    print(f"  Total Return: {results.get('total_return_pct', 0):.2f}%")
    
    print("\n✅ Backtest complete! Logs saved to logs/ directory")
    print("   View logs in Dashboard → Analytics → Logs")


if __name__ == "__main__":
    main()

